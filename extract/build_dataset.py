"""Build the training dataset from ANY folder, recursively.

Discovers Unity OST/DLC bundles and standard BeatSaver map folders, normalizes every
Standard-characteristic beatmap to the canonical schema, computes the log-mel once per
song, and writes a compact, content-independent dataset (no audio is copied):

    dataset/<song>/
        mel.npy            # cached log-mel spectrogram
        <Difficulty>.json  # canonical {notes, bombs, walls}
        meta.json          # bpm, duration, difficulties, source

Heavy work (UnityPy + librosa log-mel) runs in a multiprocessing pool. Each worker
writes its own mel/JSON/meta to disk so the only thing that crosses the process
boundary is a small status dict. The main process prints a heartbeat every few
seconds so long first tasks never look like a hang.

Usage:
    python extract/build_dataset.py [SRC ...] [--out dataset] [--force]
    python extract/build_dataset.py BeatmapLevelsData
    python extract/build_dataset.py BeatmapLevelsData --debug
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.audio import SR, log_mel  # noqa: E402
from extract.loaders import (  # noqa: E402
    iter_level_tasks,
    load_bundle_level,
    load_map_folder,
    normalize_difficulties,
    slugify,
)


# Use unbuffered stdout everywhere so a single print() never sits in a buffer
# when the parent is waiting to know we're alive. Safe on all platforms.
try:
    sys.stdout.reconfigure(line_buffering=True)                    # py3.7+
except Exception:
    pass


def _emit(msg: str) -> None:
    """Single print helper — always flushes."""
    print(msg, flush=True)


def load_mono_22k(audio) -> np.ndarray:
    kind, val = audio
    if kind == "path":
        import librosa
        y, _ = librosa.load(val, sr=SR, mono=True)
        return y.astype(np.float32)
    # wav_bytes
    import librosa
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(val))
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


def unique_id(base: str, used: set) -> str:
    sid, i = base, 2
    while sid in used:
        sid = f"{base}_{i}"; i += 1
    used.add(sid)
    return sid


def _rss_mb() -> float | None:
    """Best-effort RSS in MB.

    Uses psutil when installed; falls back to ``os.getrusage()`` on Linux/macOS
    (returns None on Windows, where ``getrusage`` is unavailable).
    """
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        pass
    if hasattr(os, "getrusage"):
        try:
            r = os.getrusage(os.RUSAGE_SELF).ru_maxrss
            return r / 1024 if sys.platform != "darwin" else r / (1024 ** 2)
        except Exception:
            return None
    return None


# Picklable worker entry — top-level so it survives spawn().
# Safe under fork too: no import-time threading.Lock / CUDA context / file handles.
def _process_one(args):
    """Process a single (kind, payload) task inside a worker.

    Critical for memory: this worker opens UnityPy / loads WAV only for the ONE
    level it was given, computes the mel, frees the audio bytes, then loops.
    Nothing leaks between tasks.
    """
    task, sid, out_root, force, debug = args
    t0 = time.perf_counter()
    pid = os.getpid()
    rss0 = _rss_mb() if debug else None
    debug_mode = bool(debug)

    def _w(msg: str) -> None:
        if debug_mode:
            print(f"[w/{pid}] {msg}", flush=True)

    song_dir = Path(out_root) / sid
    mel_path = song_dir / "mel.npy"
    meta_path = song_dir / "meta.json"
    if not force and mel_path.exists() and meta_path.exists():
        return {"status": "skip", "sid": sid, "pid": pid,
                "elapsed": time.perf_counter() - t0, "rss_mb": rss0}

    level = None
    try:
        _w(f"start {sid}")
        # ---- 1. Load the level (this is where the big WAV bytes live in RAM) ----
        t1 = time.perf_counter()
        if task[0] == "bundle":
            level = load_bundle_level(Path(task[1]))
        else:
            folder, info_name = task[1]
            level = load_map_folder(Path(folder), info_name)
        _w(f"loaded in {time.perf_counter()-t1:.1f}s")
        if level is None:
            return {"status": "empty", "sid": sid, "pid": pid,
                    "elapsed": time.perf_counter() - t0, "rss_mb": rss0}
        # Use the unique sid the parent assigned (loaders sets its own, but parent wins).
        level["song_id"] = sid
        src = level.get("source", "?")

        # ---- 2. Normalize ----
        t2 = time.perf_counter()
        canon = normalize_difficulties(level)
        canon = {k: v for k, v in canon.items() if v["notes"]}
        _w(f"normalized in {time.perf_counter()-t2:.2f}s ({len(canon)} diffs)")
        if not canon:
            del level
            return {"status": "empty", "sid": sid, "pid": pid,
                    "elapsed": time.perf_counter() - t0, "rss_mb": rss0}

        # ---- 3. Compute & save mel, then DROP the level (frees WAV bytes) ----
        bpm = level.get("bpm") or 120.0
        song_dir.mkdir(parents=True, exist_ok=True)
        if force or not mel_path.exists():
            t_mel = time.perf_counter()
            _w("load_mono_22k ...")
            y = load_mono_22k(level["audio"])
            _w(f"audio loaded: {len(y)/SR:.1f}s, log_mel ...")
            audio_secs = len(y) / SR
            mel = log_mel(y)
            _w(f"mel {mel.shape}, saving ...")
            np.save(mel_path, mel)
            del mel, y                                              # free peak NOW
            mel_secs = time.perf_counter() - t_mel
            _w(f"mel done in {mel_secs:.1f}s")
        else:
            audio_secs = np.load(mel_path).shape[0] * 512 / SR
            mel_secs = None
        del level                                                   # free WAV bytes NOW

        # ---- 4. Write diffs + meta ----
        diffs = []
        for name, c in canon.items():
            (song_dir / f"{name}.json").write_text(json.dumps(c), encoding="utf-8")
            diffs.append({"difficulty": name, "notes": len(c["notes"]),
                          "bombs": len(c["bombs"]), "walls": len(c["walls"])})
        meta = {"song": sid, "bpm": round(bpm, 3),
                "duration_sec": round(audio_secs, 2),
                "source": src, "difficulties": diffs}
        (song_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        _w("done")

        return {"status": "ok", "sid": sid, "pid": pid,
                "elapsed": time.perf_counter() - t0, "rss_mb": _rss_mb() if debug else None,
                "audio_secs": audio_secs, "mel_secs": mel_secs, "diffs": diffs, "meta": meta}
    except Exception as e:
        _w(f"EXC {e}")
        return {"status": "fail", "sid": sid, "pid": pid,
                "elapsed": time.perf_counter() - t0, "rss_mb": rss0,
                "error": str(e), "tb": traceback.format_exc(limit=1)}
    finally:
        # Belt and suspenders: ensure the level (with its big WAV bytes) is dropped
        # even if an exception was swallowed above.
        level = None


def _summarise(result: dict, debug: bool, n_done: int, n_total: int) -> str:
    sid = result["sid"]
    if result["status"] == "ok":
        if debug:
            rss = result.get("rss_mb")
            rss_s = f"{rss:.0f}MB" if isinstance(rss, (int, float)) else "n/a"
            return (f"[ok] {sid:32s} pid={result['pid']} audio={result['audio_secs']:.1f}s "
                    f"mel={result['mel_secs']:.1f}s rss={rss_s} "
                    f"total={result['elapsed']:.1f}s  [{n_done}/{n_total}]")
        n = sum(d["notes"] for d in result["diffs"])
        return (f"[ok] {sid:32s} bpm={result['meta']['bpm']:>7} "
                f"diffs={len(result['diffs'])} notes={n}")
    return f"[{result['status']}] {sid}"


def _format_eta(done: int, total: int, elapsed: float) -> str:
    if done == 0 or elapsed < 1.0:
        return "ETA --:--"
    rate = done / elapsed
    left = max(0, total - done)
    secs = left / rate
    m, s = divmod(int(secs), 60)
    return f"ETA {m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="*", default=["BeatmapLevelsData"], type=Path,
                    help="one or more folders/files to scan recursively")
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--force", action="store_true", help="recompute even if mel.npy exists")
    ap.add_argument("--jobs", "-j", type=int,
                    default=max(1, min(8, os.cpu_count() or 1)),
                    help="worker processes (default: min(8, cpu_count))")
    ap.add_argument("--chunksize", type=int, default=4,
                    help="tasks per worker batch (default: 4)")
    ap.add_argument("--debug", action="store_true",
                    help="print per-task timing, worker PID, RSS, peak RSS at end")
    ap.add_argument("--heartbeat", type=float, default=3.0,
                    help="main-loop heartbeat interval in seconds (default: 3.0)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    used_ids: set[str] = set()
    summary, ok, fail, skip, empty = [], 0, 0, 0, 0
    peak_rss = 0.0
    started = time.perf_counter()

    # Phase 1: lightweight discovery + unique-id assignment on the main process.
    # We DO NOT open UnityPy here — that would pin every WAV in this process and
    # blow up RAM. Workers load each level on demand.
    pending: list[tuple] = []
    for src in args.src:
        if not Path(src).exists():
            _emit(f"[skip] {src} does not exist")
            continue
        for task in iter_level_tasks(src):
            kind, payload = task
            base = slugify(Path(payload[0] if kind == "bundle" else payload[0]).stem)
            sid = unique_id(base, used_ids)
            pending.append((task, sid, args.out, args.force, args.debug))

    if not pending:
        (args.out / "index.json").write_text(json.dumps([], indent=1), encoding="utf-8")
        _emit(f"\ndone: nothing to process -> {args.out}")
        return

    n_total = len(pending)
    n_done = 0
    _emit(f"[..] {n_total} levels, {args.jobs} workers, chunksize={args.chunksize}"
          + (", debug on" if args.debug else ""))

    # Phase 2: parallel heavy work. imap_unordered yields as each task finishes;
    # a tiny background heartbeat thread proves the process is alive during long gaps.
    import multiprocessing as mp
    import threading
    from multiprocessing.pool import Pool

    # fork is fastest on Linux (no per-worker interpreter init), and the worker
    # is fork-safe (top-level function, no import-time state). macOS and Windows
    # require `spawn` because fork can't safely re-exec the main module on them.
    if sys.platform.startswith("linux"):
        ctx = mp.get_context("fork")
    else:
        ctx = mp.get_context("spawn")
    # maxtasksperchild=1: each worker is recycled after every task. This eliminates
    # any chance of silent memory growth inside the worker between tasks AND it
    # makes "is the worker actually doing something?" visible in the log (different
    # pid on every line under --debug).
    pool: Pool = ctx.Pool(processes=args.jobs, maxtasksperchild=1)
    finished_iter = pool.imap_unordered(_process_one, pending, chunksize=args.chunksize)

    stop_evt = threading.Event()
    counts = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    interrupted = False

    def _heartbeat():
        # Runs in a daemon thread. Reads shared counters, prints to stdout.
        # The pool's imap_unordered is driven from the main thread; this thread
        # is purely observational.
        while not stop_evt.is_set():
            stop_evt.wait(args.heartbeat)
            if stop_evt.is_set():
                return
            now = time.perf_counter()
            _emit(f"[..] {n_done}/{n_total} done ({counts['ok']} ok, {counts['skip']} skip, "
                  f"{counts['empty']} empty, {counts['fail']} fail) "
                  f"{_format_eta(n_done, n_total, now - started)}")

    hb_thread = threading.Thread(target=_heartbeat, name="heartbeat", daemon=True)
    hb_thread.start()

    def _accumulate(result: dict) -> None:
        nonlocal n_done, peak_rss, ok, skip, empty, fail
        n_done += 1
        rss = result.get("rss_mb")
        if isinstance(rss, (int, float)) and rss > peak_rss:
            peak_rss = rss
        status = result["status"]
        if status == "ok":
            summary.append(result["meta"]); ok += 1; counts["ok"] += 1
        elif status == "skip":
            skip += 1; counts["skip"] += 1
        elif status == "empty":
            empty += 1; counts["empty"] += 1
        elif status == "fail":
            fail += 1; counts["fail"] += 1
        line = _summarise(result, args.debug, n_done, n_total)
        tail = f"  {_format_eta(n_done, n_total, time.perf_counter() - started)}"
        _emit(line + tail if status == "ok" else line)
        if args.debug and status == "fail" and result.get("tb"):
            _emit(result["tb"].rstrip())

    try:
        for result in finished_iter:
            _accumulate(result)
    except KeyboardInterrupt:
        interrupted = True
        _emit("\n[!] Ctrl+C received, terminating workers ...")
        # Stop the heartbeat thread ASAP so it doesn't keep printing after we exit.
        stop_evt.set()
        # Don't try to drain — workers may be in long mel/stft and we want out now.
        # terminate() is sync; it returns once all workers are actually dead.
        try:
            pool.terminate()
        except Exception:
            pass
        try:
            hb_thread.join(timeout=1.0)
        except Exception:
            pass
        # Print the partial result and exit cleanly (don't re-raise — no traceback spam).
        (args.out / "index.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        wall = time.perf_counter() - started
        _emit(f"\ninterrupted: {ok} ok, {skip} skip, {empty} empty, {fail} failed "
              f"after {wall:.1f}s -> {args.out}")
        return
    except Exception as e:
        # Anything else — also terminate to avoid zombies, then re-raise.
        try:
            pool.terminate()
        except Exception:
            pass
        raise
    finally:
        stop_evt.set()
        if not interrupted:
            try:
                hb_thread.join(timeout=1.0)
            except Exception:
                pass
            try:
                pool.close()
                pool.join()
            except Exception:
                pass

    (args.out / "index.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    wall = time.perf_counter() - started
    tail = f", peak worker RSS={peak_rss:.0f}MB" if peak_rss > 0 else ""
    _emit(f"\ndone: {ok} ok, {skip} skip, {empty} empty, {fail} failed "
          f"in {wall:.1f}s -> {args.out}{tail}")


if __name__ == "__main__":
    main()