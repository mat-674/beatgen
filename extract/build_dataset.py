"""Build the training dataset from ANY folder, recursively.

Discovers Unity OST/DLC bundles and standard BeatSaver map folders, normalizes every
Standard-characteristic beatmap to the canonical schema, computes the log-mel once per
song, and writes a compact, content-independent dataset (no audio is copied):

    dataset/<song>/
        mel.npy            # cached log-mel spectrogram
        <Difficulty>.json  # canonical {notes, bombs, walls}
        meta.json          # bpm, duration, difficulties, source

Usage:
    python extract/build_dataset.py [SRC ...] [--out dataset] [--force]
    python extract/build_dataset.py BeatmapLevelsData
    python extract/build_dataset.py "D:/CustomLevels" BeatmapLevelsData
"""
import argparse
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.audio import SR, log_mel  # noqa: E402
from extract.loaders import discover_levels, normalize_difficulties  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="*", default=["BeatmapLevelsData"], type=Path,
                    help="one or more folders/files to scan recursively")
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--force", action="store_true", help="recompute even if mel.npy exists")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    used_ids: set[str] = set()
    summary, ok, fail = [], 0, 0

    for src in args.src:
        if not Path(src).exists():
            print(f"[skip] {src} does not exist")
            continue
        for level in discover_levels(src):
            try:
                sid = unique_id(level["song_id"], used_ids)
                song_dir = args.out / sid
                mel_path = song_dir / "mel.npy"
                canon = normalize_difficulties(level)
                canon = {k: v for k, v in canon.items() if v["notes"]}
                if not canon:
                    print(f"[empty] {sid}: no notes"); continue

                song_dir.mkdir(parents=True, exist_ok=True)
                if args.force or not mel_path.exists():
                    y = load_mono_22k(level["audio"])
                    np.save(mel_path, log_mel(y))
                    dur = round(len(y) / SR, 2)
                else:
                    dur = round(np.load(mel_path).shape[0] * 512 / SR, 2)

                diffs = []
                for name, c in canon.items():
                    (song_dir / f"{name}.json").write_text(json.dumps(c), encoding="utf-8")
                    diffs.append({"difficulty": name, "notes": len(c["notes"]),
                                  "bombs": len(c["bombs"]), "walls": len(c["walls"])})
                meta = {"song": sid, "bpm": round(level["bpm"], 3), "duration_sec": dur,
                        "source": level["source"], "difficulties": diffs}
                (song_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

                n = sum(d["notes"] for d in diffs)
                print(f"[ok] {sid:32s} bpm={meta['bpm']:>7} diffs={len(diffs)} notes={n}")
                summary.append(meta); ok += 1
            except Exception as e:
                print(f"[FAIL] {level.get('source', '?')}: {e}")
                traceback.print_exc(limit=1); fail += 1

    (args.out / "index.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\ndone: {ok} ok, {fail} failed -> {args.out}")


if __name__ == "__main__":
    main()
