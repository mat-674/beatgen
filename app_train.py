"""Gradio tab for training / fine-tuning stage1 + stage2 in the background.

Run as part of `app.py` — exposes a `build_train_tab()` that returns a gr.Column.

Design notes:
  * Training happens in a child Python process (subprocess + threading) so the UI
    stays responsive and the model weights are isolated from the inference tab.
  * The child's stdout is line-buffered (we spawn with -u) and parsed for known
    metric prefixes. Parsed metrics flow into a small gr.LinePlot, everything
    else flows into a scrolling log.
  * On exit the latest checkpoint path is reported back. The UI doesn't try to
    hot-swap the inference cache — the user just refreshes the Generate tab.
  * Only one training run at a time. We keep `proc` as a singleton attribute on
    the runner; a second Start while one is running is rejected in the UI.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "out" / "ui" / "train_state.json"

# Bound the in-memory log so the UI JSON state and the timer tick stay cheap.
LOG_MAX = 2000
LOG_KEEP = 1500

# Default device mirror: "cuda" if torch sees a GPU, else "cpu".
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Metric-line parsers. We anchor on the [stageN] prefix that stage1.py/stage2.py
# print, and capture only the named groups. Anything that doesn't match lands
# in the live log unchanged.
RE_STAGE1_EVAL = re.compile(
    r"\[stage1\]\s+ep\s+(\d+)\s+loss\s+([\d.]+)\s*\|\s*val\s+P\s+([\d.]+)\s+R\s+([\d.]+)\s+F1\s+([\d.]+)"
)
RE_STAGE2_EVAL = re.compile(
    r"\[stage2\]\s+ep\s+(\d+)\s+loss\s+([\d.]+)\s*\|\s*val\s+note-acc\s+([\d.]+)"
    r"(?:\s+bal-acc\s+([\d.]+))?"
)
RE_STAGE2_BALANCE = re.compile(
    r"\[stage2\]\s+colour balance:\s*red\s+(\d+)\s*/\s*blue\s+(\d+)\s*of\s+(\d+).*"
    r"pos_w\s+red\s+([\d.]+)\s+blue\s+([\d.]+)"
)
RE_STAGE2_XHIST = re.compile(
    r"\[stage2\]\s+x-hist\s+red\s+(\{[^}]*\})\s*\|\s*blue\s+(\{[^}]*\})"
)
RE_SAVE = re.compile(r"\[(stage\d)\]\s+saved\s+->\s+(.+)")
RE_RESUME = re.compile(r"\[(stage\d)\]\s+resumed\s+from\s+(.+)")
# Best checkpoint line printed only when a new F1/bal-acc record is set
# (see stage1.py / stage2.py). Captures the metric name + value + path so the
# UI can surface it next to `latest`.
RE_BEST = re.compile(r"\[(stage\d)\]\s+.*?\bbest\s+(\S+)\s+([\d.]+)\s+->\s+(.+)$")

RE_BUILD_OK = re.compile(r"^\[ok\]\s+(\S+)\s+bpm=([\d.]+).*notes=(\d+)")
RE_BUILD_FAIL = re.compile(r"^\[(FAIL|empty)\]\s+(\S+)")
# The build_dataset.py script prints `done: N ok, M failed` at the very end.
# When this line arrives, its totals ARE authoritative — if the last `[ok]` lines
# came in but the build crashed before that, per-line counters will undercount.
RE_BUILD_DONE = re.compile(r"^done:\s+(\d+)\s+ok,\s+(\d+)\s+failed")

MODES = ["Cold start", "Fine-tune", "Pipeline"]


def _push_log(log: list[str], line: str) -> None:
    log.append(line)
    if len(log) > LOG_MAX:
        del log[: len(log) - LOG_KEEP]


@dataclass
class TrainRunner:
    """Holds the child process and live state for a single training run."""

    proc: subprocess.Popen | None = None
    stage: str = ""
    started_at: float = 0.0
    log: list[str] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    colour: dict | None = None
    last_ckpt: str = ""        # most recent `.latest.pt` (may have overfit)
    last_best: dict | None = None   # {"metric": "F1", "value": 0.56, "path": "..."} from `best F1 0.56 -> ...`
    status: str = "idle"   # idle | running | done | failed | stopped
    was_stopped: bool = False  # set by stop() so _finalize can distinguish user-Stop from a crash
    _reader: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, stage: str, args: list[str]):
        if self.is_running():
            raise RuntimeError("Training is already running. Stop it first.")
        with self._lock:
            self.stage = stage
            self.started_at = time.time()
            self.was_stopped = False
            self.log = [f"[ui] launching: python -u models/{stage}.py {' '.join(args)}"]
            self.metrics = []
            self.colour = None
            self.last_ckpt = ""
            self.last_best = None
            self.status = "running"
        cmd = [sys.executable, "-u", str(ROOT / "models" / f"{stage}.py"), *args]
        # -u so the child flushes stdout line-by-line into our pipe.
        self.proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._persist()

    def stop(self, timeout: float = 5.0):
        if not self.is_running():
            return
        with self._lock:
            self.was_stopped = True
            _push_log(self.log, "[ui] stop requested, terminating…")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._finalize()

    def _pump(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            with self._lock:
                _push_log(self.log, line)
                self._parse(line)
        self._finalize()

    def _parse(self, line: str):
        m = RE_STAGE1_EVAL.search(line)
        if m:
            self.metrics.append({"ep": int(m.group(1)), "loss": float(m.group(2)),
                                 "P": float(m.group(3)), "R": float(m.group(4)),
                                 "F1": float(m.group(5))})
            return
        m = RE_STAGE2_EVAL.search(line)
        if m:
            self.metrics.append({"ep": int(m.group(1)), "loss": float(m.group(2)),
                                 "note_acc": float(m.group(3)),
                                 "bal_acc": float(m.group(4)) if m.group(4) else None})
            return
        m = RE_STAGE2_BALANCE.search(line)
        if m:
            self.colour = {"r_pos": int(m.group(1)), "b_pos": int(m.group(2)),
                           "steps": int(m.group(3)), "rpw": float(m.group(4)),
                           "bpw": float(m.group(5))}
            return
        m = RE_STAGE2_XHIST.search(line)
        if m:
            try:
                xr = ast.literal_eval(m.group(1))
                xb = ast.literal_eval(m.group(2))
                self.colour = (self.colour or {}) | {"x_hist_red": xr, "x_hist_blue": xb}
            except (ValueError, SyntaxError):
                pass
            return
        m = RE_SAVE.search(line)
        if m:
            self.last_ckpt = m.group(2).strip()
            return
        m = RE_BEST.search(line)
        if m:
            self.last_best = {"metric": m.group(2), "value": float(m.group(3)),
                              "path": m.group(4).strip()}
            return
        m = RE_RESUME.search(line)
        if m:
            self.log.append(f"[ui] (warm-start from {m.group(2).strip()})")

    def _finalize(self):
        rc = self.proc.returncode if self.proc else -1
        with self._lock:
            if self.was_stopped:
                self.status = "stopped"
            elif rc == 0:
                self.status = "done"
            else:
                self.status = "failed"
            _push_log(self.log, f"[ui] exited with code {rc}")
        self.proc = None
        self._persist()

    def _persist(self):
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps({
                "stage": self.stage,
                "started_at": self.started_at,
                "status": self.status,
                "last_ckpt": self.last_ckpt,
            }, indent=1), encoding="utf-8")
        except OSError:
            pass   # non-fatal; UI just loses the cross-reload marker

    def snapshot(self) -> dict:
        with self._lock:
            return {"status": self.status, "stage": self.stage,
                    "log_tail": self.log[-50:], "metrics": list(self.metrics),
                    "colour": self.colour, "last_ckpt": self.last_ckpt,
                    "last_best": self.last_best,
                    "running": self.is_running()}


RUNNER = TrainRunner()


# --- Build-dataset runner -----------------------------------------------------
# Same shape as TrainRunner, but for `extract/build_dataset.py`.

@dataclass
class BuildRunner:
    """Holds the child process and live state for a single build_dataset run."""

    proc: subprocess.Popen | None = None
    started_at: float = 0.0
    log: list[str] = field(default_factory=list)
    ok: int = 0
    fail: int = 0
    last_out: str = ""
    status: str = "idle"   # idle | running | done | failed
    was_stopped: bool = False
    _reader: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, srcs: list[str], out_dir: str, force: bool):
        if self.is_running():
            raise RuntimeError("Build is already running. Stop it first.")
        with self._lock:
            self.started_at = time.time()
            self.was_stopped = False
            self.log = [f"[ui] launching: python -u extract/build_dataset.py "
                        f"{' '.join(srcs)} --out {out_dir}" + (" --force" if force else "")]
            self.ok = 0
            self.fail = 0
            self.last_out = out_dir
            self.status = "running"
        cmd = [sys.executable, "-u", str(ROOT / "extract" / "build_dataset.py"),
               *srcs, "--out", out_dir]
        if force:
            cmd.append("--force")
        self.proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def stop(self, timeout: float = 5.0):
        if not self.is_running():
            return
        with self._lock:
            self.was_stopped = True
            _push_log(self.log, "[ui] stop requested, terminating…")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._finalize()

    def _pump(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            with self._lock:
                _push_log(self.log, line)
                self._parse(line)
        self._finalize()

    def _parse(self, line: str):
        if RE_BUILD_DONE.search(line):
            # Authoritative summary line: trust it over the per-line counters.
            m = RE_BUILD_DONE.search(line)
            self.ok = int(m.group(1))
            self.fail = int(m.group(2))
        elif RE_BUILD_OK.search(line):
            self.ok += 1
        elif RE_BUILD_FAIL.search(line):
            self.fail += 1
        # Per-line `[ok]` / `[FAIL]` lines tick the counter live while the
        # build runs; the final `done: N ok, M failed` line overrides them.
        # If the build crashes before `done:` arrives (no summary line), the
        # per-line count is the best estimate we have.

    def _finalize(self):
        rc = self.proc.returncode if self.proc else -1
        with self._lock:
            if self.was_stopped:
                self.status = "stopped"
            elif rc == 0:
                self.status = "done"
            else:
                self.status = "failed"
            _push_log(self.log, f"[ui] exited with code {rc}")
        self.proc = None

    def snapshot(self) -> dict:
        with self._lock:
            return {"status": self.status, "ok": self.ok, "fail": self.fail,
                    "last_out": self.last_out, "log_tail": self.log[-50:],
                    "running": self.is_running()}


BUILDER = BuildRunner()


# --- End-to-end pipeline runner ----------------------------------------------
# Chains BUILDER + RUNNER(stage1) + RUNNER(stage2) sequentially in one
# background thread. Reuses the existing runners so the user still gets the
# same per-step metrics / logs / checkpoint behaviour as running each step
# manually.

PIPELINE_STEPS = ("build", "stage1", "stage2")


@dataclass
class PipelineRunner:
    """Orchestrates the full build -> stage1 -> stage2 flow as one background job."""

    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = 0.0
    step: str = ""            # current step ("build" / "stage1" / "stage2") or ""
    step_idx: int = -1        # 0..len(PIPELINE_STEPS)-1, -1 when idle
    log: list[str] = field(default_factory=list)
    status: str = "idle"      # idle | running | done | failed | stopped
    last_ckpt: str = ""
    error: str = ""

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            return {"status": self.status, "step": self.step,
                    "step_idx": self.step_idx, "last_ckpt": self.last_ckpt,
                    "error": self.error, "log_tail": self.log[-50:],
                    "running": self.is_running()}

    def start(self, raw_dirs: list[str], out_dir: str, force: bool,
              epochs1: int, epochs2: int, steps1: int, bs1: int,
              lr1: float, lr2: float, device: str) -> None:
        if self.is_running():
            raise RuntimeError("Pipeline is already running.")
        if BUILDER.is_running() or RUNNER.is_running():
            raise RuntimeError("Build or Train is already running separately. "
                               "Stop it before launching the pipeline.")
        with self._lock:
            self.started_at = time.time()
            self.step = ""
            self.step_idx = -1
            self.log = [f"[ui] launching pipeline: build -> stage1 ({epochs1}ep) -> stage2 ({epochs2}ep)"]
            self.status = "running"
            self.last_ckpt = ""
            self.error = ""
        self._thread = threading.Thread(
            target=self._run, args=(raw_dirs, out_dir, force, epochs1, epochs2,
                                    steps1, bs1, lr1, lr2, device),
            daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request termination. Stops whatever sub-runner is active right now."""
        with self._lock:
            self.log.append("[ui] pipeline stop requested")
        if BUILDER.is_running():
            BUILDER.stop()
        if RUNNER.is_running():
            RUNNER.stop()

    def _run(self, raw_dirs, out_dir, force, epochs1, epochs2, steps1, bs1,
             lr1, lr2, device) -> None:
        try:
            with self._lock:
                self.step = "build"; self.step_idx = 0
                _push_log(self.log, f"[pipeline] step 1/3 — build_dataset -> {out_dir}"
                          + (" (force)" if force else ""))
            BUILDER.start(raw_dirs, out_dir, force)
            rc = self._wait_subprocess(BUILDER, "build")
            if rc != 0:
                raise RuntimeError(f"build_dataset exited with code {rc}")
            bsnap = BUILDER.snapshot()
            with self._lock:
                _push_log(self.log, f"[pipeline] build finished: {bsnap['ok']} ok, {bsnap['fail']} failed")

            with self._lock:
                self.step = "stage1"; self.step_idx = 1
            args1 = ["--epochs", str(int(epochs1)), "--data", out_dir,
                     "--device", device, "--lr", str(float(lr1)),
                     "--steps", str(int(steps1)), "--bs", str(int(bs1))]
            with self._lock:
                _push_log(self.log, f"[pipeline] step 2/3 — stage1: python -u models/stage1.py {' '.join(args1)}")
            RUNNER.start("stage1", args1)
            rc = self._wait_subprocess(RUNNER, "stage1")
            if rc != 0:
                raise RuntimeError(f"stage1 exited with code {rc}")
            ckpt1 = RUNNER.snapshot().get("last_ckpt") or ""
            with self._lock:
                self.last_ckpt = ckpt1
                _push_log(self.log, f"[pipeline] stage1 finished, last ckpt: {ckpt1}")

            with self._lock:
                self.step = "stage2"; self.step_idx = 2
            args2 = ["--epochs", str(int(epochs2)), "--data", out_dir,
                     "--device", device, "--lr", str(float(lr2))]
            if ckpt1:
                args2 += ["--resume", str(Path(ckpt1))]
                with self._lock:
                    _push_log(self.log, f"[pipeline] stage2 will warm-start from {ckpt1}")
            with self._lock:
                _push_log(self.log, f"[pipeline] step 3/3 — stage2: python -u models/stage2.py {' '.join(args2)}")
            RUNNER.start("stage2", args2)
            rc = self._wait_subprocess(RUNNER, "stage2")
            if rc != 0:
                raise RuntimeError(f"stage2 exited with code {rc}")
            ckpt2 = RUNNER.snapshot().get("last_ckpt") or ""
            with self._lock:
                self.last_ckpt = ckpt2
                _push_log(self.log, f"[pipeline] stage2 finished, last ckpt: {ckpt2}")
                self.status = "done"; self.step = ""; self.step_idx = -1
                _push_log(self.log, "[pipeline] all steps complete ✅")
        except Exception as e:
            with self._lock:
                self.status = "failed"; self.error = str(e)
                self.step = ""; self.step_idx = -1
                _push_log(self.log, f"[pipeline] FAILED: {e}")

    def _wait_subprocess(self, runner, label: str) -> int:
        """Wait until `runner`'s child exits and return its returncode.

        Polls `runner.is_running()` rather than `proc.poll()` directly so we
        don't race the sub-runner's `_finalize`, which sets `self.proc = None`.
        After the child exits we still need to wait briefly for `_finalize`
        to populate `runner.proc.returncode` before reading it.
        """
        while runner.is_running():
            time.sleep(0.5)
        for _ in range(20):                  # up to ~2s for _finalize to run
            if runner.proc is None:
                break
            time.sleep(0.1)
        # mirror the tail of the sub-runner's log so the user sees step boundaries
        snap_log = (BUILDER.snapshot().get("log_tail") if label == "build"
                    else RUNNER.snapshot().get("log_tail"))
        if snap_log:
            tail = snap_log[-1]
            with self._lock:
                if not self.log or self.log[-1] != tail:
                    _push_log(self.log, f"[{label}] {tail}")
        return runner.proc.returncode if runner.proc else -1


PIPELINE = PipelineRunner()


# --- Status formatters --------------------------------------------------------

def _format_status_block(status: str, *, running: str, done: str, failed: str,
                         idle: str, footer: str = "") -> str:
    head = {"running": running, "done": done, "failed": failed}.get(status, idle)
    if footer and status != "idle":
        return head + "\n" + footer
    return head


def _format_pipeline_status(snap: dict) -> str:
    status = snap["status"]
    if status == "running":
        idx = snap["step_idx"]; step = snap["step"]
        cur = idx + 1 if idx >= 0 else "?"
        total = len(PIPELINE_STEPS)
        running = f"### 🟡 Pipeline running — step {cur}/{total}: `{step}`"
    elif status == "done":
        running = f"### ✅ Pipeline done\n**Latest checkpoint:** `{snap.get('last_ckpt','')}`"
    elif status == "failed":
        running = (f"### ❌ Pipeline failed\n**Error:** {snap.get('error','')}\n"
                   f"**Last ckpt:** `{snap.get('last_ckpt','')}`")
    else:
        running = "### ⏹ Pipeline stopped" if status == "stopped" else "### ⚪ Idle"
    return running


def _format_build_status(snap: dict) -> str:
    status = snap["status"]
    running = f"### 🟡 Building dataset… ({snap['ok']} ok / {snap['fail']} fail)"
    done = f"### ✅ Build done — {snap['ok']} ok, {snap['fail']} failed"
    failed = f"### ❌ Build failed at {snap['ok']} ok / {snap['fail']} fail"
    stopped = f"### ⏹ Stopped at {snap['ok']} ok / {snap['fail']} fail"
    out = snap.get("last_out") or ""
    footer = f"**Output folder:** `{out}`" if out else ""
    if status == "stopped":
        return stopped + (("\n" + footer) if footer and status != "idle" else "")
    return _format_status_block(
        status, running=running, done=done, failed=failed,
        idle="### ⚪ Idle", footer=footer,
    )


def _format_status(snap: dict) -> str:
    status = snap["status"]
    stage = snap["stage"] or "—"
    ckpt = snap["last_ckpt"]
    best = snap.get("last_best")
    running = f"### 🟡 Running `{stage}`"
    done = f"### ✅ Done (`{stage}`)"
    failed = f"### ❌ Failed (`{stage}`)"
    stopped = f"### ⏹ Stopped (`{stage}`)"
    footers = []
    if ckpt:
        footers.append(f"**Latest checkpoint:** `{ckpt}`")
    if best:
        footers.append(f"**Best {best['metric']} {best['value']:.3f}:** `{best['path']}`")
    footer = "\n".join(footers)
    if status == "stopped":
        return stopped + (("\n" + footer) if footer else "")
    return _format_status_block(
        status, running=running, done=done, failed=failed,
        idle="### ⚪ Idle", footer=footer,
    )


def _metrics_to_plot(metrics: list[dict], stage: str) -> "pd.DataFrame":
    """Convert the flat metrics list into a long-form DataFrame for gr.LinePlot."""
    rows = []
    for m in metrics:
        ep = m["ep"]
        rows.append({"epoch": ep, "metric": "loss", "stage": stage, "value": m["loss"]})
        if stage == "stage1" and "F1" in m:
            rows.append({"epoch": ep, "metric": "val F1", "stage": stage, "value": m["F1"]})
        if stage == "stage2" and "note_acc" in m:
            rows.append({"epoch": ep, "metric": "val note-acc", "stage": stage,
                         "value": m["note_acc"]})
        if stage == "stage2" and m.get("bal_acc") is not None:
            rows.append({"epoch": ep, "metric": "val bal-acc", "stage": stage,
                         "value": m["bal_acc"]})
    return pd.DataFrame(rows, columns=["epoch", "metric", "stage", "value"])


def _format_colour(colour: dict | None) -> str:
    if not colour:
        return ""
    lines = [f"**Red:** {colour.get('r_pos', '?')}    **Blue:** {colour.get('b_pos', '?')}    "
             f"**Steps:** {colour.get('steps', '?')}    "
             f"**pos_w red/blue:** {colour.get('rpw', '?')} / {colour.get('bpw', '?')}"]
    xr = colour.get("x_hist_red"); xb = colour.get("x_hist_blue")
    if xr and xb:
        lines.append("\n**Column distribution (red):**")
        total = sum(xr.values()) or 1
        for col in range(4):
            pct = xr.get(col, 0) / total * 100
            bar = "█" * int(pct // 4)
            lines.append(f"  col {col}: {pct:5.1f}%  {bar}")
        lines.append("**Column distribution (blue):**")
        total = sum(xb.values()) or 1
        for col in range(4):
            pct = xb.get(col, 0) / total * 100
            bar = "█" * int(pct // 4)
            lines.append(f"  col {col}: {pct:5.1f}%  {bar}")
        # "any column over 50%?" — the classic Stage 2 collapse signature
        for label, hist in (("red", xr), ("blue", xb)):
            total = sum(hist.values()) or 1
            top = max(hist.values()) / total
            if top > 0.5:
                lines.append(f"\n⚠️  **{label}** is collapsed on one column ({top*100:.0f}%) — "
                             "the classic Stage 2 failure mode. Try a different seed or more data.")
    return "\n".join(lines)


# --- Click handlers -----------------------------------------------------------

def _err(msg: str, log: str = "") -> tuple[str, str]:
    return f"### ❌ {msg}", log


def _split_sources(raw: str) -> list[str]:
    return [s.strip() for s in re.split(r"[;,]", raw or "") if s.strip()]


def start_click(mode: str, stage: str, data_dir: str, epochs: int,
                steps: int, bs: int, lr: float, device: str,
                resume: str,
                epochs1: int, epochs2: int, lr1: float, lr2: float,
                raw_dir: str, force: bool):
    """Unified Start: Cold start, Fine-tune, or full Pipeline.

    Each mode validates its own inputs and dispatches to RUNNER.start or
    PIPELINE.start. Build lives in its own accordion and runs separately.
    """
    if RUNNER.is_running() or PIPELINE.is_running():
        rsnap = RUNNER.snapshot()
        psnap = PIPELINE.snapshot()
        if psnap["status"] == "running":
            return (_format_pipeline_status(psnap), "\n".join(psnap["log_tail"]),
                    _metrics_to_plot([], ""), "")
        return (_format_status(rsnap), "\n".join(rsnap["log_tail"]),
                _metrics_to_plot(rsnap["metrics"], rsnap["stage"]),
                _format_colour(rsnap["colour"]))

    if mode == "Pipeline":
        if not raw_dir.strip():
            return (*_err("Raw bundles path is empty"), "", _metrics_to_plot([], ""), "")
        srcs = _split_sources(raw_dir)
        missing = [s for s in srcs if not Path(s).is_dir()]
        if missing:
            return (*_err(f"Not a directory: `{'`, `'.join(missing)}`"), "", _metrics_to_plot([], ""), "")
        out_dir = data_dir.strip() or "dataset"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            PIPELINE.start(srcs, out_dir, bool(force),
                           int(epochs1), int(epochs2), int(steps), int(bs),
                           float(lr1), float(lr2), device)
        except RuntimeError as e:
            return (*_err(f"Failed to start pipeline: {e}"), "", _metrics_to_plot([], ""), "")
        snap = PIPELINE.snapshot()
        return (_format_pipeline_status(snap), "\n".join(snap["log_tail"]),
                _metrics_to_plot([], ""), _format_colour(None))

    # Cold start / Fine-tune -> a single-stage train run via TrainRunner.
    if not (data_dir or "").strip():
        return (*_err("Dataset path is empty — point it at the built dataset "
                      "folder (each song dir holds mel.npy + <Difficulty>.json)"), "",
                _metrics_to_plot([], ""), "")
    p = Path(data_dir)
    if not p.is_dir():
        return (*_err(f"`{p}` is not a directory. If you have raw .dat/UnityFS "
                      "bundles, first run: `python extract/build_dataset.py <SRC> "
                      f"--out {data_dir}`"), "", _metrics_to_plot([], ""), "")
    args = ["--epochs", str(int(epochs)), "--data", str(p),
            "--device", device, "--lr", str(float(lr))]
    if stage == "stage1":
        args += ["--steps", str(int(steps)), "--bs", str(int(bs))]
    if mode == "Fine-tune":
        if not resume or not Path(resume).is_file():
            return (*_err("Fine-tune needs a valid `--resume` path to a .pt"), "",
                    _metrics_to_plot([], ""), "")
        args += ["--resume", str(resume)]
    try:
        RUNNER.start(stage, args)
    except RuntimeError as e:
        return (*_err(f"Failed to start: {e}"), "", _metrics_to_plot([], ""), "")
    snap = RUNNER.snapshot()
    return (_format_status(snap), "\n".join(snap["log_tail"]),
            _metrics_to_plot(snap["metrics"], snap["stage"]), _format_colour(snap["colour"]))


def stop_click():
    if PIPELINE.is_running():
        PIPELINE.stop()
    RUNNER.stop()
    snap = RUNNER.snapshot()
    psnap = PIPELINE.snapshot()
    return (_format_status(snap), "\n".join(snap["log_tail"]),
            _metrics_to_plot(snap["metrics"], snap["stage"]),
            _format_pipeline_status(psnap))


def build_click(raw_dir: str, out_dir: str, force: bool):
    """Run extract/build_dataset.py; the output folder becomes your Dataset path."""
    if BUILDER.is_running():
        snap = BUILDER.snapshot()
        return (_format_build_status(snap), "\n".join(snap["log_tail"]))
    if not (raw_dir or "").strip():
        return _err("Raw bundles path is empty — point it at your Unity OST/DLC "
                    "bundles or BeatSaver map folders (multiple separated by `,` or `;`)")
    if not (out_dir or "").strip():
        return _err("Output folder is empty — this is where the built dataset lands; "
                    "paste it into `Dataset path` once the build finishes")
    srcs = _split_sources(raw_dir)
    missing = [s for s in srcs if not Path(s).is_dir()]
    if missing:
        return _err(f"Not a directory: `{'`, `'.join(missing)}`")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        BUILDER.start(srcs, out_dir, bool(force))
    except RuntimeError as e:
        return _err(f"Failed to start build: {e}")
    snap = BUILDER.snapshot()
    return (_format_build_status(snap), "\n".join(snap["log_tail"]))


def build_stop_click():
    BUILDER.stop()
    snap = BUILDER.snapshot()
    return (_format_build_status(snap), "\n".join(snap["log_tail"]))


def refresh_handlers() -> dict:
    """Snapshot all three runners for the Gradio Timer (single tick drives both tabs)."""
    r = RUNNER.snapshot()
    b = BUILDER.snapshot()
    p = PIPELINE.snapshot()
    return {
        "status_value":      _format_status(r),
        "log_value":         "\n".join(r["log_tail"]),
        "plot_value":        _metrics_to_plot(r["metrics"], r["stage"]),
        "colour_value":      _format_colour(r["colour"]),
        "build_status_value": _format_build_status(b),
        "build_log_value":    "\n".join(b["log_tail"]),
        "pipeline_status_value": _format_pipeline_status(p),
        "pipeline_log_value":    "\n".join(p["log_tail"]),
    }


# --- UI -----------------------------------------------------------------------

# Short, inline version of the training guide. Full version (per-stage
# knob recommendations, log-line interpretation, troubleshooting) lives in README.md.
RECOMMENDATIONS_SHORT = """
### Что крутить
- **Cold start**: epochs 40-60, lr 1e-3, bs 16 (stage1) — дефолты в коде разумные.
- **Fine-tune**: lr 3e-4 (в 3-5× ниже, чтобы не «забыть» выученное), 10-20 эпох обычно хватает.
- **Маленький датасет (<50 уровней)**: уменьшите stage1 `steps` до 30-40, иначе overfit за 1 эпоху.
- **Нет CUDA**: ожидайте минуты на эпоху. Stage 1 (TCN) обычно медленнее Stage 2 (GRU).

### На что смотреть
- **Stage 1 — `val F1`**: должен расти. 3 эпохи без роста → early stop.
- **Stage 2 — `colour balance`**: если red и blue различаются >3× — данные перекошены.
- **Stage 2 — `x-hist`**: если один столбец > 50% — модель коллапсировала в одну колонку.
- **Stage 2 — `val note-acc`**: должна расти к 0.7+. Застряла на 0.5 → модель не отличает цвет/колонку.

Полная версия (построчный разбор логов, troubleshooting): README.md → "Training tips".
"""


def build_train_tab() -> dict:
    """Build the Train tab. The block is appended to the existing app.

    Layout — one screen, one button:
      • Dataset   — where bundles live, where to put the built dataset, Build button
      • Train     — mode radio (Cold / Fine-tune / Pipeline) + the few fields
                    that change per mode + Start/Stop + live status/log/plot/colour

    Returns a dict of named handles; the Timer in app.py reads them on every tick.
    """
    # Most recent state file so the tab doesn't claim "Idle" after a reload.
    initial_snap = RUNNER.snapshot()
    if STATE_PATH.exists():
        try:
            persisted = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            initial_snap["status"] = persisted.get("status", initial_snap["status"])
            initial_snap["stage"] = persisted.get("stage", initial_snap["stage"])
            initial_snap["last_ckpt"] = persisted.get("last_ckpt", initial_snap["last_ckpt"])
        except (OSError, json.JSONDecodeError):
            pass

    gr.Markdown(
        "# 🏋️ Train / Fine-tune\n"
        "Один экран с тремя режимами: **Cold start** (обучение с нуля), "
        "**Fine-tune** (warm-start от чекпойнта), **Pipeline** (build → train → train сам). "
        "Чекпойнты сохраняются с timestamp-бэкапом в `models/_ckpt/`.\n\n"
        "**Dataset path** — папка **уже собранного** датасета (в каждой подпапке "
        "`mel.npy` + `<Difficulty>.json`). Если у тебя только сырые бандлы — "
        "сначала собери их ниже в секции «Dataset»."
    )

    # ---------------- Dataset section ---------------------------------------
    with gr.Accordion("📦 Dataset (raw bundles → built dataset)", open=False):
        with gr.Row():
            with gr.Column():
                d_raw = gr.Textbox(label="Raw bundles folder(s)",
                                   placeholder="BeatmapLevelsData   |   D:/maps/raw",
                                   value="BeatmapLevelsData")
                d_out = gr.Textbox(label="Output (built dataset) folder",
                                   placeholder="dataset",
                                   value="dataset")
                d_force = gr.Checkbox(value=False, label="Force rebuild (recompute mel.npy)")
                with gr.Row():
                    d_build = gr.Button("Build dataset", variant="primary")
                    d_stop = gr.Button("Stop", variant="stop")
            with gr.Column():
                build_status = gr.Markdown(_format_build_status(BUILDER.snapshot()))
                build_log = gr.Textbox(label="Build log (last 50 lines)", lines=8,
                                       max_lines=8, autoscroll=True, interactive=False,
                                       value="\n".join(BUILDER.snapshot()["log_tail"]))
        d_build.click(build_click, [d_raw, d_out, d_force],
                      [build_status, build_log])
        d_stop.click(build_stop_click, outputs=[build_status, build_log])

    # ---------------- Train section -----------------------------------------
    with gr.Accordion("🧠 Train (Cold start / Fine-tune / Pipeline)", open=True):
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(MODES, value="Cold start", label="Mode")
                # Pipeline-specific: raw bundles + output + per-stage epochs + LRs
                with gr.Group(visible=False) as p_grp:
                    p_raw = gr.Textbox(label="Raw bundles folder(s) (Pipeline mode)",
                                       placeholder="BeatmapLevelsData",
                                       value="BeatmapLevelsData")
                    p_force = gr.Checkbox(value=False,
                                          label="Force rebuild (recompute mel.npy)")
                    with gr.Row():
                        p_epochs1 = gr.Slider(1, 200, value=40, step=1, label="Stage1 epochs")
                        p_epochs2 = gr.Slider(1, 200, value=30, step=1, label="Stage2 epochs")
                    with gr.Row():
                        p_lr1 = gr.Number(value=1e-3, label="LR stage1", precision=4)
                        p_lr2 = gr.Number(value=1e-3, label="LR stage2", precision=4)
                # Single-stage fields (Cold start / Fine-tune)
                with gr.Group() as s_grp:
                    data_dir = gr.Textbox(
                        label="Dataset path (built dataset: mel.npy + *.json per song)",
                        placeholder="dataset   |   C:/data/my_dataset",
                        value="dataset")
                    stage = gr.Radio(["stage1", "stage2"], value="stage1", label="Stage")
                    epochs = gr.Slider(1, 200, value=40, step=1, label="Epochs")
                    with gr.Row():
                        steps = gr.Slider(4, 400, value=80, step=4, label="Steps/epoch (stage1)")
                        bs = gr.Slider(2, 64, value=16, step=2, label="Batch size (stage1)")
                    lr = gr.Number(value=1e-3, label="Learning rate", precision=4)
                    resume = gr.Textbox(label="Resume .pt path (Fine-tune only)",
                                        placeholder="models/_ckpt/stage1.latest.pt")
                device = gr.Radio(["cpu", "cuda"], value=DEFAULT_DEVICE, label="Device")
                with gr.Row():
                    go = gr.Button("Start", variant="primary")
                    stop = gr.Button("Stop", variant="stop")

            with gr.Column():
                status = gr.Markdown(_format_status(initial_snap))
                with gr.Tabs():
                    with gr.Tab("Live log"):
                        log = gr.Textbox(label="Live log (last 50 lines)", lines=18,
                                         max_lines=18, autoscroll=True, interactive=False,
                                         value="\n".join(initial_snap["log_tail"]))
                    with gr.Tab("Metrics"):
                        plot = gr.LinePlot(
                            x="epoch", y="value", color="metric",
                            title="Metrics per epoch", x_title="epoch", y_title="value",
                            value=_metrics_to_plot(initial_snap["metrics"],
                                                  initial_snap["stage"]))
                colour = gr.Markdown(_format_colour(initial_snap["colour"]))

    with gr.Accordion("📖 Recommendations", open=False):
        gr.Markdown(RECOMMENDATIONS_SHORT)

    # --- visibility: switch which input group is shown based on mode ---------
    def _toggle_group(mode_value: str):
        is_pipeline = mode_value == "Pipeline"
        return gr.update(visible=is_pipeline), gr.update(visible=not is_pipeline)

    mode.change(_toggle_group, inputs=[mode], outputs=[p_grp, s_grp])

    go.click(start_click,
             [mode, stage, data_dir, epochs, steps, bs, lr, device, resume,
              p_epochs1, p_epochs2, p_lr1, p_lr2, p_raw, p_force],
             [status, log, plot, colour])
    stop.click(stop_click, outputs=[status, log, plot, colour])

    return {
        "status": status, "log": log, "plot": plot, "colour": colour,
        "build_status": build_status, "build_log": build_log,
    }