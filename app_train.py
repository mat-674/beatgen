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

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "out" / "ui" / "train_state.json"

# Metric-line parsers. We anchor on the [stageN] prefix that stage1.py/stage2.py
# print, and capture only the named groups. Anything that doesn't match lands
# in the live log unchanged.
RE_STAGE1_EVAL = re.compile(
    r"\[stage1\]\s+ep\s+(\d+)\s+loss\s+([\d.]+)\s*\|\s*val\s+P\s+([\d.]+)\s+R\s+([\d.]+)\s+F1\s+([\d.]+)"
)
RE_STAGE2_EVAL = re.compile(
    r"\[stage2\]\s+ep\s+(\d+)\s+loss\s+([\d.]+)\s*\|\s*val\s+note-acc\s+([\d.]+)"
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


@dataclass
class TrainRunner:
    """Holds the child process and live state for a single training run."""

    proc: subprocess.Popen | None = None
    stage: str = ""
    started_at: float = 0.0
    log: list[str] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    colour: dict | None = None
    last_ckpt: str = ""
    status: str = "idle"   # idle | running | done | failed
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
            self.log = [f"[ui] launching: python -u models/{stage}.py {' '.join(args)}"]
            self.metrics = []
            self.colour = None
            self.last_ckpt = ""
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
            self.log.append("[ui] stop requested, terminating…")
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
                self.log.append(line)
                # keep log bounded so the JSON state file doesn't grow forever
                if len(self.log) > 2000:
                    self.log = self.log[-1500:]
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
                                 "note_acc": float(m.group(3))})
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
                xr = eval(m.group(1))
                xb = eval(m.group(2))
                self.colour = (self.colour or {}) | {"x_hist_red": xr, "x_hist_blue": xb}
            except Exception:
                pass
            return
        m = RE_SAVE.search(line)
        if m:
            self.last_ckpt = m.group(2).strip()
            return
        m = RE_RESUME.search(line)
        if m:
            self.log.append(f"[ui] (warm-start from {m.group(2).strip()})")

    def _finalize(self):
        rc = self.proc.returncode if self.proc else -1
        with self._lock:
            self.status = "done" if rc == 0 else "failed"
            self.log.append(f"[ui] exited with code {rc}")
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
                    "running": self.is_running()}


# Module-level singleton. One active training run per process; matches the UI
# flow (one tab, one start button).
RUNNER = TrainRunner()


def _format_log(lines: list[str]) -> str:
    return "\n".join(lines)


def _format_status(snap: dict) -> str:
    status = snap["status"]
    stage = snap["stage"] or "—"
    ckpt = snap["last_ckpt"]
    if status == "running":
        head = f"### 🟡 Running `{stage}`"
    elif status == "done":
        head = f"### ✅ Done (`{stage}`)"
    elif status == "failed":
        head = f"### ❌ Failed (`{stage}`)"
    else:
        head = "### ⚪ Idle"
    ckpt_line = f"\n**Latest checkpoint:** `{ckpt}`" if ckpt else ""
    return head + ckpt_line


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
    return pd.DataFrame(rows, columns=["epoch", "metric", "stage", "value"])


def _format_colour(colour: dict | None) -> str:
    if not colour:
        return ""
    lines = [f"**Red:** {colour.get('r_pos', '?')}    **Blue:** {colour.get('b_pos', '?')}    "
             f"**Steps:** {colour.get('steps', '?')}    "
             f"**pos_w red/blue:** {colour.get('rpw', '?')} / {colour.get('bpw', '?')}"]
    xr = colour.get("x_hist_red"); xb = colour.get("x_hist_blue")
    if xr and xb:
        total = sum(xr.values()) or 1
        lines.append("\n**Column distribution (red):**")
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
        max_pct = max(xr.values() + xb.values()) / max(1, max(xr.values() + xb.values()) + sum(xr.values()) + sum(xb.values()) - max(xr.values() + xb.values()))
        # crude "any column over 50%?" warning
        for label, hist in (("red", xr), ("blue", xb)):
            total = sum(hist.values()) or 1
            top = max(hist.values()) / total
            if top > 0.5:
                lines.append(f"\n⚠️  **{label}** is collapsed on one column ({top*100:.0f}%) — "
                             f"the classic Stage 2 failure mode. Try a different seed or more data.")
    return "\n".join(lines)


RECOMMENDATIONS = """
### Что крутить
- **Cold start**: `--epochs 40-60`, `--lr 1e-3`, `--bs 16` (stage1) — дефолты в коде разумные.
- **Fine-tune**: `--lr 3e-4` (в 3-5× ниже, чтобы не «забыть» выученное), `--epochs 10-20` обычно хватает.
- **Маленький датасет (<50 уровней)**: уменьшите `--steps` (stage1) до 30-40, иначе overfit за 1 эпоху.
- **Нет CUDA**: ожидайте минуты на эпоху. Stage 1 (TCN) обычно медленнее, чем Stage 2 (GRU на коротких последовательностях).

### На что смотреть в логах

**Stage 1** — каждая `ep N loss X | val P R F1`:
- `val F1` должен расти. Если 3 эпохи подряд не растёт — early stop.
- `pos_rate` > 0.10 и `pos_weight` > 10 — датасет богат нотами, BCE агрессивный; это нормально.
- Loss падает, F1 не растёт → переобучение, уменьшите `--steps` или `--epochs`.
- Val F1 застрял на ~0.3-0.5 → onset detector вырождается в «всегда on/off». Проверьте, что в датасете реально есть `notes` (`build_dataset.py` печатает `notes=N` для каждого уровня).

**Stage 2** — `colour balance` / `x-hist` (печатается один раз в начале) и `val note-acc`:
- **`colour balance`**: если red << blue (или наоборот) больше чем в 3 раза — данные перекошены, модель будет перекошена. **Это именно тот коллапс, который недавно чинили** через `_sample_cat` в `generate.py`.
- **`x-hist`**: распределение нот по 4 колонкам. Если один столбец > 50% — модель коллапсировала в одну колонку. Fine-tune не поможет, нужен cold-start с другим seed или больше данных.
- **`val note-acc`** должна расти к 0.7+. Если застряла на 0.5 — модель не отличает цвет/колонку, пересмотрите `CTX_RADIUS` в `models/stage2.py`.

### Если что-то пошло не так
- Чекпойнт не грузится → `models/_ckpt/<name>.latest.pt` это новый формат, legacy `models/_ckpt/<name>.pt` (без `.latest`) — старый. UI пишет оба.
- Train вкладка зависла → нажмите Stop; если не помогает — перезапуск `app.py`.
- Live-лог не обновляется → обновите страницу; persistent state в `out/ui/train_state.json` покажет последний статус.
"""


def start_click(data_dir, stage, mode, resume, epochs, steps, bs, lr, device):
    if RUNNER.is_running():
        return (_format_status(RUNNER.snapshot()),
                _format_log(RUNNER.log), [])
    if not data_dir or not Path(data_dir).is_dir():
        return ("### ❌ Path is empty or not a directory",
                "", [])
    args = ["--epochs", str(int(epochs)), "--data", str(data_dir),
            "--device", device, "--lr", str(float(lr))]
    if stage == "stage1":
        args += ["--steps", str(int(steps)), "--bs", str(int(bs))]
    if mode == "Fine-tune from .pt":
        if not resume or not Path(resume).is_file():
            return ("### ❌ Fine-tune needs a valid `--resume` path to a .pt", "", [])
        args += ["--resume", str(resume)]
    try:
        RUNNER.start(stage, args)
    except Exception as e:
        return (f"### ❌ Failed to start: `{e}`", "", [])
    return (_format_status(RUNNER.snapshot()), _format_log(RUNNER.log), [])


def stop_click():
    RUNNER.stop()
    snap = RUNNER.snapshot()
    return (_format_status(snap), _format_log(snap["log_tail"]), _metrics_to_plot(snap["metrics"], snap["stage"]))


def refresh():
    snap = RUNNER.snapshot()
    return (_format_status(snap), _format_log(snap["log_tail"]),
            _metrics_to_plot(snap["metrics"], snap["stage"]), _format_colour(snap["colour"]))


def build_train_tab() -> None:
    """Build the Train tab. The block is appended to the existing app."""
    # Read the most recent state file so the tab doesn't claim "Idle" after a reload
    initial_snap = RUNNER.snapshot()
    if STATE_PATH.exists():
        try:
            persisted = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            initial_snap["status"] = persisted.get("status", initial_snap["status"])
            initial_snap["stage"] = persisted.get("stage", initial_snap["stage"])
            initial_snap["last_ckpt"] = persisted.get("last_ckpt", initial_snap["last_ckpt"])
        except (OSError, json.JSONDecodeError):
            pass

    gr.Markdown("# 🏋️ Train / Fine-tune\n"
                "Запускает `models/stage1.py` или `models/stage2.py` в фоне, "
                "стримит логи и метрики в UI. Чекпойнт сохраняется с timestamp-бэкапом.")

    with gr.Row():
        with gr.Column():
            data_dir = gr.Textbox(label="Dataset path (folder with mel.npy + *.json)",
                                  placeholder="e.g. C:/data/bs_dataset или просто `dataset`")
            stage = gr.Radio(["stage1", "stage2"], value="stage1", label="Stage")
            mode = gr.Radio(["Cold start (from scratch)", "Fine-tune from .pt"],
                            value="Cold start (from scratch)", label="Mode")
            resume = gr.Textbox(label="Resume .pt path (only for fine-tune)",
                                placeholder="models/_ckpt/stage1.latest.pt")
            epochs = gr.Slider(1, 200, value=5, step=1, label="Epochs")
            with gr.Row():
                steps = gr.Slider(4, 400, value=80, step=4, label="Steps/epoch (stage1)")
                bs = gr.Slider(2, 64, value=16, step=2, label="Batch size (stage1)")
            lr = gr.Number(value=1e-3, label="Learning rate", precision=4)
            device = gr.Radio(["cpu", "cuda"], value="cpu", label="Device")
            with gr.Row():
                start = gr.Button("Start training", variant="primary")
                stop = gr.Button("Stop", variant="stop")

        with gr.Column():
            status = gr.Markdown(_format_status(initial_snap))
            colour = gr.Markdown()
            log = gr.Textbox(label="Live log (last 50 lines)", lines=20,
                             max_lines=20, autoscroll=True, interactive=False,
                             value=_format_log(initial_snap["log_tail"]))
            plot = gr.LinePlot(x="epoch", y="value", color="metric",
                               title="Metrics per epoch", x_title="epoch", y_title="value",
                               value=_metrics_to_plot(initial_snap["metrics"],
                                                      initial_snap["stage"]))

    with gr.Accordion("📖 Рекомендации", open=False):
        gr.Markdown(RECOMMENDATIONS)

    # Wire events. refresh() is wired to a Timer in app.py (it lives outside the
    # tab to keep the interval simple). Here we just register the click handlers.
    start.click(start_click,
                [data_dir, stage, mode, resume, epochs, steps, bs, lr, device],
                [status, log, plot])
    stop.click(stop_click, outputs=[status, log, plot])
    return {"status": status, "log": log, "plot": plot, "colour": colour}


def refresh_handlers():
    """Returns the (inputs, outputs) lists for the Gradio Timer that polls state."""
    snap = RUNNER.snapshot()
    return {"status_value": _format_status(snap),
            "log_value": _format_log(snap["log_tail"]),
            "plot_value": _metrics_to_plot(snap["metrics"], snap["stage"]),
            "colour_value": _format_colour(snap["colour"])}
