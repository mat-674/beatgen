"""Gradio UI for beatgen inference.

    python app.py                 # then open the printed local URL

Upload a song, pick a difficulty/density, get a downloadable BeatSaver map (.zip).
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

# Gradio 5.x on a newer Starlette still emits this DeprecationWarning on every
# queue-join request — harmless but floods the UI log with identical lines.
# Silence the specific source until Gradio updates its symbol.
warnings.filterwarnings(
    "ignore",
    message=r"HTTP_422_UNPROCESSABLE_ENTITY.*deprecated.*HTTP_422_UNPROCESSABLE_CONTENT",
    category=DeprecationWarning,
    module=r"gradio\.routes",
)

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.common import DIFFICULTIES  # noqa: E402
from generate import analyze, generate_notes, load_models, models_available  # noqa: E402
from output.beatsaver import write_map  # noqa: E402
from app_train import build_train_tab, refresh_handlers  # noqa: E402
from utils.progress import UiProgress  # noqa: E402

UI_OUT = ROOT / "out" / "ui"
_CACHE: dict[str, dict] = {}   # device -> loaded models


def get_models(device: str):
    if device not in _CACHE:
        _CACHE[device] = load_models(device)
    return _CACHE[device]


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:48] or "song"


def _ui_callback(gr_progress):
    """Build a UiProgress-style callback that drives ``gr.Progress`` only.

    The status line lives in the generator (we yield a fresh string per
    phase) — keeping the callback in charge of ``gr.Progress`` only means
    we don't fight Gradio's re-render debounce, and the Markdown update
    in the generator is the single source of truth the user reads.
    """

    def _cb(fraction, phase, postfix):
        if gr_progress is None:
            return   # called outside a generator (unit tests, CLI shim)
        if fraction is not None:
            gr_progress(fraction, desc=str(phase))
        else:
            gr_progress(0, desc=str(phase))

    return _cb


def _format_status(phase: str, postfix: dict, started_at: float) -> str:
    """One-line Markdown for the live status block above the download button."""
    elapsed = time.monotonic() - started_at
    m, s = divmod(int(elapsed), 60)
    bits = "  ".join(f"{k}={v}" for k, v in postfix.items()) if postfix else ""
    line = f"⏳ **{phase}**" + (f"  {bits}" if bits else "") + f"  _(elapsed {m:02d}:{s:02d})_"
    return line


def generate_ui(audio_path, difficulties, threshold, bpm, device,
                title, artist, mapper, cover, progress=gr.Progress(track_tqdm=True)):
    """Generate map(s) for the uploaded song. A *generator* — yields UI updates.

    The handler is a generator on purpose: Gradio repaints its outputs every
    time the body ``yield``s, so the user sees the long Stage 2 decode loop
    tick smoothly via:

    - ``progress`` (``gr.Progress``) — the inline progress bar (driven by
      the UiProgress callback so it ticks per-frame on the slow decode).
    - ``out_status`` (``gr.Markdown``) — short human-readable phase string
      (load_models / analyze / decode(d) / pack) with postfix counters
      and elapsed time.
    - ``out_md`` / ``out_zip`` — empty until the final yield, which carries
      the summary block and the playable .zip, matching the legacy handler.

    Yields ``(status, "", None)`` on every tick, then
    ``(status, summary_md, zip_path)`` once. ``track_tqdm=True`` is a
    belt-and-braces fallback — if any nested tqdm somehow leaks it is
    still picked up by Gradio.
    """
    # 1. validation -------------------------------------------------------
    if not audio_path:
        yield "⬆️ Upload an audio file first (.wav / .ogg / .flac).", "", None
        return
    if not difficulties:
        yield "⬆️ Pick at least one difficulty.", "", None
        return
    if not models_available():
        yield ("❌ No trained models found in `models/_ckpt/`.\n\n"
               "Train them first:\n```\npython models/stage1.py\npython models/stage2.py\n```"), "", None
        return

    started_at = time.monotonic()
    n_diff = len(difficulties)
    # Per-tick counter: 1 (load) + 1 (analyze) + n_diff * (decode) + 1 (pack).
    # The decode phase is re-sized inside ``generate_notes`` via
    # ``UiProgress.set_phase`` so per-frame ticks land in the same bar.
    ui = UiProgress(total=3 + n_diff, desc="generate",
                    callback=_ui_callback(progress))

    try:
        # 2. models -----------------------------------------------------------
        ui.set_phase("load_models", total=1)
        models = get_models(device)
        ui.set_postfix(device=device)
        ui.update(1)
        yield _format_status("load_models", {"device": device}, started_at), "", None

        stem = safe_name(Path(audio_path).stem)
        out_dir = UI_OUT / stem
        if out_dir.exists():
            shutil.rmtree(out_dir)

        # 3. analyze (shared) ------------------------------------------------
        ui.set_phase("analyze audio", total=1)
        analysis = analyze(audio_path, bpm=(float(bpm) if bpm else None))
        bpm_val = analysis[2]
        dur_sec = len(analysis[0]) / 22050
        ui.update(1)
        yield _format_status("analyze",
                             {"bpm": round(bpm_val, 1), "secs": round(dur_sec, 1)},
                             started_at), "", None

        # 4. per-difficulty phases ------------------------------------------
        beatmaps, per = {}, {}
        for i, diff in enumerate(difficulties):
            ui.set_phase(f"decode ({diff})", total=1)
            canon, stats = generate_notes(audio_path, diff, models=models,
                                          thr=float(threshold), analysis=analysis,
                                          progress=ui)
            beatmaps[diff] = canon
            per[diff] = stats
            ui.update(1)
            yield _format_status(
                f"decode ({diff})  {i + 1}/{n_diff}",
                {"notes": stats["notes"], "onsets": stats["onsets"],
                 "🔴": stats["red"], "🔵": stats["blue"]},
                started_at), "", None

        # 5. write -------------------------------------------------------
        ui.set_phase("pack", total=1)
        song_name = (title or "").strip() or Path(audio_path).stem
        song_author = (artist or "").strip() or "Unknown"
        level_author = (mapper or "").strip() or "beatgen-ai"
        write_map(out_dir, beatmaps, song_name=song_name, bpm=bpm_val,
                  song_author=song_author, level_author=level_author,
                  audio_src=audio_path, cover_src=cover or None)
        ui.update(1)
        zip_path = shutil.make_archive(str(UI_OUT / stem), "zip", root_dir=out_dir)
        ui.close()

        # 6. summary ----------------------------------------------------------
        order = sorted(beatmaps, key=lambda d: DIFFICULTIES.index(d))
        lines = [f"### ✅ Map generated — {len(beatmaps)} difficulty(ies)",
                 f"- **{song_name}** — {song_author}  |  **BPM:** {round(bpm_val, 2)}  "
                 f"(threshold {float(threshold):.2f})"]
        for diff in order:
            s = per[diff]
            dirs = " ".join(f"{d}×{c}" for d, c in s["cut_directions"].items())
            lines.append(f"- **{diff}:** {s['notes']} notes "
                         f"(🔴 {s['red']} / 🔵 {s['blue']}), {s['notes_per_sec']}/s, "
                         f"onsets {s['onsets']} · cuts {dirs}")
        md = ("\n".join(lines) + "\n\nDownload the `.zip`, unzip into your Beat Saber "
              "`CustomLevels` folder (or open in ChroMapper).")
        elapsed = time.monotonic() - started_at
        m, s = divmod(int(elapsed), 60)
        yield (f"✅ **done**  {len(beatmaps)} difficulty(ies)  "
               f"_(elapsed {m:02d}:{s:02d})_"), md, zip_path
    except Exception as e:
        traceback.print_exc()
        ui.close()
        yield f"❌ Generation failed: `{e}`", "", None


def build():
    cuda = torch.cuda.is_available()
    devices = (["cuda", "cpu"] if cuda else ["cpu"])
    with gr.Blocks(title="beatgen — AI Beat Saber maps") as demo:
        gr.Markdown("# 🎵 beatgen — AI Beat Saber map generator\n"
                    "Upload a track → get a playable map. Two-stage model "
                    "(onset TCN + note GRU) trained on the official OST.")
        with gr.Tabs():
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column():
                        audio = gr.Audio(type="filepath", label="Song (.wav / .ogg / .flac)")
                        difficulty = gr.CheckboxGroup(DIFFICULTIES, value=["Expert"],
                                                      label="Difficulties (packed into one level)")
                        threshold = gr.Slider(0.30, 0.97, value=0.85, step=0.01,
                                              label="Note density threshold (higher = fewer notes)")
                        bpm = gr.Number(value=0, label="BPM (0 = auto-detect)", precision=2)
                        device = gr.Radio(devices, value=devices[0], label="Runtime")
                        with gr.Accordion("Metadata (optional)", open=False):
                            title = gr.Textbox(label="Song title", placeholder="(blank = filename)")
                            artist = gr.Textbox(label="Artist / song author", placeholder="Unknown")
                            mapper = gr.Textbox(label="Mapper / level author", value="beatgen-ai")
                            cover = gr.Image(type="filepath", label="Cover image (square preferred)")
                        go = gr.Button("Generate map", variant="primary")
                    with gr.Column():
                        # Live status line — ``generate_ui`` is a generator
                        # so it yields a fresh string here every phase
                        # boundary (load_models / analyze / decode(d) / pack).
                        out_status = gr.Markdown("⬆️ Upload an audio file to start.")
                        out_zip = gr.File(label="Generated map (.zip)")
                        out_md = gr.Markdown()
                # Handler yields (status_md, summary_md, zip_path) on every
                # tick; the summary block stays empty until the final yield.
                go.click(generate_ui,
                         [audio, difficulty, threshold, bpm, device,
                          title, artist, mapper, cover],
                         [out_status, out_md, out_zip])
                if not cuda:
                    gr.Markdown("_Running on CPU. Reinstall the CUDA torch wheel "
                                "(`python install.py --runtime cuda`) for GPU._")
            with gr.Tab("Train"):
                handles = build_train_tab()
                # Timer drives the live log + metrics + build + fetch status
                # while a run is alive. ``plot`` and ``colour`` stay at indices
                # 2/3 so the Metrics zone keeps updating from the same 2 s tick.
                timer = gr.Timer(2.0, active=True)
                def _tick():
                    s = refresh_handlers()
                    return (s["status_value"], s["log_value"], s["plot_value"],
                            s["colour_value"],
                            s["build_status_value"], s["build_log_value"],
                            s["fetch_status_value"], s["fetch_log_value"])
                timer.tick(_tick,
                           outputs=[handles["status"], handles["log"],
                                    handles["plot"], handles["colour"],
                                    handles["build_status"], handles["build_log"],
                                    handles["fetch_status"], handles["fetch_log"]])
    return demo


def _launch_kwargs() -> dict:
    """Pick the right ``.launch()`` kwargs for the current environment.

    Local machine: ``share=True`` so the user gets a public ``*.gradio.live``
    URL (current behaviour, unchanged).

    Kaggle / Colab: bind to ``0.0.0.0:7860`` and disable the share tunnel —
    the notebook platform proxies its own public URL to the kernel, and the
    share tunnel would only confuse the port mapping.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("COLAB_GPU"):
        return {"server_name": "0.0.0.0", "server_port": 7860, "share": False}
    return {"share": True}


if __name__ == "__main__":
    UI_OUT.mkdir(parents=True, exist_ok=True)
    build().launch(**_launch_kwargs())
