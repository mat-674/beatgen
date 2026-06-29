"""Gradio UI for beatgen inference.

    python app.py                 # then open the printed local URL

Upload a song, pick a difficulty/density, get a downloadable BeatSaver map (.zip).
"""
from __future__ import annotations

import shutil
import sys
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

UI_OUT = ROOT / "out" / "ui"
_CACHE: dict[str, dict] = {}   # device -> loaded models


def get_models(device: str):
    if device not in _CACHE:
        _CACHE[device] = load_models(device)
    return _CACHE[device]


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:48] or "song"


def generate_ui(audio_path, difficulties, threshold, bpm, device,
                title, artist, mapper, cover):
    if not audio_path:
        return None, "⬆️ Upload an audio file first (.wav / .ogg / .flac)."
    if not difficulties:
        return None, "⬆️ Pick at least one difficulty."
    if not models_available():
        return None, ("❌ No trained models found in `models/_ckpt/`.\n\n"
                      "Train them first:\n```\npython models/stage1.py\npython models/stage2.py\n```")
    try:
        models = get_models(device)
        stem = safe_name(Path(audio_path).stem)
        out_dir = UI_OUT / stem
        if out_dir.exists():
            shutil.rmtree(out_dir)

        # audio is decoded once and reused across every difficulty
        analysis = analyze(audio_path, bpm=(float(bpm) if bpm else None))
        bpm_val = analysis[2]
        beatmaps, per = {}, {}
        for diff in difficulties:
            canon, stats = generate_notes(audio_path, diff, models=models,
                                          thr=float(threshold), analysis=analysis)
            beatmaps[diff] = canon
            per[diff] = stats

        song_name = (title or "").strip() or Path(audio_path).stem
        song_author = (artist or "").strip() or "Unknown"
        level_author = (mapper or "").strip() or "beatgen-ai"
        write_map(out_dir, beatmaps, song_name=song_name, bpm=bpm_val,
                  song_author=song_author, level_author=level_author,
                  audio_src=audio_path, cover_src=cover or None)
        zip_path = shutil.make_archive(str(UI_OUT / stem), "zip", root_dir=out_dir)

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
        return zip_path, md
    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Generation failed: `{e}`"


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
                        out_zip = gr.File(label="Generated map (.zip)")
                        out_md = gr.Markdown()
                go.click(generate_ui,
                         [audio, difficulty, threshold, bpm, device, title, artist, mapper, cover],
                         [out_zip, out_md])
                if not cuda:
                    gr.Markdown("_Running on CPU. Reinstall the CUDA torch wheel "
                                "(`python install.py --runtime cuda`) for GPU._")
            with gr.Tab("Train"):
                handles = build_train_tab()
                # Timer drives the live log + metrics + build status while a run is alive.
                timer = gr.Timer(2.0, active=True)
                def _tick():
                    s = refresh_handlers()
                    return (s["status_value"], s["log_value"], s["plot_value"],
                            s["colour_value"],
                            s["build_status_value"], s["build_log_value"])
                timer.tick(_tick,
                           outputs=[handles["status"], handles["log"],
                                    handles["plot"], handles["colour"],
                                    handles["build_status"], handles["build_log"]])
    return demo


if __name__ == "__main__":
    UI_OUT.mkdir(parents=True, exist_ok=True)
    build().launch()
