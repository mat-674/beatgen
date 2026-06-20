"""Gradio UI for beatgen inference.

    python app.py                 # then open the printed local URL

Upload a song, pick a difficulty/density, get a downloadable BeatSaver map (.zip).
"""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.common import DIFFICULTIES  # noqa: E402
from generate import load_models, models_available, run  # noqa: E402

UI_OUT = ROOT / "out" / "ui"
_CACHE: dict[str, dict] = {}   # device -> loaded models


def get_models(device: str):
    if device not in _CACHE:
        _CACHE[device] = load_models(device)
    return _CACHE[device]


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:48] or "song"


def generate_ui(audio_path, difficulty, threshold, bpm, device):
    if not audio_path:
        return None, "⬆️ Upload an audio file first (.wav / .ogg / .flac)."
    if not models_available():
        return None, ("❌ No trained models found in `models/_ckpt/`.\n\n"
                      "Train them first:\n```\npython models/stage1.py\npython models/stage2.py\n```")
    try:
        models = get_models(device)
        stem = safe_name(Path(audio_path).stem)
        out_dir = UI_OUT / f"{stem}_{difficulty}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir, stats = run(audio_path, difficulty, out_dir,
                             bpm=(float(bpm) if bpm else None),
                             thr=float(threshold), models=models)
        zip_base = UI_OUT / f"{stem}_{difficulty}"
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=out_dir)

        dirs = " ".join(f"{d}×{c}" for d, c in stats["cut_directions"].items())
        md = (f"### ✅ Map generated\n"
              f"- **difficulty:** {stats['difficulty']}  |  **BPM:** {stats['bpm']}  "
              f"(threshold {stats['threshold']})\n"
              f"- **notes:** {stats['notes']}  "
              f"(🔴 {stats['red']} / 🔵 {stats['blue']})  →  "
              f"**{stats['notes_per_sec']} notes/sec** over {stats['duration_sec']}s\n"
              f"- **onsets detected:** {stats['onsets']}\n"
              f"- **cut directions:** {dirs}\n\n"
              f"Download the `.zip`, unzip into your Beat Saber `CustomLevels` folder "
              f"(or open in ChroMapper).")
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
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(type="filepath", label="Song (.wav / .ogg / .flac)")
                difficulty = gr.Dropdown(DIFFICULTIES, value="Expert", label="Difficulty")
                threshold = gr.Slider(0.30, 0.97, value=0.85, step=0.01,
                                      label="Note density threshold (higher = fewer notes)")
                bpm = gr.Number(value=0, label="BPM (0 = auto-detect)", precision=2)
                device = gr.Radio(devices, value=devices[0], label="Runtime")
                go = gr.Button("Generate map", variant="primary")
            with gr.Column():
                out_zip = gr.File(label="Generated map (.zip)")
                out_md = gr.Markdown()
        go.click(generate_ui, [audio, difficulty, threshold, bpm, device], [out_zip, out_md])
        if not cuda:
            gr.Markdown("_Running on CPU. Reinstall the CUDA torch wheel "
                        "(`python install.py --runtime cuda`) for GPU._")
    return demo


if __name__ == "__main__":
    UI_OUT.mkdir(parents=True, exist_ok=True)
    build().launch()
