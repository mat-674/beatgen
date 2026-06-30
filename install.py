"""beatgen installer — creates a venv and installs deps for CPU or CUDA.

    python install.py                      # interactive: asks CPU or CUDA
    python install.py --runtime cpu
    python install.py --runtime cuda --cuda cu124
    python install.py --runtime cuda --no-venv   # install into the current environment

CUDA wheels bundle their own CUDA runtime, so you only need a recent NVIDIA driver
(not a matching system CUDA toolkit). Pick the wheel tag closest to your driver.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ["numpy", "librosa", "soundfile", "UnityPy", "gradio"]
CUDA_TAGS = ["cu118", "cu121", "cu124", "cu126", "cu128"]


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str]):
    print("  $", " ".join(cmd))
    subprocess.check_call(cmd)


def ask_runtime() -> str:
    print("Select runtime:\n  [1] CPU\n  [2] CUDA (NVIDIA GPU)")
    return "cuda" if input("Enter 1 or 2 [1]: ").strip() == "2" else "cpu"


def main():
    ap = argparse.ArgumentParser(description="Install beatgen dependencies.")
    ap.add_argument("--runtime", choices=["cpu", "cuda"], help="default: ask")
    ap.add_argument("--cuda", default="cu124", choices=CUDA_TAGS,
                    help="CUDA wheel tag (default cu124)")
    ap.add_argument("--venv", default=".venv", help="venv directory (default .venv)")
    ap.add_argument("--no-venv", action="store_true", help="use the current interpreter")
    args = ap.parse_args()

    runtime = args.runtime or ask_runtime()

    if args.no_venv:
        py = Path(sys.executable)
        print(f"Using current interpreter: {py}")
    else:
        venv = (ROOT / args.venv).resolve()
        if not venv_python(venv).exists():
            print(f"Creating venv at {venv}")
            run([sys.executable, "-m", "venv", str(venv)])
        py = venv_python(venv)

    pip = [str(py), "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip"])

    print(f"\nInstalling PyTorch ({runtime})...")
    if runtime == "cuda":
        run(pip + ["install", "torch", "--index-url",
                   f"https://download.pytorch.org/whl/{args.cuda}"])
    else:
        run(pip + ["install", "torch"])

    print("\nInstalling the rest...")
    run(pip + ["install", "--upgrade", *DEPS])

    print("\nVerifying...")
    run([str(py), "-c",
         "import torch, librosa, soundfile, UnityPy, gradio; "
         "print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); "
         "print('gradio', gradio.__version__)"])

    rel = py if args.no_venv else venv_python(ROOT / args.venv)
    print("\n✅ Done. Next steps:")
    # Default `SRC` is BeatmapLevelsData/ — the official OST/DLC Unity bundles
    # the README ships alongside. Pass a different folder (or several) to
    # point at community maps or a different bundle layout.
    print(f"  Build dataset : {rel} extract/build_dataset.py BeatmapLevelsData --out dataset")
    print(f"  Train stage 1 : {rel} models/stage1.py --epochs 30")
    print(f"  Train stage 2 : {rel} models/stage2.py --epochs 30")
    print(f"  Launch UI     : {rel} app.py")
    if runtime == "cuda":
        print("  (training/inference will use the GPU automatically)")


if __name__ == "__main__":
    main()
