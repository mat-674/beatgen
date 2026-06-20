"""End-to-end inference: audio -> Stage1 (onsets) -> Stage2 (notes) -> validate -> map.

Reusable API (used by both the CLI and the Gradio app):
    models = load_models(device)
    out_dir, stats = run(audio_path, "Expert", "out/song", bpm=None, thr=0.85, models=models)

CLI:
    python generate.py <audio.wav|ogg> --difficulty Expert --thr 0.85 --out out/song
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features.audio import HOP, SR, load_audio, log_mel  # noqa: E402
from models.common import DIFF_IDX  # noqa: E402
from models.stage1 import Stage1Net  # noqa: E402
from models.stage2 import NOTE_VEC, Stage2Net, encode_prev, mel_context  # noqa: E402
from validate.playability import validate  # noqa: E402
from output.beatsaver import write_map  # noqa: E402

CKPT1 = Path(__file__).resolve().parent / "models/_ckpt/stage1.pt"
CKPT2 = Path(__file__).resolve().parent / "models/_ckpt/stage2.pt"


def standardize(mel, mean, std):
    return ((mel - mean) / std).astype(np.float32)


def pick_onsets(prob, thr, min_gap=2):
    """Local-max peak picking above threshold with a minimum frame gap."""
    frames, last = [], -10_000
    for f in range(len(prob)):
        if prob[f] < thr:
            continue
        lo, hi = max(0, f - 1), min(len(prob), f + 2)
        if prob[f] >= prob[lo:hi].max() and f - last >= min_gap:
            frames.append(f); last = f
    return frames


def estimate_bpm(y):
    import warnings
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = librosa.beat.tempo(y=y, sr=SR, aggregate=None)
    return float(np.median(t)) if len(t) else 120.0


def models_available() -> bool:
    return CKPT1.exists() and CKPT2.exists()


def load_models(device: str = "cpu") -> dict:
    """Load both stage checkpoints once; reuse across many generations."""
    if not models_available():
        raise FileNotFoundError(
            f"missing checkpoints ({CKPT1.name}/{CKPT2.name}) — train first "
            "(models/stage1.py, models/stage2.py)")
    s1 = torch.load(CKPT1, map_location=device, weights_only=False)
    m1 = Stage1Net().to(device); m1.load_state_dict(s1["model"]); m1.eval()
    s2 = torch.load(CKPT2, map_location=device, weights_only=False)
    m2 = Stage2Net().to(device); m2.load_state_dict(s2["model"]); m2.eval()
    return {"device": device, "m1": m1, "s1": s1, "m2": m2, "s2": s2}


@torch.no_grad()
def run(audio_path, difficulty="Expert", out_dir="out/generated", *,
        bpm=None, thr=0.85, models=None, device="cpu", write_audio=True):
    """Generate a playable map. Returns (out_dir: Path, stats: dict)."""
    if models is None:
        models = load_models(device)
    device = models["device"]
    diff_idx = DIFF_IDX[difficulty]

    y = load_audio(audio_path)
    mel_raw = log_mel(y)
    if not bpm:
        bpm = estimate_bpm(y)

    # Stage 1: onsets
    m1, s1 = models["m1"], models["s1"]
    mel1 = standardize(mel_raw, s1["mean"], s1["std"])
    prob = torch.sigmoid(m1(torch.tensor(mel1[None], device=device),
                            torch.tensor([diff_idx], device=device)))[0].cpu().numpy()
    frames = pick_onsets(prob, thr)
    if not frames:
        raise RuntimeError("no onsets found — lower the threshold")

    # Stage 2: notes (autoregressive over actions)
    m2, s2 = models["m2"], models["s2"]
    mel2 = standardize(mel_raw, s2["mean"], s2["std"])
    diff_t = torch.tensor([diff_idx], device=device)
    notes, h = [], None
    prev = np.zeros(NOTE_VEC, dtype=np.float32)
    for f in frames:
        ctx = torch.tensor(mel_context(mel2, f)[None, None], device=device)
        out, h = m2(ctx, diff_t, torch.tensor(prev[None, None], device=device), h)
        (rp, rx, ry, rd), (bp, bx, by, bd) = Stage2Net.split(out)
        rp, bp = torch.sigmoid(rp).item(), torch.sigmoid(bp).item()
        red = blue = None
        if rp >= 0.5:
            red = (int(rx.argmax()), int(ry.argmax()), int(rd.argmax()))
        if bp >= 0.5:
            blue = (int(bx.argmax()), int(by.argmax()), int(bd.argmax()))
        if red is None and blue is None:            # an onset must carry a note
            if rp >= bp:
                red = (int(rx.argmax()), int(ry.argmax()), int(rd.argmax()))
            else:
                blue = (int(bx.argmax()), int(by.argmax()), int(bd.argmax()))
        beat = (f * HOP / SR) * bpm / 60.0
        for col, note in ((0, red), (1, blue)):
            if note is not None:
                notes.append({"b": beat, "x": note[0], "y": note[1], "c": col, "d": note[2]})
        prev = encode_prev(red, blue)

    canon = validate({"notes": notes, "bombs": [], "walls": []})
    out_dir = Path(out_dir)
    write_map(out_dir, canon, song_name=Path(audio_path).stem, bpm=bpm,
              difficulty=difficulty, audio=(y, SR) if write_audio else None)

    n = canon["notes"]
    dur = len(y) / SR
    stats = {"out_dir": str(out_dir), "bpm": round(float(bpm), 2),
             "difficulty": difficulty, "threshold": thr,
             "onsets": len(frames), "notes": len(n),
             "duration_sec": round(dur, 1),
             "notes_per_sec": round(len(n) / dur, 2) if dur else 0,
             "red": sum(1 for z in n if z["c"] == 0),
             "blue": sum(1 for z in n if z["c"] == 1),
             "cut_directions": dict(sorted(Counter(z["d"] for z in n).items()))}
    return out_dir, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--difficulty", default="Expert", choices=list(DIFF_IDX))
    ap.add_argument("--out", type=Path, default=Path("out/generated"))
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir, stats = run(args.audio, args.difficulty, args.out,
                         bpm=args.bpm, thr=args.thr, device=args.device)
    for k, v in stats.items():
        print(f"{k:16s}: {v}")
    print(f"\nmap -> {out_dir}")


if __name__ == "__main__":
    main()
