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

_CKPT_DIR = Path(__file__).resolve().parent / "models/_ckpt"
# Prefer .best.pt (the model the trainer actually validated on) over .latest.pt
# (which is just the last-epoch weights and may be overfit). Fall back to .pt
# so older training runs that never wrote a .best still load.
def _pick_ckpt(stage: str) -> Path:
    for name in (f"{stage}.best.pt", f"{stage}.latest.pt", f"{stage}.pt"):
        p = _CKPT_DIR / name
        if p.exists():
            return p
    return _CKPT_DIR / f"{stage}.pt"   # the missing-file path used in error msgs

CKPT1 = _pick_ckpt("stage1")
CKPT2 = _pick_ckpt("stage2")


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


def analyze(audio_path, bpm=None):
    """Load audio once -> (mono 22k samples, log-mel, bpm). Shared across difficulties."""
    y = load_audio(audio_path)
    mel_raw = log_mel(y)
    if not bpm:
        bpm = estimate_bpm(y)
    return y, mel_raw, float(bpm)


def _sample_cat(logits, temperature, rng):
    """Pick an index from category logits. temperature<=0 -> argmax; else softmax sample.

    Sampling (instead of argmax) is what spreads notes across columns/directions — a
    near-collapsed distribution argmaxes to the same lane every time, but sampling from
    it still yields variety and breaks the deterministic teacher-forcing feedback loop.
    """
    z = logits.reshape(-1).float()
    if temperature <= 0:
        return int(z.argmax())
    p = torch.softmax(z / temperature, dim=-1).cpu().numpy().astype("float64")
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


# Note sampling temperature. Hardcoded to 1.0 because the GRU frequently
# over-confidently picks one lane (teacher-forcing feedback loop), so argmax
# (T=0) collapses to a single column on out-of-distribution songs. T=1.0
# samples proportionally to the logits and reliably recovers all 4 columns
# + 9 cut directions. Advanced callers can still pass `temperature=` to
# generate_notes / run to override.
DEFAULT_TEMPERATURE = 1.0


@torch.no_grad()
def generate_notes(audio_path, difficulty="Expert", *, bpm=None, thr=0.85,
                   models=None, device="cpu", temperature=DEFAULT_TEMPERATURE,
                   seed=None, analysis=None):
    """Run both stages for one difficulty. Returns (canon: dict, stats: dict). No file I/O."""
    if models is None:
        models = load_models(device)
    device = models["device"]
    diff_idx = DIFF_IDX[difficulty]
    rng = np.random.default_rng(seed)

    if analysis is None:
        analysis = analyze(audio_path, bpm)
    y, mel_raw, bpm = analysis

    # Stage 1: onsets
    m1, s1 = models["m1"], models["s1"]
    mel1 = standardize(mel_raw, s1["mean"], s1["std"])
    prob = torch.sigmoid(m1(torch.tensor(mel1[None], device=device),
                            torch.tensor([diff_idx], device=device)))[0].cpu().numpy()
    frames = pick_onsets(prob, thr)
    if not frames:
        raise RuntimeError("no onsets found — lower the threshold")

    # Stage 2: notes (autoregressive over actions), with temperature sampling
    m2, s2 = models["m2"], models["s2"]
    mel2 = standardize(mel_raw, s2["mean"], s2["std"])
    diff_t = torch.tensor([diff_idx], device=device)
    notes, h = [], None
    prev = np.zeros(NOTE_VEC, dtype=np.float32)
    last_color = None                                   # for fallback alternation nudge
    for f in frames:
        ctx = torch.tensor(mel_context(mel2, f)[None, None], device=device)
        out, h = m2(ctx, diff_t, torch.tensor(prev[None, None], device=device), h)
        (rp, rx, ry, rd), (bp, bx, by, bd) = Stage2Net.split(out)
        rp, bp = torch.sigmoid(rp).item(), torch.sigmoid(bp).item()

        def emit(lx, ly, ld):
            return (_sample_cat(lx, temperature, rng),
                    _sample_cat(ly, temperature, rng),
                    _sample_cat(ld, temperature, rng))

        red = emit(rx, ry, rd) if rp >= 0.5 else None
        blue = emit(bx, by, bd) if bp >= 0.5 else None
        if red is None and blue is None:                # an onset must carry a note
            # weighted-random by (rp, bp) with a light nudge toward alternating colour,
            # instead of the old deterministic `if rp >= bp` that collapsed to one side.
            wr, wb = rp, bp
            if last_color == 0:
                wb *= 1.3
            elif last_color == 1:
                wr *= 1.3
            tot = wr + wb
            if tot <= 0 or rng.random() < wr / tot:
                red = emit(rx, ry, rd)
            else:
                blue = emit(bx, by, bd)

        beat = (f * HOP / SR) * bpm / 60.0
        for col, note in ((0, red), (1, blue)):
            if note is not None:
                notes.append({"b": beat, "x": note[0], "y": note[1], "c": col, "d": note[2]})
        if red is not None and blue is None:
            last_color = 0
        elif blue is not None and red is None:
            last_color = 1
        prev = encode_prev(red, blue)

    canon = validate({"notes": notes, "bombs": [], "walls": []})
    n = canon["notes"]
    dur = len(y) / SR
    stats = {"bpm": round(float(bpm), 2), "difficulty": difficulty, "threshold": thr,
             "onsets": len(frames), "notes": len(n),
             "duration_sec": round(dur, 1),
             "notes_per_sec": round(len(n) / dur, 2) if dur else 0,
             "red": sum(1 for z in n if z["c"] == 0),
             "blue": sum(1 for z in n if z["c"] == 1),
             "cut_directions": dict(sorted(Counter(z["d"] for z in n).items()))}
    return canon, stats


@torch.no_grad()
def run(audio_path, difficulty="Expert", out_dir="out/generated", *,
        bpm=None, thr=0.85, models=None, device="cpu", write_audio=True,
        temperature=DEFAULT_TEMPERATURE, seed=None):
    """Single-difficulty wrapper: generate + pack a playable map. (out_dir, stats)."""
    if models is None:
        models = load_models(device)
    canon, stats = generate_notes(audio_path, difficulty, bpm=bpm, thr=thr,
                                  models=models, device=device,
                                  temperature=temperature, seed=seed)
    out_dir = Path(out_dir)
    write_map(out_dir, {difficulty: canon}, song_name=Path(audio_path).stem,
              bpm=stats["bpm"], audio_src=audio_path if write_audio else None)
    stats["out_dir"] = str(out_dir)
    return out_dir, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--difficulty", default="Expert", choices=list(DIFF_IDX))
    ap.add_argument("--out", type=Path, default=Path("out/generated"))
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir, stats = run(args.audio, args.difficulty, args.out,
                         bpm=args.bpm, thr=args.thr, device=args.device,
                         seed=args.seed)
    for k, v in stats.items():
        print(f"{k:16s}: {v}")
    print(f"\nmap -> {out_dir}")


if __name__ == "__main__":
    main()
