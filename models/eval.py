"""Standalone evaluation for Stage 1 (onset) and Stage 2 (note picker).

Mirrors the validation step that runs inside the training loop of
``models/stage1.py`` and ``models/stage2.py`` but launches no training, takes a
checkpoint path, and prints one set of metrics over a chosen split. Useful for
sanity-checking a fine-tuned model, comparing two checkpoints, or reporting
metrics on the *full* dataset (the training scripts only validate on
``VAL_SONGS``).

The two stages use different metrics:
  * Stage 1 — per-frame onset: ``(precision, recall, F1)`` at ``--threshold``.
  * Stage 2 — per-step note presence: ``note_acc`` (legacy) and ``bal_acc``.

Usage:
    python models/eval.py --stage 1 --resume models/_ckpt/stage1.best.pt
    python models/eval.py --stage 2 --resume models/_ckpt/stage2.best.pt --bs 32
    python models/eval.py --stage 1 --resume models/_ckpt/stage1.best.pt --split all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.audio import N_MELS  # noqa: E402
from models.common import MelCache, N_DIFF, list_beatmaps  # noqa: E402
from models.stage1 import CROP, Stage1Net, build_labels, evaluate as eval_stage1_fn  # noqa: E402
from models.stage2 import (  # noqa: E402
    CTX_RADIUS, Stage2Net, build_sequences, eval_presence as eval_stage2_fn,
)

# Architecture defaults — must match the values the training scripts write
# into ckpt["hparams"], otherwise load_state_dict raises shape errors.
_DEFAULTS = {
    1: {"hid": 384, "demb": 16},
    2: {"hid": 384, "demb": 16, "layers": 3, "ctx_radius": CTX_RADIUS},
}


def _maybe_compile(model, device, enabled: bool):
    """Same ``aot_eager`` policy as the training scripts: try compile, fall
    back to eager on any failure (incl. Windows + missing Triton)."""
    if not enabled or device != "cuda":
        return model, False
    import torch._dynamo as _d
    _d.config.suppress_errors = True
    try:
        return torch.compile(model, backend="aot_eager"), True
    except Exception as e:
        print(f"[eval] torch.compile(aot_eager) failed ({e!r}); running eager.",
              flush=True)
        return model, False


def _ckpt_hparam(ckpt, key, default):
    """Read a single hparam from the checkpoint, falling back to *default*."""
    h = ckpt.get("hparams") or {}
    return h.get(key, default)


def _split(beatmaps, want: str, data_dir: Path):
    if want == "all":
        return beatmaps
    val = [b for b in beatmaps if b["is_val"]]
    if not val:
        print(f"[eval] WARNING: --split val but no is_val beatmaps in {data_dir} "
              f"(VAL_SONGS=beatsaber/crabrave/turnmeon absent). "
              f"Falling back to all beatmaps.", flush=True)
        return beatmaps
    return val


def _print_header(args, beatmaps, ckpt, stage_extra: str = ""):
    train = [b for b in beatmaps if not b["is_val"]]
    val = [b for b in beatmaps if b["is_val"]]
    print(f"[eval] checkpoint: {args.resume}", flush=True)
    if args.stage == 1:
        print(f"[eval] hparams: hid={args.hid} demb={args.demb} crop={CROP}{stage_extra}",
              flush=True)
    else:
        print(f"[eval] hparams: hid={args.hid} layers={args.layers} "
              f"demb={args.demb} ctx_radius={args.ctx_radius}{stage_extra}",
              flush=True)
    print(f"[eval] beatmaps: {len(train)} train / {len(val)} val", flush=True)
    print(f"[eval] stage{args.stage} ckpt hparams: {ckpt.get('hparams', {})}", flush=True)


def run_stage1(args, pool, cache, ckpt, device):
    # Build labels for the requested pool only — we score only those beatmaps.
    labels = build_labels(pool, cache)
    model = Stage1Net(n_mels=N_MELS, n_diff=N_DIFF,
                      demb=args.demb, hid=args.hid).to(device)
    model, _compiled = _maybe_compile(model, device, enabled=not args.no_compile)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(ckpt["model"])
    print(f"[eval] loaded {args.resume}", flush=True)

    p, r, f1 = eval_stage1_fn(model, pool, labels, cache, device,
                              thr=args.threshold)
    print(f"[eval] stage1 split={args.split} n_beatmaps={len(pool)} "
          f"thr={args.threshold:.2f} | "
          f"P {p:.3f} R {r:.3f} F1 {f1:.3f}", flush=True)


def run_stage2(args, pool, cache, ckpt, device):
    seqs = build_sequences(pool, cache)
    model = Stage2Net(n_diff=N_DIFF, demb=args.demb, hid=args.hid,
                      layers=args.layers, ctx_dim=2 * N_MELS).to(device)
    model, _compiled = _maybe_compile(model, device, enabled=not args.no_compile)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(ckpt["model"])
    print(f"[eval] loaded {args.resume}", flush=True)

    bs = max(args.bs, 1)
    acc = eval_stage2_fn(model, seqs, device, bs=bs)
    print(f"[eval] stage2 split={args.split} n_sequences={len(seqs)} "
          f"bs={bs} | "
          f"note_acc {acc['note_acc']:.3f} bal_acc {acc['bal_acc']:.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True,
                    help="which stage to evaluate")
    ap.add_argument("--resume", type=Path, required=True,
                    help="path to a .pt produced by save_with_backup "
                         "(stage1.best.pt / stage1.latest.pt / stage2.*.pt)")
    ap.add_argument("--data", type=Path, default=Path("dataset"),
                    help="dataset directory (one subdir per song, with mel.npy "
                         "and <Difficulty>.json files)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile (eager mode)")

    # Eval-specific knobs.
    ap.add_argument("--split", choices=("val", "all"), default="val",
                    help="evaluate on the held-out VAL_SONGS only (default, "
                         "matches the train-loop metric) or on the full dataset")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="stage 1 only: sigmoid threshold for F1/P/R (default 0.5)")
    ap.add_argument("--bs", type=int, default=16,
                    help="stage 2 only: packed-sequence batch size (default 16)")

    # Optional architecture overrides (default: read from ckpt["hparams"]).
    ap.add_argument("--hid", type=int, default=None,
                    help="hidden width (overrides ckpt hparams)")
    ap.add_argument("--demb", type=int, default=None,
                    help="difficulty embedding width (overrides ckpt hparams)")
    ap.add_argument("--layers", type=int, default=None,
                    help="stage 2 only: GRU layers (overrides ckpt hparams)")
    ap.add_argument("--ctx-radius", type=int, default=None,
                    help="stage 2 only: mel_context window half-width in frames")

    args = ap.parse_args()

    # ---- Resolve architecture ---------------------------------------------
    # We load the checkpoint first because hparams live inside it; if the user
    # passed --hid/--demb/--layers/--ctx-radius those win.
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        sys.exit(f"[eval] {args.resume} is not a beatgen checkpoint "
                 f"(no 'model' key).")

    defaults = _DEFAULTS[args.stage]
    args.hid  = args.hid  if args.hid  is not None else _ckpt_hparam(ckpt, "hid", defaults["hid"])
    args.demb = args.demb if args.demb is not None else _ckpt_hparam(ckpt, "demb", defaults["demb"])
    if args.stage == 2:
        args.layers      = args.layers      if args.layers      is not None \
            else _ckpt_hparam(ckpt, "layers", defaults["layers"])
        args.ctx_radius  = args.ctx_radius  if args.ctx_radius  is not None \
            else _ckpt_hparam(ckpt, "ctx_radius", defaults["ctx_radius"])
        # CTX_RADIUS is read by mel_context() from the module global; update it
        # so the rebuilt sequences match the checkpoint.
        import models.stage2 as _s2
        _s2.CTX_RADIUS = args.ctx_radius

    # ---- Dataset -----------------------------------------------------------
    beatmaps = list_beatmaps(args.data)
    if not beatmaps:
        sys.exit(f"[eval] no beatmaps under {args.data.resolve()} — a song dir "
                 f"must contain mel.npy and at least one <Difficulty>.json. "
                 f"Build with: python extract/build_dataset.py BeatmapLevelsData "
                 f"--out {args.data}")
    # Filter once by split here so both stages run on the same set; the
    # canonical val split is VAL_SONGS = {beatsaber, crabrave, turnmeon}.
    pool = _split(beatmaps, args.split, args.data)
    cache = MelCache()
    cache.fit_norm([b["mel_path"] for b in beatmaps])
    # Honour the checkpoint's mean/std: training resumes do the same so a
    # resumed model continues to read inputs the way it was trained on. A
    # large drift (>5 %) is worth surfacing so the user knows the metric is
    # only comparable to the training run if they keep this in sync.
    old_mean = float(ckpt.get("mean", cache.mean))
    old_std  = float(ckpt.get("std",  cache.std))
    if cache.mean and abs(old_mean - cache.mean) > 0.05 * abs(cache.mean):
        print(f"[eval] NOTE: ckpt mean {old_mean:.4f} != dataset mean "
              f"{cache.mean:.4f} (>{5}% drift) — using ckpt's mean/std "
              f"to match the trained model.", flush=True)
    cache.mean, cache.std = old_mean, old_std

    # ---- Dispatch ----------------------------------------------------------
    _print_header(args, beatmaps, ckpt)
    if args.stage == 1:
        run_stage1(args, pool, cache, ckpt, args.device)
    else:
        run_stage2(args, pool, cache, ckpt, args.device)


if __name__ == "__main__":
    main()

