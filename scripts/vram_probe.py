"""VRAM probe — measure peak CUDA memory for candidate training configs.

Quickly validates that a (hid, bs, [layers], [max-len]) combination actually
fits in the user's GPU budget BEFORE kicking off a 50-epoch run. Each probe
allocates a synthetic batch of the right shape, runs a few forward+backward
steps under bf16 autocast (matching what the trainer does), and prints
``peak_allocated_mb`` per combo.

Usage:
    .venv/bin/python scripts/vram_probe.py --stage 1 \
        --configs "hid=256,bs=32" "hid=384,bs=64"

    .venv/bin/python scripts/vram_probe.py --stage 2 \
        --configs "hid=256,layers=2,bs=8,len=4096" \
                 "hid=384,layers=3,bs=16,len=4096"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow ``python scripts/vram_probe.py`` from anywhere inside the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from features.audio import N_MELS  # noqa: E402
from models.stage1 import CROP, Stage1Net  # noqa: E402
from models.stage2 import NOTE_VEC, CTX_DIM, Stage2Net  # noqa: E402

CROP_FRAMES = CROP  # 1024 by default in models/stage1.py
DEFAULT_BUDGET_MB = 10240  # 10 GB on a 12 GB card — leave 2 GB for OS + compile


@dataclass
class Config:
    """One candidate training combo."""
    stage: int
    hid: int
    bs: int
    layers: int = 0          # stage2 only
    length: int = 4096       # stage2 only — bucket L_max
    n_diff: int = 5          # Beat Saber has 5 difficulties


def parse_configs(stage: int, raw: list[str]) -> list[Config]:
    """Parse "key=value,key=value" specs into Config dataclasses.

    ``bs`` is required for both stages; ``layers``/``length`` are required
    only for stage 2.
    """
    out: list[Config] = []
    for s in raw:
        kv = dict(p.split("=", 1) for p in s.split(",") if "=" in p)
        if stage == 1:
            assert "hid" in kv and "bs" in kv, f"stage1 needs hid=, bs=  (got {s!r})"
            out.append(Config(stage=1, hid=int(kv["hid"]), bs=int(kv["bs"])))
        else:
            assert "hid" in kv and "bs" in kv, f"stage2 needs hid=, bs=  (got {s!r})"
            out.append(Config(
                stage=2,
                hid=int(kv["hid"]),
                bs=int(kv["bs"]),
                layers=int(kv.get("layers", 2)),
                length=int(kv.get("len", 4096)),
            ))
    return out


def probe_stage1(cfg: Config, device: torch.device) -> float:
    """Stage 1: dilated TCN over (bs, CROP, N_MELS) + difficulty emb.

    Mirrors the forward in models/stage1.py:217-225 inside bf16 autocast,
    with a synthetic batch of the same shape. 3 forward+backward steps are
    enough to settle cudnn benchmark and ``try_compile``-style kernels.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = Stage1Net(hid=cfg.hid, n_diff=cfg.n_diff).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    mel = torch.randn(cfg.bs, CROP_FRAMES, N_MELS, device=device)
    diff = torch.randint(0, cfg.n_diff, (cfg.bs,), device=device)
    # Stage1Net returns (bs, T) — single-channel logits, no trailing 1.
    lab = torch.randint(0, 2, (cfg.bs, CROP_FRAMES), device=device).float()
    for _ in range(3):
        opt.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(mel, diff)
        loss = lossf(out, lab)
        loss.backward()
        opt.step()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def probe_stage2(cfg: Config, device: torch.device) -> float:
    """Stage 2: GRU over (bs, length) packed action sequences + ctx window.

    Mirrors models/stage2.py:389-403: pad each "song" to ``length`` (we
    faked one bucket at exactly L_max = ``length``) and run forward+backward
    in bf16 autocast.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # Stage2Net accepts ctx_dim= at construction; default (CTX_DIM = 160) is
    # the post-mel_context window (mean||max over window of +/-CTX_RADIUS=6).
    ctx_dim = 2 * N_MELS  # = 160 — the stage2 model's default
    model = Stage2Net(hid=cfg.hid, layers=cfg.layers,
                      n_diff=cfg.n_diff, ctx_dim=ctx_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    ctx = torch.randn(cfg.bs, cfg.length, ctx_dim, device=device)
    prev = torch.zeros(cfg.bs, cfg.length, NOTE_VEC, device=device)
    # Per-hand presence as the only loss we actually care about for the
    # memory-side question (the x/y/d heads add constants — irrelevant for
    # the bisect).
    tgt = torch.randint(0, 2, (cfg.bs, cfg.length, 2), device=device).float()
    diff = torch.randint(0, cfg.n_diff, (cfg.bs,), device=device)
    for _ in range(3):
        opt.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, _ = model(ctx, diff, prev)
            # out shape: (bs, length, 2*NOTE_VEC) — split into per-hand presence
            # (matches seq_loss in models/stage2.py which only BCElogs on the
            # first dim). We just take the first slice for the probe.
            pred = out[..., 0]
        loss = lossf(pred, tgt[..., 0])
        loss.backward()
        opt.step()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="one or more 'key=val,...' specs to test")
    ap.add_argument("--budget-mb", type=float, default=DEFAULT_BUDGET_MB,
                    help=f"warn if a config exceeds this many MB "
                         f"(default {DEFAULT_BUDGET_MB} = 10 GB)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — VRAM probe needs a GPU.", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)} "
          f"({torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB total)")

    configs = parse_configs(args.stage, args.configs)
    over_budget = 0
    for cfg in configs:
        if cfg.stage == 1:
            peak = probe_stage1(cfg, device)
            label = f"hid={cfg.hid} bs={cfg.bs}"
        else:
            peak = probe_stage2(cfg, device)
            label = (f"hid={cfg.hid} layers={cfg.layers} "
                     f"bs={cfg.bs} len={cfg.length}")
        verdict = "OK" if peak < args.budget_mb else "OVER"
        if verdict == "OVER":
            over_budget += 1
        print(f"  stage{cfg.stage} {label:40s} peak {peak:7.0f} MB  {verdict}")
    return 1 if over_budget else 0


if __name__ == "__main__":
    sys.exit(main())
