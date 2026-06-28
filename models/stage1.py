"""Stage 1: per-frame "is there a note action here?" (onset/density).

CNN over mel + difficulty embedding -> BiGRU -> per-frame logit.

    python models/stage1.py --epochs 40                              # train from scratch
    python models/stage1.py --resume models/_ckpt/stage1.latest.pt  # fine-tune
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.audio import time_to_frame  # noqa: E402
from features.audio import N_MELS  # noqa: E402
from models.common import (MelCache, N_DIFF, list_beatmaps, load_canonical,  # noqa: E402
                          save_with_backup)

CROP = 1024
CKPT = Path("models/_ckpt/stage1.pt")


class ResBlock(nn.Module):
    """Dilated 1D conv residual block (TCN). Fast + parallel over time on CPU."""

    def __init__(self, ch, dil):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class Stage1Net(nn.Module):
    """TCN: per-frame onset logits from mel + difficulty embedding."""

    def __init__(self, n_mels=N_MELS, n_diff=N_DIFF, demb=16, hid=256,
                 dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.diff_emb = nn.Embedding(n_diff, demb)
        self.inp = nn.Conv1d(n_mels + demb, hid, 3, padding=1)
        self.blocks = nn.ModuleList(ResBlock(hid, d) for d in dilations)
        self.head = nn.Conv1d(hid, 1, 1)

    def forward(self, mel, diff):           # mel: (B,T,M)  diff: (B,)
        de = self.diff_emb(diff)[:, :, None].expand(-1, -1, mel.size(1))  # (B,demb,T)
        x = torch.cat([mel.transpose(1, 2), de], dim=1)                   # (B,M+demb,T)
        x = torch.relu(self.inp(x))
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(1)      # (B,T) logits


def build_labels(beatmaps, cache):
    """Per beatmap -> float32 label vector aligned to its mel length."""
    labels = {}
    for bm in beatmaps:
        T = cache.get_raw(bm["mel_path"]).shape[0]
        lab = np.zeros(T, dtype=np.float32)
        for n in load_canonical(bm["json_path"])["notes"]:
            f = time_to_frame(float(n["t"]))
            if 0 <= f < T:
                lab[f] = 1.0
        labels[(bm["song"], bm["diff"])] = lab
    return labels


def sample_batch(beatmaps, labels, cache, bs, device, rng):
    mels, diffs, labs = [], [], []
    for _ in range(bs):
        bm = beatmaps[rng.integers(len(beatmaps))]
        mel = cache.get(bm["mel_path"])
        lab = labels[(bm["song"], bm["diff"])]
        T = mel.shape[0]
        if T <= CROP:
            pad = CROP - T
            mel = np.pad(mel, ((0, pad), (0, 0)))
            lab = np.pad(lab, (0, pad))
            s = 0
        else:
            s = rng.integers(T - CROP)
        mels.append(mel[s:s + CROP])
        labs.append(lab[s:s + CROP])
        diffs.append(bm["diff_idx"])
    return (torch.tensor(np.stack(mels), device=device),
            torch.tensor(np.array(diffs), device=device),
            torch.tensor(np.stack(labs), device=device))


@torch.no_grad()
def evaluate(model, beatmaps, labels, cache, device, thr=0.5):
    model.eval()
    tp = fp = fn = 0
    for bm in beatmaps:
        mel = torch.tensor(cache.get(bm["mel_path"])[None], device=device)
        diff = torch.tensor([bm["diff_idx"]], device=device)
        prob = torch.sigmoid(model(mel, diff))[0].cpu().numpy()
        pred = prob > thr
        gt = labels[(bm["song"], bm["diff"])].astype(bool)
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--resume", type=Path, default=None,
                    help="warm-start from a .pt (same dict format as save_with_backup)")
    ap.add_argument("--out-dir", type=Path, default=CKPT.parent,
                    help="where to write <name>.latest.pt and <name>.bak-<UTC>.pt")
    args = ap.parse_args()
    device = args.device
    rng = np.random.default_rng(0)

    beatmaps = list_beatmaps(args.data)
    train = [b for b in beatmaps if not b["is_val"]]
    val = [b for b in beatmaps if b["is_val"]]
    cache = MelCache()
    cache.fit_norm([b["mel_path"] for b in beatmaps])
    labels = build_labels(beatmaps, cache)

    # global positive rate -> pos_weight
    pos = sum(l.sum() for l in labels.values()); tot = sum(l.size for l in labels.values())
    pos_w = float((tot - pos) / pos)
    print(f"[stage1] beatmaps: {len(train)} train / {len(val)} val | "
          f"pos_rate={pos/tot:.4f} pos_weight={pos_w:.1f}", flush=True)

    model = Stage1Net().to(device)
    if args.resume is not None:
        # weights_only=False: our own checkpoint format ({"model", "mean", "std"}),
        # saved with save_with_backup. Trusted local file, not untrusted download.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # keep the resumed mel-std in sync with the cache so the loss is comparable
        cache.mean = float(ckpt.get("mean", cache.mean))
        cache.std = float(ckpt.get("std", cache.std))
        print(f"[stage1] resumed from {args.resume}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))

    for ep in range(1, args.epochs + 1):
        model.train(); tot_loss = 0.0
        for _ in range(args.steps):
            mel, diff, lab = sample_batch(train, labels, cache, args.bs, device, rng)
            opt.zero_grad()
            loss = lossf(model(mel, diff), lab)
            loss.backward(); opt.step()
            tot_loss += loss.item()
        if ep % 5 == 0 or ep == args.epochs:
            p, r, f1 = evaluate(model, val, labels, cache, device)
            print(f"[stage1] ep {ep:3d} loss {tot_loss/args.steps:.4f} | "
                  f"val P {p:.3f} R {r:.3f} F1 {f1:.3f}", flush=True)
            latest = save_with_backup(
                {"model": model.state_dict(), "mean": cache.mean, "std": cache.std},
                args.out_dir, "stage1",
            )
            # keep the legacy path up to date for back-compat with the inference loader
            torch.save({"model": model.state_dict(), "mean": cache.mean, "std": cache.std}, CKPT)
            print(f"[stage1] saved -> {latest}", flush=True)

    print(f"[stage1] done -> {args.out_dir / 'stage1.latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
