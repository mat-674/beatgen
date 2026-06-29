"""Stage 2: given action times, pick the note(s) — color/x/y/direction.

A GRU over the (short) action sequence. Per step it predicts, for red and blue:
presence + lane x(4) + layer y(3) + cut direction d(9). Conditioned on local audio
context, difficulty, and the previous emitted note (teacher forcing -> flow/parity).

    python models/stage2.py --epochs 30                                # train from scratch
    python models/stage2.py --resume models/_ckpt/stage2.latest.pt    # fine-tune
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.audio import N_MELS, time_to_frame  # noqa: E402
from models.common import (MelCache, N_DIFF, list_beatmaps, load_canonical,  # noqa: E402
                          save_with_backup)

CKPT = Path("models/_ckpt/stage2.pt")

# Per-note encoding layout: one-hot presence + x(4) + y(3) + d(9) = 17 per hand.
N_X, N_Y, N_D, N_PRESENT = 4, 3, 9, 1
PER_NOTE_VEC = N_PRESENT + N_X + N_Y + N_D      # 17
RED_OFFSET, BLUE_OFFSET = 0, PER_NOTE_VEC       # 0, 17
NOTE_VEC = 2 * PER_NOTE_VEC                     # 34 (encoder width)
CTX_RADIUS = 4                                  # mel_context window half-width (frames, ~210 ms)
CTX_DIM = 2 * N_MELS                            # mel_context = concat(mean, max) over the window

# Slice offsets inside one hand's PER_NOTE_VEC block, for readability at call sites.
SLICE_X = slice(N_PRESENT, N_PRESENT + N_X)              # 1..5
SLICE_Y = slice(N_PRESENT + N_X, N_PRESENT + N_X + N_Y)  # 5..8
SLICE_D = slice(N_PRESENT + N_X + N_Y, PER_NOTE_VEC)     # 8..17


def encode_prev(red, blue) -> np.ndarray:
    """red/blue = None or (x,y,d). -> 34-dim teacher-forcing vector."""
    v = np.zeros(NOTE_VEC, dtype=np.float32)
    for off, note in ((RED_OFFSET, red), (BLUE_OFFSET, blue)):
        if note is not None:
            x, y, d = note
            v[off] = 1.0
            v[off + SLICE_X.start + x] = 1.0
            v[off + SLICE_Y.start + y] = 1.0
            v[off + SLICE_D.start + d] = 1.0
    return v


def actions_from_beatmap(notes):
    """Group notes by frame -> ordered list of (frame, red, blue)."""
    by_frame = {}
    for n in notes:
        f = time_to_frame(float(n["t"]))
        slot = by_frame.setdefault(f, {})
        key = "red" if int(n["c"]) == 0 else "blue"
        if key not in slot:   # keep first of each color at this time
            slot[key] = (int(n["x"]), int(n["y"]), int(n["d"]))
    acts = []
    for f in sorted(by_frame):
        s = by_frame[f]
        acts.append((f, s.get("red"), s.get("blue")))
    return acts


def mel_context(mel: np.ndarray, frame: int, r: int = CTX_RADIUS) -> np.ndarray:
    """Local audio summary around an action: concat(mean, max) over a +/-r window.

    A plain mean over +/-1 frame gave the GRU almost no music to ground colour/position
    on, so it leaned on the previous-note teacher forcing and collapsed (all one colour
    / one column). A wider window plus a max channel restores real audio discrimination.
    """
    T = mel.shape[0]
    if T == 0:
        return np.zeros(2 * mel.shape[1], dtype=np.float32)
    f = min(max(int(frame), 0), T - 1)               # clamp so the window is never empty
    a, b = max(0, f - r), min(T, f + r + 1)
    win = mel[a:b]
    return np.concatenate([win.mean(axis=0), win.max(axis=0)]).astype(np.float32)


class Stage2Net(nn.Module):
    def __init__(self, n_diff=N_DIFF, demb=16, hid=256, layers=2, ctx_dim=CTX_DIM):
        super().__init__()
        self.diff_emb = nn.Embedding(n_diff, demb)
        self.ctx_proj = nn.Linear(ctx_dim, 128)
        self.gru = nn.GRU(128 + demb + NOTE_VEC, hid, num_layers=layers,
                          batch_first=True, dropout=0.1)
        self.head = nn.Linear(hid, NOTE_VEC)

    def forward(self, ctx, diff, prev, h=None):
        # ctx:(B,L,M) diff:(B,) prev:(B,L,34)
        de = self.diff_emb(diff)[:, None, :].expand(-1, ctx.size(1), -1)
        x = torch.cat([torch.relu(self.ctx_proj(ctx)), de, prev], dim=-1)
        y, h = self.gru(x, h)
        return self.head(y), h     # (B,L,NOTE_VEC)

    @staticmethod
    def split(out):
        r, b = out[..., :PER_NOTE_VEC], out[..., PER_NOTE_VEC:]
        def parts(z):
            return z[..., 0], z[..., SLICE_X], z[..., SLICE_Y], z[..., SLICE_D]  # present,x,y,d
        return parts(r), parts(b)


def build_sequences(beatmaps, cache):
    seqs = []
    for bm in beatmaps:
        mel = cache.get(bm["mel_path"])
        acts = actions_from_beatmap(load_canonical(str(bm["json_path"]))["notes"])
        # need at least 2 actions for the GRU to learn a transition; shorter sequences
        # have no teacher-forcing history and would dilute the loss.
        if len(acts) < 2:
            continue
        ctx = np.stack([mel_context(mel, f) for f, _, _ in acts]).astype(np.float32)
        prev = np.zeros((len(acts), NOTE_VEC), dtype=np.float32)
        tgt = np.zeros((len(acts), 8), dtype=np.int64)  # rP,rx,ry,rd,bP,bx,by,bd
        last_r = last_b = None
        for i, (_, red, blue) in enumerate(acts):
            prev[i] = encode_prev(last_r, last_b)
            if red is not None:
                tgt[i, 0] = 1; tgt[i, 1:4] = red
            if blue is not None:
                tgt[i, 4] = 1; tgt[i, 5:8] = blue
            last_r, last_b = red, blue
        seqs.append({"diff": bm["diff_idx"], "ctx": ctx, "prev": prev, "tgt": tgt})
    return seqs


def seq_loss(out, tgt, device, pos_weight=None):
    (rp, rx, ry, rd), (bp, bx, by, bd) = Stage2Net.split(out)
    t = torch.tensor(tgt, device=device)
    rw, bw = (pos_weight if pos_weight is not None else (None, None))
    loss = F.binary_cross_entropy_with_logits(rp, t[:, :, 0].float(), pos_weight=rw)
    loss = loss + F.binary_cross_entropy_with_logits(bp, t[:, :, 4].float(), pos_weight=bw)

    def ce(logits, ti, present):
        m = present.bool()
        if m.any():
            return F.cross_entropy(logits[m], t[:, :, ti][m])
        return torch.zeros((), device=device)
    pr, pb = t[:, :, 0], t[:, :, 4]
    loss = loss + ce(rx, 1, pr) + ce(ry, 2, pr) + ce(rd, 3, pr)
    loss = loss + ce(bx, 5, pb) + ce(by, 6, pb) + ce(bd, 7, pb)
    return loss


@torch.no_grad()
def eval_presence(model, seqs, device):
    """Balanced accuracy over per-step present/absent predictions.

    Returns a dict with `note_acc` (accuracy on present steps — the legacy metric)
    and `bal_acc` (mean of present + absent accuracy — robust to class imbalance).
    The `note_acc` field is kept for backward compatibility with the UI plot.
    """
    model.eval()
    correct_p = total_p = correct_a = total_a = 0
    for s in seqs:
        ctx = torch.tensor(s["ctx"][None], device=device)
        prev = torch.tensor(s["prev"][None], device=device)
        diff = torch.tensor([s["diff"]], device=device)
        out, _ = model(ctx, diff, prev)
        (rp, _, _, _), (bp, _, _, _) = Stage2Net.split(out)
        for logits, pres in ((rp, s["tgt"][:, 0]), (bp, s["tgt"][:, 4])):
            pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
            truth = pres.astype(bool)
            correct_p += int((pred & truth).sum())
            total_p   += int(truth.sum())
            correct_a += int((~pred & ~truth).sum())
            total_a   += int((~truth).sum())
    note_acc = correct_p / max(1, total_p)
    bal_acc  = (correct_p / max(1, total_p) + correct_a / max(1, total_a)) / 2
    return {"note_acc": note_acc, "bal_acc": bal_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--resume", type=Path, default=None,
                    help="warm-start from a .pt (same dict format as save_with_backup)")
    ap.add_argument("--out-dir", type=Path, default=CKPT.parent,
                    help="where to write <name>.latest.pt and <name>.bak-<UTC>.pt")
    args = ap.parse_args()
    device = args.device
    rng = np.random.default_rng()

    beatmaps = list_beatmaps(args.data)
    if not beatmaps:
        print(f"[stage2] no beatmaps under {args.data.resolve()} — "
              f"a song dir must contain mel.npy and at least one <Difficulty>.json. "
              f"Build the dataset with: python extract/build_dataset.py BeatmapLevelsData --out dataset",
              flush=True)
        sys.exit(1)
    train = [b for b in beatmaps if not b["is_val"]]
    val = [b for b in beatmaps if b["is_val"]]
    cache = MelCache(); cache.fit_norm([b["mel_path"] for b in beatmaps])
    tr_seq = build_sequences(train, cache)
    va_seq = build_sequences(val, cache)
    print(f"[stage2] sequences: {len(tr_seq)} train / {len(va_seq)} val", flush=True)

    # colour balance -> per-colour pos_weight (mirrors Stage 1's onset pos_weight) +
    # a column histogram, so a future "all blue / one column" collapse is visible in logs.
    steps = sum(len(s["tgt"]) for s in tr_seq)
    r_pos = sum(int(s["tgt"][:, 0].sum()) for s in tr_seq)
    b_pos = sum(int(s["tgt"][:, 4].sum()) for s in tr_seq)
    if steps == 0 or (r_pos == 0 and b_pos == 0):
        print(f"[stage2] WARNING: empty training set (steps={steps}, "
              f"r_pos={r_pos}, b_pos={b_pos}). Falling back to pos_weight=1.0.", flush=True)
        rpw = bpw = 1.0
    else:
        rpw = ((steps - r_pos) / r_pos) if r_pos else 1.0
        bpw = ((steps - b_pos) / b_pos) if b_pos else 1.0
    pos_weight = (torch.tensor(rpw, device=device), torch.tensor(bpw, device=device))
    xr, xb = Counter(), Counter()
    for s in tr_seq:
        t = s["tgt"]
        for row in t:
            if row[0]:
                xr[int(row[1])] += 1
            if row[4]:
                xb[int(row[5])] += 1
    print(f"[stage2] colour balance: red {r_pos} / blue {b_pos} of {steps} steps | "
          f"pos_w red {rpw:.1f} blue {bpw:.1f}", flush=True)
    print(f"[stage2] x-hist red {dict(sorted(xr.items()))} | blue {dict(sorted(xb.items()))}",
          flush=True)

    model = Stage2Net().to(device)
    if args.resume is not None:
        # weights_only=False: our own checkpoint format ({"model", "mean", "std"}),
        # saved with save_with_backup. Trusted local file, not untrusted download.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        old_mean = float(ckpt.get("mean", cache.mean))
        old_std = float(ckpt.get("std", cache.std))
        if (cache.mean and abs(old_mean - cache.mean) > 0.05 * abs(cache.mean)
                or abs(old_std - cache.std) > 0.05 * abs(cache.std or 1.0)):
            print(f"[stage2] WARNING: resume ckpt was fitted on a different dataset "
                  f"(mean {old_mean:.4f} -> {cache.mean:.4f}, std {old_std:.4f} -> {cache.std:.4f}). "
                  f"Loss will be miscalibrated until lr decays.", flush=True)
        cache.mean = old_mean
        cache.std = old_std
        print(f"[stage2] resumed from {args.resume}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(len(tr_seq)); tot = 0.0
        for i in order:
            s = tr_seq[i]
            ctx = torch.tensor(s["ctx"][None], device=device)
            prev = torch.tensor(s["prev"][None], device=device)
            diff = torch.tensor([s["diff"]], device=device)
            out, _ = model(ctx, diff, prev)
            loss = seq_loss(out, s["tgt"][None], device, pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 5 == 0 or ep == args.epochs:
            acc = eval_presence(model, va_seq, device)
            print(f"[stage2] ep {ep:3d} loss {tot/len(tr_seq):.4f} | "
                  f"val note-acc {acc['note_acc']:.3f} bal-acc {acc['bal_acc']:.3f}",
                  flush=True)
            latest = save_with_backup(
                {"model": model.state_dict(), "mean": cache.mean, "std": cache.std},
                args.out_dir, "stage2",
            )
            ckpt_state = {"model": model.state_dict(), "mean": cache.mean, "std": cache.std}
            tmp = CKPT.with_suffix(".tmp.pt")
            torch.save(ckpt_state, tmp)
            os.replace(tmp, CKPT)
            print(f"[stage2] saved -> {latest}", flush=True)

    print(f"[stage2] done -> {args.out_dir / 'stage2.latest.pt'}", flush=True)


if __name__ == "__main__":
    main()