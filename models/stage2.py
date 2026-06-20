"""Stage 2: given action times, pick the note(s) — color/x/y/direction.

A GRU over the (short) action sequence. Per step it predicts, for red and blue:
presence + lane x(4) + layer y(3) + cut direction d(9). Conditioned on local audio
context, difficulty, and the previous emitted note (teacher forcing -> flow/parity).

    python models/stage2.py --epochs 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.audio import N_MELS, time_to_frame  # noqa: E402
from models.common import (MelCache, N_DIFF, list_beatmaps, load_canonical)  # noqa: E402

CKPT = Path("models/_ckpt/stage2.pt")
NOTE_VEC = 2 * (1 + 4 + 3 + 9)   # per-note encoding width = 34


def encode_prev(red, blue) -> np.ndarray:
    """red/blue = None or (x,y,d). -> 34-dim teacher-forcing vector."""
    v = np.zeros(NOTE_VEC, dtype=np.float32)
    for off, note in ((0, red), (17, blue)):
        if note is not None:
            x, y, d = note
            v[off] = 1.0
            v[off + 1 + x] = 1.0
            v[off + 5 + y] = 1.0
            v[off + 8 + d] = 1.0
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


def mel_context(mel: np.ndarray, frame: int, r: int = 1) -> np.ndarray:
    a, b = max(0, frame - r), min(mel.shape[0], frame + r + 1)
    return mel[a:b].mean(axis=0)


class Stage2Net(nn.Module):
    def __init__(self, n_mels=N_MELS, n_diff=N_DIFF, demb=16, hid=128):
        super().__init__()
        self.diff_emb = nn.Embedding(n_diff, demb)
        self.ctx_proj = nn.Linear(n_mels, 64)
        self.gru = nn.GRU(64 + demb + NOTE_VEC, hid, batch_first=True)
        # heads: for red & blue -> present(1)+x(4)+y(3)+d(9) = 17 each
        self.head = nn.Linear(hid, 2 * 17)

    def forward(self, ctx, diff, prev, h=None):
        # ctx:(B,L,M) diff:(B,) prev:(B,L,34)
        de = self.diff_emb(diff)[:, None, :].expand(-1, ctx.size(1), -1)
        x = torch.cat([torch.relu(self.ctx_proj(ctx)), de, prev], dim=-1)
        y, h = self.gru(x, h)
        return self.head(y), h     # (B,L,34)

    @staticmethod
    def split(out):
        r, b = out[..., :17], out[..., 17:]
        def parts(z):
            return z[..., 0], z[..., 1:5], z[..., 5:8], z[..., 8:17]  # present,x,y,d
        return parts(r), parts(b)


def build_sequences(beatmaps, cache):
    seqs = []
    for bm in beatmaps:
        mel = cache.get(bm["mel_path"])
        acts = actions_from_beatmap(load_canonical(bm["json_path"])["notes"])
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


def seq_loss(out, tgt, device):
    (rp, rx, ry, rd), (bp, bx, by, bd) = Stage2Net.split(out)
    t = torch.tensor(tgt, device=device)
    loss = F.binary_cross_entropy_with_logits(rp, t[:, :, 0].float())
    loss = loss + F.binary_cross_entropy_with_logits(bp, t[:, :, 4].float())

    def ce(logits, ti, present):
        m = present.bool()
        if m.any():
            return F.cross_entropy(logits[m], t[:, :, ti][m])
        return torch.zeros((), device=device)
    pr, pb = t[:, :, 0], t[:, :, 4]
    loss = loss + ce(rx, 1, pr) + ce(ry, 2, pr) + ce(rd, 3, pr)
    loss = loss + ce(bx, 5, pb) + ce(by, 6, pb) + ce(bd, 7, pb)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    args = ap.parse_args()
    device = args.device
    rng = np.random.default_rng(0)

    beatmaps = list_beatmaps(args.data)
    train = [b for b in beatmaps if not b["is_val"]]
    val = [b for b in beatmaps if b["is_val"]]
    cache = MelCache(); cache.fit_norm([b["mel_path"] for b in beatmaps])
    tr_seq = build_sequences(train, cache)
    va_seq = build_sequences(val, cache)
    print(f"sequences: {len(tr_seq)} train / {len(va_seq)} val")

    model = Stage2Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(len(tr_seq)); tot = 0.0
        for i in order:
            s = tr_seq[i]
            ctx = torch.tensor(s["ctx"][None], device=device)
            prev = torch.tensor(s["prev"][None], device=device)
            diff = torch.tensor([s["diff"]], device=device)
            out, _ = model(ctx, diff, prev)
            loss = seq_loss(out, s["tgt"][None], device)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 5 == 0 or ep == args.epochs:
            acc = eval_presence(model, va_seq, device)
            print(f"ep {ep:3d} loss {tot/len(tr_seq):.4f} | val note-acc {acc:.3f}")

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": cache.mean, "std": cache.std}, CKPT)
    print(f"saved -> {CKPT}")


@torch.no_grad()
def eval_presence(model, seqs, device):
    model.eval(); correct = total = 0
    for s in seqs:
        ctx = torch.tensor(s["ctx"][None], device=device)
        prev = torch.tensor(s["prev"][None], device=device)
        diff = torch.tensor([s["diff"]], device=device)
        out, _ = model(ctx, diff, prev)
        (rp, rx, ry, rd), (bp, bx, by, bd) = Stage2Net.split(out)
        t = s["tgt"]
        for logits, ti, pres in ((rx, 1, t[:, 0]), (ry, 2, t[:, 0]), (rd, 3, t[:, 0]),
                                 (bx, 5, t[:, 4]), (by, 6, t[:, 4]), (bd, 7, t[:, 4])):
            m = pres.astype(bool)
            if m.any():
                pred = logits[0].argmax(-1).cpu().numpy()[m]
                correct += int((pred == t[:, ti][m]).sum()); total += int(m.sum())
    return correct / max(1, total)


if __name__ == "__main__":
    main()
