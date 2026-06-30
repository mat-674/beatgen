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
import shutil
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
                          save_with_backup, check_hparams)

CKPT = Path("models/_ckpt/stage2.pt")

# Per-note encoding layout: one-hot presence + x(4) + y(3) + d(9) = 17 per hand.
N_X, N_Y, N_D, N_PRESENT = 4, 3, 9, 1
PER_NOTE_VEC = N_PRESENT + N_X + N_Y + N_D      # 17
RED_OFFSET, BLUE_OFFSET = 0, PER_NOTE_VEC       # 0, 17
NOTE_VEC = 2 * PER_NOTE_VEC                     # 34 (encoder width)
CTX_RADIUS = 6                                  # mel_context window half-width (frames, ~315 ms at HOP=512/44100)
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


def mel_context(mel: np.ndarray, frame: int, r: int | None = None) -> np.ndarray:
    """Local audio summary around an action: concat(mean, max) over a +/-r window.

    A plain mean over +/-1 frame gave the GRU almost no music to ground colour/position
    on, so it leaned on the previous-note teacher forcing and collapsed (all one colour
    / one column). A wider window plus a max channel restores real audio discrimination.

    `r=None` falls back to the module-level `CTX_RADIUS` (current default 6 — was 4
    in the 327-song run; bumped because ranked maps have more chromatic detail that
    benefits from a slightly larger audio window).
    """
    if r is None:
        r = CTX_RADIUS
    T = mel.shape[0]
    if T == 0:
        return np.zeros(2 * mel.shape[1], dtype=np.float32)
    f = min(max(int(frame), 0), T - 1)               # clamp so the window is never empty
    a, b = max(0, f - r), min(T, f + r + 1)
    win = mel[a:b]
    return np.concatenate([win.mean(axis=0), win.max(axis=0)]).astype(np.float32)


class Stage2Net(nn.Module):
    def __init__(self, n_diff=N_DIFF, demb=16, hid=384, layers=3, ctx_dim=CTX_DIM):
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


def seq_loss(out, tgt, device, pos_weight=None, lengths=None):
    """Per-sequence loss. `tgt` is a numpy array of shape (B, L, 8) for the *unpadded*
    longest sequence in the batch; `lengths` (1-D tensor/array of ints, length B)
    tells which prefix of each row is real (the rest is padding from pad_sequence).

    When `lengths is None` we treat the whole batch as one real sequence (bs=1
    legacy path) — keeps the original single-sequence training mode working.
    """
    (rp, rx, ry, rd), (bp, bx, by, bd) = Stage2Net.split(out)
    t = torch.as_tensor(tgt, device=device)
    if lengths is None:
        # Legacy single-sequence mode: every position is valid, keep old math.
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

    # ---- Packed mode: mask the padding before reducing ----
    B, L = rp.shape
    lens = torch.as_tensor(lengths, device=device).clamp(min=1, max=L)
    # mask[i, j] = 1 for j < lens[i], 0 otherwise. (B, L) float.
    mask = (torch.arange(L, device=device)[None, :] < lens[:, None]).float()
    rw, bw = (pos_weight if pos_weight is not None else
              (torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)))
    # Mean over valid positions only — divide by mask.sum(), not B*L.
    def bce_masked(logits, target, w):
        # bf16-safe: compute in fp32 by casting logits/target inside the reduction.
        per = F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), pos_weight=w, reduction="none"
        )  # (B, L)
        return (per * mask).sum() / mask.sum().clamp(min=1.0)
    loss = bce_masked(rp, t[:, :, 0], rw) + bce_masked(bp, t[:, :, 4], bw)

    def ce_masked(logits, ti, present):
        m = present.bool() & mask.bool()      # valid AND hand is present
        if m.any():
            return F.cross_entropy(logits[m].float(), t[:, :, ti][m])
        return torch.zeros((), device=device)
    pr, pb = t[:, :, 0], t[:, :, 4]
    loss = loss + ce_masked(rx, 1, pr) + ce_masked(ry, 2, pr) + ce_masked(rd, 3, pr)
    loss = loss + ce_masked(bx, 5, pb) + ce_masked(by, 6, pb) + ce_masked(bd, 7, pb)
    return loss


def pack_batch(seqs, device):
    """Pad a list of dicts {diff, ctx, prev, tgt} (already sorted by caller) into a
    padded batch tensor. Returns:
        ctx:  (B, L_max, 2*N_MELS)  fp32
        prev: (B, L_max, NOTE_VEC)  fp32
        tgt:  (B, L_max, 8)         int64   (rows padded with 0)
        diff: (B,)                  int64
        lens: (B,)                  int64   (real length of each row)
    """
    import torch.nn.utils.rnn as rnn
    ctx = rnn.pad_sequence([torch.as_tensor(s["ctx"]) for s in seqs],
                           batch_first=True).to(device)
    prev = rnn.pad_sequence([torch.as_tensor(s["prev"]) for s in seqs],
                            batch_first=True).to(device)
    tgt = rnn.pad_sequence([torch.as_tensor(s["tgt"]) for s in seqs],
                           batch_first=True).to(device)
    lens = torch.tensor([len(s["tgt"]) for s in seqs], dtype=torch.long, device=device)
    diff = torch.tensor([s["diff"] for s in seqs], dtype=torch.long, device=device)
    return ctx, prev, tgt, diff, lens


@torch.no_grad()
def eval_presence(model, seqs, device, bs: int = 1):
    """Balanced accuracy over per-step present/absent predictions.

    Returns a dict with `note_acc` (accuracy on present steps — the legacy metric)
    and `bal_acc` (mean of present + absent accuracy — robust to class imbalance).
    The `note_acc` field is kept for backward compatibility with the UI plot.

    bs=1 is the legacy per-sequence path; bs>1 sorts val by length, packs in
    buckets, and skips padded positions via the `lens` from pack_batch (a
    padded row's `tgt[:, 0]` and `tgt[:, 4]` are both 0, but we never look at
    them because we iterate `for k, ln in enumerate(lens.cpu().numpy().tolist())`
    and slice `[:, :ln]`).
    """
    model.eval()
    correct_p = total_p = correct_a = total_a = 0
    if bs <= 1:
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
    else:
        # Sort by length so each bucket has low padding overhead.
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["tgt"]))
        i = 0
        while i < len(seqs):
            j = min(i + bs, len(seqs))
            batch = [seqs[k] for k in order[i:j]]
            ctx, prev, tgt, diff, lens = pack_batch(batch, device)
            out, _ = model(ctx, diff, prev)
            (rp, _, _, _), (bp, _, _, _) = Stage2Net.split(out)
            rp_b = (torch.sigmoid(rp) > 0.5).cpu().numpy()
            bp_b = (torch.sigmoid(bp) > 0.5).cpu().numpy()
            t_b = tgt.cpu().numpy()
            for k, ln in enumerate(lens.cpu().numpy().tolist()):
                for logits_row, truth in (
                    (rp_b[k, :ln], t_b[k, :ln, 0].astype(bool)),
                    (bp_b[k, :ln], t_b[k, :ln, 4].astype(bool)),
                ):
                    correct_p += int((logits_row & truth).sum())
                    total_p   += int(truth.sum())
                    correct_a += int((~logits_row & ~truth).sum())
                    total_a   += int((~truth).sum())
            i = j
    note_acc = correct_p / max(1, total_p)
    bal_acc  = (correct_p / max(1, total_p) + correct_a / max(1, total_a)) / 2
    return {"note_acc": note_acc, "bal_acc": bal_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=384,
                    help="GRU hidden width (was 256 in the 327-song run; "
                         "bumped to 384 for the 11× dataset).")
    ap.add_argument("--layers", type=int, default=3,
                    help="GRU layers (was 2; bumped to 3 for deeper note-pattern modelling).")
    ap.add_argument("--demb", type=int, default=16,
                    help="difficulty embedding width (rarely worth changing).")
    ap.add_argument("--bs", type=int, default=16,
                    help="packed-sequence batch size (was 1 — one sequence per step; "
                         "with 11× more data the GPU was idle).")
    ap.add_argument("--ctx-radius", type=int, default=CTX_RADIUS,
                    help=f"mel_context window half-width in frames "
                         f"(default {CTX_RADIUS} ≈ 315 ms; was 4 in the 327-song run).")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile.")
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

    model = Stage2Net(hid=args.hid, layers=args.layers, demb=args.demb).to(device)
    # torch.compile gives a clean 2-3× on small GRUs because the kernel-launch
    # overhead is a meaningful fraction of per-step time at bs=16 / L≤2048.
    # On Windows the default backend (inductor) needs Triton, which is often
    # missing or unusable in the local torch build. `aot_eager` backend runs
    # dynamo's AOT autograd but evaluates in plain eager mode — no kernel
    # codegen, no triton. If even that fails, suppress_errors=True forces
    # fallback to pure eager.
    import torch._dynamo as _d
    if (not args.no_compile) and (device == "cuda"):
        _d.config.suppress_errors = True
        try:
            model = torch.compile(model, backend="aot_eager")
            compiled = True
        except Exception as e:
            print(f"[stage2] torch.compile(aot_eager) failed ({e!r}); "
                  "running eager.", flush=True)
            compiled = False
    else:
        compiled = False
    if args.resume is not None:
        # weights_only=False: our own checkpoint format ({"model", "mean", "std", "hparams"}),
        # saved with save_with_backup. Trusted local file, not untrusted download.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        target = model._orig_mod if compiled and hasattr(model, "_orig_mod") else model
        target.load_state_dict(ckpt["model"])
        check_hparams(ckpt.get("hparams"),
                      {"hid": args.hid, "layers": args.layers, "demb": args.demb,
                       "ctx_radius": args.ctx_radius},
                      label="stage2")
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
    use_amp = (device == "cuda")

    # Track the best balanced-accuracy checkpoint and stop early when it stops
    # improving — see stage1.py for the rationale.
    best_bal = -1.0
    EARLY_STOP_PATIENCE_EVALS = 3
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        # Sort by length each epoch so each packed bucket has low padding waste.
        order = sorted(range(len(tr_seq)), key=lambda i: len(tr_seq[i]["tgt"]))
        tot = 0.0
        n_loss = 0
        i = 0
        while i < len(order):
            batch = [tr_seq[k] for k in order[i:i + args.bs]]
            i += args.bs
            ctx, prev, tgt, diff, lens = pack_batch(batch, device)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out, _ = model(ctx, diff, prev)
                # Loss stays fp32 — seq_loss already casts logits/target inside.
                loss = seq_loss(out, tgt, device, pos_weight, lengths=lens)
            else:
                out, _ = model(ctx, diff, prev)
                loss = seq_loss(out, tgt, device, pos_weight, lengths=lens)
            loss.backward(); opt.step()
            tot += loss.item(); n_loss += 1
        if ep % 5 == 0 or ep == args.epochs:
            acc = eval_presence(model, va_seq, device, bs=max(args.bs, 1))
            bal = acc["bal_acc"]
            # When compiled, the canonical state_dict lives behind `_orig_mod`.
            state = (model._orig_mod.state_dict()
                     if (compiled and hasattr(model, "_orig_mod"))
                     else model.state_dict())
            if bal > best_bal:
                best_bal = bal
                no_improve = 0
                best_path = args.out_dir / "stage2.best.pt"
                tmp = best_path.with_suffix(".tmp.pt")
                torch.save({"model": state,
                            "mean": cache.mean, "std": cache.std,
                            "hparams": {"hid": args.hid, "layers": args.layers,
                                        "demb": args.demb, "ctx_radius": args.ctx_radius}},
                           tmp)
                os.replace(tmp, best_path)
                best_tag = f" best bal-acc {bal:.3f} -> {best_path}"
                # Mirror best -> legacy stage2.pt so hardcoded-path callers
                # load the validated-best model instead of the latest one
                # (which is just the last-epoch weights and may be overfit).
                shutil.copy2(best_path, CKPT)
            else:
                no_improve += 1
                best_tag = f" (best bal-acc {best_bal:.3f}, no improve {no_improve}/{EARLY_STOP_PATIENCE_EVALS})"
                if no_improve >= EARLY_STOP_PATIENCE_EVALS:
                    print(f"[stage2] ep {ep:3d} loss {tot/max(1,n_loss):.4f} | "
                          f"val note-acc {acc['note_acc']:.3f} bal-acc {bal:.3f}{best_tag}",
                          flush=True)
                    print(f"[stage2] early stop: no bal-acc improvement for "
                          f"{EARLY_STOP_PATIENCE_EVALS * 5} epochs", flush=True)
                    break
            print(f"[stage2] ep {ep:3d} loss {tot/max(1,n_loss):.4f} | "
                  f"val note-acc {acc['note_acc']:.3f} bal-acc {bal:.3f}{best_tag}",
                  flush=True)
            latest = save_with_backup(
                {"model": state, "mean": cache.mean, "std": cache.std,
                 "hparams": {"hid": args.hid, "layers": args.layers,
                             "demb": args.demb, "ctx_radius": args.ctx_radius}},
                args.out_dir, "stage2",
            )
            print(f"[stage2] saved -> {latest}", flush=True)

    print(f"[stage2] done -> {args.out_dir / 'stage2.latest.pt'} "
          f"(best bal-acc {best_bal:.3f} -> {args.out_dir / 'stage2.best.pt'})", flush=True)


if __name__ == "__main__":
    main()