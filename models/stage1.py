"""Stage 1: per-frame "is there a note action here?" (onset/density).

CNN over mel + difficulty embedding -> BiGRU -> per-frame logit.

    python models/stage1.py --epochs 40                              # train from scratch
    python models/stage1.py --resume models/_ckpt/stage1.latest.pt  # fine-tune
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.audio import time_to_frame  # noqa: E402
from features.audio import N_MELS  # noqa: E402
from models.common import (MelCache, N_DIFF, list_beatmaps, load_canonical,  # noqa: E402
                          save_with_backup, check_hparams)

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

    def __init__(self, n_mels=N_MELS, n_diff=N_DIFF, demb=16, hid=384,
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
        for n in load_canonical(str(bm["json_path"]))["notes"]:
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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=384,
                    help="TCN hidden width (was 256 in the original 327-song run; "
                         "bumped to 384 for the 11× dataset — still <1GB VRAM).")
    ap.add_argument("--demb", type=int, default=16,
                    help="difficulty embedding width (rarely worth changing).")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile (useful for debugging or older torch).")
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
        print(f"[stage1] no beatmaps under {args.data.resolve()} — "
              f"a song dir must contain mel.npy and at least one <Difficulty>.json. "
              f"Build the dataset with: python extract/build_dataset.py BeatmapLevelsData --out dataset",
              flush=True)
        sys.exit(1)
    train = [b for b in beatmaps if not b["is_val"]]
    val = [b for b in beatmaps if b["is_val"]]
    cache = MelCache()
    cache.fit_norm([b["mel_path"] for b in beatmaps])
    labels = build_labels(beatmaps, cache)

    # global positive rate -> pos_weight (with zero-guard for an empty/note-less dataset)
    pos = sum(l.sum() for l in labels.values()); tot = sum(l.size for l in labels.values())
    if tot == 0 or pos == 0:
        print(f"[stage1] WARNING: dataset has no notes (pos={pos}, tot={tot}). "
              f"Falling back to pos_weight=1.0; the model will not learn.", flush=True)
        pos_w = 1.0
    else:
        pos_w = float((tot - pos) / pos)
    print(f"[stage1] beatmaps: {len(train)} train / {len(val)} val | "
          f"pos_rate={(pos/tot if tot else 0):.4f} pos_weight={pos_w:.1f}", flush=True)

    model = Stage1Net(hid=args.hid, demb=args.demb).to(device)
    # cudnn.benchmark=True picks the fastest Conv1d algo for the fixed CROP=1024 shape.
    # Safe here — the input shape never varies across the training loop.
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    # torch.compile: 2-3× speedup on TCN forward, but on Windows the default
    # backend (inductor) needs Triton — which often isn't installed/usable in
    # the local torch build. Use `aot_eager` backend instead: it runs dynamo's
    # AOT autograd but evaluates the resulting graph in plain eager mode, so
    # no kernel codegen / no triton dependency. This is a "free correctness"
    # pass with most of the per-step Python overhead removed.
    # If even that fails, suppress_errors=True makes any later compile failure
    # fall back to pure eager (slow but works).
    import torch._dynamo as _d
    if (not args.no_compile) and (device == "cuda"):
        _d.config.suppress_errors = True
        try:
            model = torch.compile(model, backend="aot_eager")
            compiled = True
        except Exception as e:
            print(f"[stage1] torch.compile(aot_eager) failed ({e!r}); "
                  "running eager.", flush=True)
            compiled = False
    else:
        compiled = False
    if args.resume is not None:
        # weights_only=False: our own checkpoint format ({"model", "mean", "std", "hparams"}),
        # saved with save_with_backup. Trusted local file, not untrusted download.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # When the model is compiled, state_dict lives behind `model._orig_mod`.
        target = model._orig_mod if compiled and hasattr(model, "_orig_mod") else model
        target.load_state_dict(ckpt["model"])
        check_hparams(ckpt.get("hparams"),
                      {"hid": args.hid, "demb": args.demb, "crop": CROP},
                      label="stage1")
        # keep the resumed mel-std in sync with the cache so the loss is comparable
        old_mean = float(ckpt.get("mean", cache.mean))
        old_std = float(ckpt.get("std", cache.std))
        if (cache.mean and abs(old_mean - cache.mean) > 0.05 * abs(cache.mean)
                or abs(old_std - cache.std) > 0.05 * abs(cache.std or 1.0)):
            print(f"[stage1] WARNING: resume ckpt was fitted on a different dataset "
                  f"(mean {old_mean:.4f} -> {cache.mean:.4f}, std {old_std:.4f} -> {cache.std:.4f}). "
                  f"Loss will be miscalibrated until lr decays.", flush=True)
        cache.mean = old_mean
        cache.std = old_std
        print(f"[stage1] resumed from {args.resume}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # pos_weight stays fp32 and the loss itself is kept OUTSIDE autocast: BCE
    # reduction in bf16 is fine on RTX 30/40 but mixing pos_weight dtype with the
    # autocast logits can flip underflow patterns. Standard pattern is autocast
    # around the forward only, fp32 loss outside.
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    use_amp = (device == "cuda")

    # Track the best F1 we have seen so we can save a stable `stage1.best.pt`
    # (the .latest.pt checkpoint always reflects the last epoch, which may have
    # overfit and be worse than an earlier one — `best.pt` is what inference
    # should actually load).
    best_f1 = -1.0
    # Patience for early stop: how many consecutive evals (each eval = every 5
    # epochs) we allow without improvement before bailing. 3 evals = 15 epochs
    # of grace, long enough to ride out the noisy plateau before F1 finally
    # ticks up again (see train_stage1.log: F1 was flat for ~25 epochs before
    # resuming growth).
    EARLY_STOP_PATIENCE_EVALS = 3
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train(); tot_loss = 0.0
        for _ in range(args.steps):
            mel, diff, lab = sample_batch(train, labels, cache, args.bs, device, rng)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(mel, diff)
                loss = lossf(logits, lab)
            else:
                loss = lossf(model(mel, diff), lab)
            loss.backward(); opt.step()
            tot_loss += loss.item()
        if ep % 5 == 0 or ep == args.epochs:
            p, r, f1 = evaluate(model, val, labels, cache, device)
            # best.pt: only overwrite on strict improvement so it never regresses.
            # When the model is compiled, state_dict() walks through `_orig_mod`.
            state = model.state_dict() if not (compiled and hasattr(model, "_orig_mod")) \
                else model._orig_mod.state_dict()
            if f1 > best_f1:
                best_f1 = f1
                no_improve = 0
                best_path = args.out_dir / "stage1.best.pt"
                tmp = best_path.with_suffix(".tmp.pt")
                torch.save({"model": state,
                            "mean": cache.mean, "std": cache.std,
                            "hparams": {"hid": args.hid, "demb": args.demb, "crop": CROP}},
                           tmp)
                os.replace(tmp, best_path)
                best_tag = f" best F1 {f1:.3f} -> {best_path}"
                # Mirror best -> legacy stage1.pt so callers that hardcode that
                # path load the validated-best weights, not the latest-epoch
                # ones (which may be overfit and worse). Cheap: copy not save.
                shutil.copy2(best_path, CKPT)
            else:
                no_improve += 1
                best_tag = f" (best F1 {best_f1:.3f}, no improve {no_improve}/{EARLY_STOP_PATIENCE_EVALS})"
                if no_improve >= EARLY_STOP_PATIENCE_EVALS:
                    print(f"[stage1] ep {ep:3d} loss {tot_loss/args.steps:.4f} | "
                          f"val P {p:.3f} R {r:.3f} F1 {f1:.3f}{best_tag}",
                          flush=True)
                    print(f"[stage1] early stop: no F1 improvement for "
                          f"{EARLY_STOP_PATIENCE_EVALS * 5} epochs", flush=True)
                    break
            print(f"[stage1] ep {ep:3d} loss {tot_loss/args.steps:.4f} | "
                  f"val P {p:.3f} R {r:.3f} F1 {f1:.3f}{best_tag}", flush=True)
            latest = save_with_backup(
                {"model": state, "mean": cache.mean, "std": cache.std,
                 "hparams": {"hid": args.hid, "demb": args.demb, "crop": CROP}},
                args.out_dir, "stage1",
            )
            print(f"[stage1] saved -> {latest}", flush=True)

    print(f"[stage1] done -> {args.out_dir / 'stage1.latest.pt'} "
          f"(best F1 {best_f1:.3f} -> {args.out_dir / 'stage1.best.pt'})", flush=True)


if __name__ == "__main__":
    main()
