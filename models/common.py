"""Shared helpers for both stages: difficulty ids, data loading, mel cache."""
from __future__ import annotations

import json
import os
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

DIFFICULTIES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
DIFF_IDX = {d: i for i, d in enumerate(DIFFICULTIES)}
N_DIFF = len(DIFFICULTIES)

# held-out songs for validation (never trained on)
VAL_SONGS = {"beatsaber", "crabrave", "turnmeon"}


def list_beatmaps(data_dir: Path):
    """-> list of dicts {song, diff, diff_idx, json_path, mel_path, is_val}."""
    out = []
    for sd in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        mel = sd / "mel.npy"
        if not mel.exists():
            continue
        for jp in sorted(sd.glob("*.json")):
            if jp.name == "meta.json":
                continue
            diff = jp.stem
            if diff not in DIFF_IDX:
                continue
            out.append({"song": sd.name, "diff": diff, "diff_idx": DIFF_IDX[diff],
                        "json_path": jp, "mel_path": mel, "is_val": sd.name in VAL_SONGS})
    return out


@lru_cache(maxsize=128)
def load_canonical(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class MelCache:
    """Lazily load + globally standardize mel arrays, kept in memory.

    FIFO eviction when the cache exceeds _MAX_MELS (~1-2 GB at typical mel sizes
    on the 327-song dataset). Keeps memory bounded across long training runs.
    """

    _MAX_MELS = 512

    def __init__(self):
        self._mels: dict[str, np.ndarray] = {}
        self.mean = 0.0
        self.std = 1.0

    def get_raw(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self._mels:
            if len(self._mels) >= self._MAX_MELS:
                self._mels.pop(next(iter(self._mels)))
            self._mels[key] = np.load(path)
        return self._mels[key]

    def fit_norm(self, paths):
        if not paths:
            raise ValueError(
                "MelCache.fit_norm: no beatmaps found. "
                "Run `python extract/build_dataset.py BeatmapLevelsData --out dataset` "
                "first, then pass --data dataset (not BeatmapLevelsData)."
            )
        acc, sq, n = 0.0, 0.0, 0
        for p in paths:
            m = self.get_raw(p)
            acc += m.sum(dtype=np.float64)
            sq += (m.astype(np.float64) ** 2).sum()
            n += m.size
        if n == 0:
            raise ValueError(
                "MelCache.fit_norm: all mel arrays are empty (total size = 0). "
                "The dataset folder exists but contains no usable mel.npy."
            )
        self.mean = acc / n
        self.std = float(np.sqrt(sq / n - self.mean ** 2)) or 1.0

    def get(self, path: Path) -> np.ndarray:
        return ((self.get_raw(path) - self.mean) / self.std).astype(np.float32)


def save_with_backup(state: dict, out_dir: Path, name: str) -> Path:
    """Write <name>.latest.pt (overwrite) and <name>.bak-<UTC ts>.pt (unique copy).

    The caller passes the same dict shape used elsewhere in the project:
        {"model": state_dict, "mean": float|ndarray, "std": float|ndarray,
         "hparams": {"hid": ..., "layers": ..., ...}}    # optional, recommended

    `hparams` records the architecture/training choices that produced this
    checkpoint so inference (or a future resume) can rebuild the model with
    matching shape — the previous format silently assumed fixed hid=256 etc.
    Missing keys are tolerated: older checkpoints without `hparams` still load.

    Returns the path to the latest file. Used by both training stages so the UI can
    see a fresh checkpoint after every eval and also keep a timestamped history.

    Uses a hardlink for the backup (O(1) on the same filesystem, NTFS since 1809)
    and falls back to a full copy on cloud-synced folders (OneDrive) where hardlinks
    are rejected with OSError(1314).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / f"{name}.latest.pt"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bak = out_dir / f"{name}.bak-{ts}.pt"
    torch.save(state, latest)
    try:
        os.link(latest, bak)
    except OSError:
        shutil.copy2(latest, bak)
    return latest


def check_hparams(ckpt_hparams: dict | None, current: dict, label: str) -> None:
    """Warn (not fail) if a resumed checkpoint's hparams drift from current CLI args.

    Called by both stage1/stage2 right after `model.load_state_dict(ckpt["model"])`.
    A drift means we're loading weights trained with a different shape: many keys
    will still load via `strict=False`-style silent partial loads, but anything
    that *did* load will produce garbage at shapes that don't match the running
    model's expectations. Loud warning + flushed stdout keeps this visible in
    the UI log so the user can decide whether to `--resume` with matching args.

    Only logs differences; never raises.
    """
    if not ckpt_hparams:
        return
    drift = {k: (ckpt_hparams.get(k), current.get(k))
             for k in current
             if ckpt_hparams.get(k) != current.get(k)}
    if drift:
        pretty = ", ".join(f"{k}: ckpt={a} != current={b}" for k, (a, b) in drift.items())
        print(f"[{label}] WARNING: hparams drift — {pretty}", flush=True)