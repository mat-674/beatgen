"""Shared helpers for both stages: difficulty ids, data loading, mel cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def load_canonical(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class MelCache:
    """Lazily load + globally standardize mel arrays, kept in memory."""

    def __init__(self):
        self._mels: dict[str, np.ndarray] = {}
        self.mean = 0.0
        self.std = 1.0

    def get_raw(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self._mels:
            self._mels[key] = np.load(path)
        return self._mels[key]

    def fit_norm(self, paths):
        acc, n = 0.0, 0
        sq = 0.0
        for p in paths:
            m = self.get_raw(p)
            acc += m.sum(dtype=np.float64)
            sq += (m.astype(np.float64) ** 2).sum()
            n += m.size
        self.mean = acc / n
        self.std = float(np.sqrt(sq / n - self.mean ** 2)) or 1.0

    def get(self, path: Path) -> np.ndarray:
        return ((self.get_raw(path) - self.mean) / self.std).astype(np.float32)
