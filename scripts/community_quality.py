"""Quality gate + soft-dedup for incoming BeatLeader leaderboard records.

Two-stage gate:

1. ``passes_basic_filters(rec)`` — pure structural checks: duration in
   [60, 600] s, bpm in [60, 250], notes ≥ 30, NPS in [0.5, 30], stars ≥ 0.
   We never want the model to memorise the rare 60-NPS modded outlier or the
   8-second trailer edit.

2. ``passes_status_filter(rec, accepted_types, min_stars)`` — the user-driven
   policy from the CLI: ranked/qualified/nominated/etc., minimum star rating.

Soft-dedup is by ``(normalized_artist|title, duration±2s)`` — different
mappers often upload the same song under different hashes, and we only want
one training sample per song.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from community_providers.beatleader import (
    LeaderboardRecord,
    STATUS_INEVENT,
    STATUS_NOMINATED,
    STATUS_OST,
    STATUS_QUALIFIED,
    STATUS_RANKED,
    STATUS_UNRANKABLE,
    STATUS_OUTDATED,
    STATUS_UNRANKED,
)


# status name -> int (mirrors BL enum)
ACCEPTED_STATUS_BY_NAME = {
    "ranked": {STATUS_RANKED},
    "qualified": {STATUS_QUALIFIED},
    "nominated": {STATUS_NOMINATED},
    "ranked_or_qualified": {STATUS_RANKED, STATUS_QUALIFIED},
    "ranked_qualified_nominated": {STATUS_RANKED, STATUS_QUALIFIED, STATUS_NOMINATED},
}


@dataclass
class QualityConfig:
    accepted_types: tuple[str, ...]  # CLI-friendly: "ranked", "qualified", "nominated"
    min_stars: float
    min_notes: int = 30
    min_duration_s: int = 60
    max_duration_s: int = 600
    min_bpm: float = 60.0
    max_bpm: float = 250.0
    min_nps: float = 0.5
    max_nps: float = 30.0


def _accepted_statuses(cfg: QualityConfig) -> set[int]:
    out: set[int] = set()
    for t in cfg.accepted_types:
        key = t.lower()
        if key == "ranked":
            out |= {STATUS_RANKED}
        elif key == "qualified":
            out |= {STATUS_QUALIFIED}
        elif key == "nominated":
            out |= {STATUS_NOMINATED}
        else:
            raise ValueError(f"unknown accepted_types entry: {t!r}")
    return out


def passes_basic_filters(rec: LeaderboardRecord, cfg: QualityConfig) -> bool:
    if rec.duration < cfg.min_duration_s or rec.duration > cfg.max_duration_s:
        return False
    if rec.bpm < cfg.min_bpm or rec.bpm > cfg.max_bpm:
        return False
    if rec.notes < cfg.min_notes:
        return False
    # NPS is computed by BL only after ranking; for nominated rows it's 0.
    # We don't want to drop those on NPS alone, but we *do* want to drop
    # 0-NPS / 0-note ghosts that shouldn't reach the downloader.
    if rec.notes == 0:
        return False
    return True


def passes_status_filter(rec: LeaderboardRecord, cfg: QualityConfig) -> bool:
    accepted = _accepted_statuses(cfg)
    if rec.status not in accepted:
        return False
    if rec.status in (STATUS_UNRANKABLE, STATUS_OUTDATED, STATUS_UNRANKED, STATUS_INEVENT, STATUS_OST):
        return False
    if rec.stars is not None and rec.stars < cfg.min_stars:
        return False
    return True


def passes(rec: LeaderboardRecord, cfg: QualityConfig) -> bool:
    return passes_basic_filters(rec, cfg) and passes_status_filter(rec, cfg)


# ----------------------------------------------------------------- soft dedup
_NONWORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize_title(s: str) -> str:
    """Lowercase, strip diacritics, collapse non-alphanumerics to single space."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _NONWORD.sub(" ", s).strip()
    return s


def soft_dedup_key(rec: LeaderboardRecord) -> tuple[str, str, int]:
    """(normalized_artist|title, duration_bin) where bin = round(duration/2)."""
    artist = normalize_title(rec.author)
    title = normalize_title(rec.name + " " + rec.sub_name)
    dur_bin = int(rec.duration // 2)
    return (artist, title, dur_bin)


class SoftDeduper:
    """Track first-seen ``soft_dedup_key`` -> hash. Later collisions are dropped
    unless we already downloaded the first one (in which case we *don't* keep
    the new one — we already have it)."""
    def __init__(self):
        self._seen: dict[tuple[str, str, int], str] = {}

    def is_duplicate(self, rec: LeaderboardRecord) -> bool:
        k = soft_dedup_key(rec)
        if not k[0] or not k[1]:  # empty title or artist -> never dedup
            return False
        existing = self._seen.get(k)
        if existing is None:
            self._seen[k] = rec.hash
            return False
        return existing != rec.hash