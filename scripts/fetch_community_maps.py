"""Fetch quality-rated community Beat Saber maps from BeatLeader.

Defaults to "everything that's currently ranked" sorted by playCount (the
strongest in-the-wild signal we have). Drops anything that isn't Standard,
and dedupes per (artist|title, duration±2s) so a song reuploaded by a dozen
mappers only contributes one training sample.

Usage:
    python scripts/fetch_community_maps.py --max-songs 5000
    python scripts/fetch_community_maps.py --types ranked,qualified --min-stars 3
    python scripts/fetch_community_maps.py --resume
    python scripts/fetch_community_maps.py --out data/community_maps --rate 50

The downloaded maps land under ``<out>/<songId>_<slug>/<hash>/`` and are
ready to be picked up by ``extract/build_dataset.py``:
    python extract/build_dataset.py data/community_maps BeatmapLevelsData
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make this script runnable both as ``python scripts/fetch_community_maps.py``
# and ``python -m scripts.fetch_community_maps`` from the project root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from community_providers.beatleader import (  # noqa: E402
    LeaderboardRecord, RateLimiter, iter_leaderboards, fetch_song,
)
from community_quality import (  # noqa: E402
    QualityConfig, SoftDeduper, passes,
)
from community_manifest import Manifest  # noqa: E402
from community_downloader import download_and_extract  # noqa: E402

log = logging.getLogger("fetch_community_maps")


# ----------------------------------------------------------------- counters
class Counters:
    __slots__ = ("seen", "passed", "soft_dup", "downloaded", "cached", "failed")

    def __init__(self):
        self.seen = 0
        self.passed = 0
        self.soft_dup = 0
        self.downloaded = 0
        self.cached = 0
        self.failed = 0


# ----------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=_ROOT / "data" / "community_maps",
                   help="destination directory (default: data/community_maps)")
    p.add_argument("--max-songs", type=int, default=5000,
                   help="stop after this many downloaded songs (default: 5000)")
    p.add_argument("--types", default="ranked",
                   help="comma-separated BeatLeader types: ranked, qualified, nominated "
                        "(default: ranked). Each becomes a separate type= param.")
    p.add_argument("--sort-by", default="playCount",
                   help="MapSortBy field; one of: stars, playCount, passRating, "
                        "accRating, techRating, name, timestamp, bPM, duration, ... "
                        "(default: playCount)")
    p.add_argument("--order", default="desc", choices=("asc", "desc"),
                   help="sort order (default: desc)")
    p.add_argument("--min-stars", type=float, default=0.0,
                   help="drop difficulties with stars < this (default: 0; "
                        "nominated maps have null stars which pass)")
    p.add_argument("--min-notes", type=int, default=30,
                   help="drop difficulties with fewer than N notes (default: 30)")
    p.add_argument("--rate", type=int, default=50,
                   help="BL requests per window (default: 50; matches BL's budget)")
    p.add_argument("--rate-window", type=float, default=10.0,
                   help="rate-limit window in seconds (default: 10)")
    p.add_argument("--page-size", type=int, default=100,
                   help="page size for /leaderboards (default: 100)")
    p.add_argument("--max-pages", type=int, default=None,
                   help="stop after N pages of /leaderboards (debug aid)")
    p.add_argument("--resume", action="store_true",
                   help="reuse the existing _index.sqlite (default: True behaviour; "
                        "use --no-resume to start over)")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


# ----------------------------------------------------------------- orchestrator
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = QualityConfig(
        accepted_types=tuple(t.strip() for t in args.types.split(",") if t.strip()),
        min_stars=args.min_stars,
        min_notes=args.min_notes,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "_index.sqlite"
    if not args.resume and manifest_path.exists():
        log.warning("removing existing manifest %s", manifest_path)
        manifest_path.unlink()

    manifest = Manifest(manifest_path)
    limiter = RateLimiter(rate=args.rate, window=args.rate_window)
    deduper = SoftDeduper()
    counters = Counters()

    log.info("seed: types=%s sortBy=%s order=%s maxSongs=%d",
             cfg.accepted_types, args.sort_by, args.order, args.max_songs)
    t0 = time.monotonic()

    try:
        for rec in iter_leaderboards(
            types=cfg.accepted_types,
            sort_by=args.sort_by,
            order=args.order,
            page_size=args.page_size,
            max_pages=args.max_pages,
            limiter=limiter,
        ):
            counters.seen += 1
            if not passes(rec, cfg):
                continue
            counters.passed += 1
            if manifest.is_downloaded(rec.hash, rec.difficulty_id):
                counters.cached += 1
                if counters.downloaded + counters.cached >= args.max_songs:
                    log.info("reached --max-songs via cache, stopping")
                    break
                continue
            if deduper.is_duplicate(rec):
                counters.soft_dup += 1
                manifest.record_filtered(rec, "soft-dedup (artist|title,duration±2s)")
                continue
            try:
                res = download_and_extract(rec, args.out)
            except Exception as e:
                counters.failed += 1
                manifest.record_failed(rec, str(e))
                log.error("[fail] %s (%s) — %s", rec.name, rec.difficulty_name, e)
                continue
            manifest.record_ok(rec, res.path)
            counters.downloaded += 1
            elapsed = time.monotonic() - t0
            log.info("[ok %4d/%d] %s — %s (%.1fs; %.1f songs/min)",
                     counters.downloaded, args.max_songs, rec.name,
                     rec.difficulty_name, elapsed,
                     counters.downloaded / max(elapsed / 60, 1e-3))
            if counters.downloaded >= args.max_songs:
                log.info("reached --max-songs, stopping")
                break
    finally:
        manifest.close()

    elapsed = time.monotonic() - t0
    log.info(
        "done in %.1fs: seen=%d passed=%d soft_dup=%d cached=%d downloaded=%d failed=%d",
        elapsed, counters.seen, counters.passed, counters.soft_dup,
        counters.cached, counters.downloaded, counters.failed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())