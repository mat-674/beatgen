"""Atomic zip downloader for community Beat Saber maps.

For each ``LeaderboardRecord`` we:

  1. ``GET`` ``download_url`` into a temp file ``tmp/<hash>.zip.part``.
  2. Verify the zip opens and contains at least an ``Info.dat``.
  3. Unzip into a staging dir ``tmp/<hash>/``.
  4. Move the staging dir to ``<out>/<song_id>_<slug>/<hash>/`` via
     ``os.replace`` (atomic on the same filesystem).

The destination layout is::

    <out>/<song_id>_<slug>/<hash>/
        Info.dat
        EasyStandard.dat / NormalStandard.dat / HardStandard.dat / ...
        song.egg | song.ogg
        cover.jpg (optional, renamed from BeatSaver's cover image)

``extract/loaders.load_map_folder`` ([extract/loaders.py:130-161]) walks any
folder looking for ``Info.dat`` — this layout is exactly what it expects.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from community_providers.beatleader import LeaderboardRecord

log = logging.getLogger(__name__)

USER_AGENT = "beatgen-fetcher/1.0"


# ----------------------------------------------------------------- helpers
def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s or "").strip("_")
    return (s[:n] or "song").strip("_-")


def _safe_song_dir(out: Path, rec: LeaderboardRecord) -> Path:
    """``<out>/<song_id>_<slug>/<hash>/`` — different difficulties of the same
    song share the song dir; the hash is a per-version subdir."""
    name = f"{rec.song_id}_{_slug(rec.name)}" if rec.song_id else _slug(rec.name)
    return out / name / rec.hash


def _parse_info_metadata(info_path: Path) -> dict:
    try:
        with info_path.open("rb") as f:
            return json.loads(f.read().decode("utf-8", errors="surrogateescape"))
    except (OSError, json.JSONDecodeError):
        return {}


def _expected_difficulty_files(info: dict) -> list[str]:
    """Return the list of relative ``.dat`` paths the map should expose
    (Standard characteristic only — beatgen ignores the rest)."""
    out: list[str] = []
    for dset in info.get("_difficultyBeatmapSets", []):
        if dset.get("_beatmapCharacteristicName") != "Standard":
            continue
        for dbm in dset.get("_difficultyBeatmaps", []):
            fn = dbm.get("_beatmapFilename")
            if fn:
                out.append(fn)
    return out


def _download_zip(url: str, dest: Path, timeout: float = 60.0,
                  chunk: int = 64 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, \
                 open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=chunk)
            os.replace(tmp, dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            wait = min(2 ** attempt, 8)
            log.warning("download %s failed (%s) — retry in %.1fs", url[:60], e, wait)
            time.sleep(wait)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    raise RuntimeError(f"failed to download {url}: {last_err}")


# ----------------------------------------------------------------- public API
class DownloadResult:
    __slots__ = ("path", "skipped_reason", "info")

    def __init__(self, path: Path | None, skipped_reason: str | None = "",
                 info: dict | None = None):
        self.path = path
        self.skipped_reason = skipped_reason
        self.info = info or {}


def download_and_extract(rec: LeaderboardRecord, out: Path) -> DownloadResult:
    """Download ``rec.download_url`` and atomically unpack into ``out``.

    Returns ``DownloadResult.path`` on success or ``DownloadResult.skipped_reason``
    if the map already exists / is malformed.
    """
    out.mkdir(parents=True, exist_ok=True)

    final_dir = _safe_song_dir(out, rec)
    if (final_dir / "Info.dat").exists():
        log.debug("[cache] hit %s", final_dir)
        return DownloadResult(final_dir, "cached")

    with tempfile.TemporaryDirectory(prefix="bldl_", dir=out) as stage_root:
        stage = Path(stage_root)
        zip_path = stage / "map.zip"
        try:
            _download_zip(rec.download_url, zip_path)
        except Exception as e:
            log.error("[dl-fail] %s: %s", rec.name, e)
            raise

        # Inspect zip first — fail fast on junk.
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                info_name = next((n for n in names if n.lower().endswith("info.dat")), None)
                if info_name is None:
                    raise RuntimeError("zip has no Info.dat")
                # Extract everything to staging dir; we'll trim afterwards.
                zf.extractall(stage)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"bad zip from {rec.download_url}: {e}") from e

        info_path = stage / "Info.dat"
        if not info_path.exists():
            # Some zips put Info.dat in a subfolder — walk and pick the first.
            for p in stage.rglob("Info.dat"):
                if p.is_file():
                    info_path = p
                    break
        if not info_path.exists():
            raise RuntimeError("Info.dat not found after unzip")

        info = _parse_info_metadata(info_path)
        expected_dats = _expected_difficulty_files(info)
        song_filename = info.get("_songFilename") or ""
        if not song_filename or not (stage / song_filename).exists():
            # Some packs put the audio in a nested folder — search.
            for p in stage.rglob("*"):
                if p.is_file() and p.name == song_filename:
                    break
            else:
                raise RuntimeError(f"audio file {song_filename!r} not in zip")

        # Move/rename to canonical layout under final_dir (atomic).
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        # ``stage`` -> ``final_dir``
        if final_dir.exists():
            # Race: another worker beat us to it. Accept that result.
            log.info("[race] %s already exists, using cached", final_dir)
            return DownloadResult(final_dir, "cached")

        try:
            os.replace(str(stage), str(final_dir))
        except OSError:
            # Cross-device move: fall back to copy + remove.
            shutil.move(str(stage), str(final_dir))

    # Best-effort: rename cover image if present (BeatSaver stores ``cover.jpg``
    # in some zips and ``<hash>.jpg`` in others).
    _try_normalize_cover(final_dir)
    # ``mel.npy`` etc. would NOT belong here — the fetcher stays purely
    # about transport; ``extract/build_dataset.py`` computes the mel later.
    return DownloadResult(final_dir, "", info)


def _try_normalize_cover(folder: Path) -> None:
    if (folder / "cover.jpg").exists():
        return
    for cand in folder.glob("*.jpg"):
        try:
            cand.replace(folder / "cover.jpg")
            return
        except OSError:
            continue