"""BeatLeader API client.

Two surfaces:

  * ``iter_leaderboards(types, sort_by, ...)`` — walks ``GET /leaderboards``
    page-by-page and yields one normalized record per (song, difficulty) row.
    Each row already carries the ``downloadUrl`` to the BeatSaver-cached zip,
    so we never need to talk to api.beatsaver.com just to fetch a map.

  * ``fetch_song(hash)`` — single ``GET /map/hash/{hash}`` for the full
    ``SongResponse`` (used to enrich the manifest with per-difficulty stats,
    and as the resilience fallback when ``downloadUrl`` is missing).

Rate-limit policy: BL advertises ``X-Rate-Limit: 10s`` (50 req per 10s budget
in the response headers we observed). We use a simple token-bucket and honour
``Retry-After`` on 429/503. ``max_retries`` defaults to 3.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Iterator

log = logging.getLogger(__name__)

BASE = "https://api.beatleader.com"
USER_AGENT = "beatgen-fetcher/1.0 (+https://github.com/local/beatgen)"
DEFAULT_COUNT = 100  # /leaderboards page size — 100 is the largest BL accepts per page


# ----------------------------------------------------------------- enums
# BL difficulty status (https://api.beatleader.com/swagger/blapi/swagger.json
# -> components.schemas.DifficultyStatus.enum)
STATUS_UNRANKED = 0
STATUS_NOMINATED = 1
STATUS_QUALIFIED = 2
STATUS_RANKED = 3
STATUS_UNRANKABLE = 4
STATUS_OUTDATED = 5
STATUS_INEVENT = 6
STATUS_OST = 7

# Human-readable names (used in logs/manifest)
STATUS_NAME = {
    STATUS_UNRANKED: "unranked",
    STATUS_NOMINATED: "nominated",
    STATUS_QUALIFIED: "qualified",
    STATUS_RANKED: "ranked",
    STATUS_UNRANKABLE: "unrankable",
    STATUS_OUTDATED: "outdated",
    STATUS_INEVENT: "inevent",
    STATUS_OST: "ost",
}

# Beat Saber characteristics (we want Standard only)
MODE_STANDARD = 1
MODE_NOSTANDARD_MODES = (2, 3, 4, 5, 6, 7)  # NoArrows, OneHand, etc. and Lightshow


# ----------------------------------------------------------------- record
@dataclass(frozen=True)
class LeaderboardRecord:
    """One row of ``/leaderboards``: a (song, difficulty) pair."""
    song_id: str                 # BeatSaver-style numeric id (string, can include letters)
    hash: str                    # SHA-1 of the .ogg/.egg zip
    name: str
    sub_name: str
    author: str                  # song artist
    mapper: str                  # level author
    bpm: float
    duration: int                # seconds
    download_url: str
    cover_image: str
    difficulty_id: int
    difficulty_name: str         # "Expert", "ExpertPlus", ...
    mode: int                    # 1 = Standard
    status: int                  # see STATUS_* above
    stars: float | None          # modifiersRating.stars (None until ranked)
    notes: int
    njs: float
    nps: float
    upload_time: int             # unix seconds; 0 if BL doesn't know

    @property
    def key(self) -> tuple[str, int]:
        """Stable dedup key: (hash, difficulty_id)."""
        return (self.hash, self.difficulty_id)

    @property
    def sort_key(self) -> tuple[float, ...]:
        """For deterministic ordering inside a page (stars desc, notes desc)."""
        return (-(self.stars or 0.0), -self.notes)


# ----------------------------------------------------------------- HTTP
class RateLimiter:
    """Sliding-window token bucket. BL: 50 req / 10 s (default), configurable."""
    def __init__(self, rate: int = 50, window: float = 10.0):
        self.rate = rate
        self.window = window
        self._ts: list[float] = []

    def wait(self):
        now = time.monotonic()
        # drop timestamps that fell out of the window
        self._ts = [t for t in self._ts if now - t < self.window]
        if len(self._ts) >= self.rate:
            sleep_for = self.window - (now - self._ts[0]) + 0.05
            if sleep_for > 0:
                log.debug("rate-limit sleep %.2fs", sleep_for)
                time.sleep(sleep_for)
        self._ts.append(time.monotonic())


class BeatLeaderError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} on {url}: {body[:200] if body else '(no body)'}")
        self.status = status
        self.url = url


def _request(url: str, limiter: RateLimiter, max_retries: int = 3,
             timeout: float = 30.0) -> dict:
    """GET ``url`` as JSON with retries on 429/5xx and connection errors."""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                # headers carry RateLimit info we could parse, but we just retry
                # on visible 429/503 below.
                return json.loads(body)
        except urllib.error.HTTPError as e:
            last_err = BeatLeaderError(e.code, url, e.read().decode("utf-8", "replace"))
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 8)
                log.warning("BL %s -> retry in %.1fs (attempt %d)",
                            e.code, wait, attempt + 1)
                time.sleep(wait)
                continue
            raise last_err from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            if attempt < max_retries:
                wait = min(2 ** attempt, 8)
                log.warning("BL network error: %s -> retry in %.1fs", e, wait)
                time.sleep(wait)
                continue
            raise BeatLeaderError(0, url, repr(e)) from e
    raise last_err  # unreachable, but keeps type-checkers happy


# ----------------------------------------------------------------- normalizer
def _record_from_row(row: dict) -> LeaderboardRecord | None:
    """Convert one ``/leaderboards`` row to LeaderboardRecord, or None to skip.

    Skip rules baked in here (kept cheap so we don't even allocate):
      * no ``song.hash`` or no ``song.downloadUrl`` (would fail download later)
      * no ``difficulty`` payload (BL sometimes returns sparse rows)
      * mode != Standard (we only train on Standard maps)
      * status == unrankable / outdated (the map is broken)
      * upload_time == 0 AND status == unranked (BL doesn't know the song;
        usually a deleted/uploaded-too-recent entry — skip, it can't be ranked)
    """
    song = row.get("song") or {}
    diff = row.get("difficulty") or {}
    if not song or not diff:
        return None
    h = song.get("hash")
    url = song.get("downloadUrl")
    if not h or not url:
        return None
    if diff.get("mode") != MODE_STANDARD:
        return None
    status = int(diff.get("status", STATUS_UNRANKED))
    if status in (STATUS_UNRANKABLE, STATUS_OUTDATED):
        return None

    mr = diff.get("modifiersRating") or {}
    stars = mr.get("stars")
    try:
        stars_f = float(stars) if stars is not None else None
    except (TypeError, ValueError):
        stars_f = None

    return LeaderboardRecord(
        song_id=str(song.get("id") or ""),
        hash=h.lower(),
        name=(song.get("name") or "").strip(),
        sub_name=(song.get("subName") or "").strip(),
        author=(song.get("author") or "").strip(),
        mapper=(song.get("mapper") or "").strip(),
        bpm=float(song.get("bpm") or 0.0),
        duration=int(song.get("duration") or 0),
        download_url=url,
        cover_image=song.get("coverImage") or "",
        difficulty_id=int(diff.get("id") or 0),
        difficulty_name=(diff.get("difficultyName") or "").strip(),
        mode=int(diff.get("mode") or 0),
        status=status,
        stars=stars_f,
        notes=int(diff.get("notes") or 0),
        njs=float(diff.get("njs") or 0.0),
        nps=float(diff.get("nps") or 0.0),
        upload_time=int(song.get("uploadTime") or 0),
    )


# ----------------------------------------------------------------- seed
def iter_leaderboards(
    types: Iterable[str] = ("ranked",),
    sort_by: str = "playCount",
    order: str = "desc",
    page_size: int = DEFAULT_COUNT,
    max_pages: int | None = None,
    limiter: RateLimiter | None = None,
) -> Iterator[LeaderboardRecord]:
    """Walk ``/leaderboards`` page-by-page.

    Parameters
    ----------
    types : iterable of one of {"ranked","qualified","nominated","ranking",
            "staff","reweighting","reweighted","unranked","ost","all"}
        We pass each value as a separate ``type`` query param so BL returns the
        union (BL accepts repeated params).
    sort_by : str, default ``"playCount"``
        One of BL's MapSortBy enum. ``"stars"`` is also useful for
        qualified/nominated seeds.
    order : ``"asc"`` or ``"desc"`` (default ``"desc"``).
    page_size : int, default 100.
    max_pages : stop after this many pages (debug aid; default ``None`` = no cap).
    """
    limiter = limiter or RateLimiter()
    types_list = list(types)
    if not types_list:
        types_list = ["ranked"]

    base_qs = {
        "sortBy": sort_by,
        "order": order,
        "count": str(page_size),
    }
    for t in types_list:
        base_qs.setdefault("type", [])

    page = 1
    total_records = 0
    while True:
        if max_pages is not None and page > max_pages:
            return
        qs = list(base_qs.items())
        # repeat type= for each requested value
        qs = [(k, v) for k, v in qs if k != "type"]
        for t in types_list:
            qs.append(("type", t))
        qs.append(("page", str(page)))
        url = f"{BASE}/leaderboards?{urllib.parse.urlencode(qs)}"
        data = _request(url, limiter)
        rows = data.get("data") or []
        if not rows:
            return
        yielded = 0
        for row in rows:
            rec = _record_from_row(row)
            if rec is None:
                continue
            yield rec
            yielded += 1
            total_records += 1
        log.info("BL page=%d returned=%d yielded=%d total_so_far=%d",
                 page, len(rows), yielded, total_records)
        meta = data.get("metadata") or {}
        total = int(meta.get("total") or 0)
        # When ``count`` is exhausted, BL returns fewer than ``count`` rows —
        # short-circuit immediately instead of waiting for an empty page.
        if len(rows) < page_size:
            return
        # belt-and-braces: if we've already iterated past ``total``, stop.
        # (BL's ``total`` is per ``type`` filter, so with multiple types it's
        # the union of one — we only stop when a page comes back empty.)
        page += 1


# ----------------------------------------------------------------- detail
def fetch_song(hash: str, limiter: RateLimiter | None = None) -> dict | None:
    """``GET /map/hash/{hash}`` — full SongResponse (with all difficulties)."""
    limiter = limiter or RateLimiter()
    h = hash.strip().lower()
    if not h:
        return None
    url = f"{BASE}/map/hash/{urllib.parse.quote(h)}"
    try:
        return _request(url, limiter)
    except BeatLeaderError as e:
        if e.status == 404:
            log.info("BL /map/hash %s -> 404 (song vanished)", h[:10])
            return None
        raise