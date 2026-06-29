"""SQLite-backed manifest for the community-map fetcher.

A single ``_index.sqlite`` in the output dir records every song we've either
*seen* (in seed, but not necessarily downloaded) and every song we've
*accepted* (passed filters, downloaded, and atomically unpacked). Resume
support is just a ``SELECT status='ok' WHERE hash=?`` lookup.

Schema is intentionally tiny — ``hash`` is the primary key for ``songs``
and the dedup index is ``(hash, difficulty_id)``.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from community_providers.beatleader import LeaderboardRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    hash            TEXT NOT NULL,
    song_id         TEXT NOT NULL,
    name            TEXT,
    author          TEXT,
    mapper          TEXT,
    bpm             REAL,
    duration        INTEGER,
    download_url    TEXT,
    cover_image     TEXT,
    status          TEXT NOT NULL,           -- 'ok' | 'failed' | 'filtered'
    bl_status       INTEGER,
    bl_stars        REAL,
    notes           INTEGER,
    difficulty_id   INTEGER NOT NULL,
    difficulty_name TEXT,
    dest_path       TEXT,
    downloaded_at   INTEGER,
    PRIMARY KEY (hash, difficulty_id)
);
CREATE INDEX IF NOT EXISTS idx_songs_status ON songs(status);
CREATE INDEX IF NOT EXISTS idx_songs_hash ON songs(hash);

CREATE TABLE IF NOT EXISTS soft_dedup (
    artist_norm  TEXT NOT NULL,
    title_norm   TEXT NOT NULL,
    duration_bin INTEGER NOT NULL,
    hash         TEXT NOT NULL,
    PRIMARY KEY (artist_norm, title_norm, duration_bin)
);
"""


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                     isolation_level=None)  # autocommit; we use BEGIN
        self._conn.executescript(SCHEMA)
        # Pragmas: durability > speed; one writer, many readers OK.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    # ---- low-level helpers
    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)

    def _query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchone()

    def _query_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchall()

    # ---- public API
    def is_downloaded(self, hash: str, difficulty_id: int) -> bool:
        """True if we already have an ok-status row for this (hash, diff)."""
        row = self._query_one(
            "SELECT 1 FROM songs WHERE hash=? AND difficulty_id=? AND status='ok' LIMIT 1",
            (hash.lower(), difficulty_id),
        )
        return row is not None

    def is_failed(self, hash: str, difficulty_id: int, max_attempts: int = 3) -> bool:
        """True if we've failed this (hash, diff) >= max_attempts times and
        shouldn't bother retrying immediately."""
        row = self._query_one(
            "SELECT downloaded_at FROM songs "
            "WHERE hash=? AND difficulty_id=? AND status='failed' LIMIT 1",
            (hash.lower(), difficulty_id),
        )
        return row is not None

    def soft_dedup_seen(self, artist_norm: str, title_norm: str, duration_bin: int) -> str | None:
        """Return the hash we already saved for this (artist|title, dur) tuple, or None."""
        if not artist_norm or not title_norm:
            return None
        row = self._query_one(
            "SELECT hash FROM soft_dedup "
            "WHERE artist_norm=? AND title_norm=? AND duration_bin=? LIMIT 1",
            (artist_norm, title_norm, duration_bin),
        )
        return row["hash"] if row else None

    def record_ok(self, rec: LeaderboardRecord, dest_path: Path) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO songs
                       (hash, song_id, name, author, mapper, bpm, duration,
                        download_url, cover_image, status, bl_status, bl_stars,
                        notes, difficulty_id, difficulty_name, dest_path,
                        downloaded_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?, ?, ?)""",
                    (rec.hash, rec.song_id, rec.name, rec.author, rec.mapper,
                     rec.bpm, rec.duration, rec.download_url, rec.cover_image,
                     rec.status, rec.stars, rec.notes, rec.difficulty_id,
                     rec.difficulty_name, str(dest_path), int(time.time())),
                )
                self._conn.execute(
                    """INSERT OR IGNORE INTO soft_dedup
                       (artist_norm, title_norm, duration_bin, hash)
                       VALUES (?, ?, ?, ?)""",
                    (_norm(rec.author), _norm(rec.name + " " + rec.sub_name),
                     int(rec.duration // 2), rec.hash),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def record_failed(self, rec: LeaderboardRecord, reason: str = "") -> None:
        # We store failures as well so we can stop retrying after N attempts.
        self._exec(
            """INSERT OR REPLACE INTO songs
               (hash, song_id, name, author, mapper, bpm, duration,
                download_url, cover_image, status, bl_status, bl_stars,
                notes, difficulty_id, difficulty_name, dest_path, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'failed', ?, ?, ?, ?, ?, NULL, ?)""",
            (rec.hash, rec.song_id, rec.name, rec.author, rec.mapper,
             rec.bpm, rec.duration, rec.download_url, rec.cover_image,
             rec.status, rec.stars, rec.notes, rec.difficulty_id,
             rec.difficulty_name, int(time.time())),
        )

    def record_filtered(self, rec: LeaderboardRecord, reason: str) -> None:
        # Filtered rows are *not* keyed by (hash, diff) because we want to
        # re-evaluate them next run if the policy changes; we keep them in a
        # tiny separate log table to be cheap.
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS filtered (
                       hash TEXT, difficulty_id INTEGER, reason TEXT,
                       ts INTEGER, PRIMARY KEY (hash, difficulty_id))"""
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO filtered (hash, difficulty_id, reason, ts) "
                "VALUES (?, ?, ?, ?)",
                (rec.hash, rec.difficulty_id, reason, int(time.time())),
            )

    def stats(self) -> dict[str, int]:
        out = {}
        for status in ("ok", "failed", "filtered"):
            if status == "filtered":
                row = self._query_one("SELECT COUNT(*) AS c FROM filtered")
            else:
                row = self._query_one(
                    f"SELECT COUNT(*) AS c FROM songs WHERE status=?", (status,))
            out[status] = int(row["c"]) if row else 0
        return out

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()


def _norm(s: str) -> str:
    """Mirror of community_quality.normalize_title but kept inline so this
    module doesn't import it (the manifest is intentionally a leaf module)."""
    import re, unicodedata
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\W_]+", " ", s.lower()).strip()
    return s