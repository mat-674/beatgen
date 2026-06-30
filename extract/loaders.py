"""Omnivorous, recursive level discovery.

Walks any folder and yields normalized "level" dicts from whatever it finds:
  - Unity asset bundles (UnityFS magic, any/no extension)  -> Beat Saber OST/DLC
  - standard BeatSaver map folders (Info.dat + .dat + .ogg/.egg)  -> custom maps

A level dict:
    {
      "song_id": str,                 # unique slug
      "bpm": float,
      "audio": ("path", str) | ("wav_bytes", bytes),
      "difficulties": {DiffName: canonical_beatmap_dict},
      "source": str,                  # where it came from (for logging)
    }
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from schema.canonical import (DIFFICULTY_NAMES, bpm_from_audio_data, parse_beatmap)

STANDARD = "Standard"
SKIP_DIRS = {".git", ".venv", "__pycache__", "dataset", "out", "node_modules"}
UNITY_MAGIC = b"UnityFS\x00"
VALID_DIFFS = set(DIFFICULTY_NAMES.values())


# ---------------------------------------------------------------- helpers
def slugify(name: str, n: int = 48) -> str:
    s = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    return (s[:n] or "song").lower()


def load_json_textasset(blob: bytes) -> dict:
    if blob[:2] == b"\x1f\x8b":            # gzip
        blob = gzip.decompress(blob)
    return json.loads(blob.decode("utf-8", "surrogateescape"))


def is_unity_bundle(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == UNITY_MAGIC
    except OSError:
        return False


# ---------------------------------------------------------------- bundle
def _to_bytes(script) -> bytes:
    if isinstance(script, bytes):
        return script
    if isinstance(script, str):
        return script.encode("utf-8", "surrogateescape")
    return bytes(script)


def load_bundle_level(path: Path) -> dict | None:
    import UnityPy
    env = UnityPy.load(str(path))
    audio_wav = None
    level_mb = None
    text_by_pid: dict[int, tuple[str, bytes]] = {}

    for obj in env.objects:
        t = obj.type.name
        if t == "AudioClip" and audio_wav is None:
            clip = obj.read()
            for _, wav in clip.samples.items():
                audio_wav = wav
                break
        elif t == "MonoBehaviour":
            try:
                if obj.serialized_type and obj.serialized_type.node:
                    tree = obj.read_typetree()
                    if "_difficultyBeatmapSets" in tree:
                        level_mb = tree
            except Exception:
                pass
        elif t == "TextAsset":
            ta = obj.read()
            text_by_pid[obj.path_id] = (getattr(ta, "m_Name", ""),
                                        _to_bytes(getattr(ta, "m_Script", b"")))
    if level_mb is None or audio_wav is None:
        return None

    bpm = 0.0
    for name, blob in text_by_pid.values():
        if ".audio" in name:
            try:
                bpm = bpm_from_audio_data(load_json_textasset(blob))
            except Exception:
                pass
            break

    difficulties = {}
    for dset in level_mb.get("_difficultyBeatmapSets", []):
        if dset.get("_beatmapCharacteristicSerializedName") != STANDARD:
            continue
        for dbm in dset.get("_difficultyBeatmaps", []):
            pid = dbm.get("_beatmapAsset", {}).get("m_PathID", 0)
            name = DIFFICULTY_NAMES.get(int(dbm.get("_difficulty", -1)))
            if name and pid in text_by_pid:
                difficulties[name] = load_json_textasset(text_by_pid[pid][1])

    if not difficulties:
        return None
    if bpm <= 0:
        bpm = _fallback_bpm(difficulties)
    return {"song_id": slugify(path.stem), "bpm": bpm,
            "audio": ("wav_bytes", audio_wav), "difficulties": difficulties,
            "source": str(path)}


def _fallback_bpm(difficulties: dict, default: float = 120.0) -> float:
    """No BPM asset (rare legacy V2): assume notes land on integer-ish beats -> guess 120."""
    return default


# ---------------------------------------------------------------- map folder
def _find_info(filenames) -> str | None:
    for fn in filenames:
        if fn.lower() == "info.dat":
            return fn
    return None


def load_map_folder(folder: Path, info_name: str) -> dict | None:
    info = json.loads((folder / info_name).read_text(encoding="utf-8", errors="surrogateescape"))
    bpm = float(info.get("_beatsPerMinute", 0) or 0)
    audio_file = info.get("_songFilename")
    if not audio_file or bpm <= 0:
        return None
    audio_path = folder / audio_file
    if not audio_path.exists():
        return None

    difficulties = {}
    for dset in info.get("_difficultyBeatmapSets", []):
        if dset.get("_beatmapCharacteristicName") != STANDARD:
            continue
        for dbm in dset.get("_difficultyBeatmaps", []):
            name = dbm.get("_difficulty")
            fn = dbm.get("_beatmapFilename")
            if name not in VALID_DIFFS or not fn:
                continue
            dat = folder / fn
            if not dat.exists():
                continue
            try:
                raw = json.loads(dat.read_text(encoding="utf-8", errors="surrogateescape"))
                difficulties[name] = raw
            except Exception:
                pass
    if not difficulties:
        return None
    return {"song_id": slugify(folder.name), "bpm": bpm,
            "audio": ("path", str(audio_path)), "difficulties": difficulties,
            "source": str(folder)}


# ---------------------------------------------------------------- discovery
def discover_levels(root: Path):
    """Yield raw level dicts (difficulties still in their native schema)."""
    root = Path(root)
    if root.is_file():
        if is_unity_bundle(root):
            lvl = load_bundle_level(root)
            if lvl:
                yield lvl
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        d = Path(dirpath)
        info = _find_info(filenames)
        if info:
            lvl = load_map_folder(d, info)
            if lvl:
                yield lvl
            continue                      # don't also scan a map folder for bundles
        for fn in filenames:
            p = d / fn
            if is_unity_bundle(p):
                lvl = load_bundle_level(p)
                if lvl:
                    yield lvl


# ---------------------------------------------------------------- lightweight discovery
# `discover_levels` above opens each bundle via UnityPy in the calling process — fine for
# a one-shot script, but it pins ~50 MB of WAV bytes per level into memory simultaneously.
# When called from a parallel pool with hundreds of OST bundles, that explodes RAM.
# The two iterators below are cheap: they only walk the filesystem and look at file
# headers; the heavy UnityPy load happens later, in the worker that owns the level.

def iter_level_tasks(root: Path):
    """Yield light-weight "level tasks" — each is (kind, payload).

    `kind` is one of:
      - "bundle":   payload = absolute bundle path (Unity bundle, magic-checked)
      - "map":      payload = (folder, info_filename)

    UnityPy / librosa are NOT touched here — call `load_bundle_level` /
    `load_map_folder` on the worker side.
    """
    root = Path(root)
    if root.is_file():
        if is_unity_bundle(root):
            yield ("bundle", str(root))
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        d = Path(dirpath)
        info = _find_info(filenames)
        if info:
            yield ("map", (str(d), info))
            continue                      # map folders don't host bundles
        for fn in filenames:
            p = d / fn
            if is_unity_bundle(p):
                yield ("bundle", str(p))


def normalize_difficulties(level: dict) -> dict[str, dict]:
    """Convert each native beatmap to the canonical schema using the level BPM."""
    bpm = level["bpm"]
    return {name: parse_beatmap(raw, bpm) for name, raw in level["difficulties"].items()}
