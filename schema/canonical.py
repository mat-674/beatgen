"""Canonical beatmap representation, unifying BeatSaber V2 (plain) and V4 (gzip).

Canonical note:  {"b": beat, "t": sec, "x": 0-3, "y": 0-2, "c": 0|1, "d": 0-8}
Canonical bomb:  {"b": beat, "t": sec, "x": 0-3, "y": 0-2}
Canonical wall:  {"b": beat, "t": sec, "dur": beats, "x": 0-3, "y": 0-2, "w": w, "h": h}

c: color 0=red(left) 1=blue(right).  d: cut direction 0-8 (8=any/dot).
"""
from __future__ import annotations

# BeatSaber difficulty enum index -> name (matches _difficulty in BeatmapLevelData)
DIFFICULTY_NAMES = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Expert", 4: "ExpertPlus"}
DIFFICULTY_INDEX = {v: k for k, v in DIFFICULTY_NAMES.items()}


def _sec(beat: float, bpm: float) -> float:
    return beat * 60.0 / bpm if bpm else 0.0


def parse_v2(data: dict, bpm: float) -> dict:
    """Parse a V2.x.x beatmap dict -> canonical {notes, bombs, walls}."""
    notes, bombs = [], []
    for n in data.get("_notes", []):
        b = float(n.get("_time", 0.0))
        x = int(n.get("_lineIndex", 0))
        y = int(n.get("_lineLayer", 0))
        t = int(n.get("_type", 0))
        if t == 3:  # bomb
            bombs.append({"b": b, "t": _sec(b, bpm), "x": x, "y": y})
        elif t in (0, 1):
            notes.append({"b": b, "t": _sec(b, bpm), "x": x, "y": y,
                          "c": t, "d": int(n.get("_cutDirection", 8))})
    walls = []
    for o in data.get("_obstacles", []):
        b = float(o.get("_time", 0.0))
        full = int(o.get("_type", 0)) == 0
        walls.append({"b": b, "t": _sec(b, bpm), "dur": float(o.get("_duration", 0.0)),
                      "x": int(o.get("_lineIndex", 0)), "y": 0 if full else 2,
                      "w": int(o.get("_width", 1)), "h": 5 if full else 3})
    return {"notes": notes, "bombs": bombs, "walls": walls}


def _deref(elements, data, fields):
    """V4 stores objects as element{b,i,...} referencing data[i]{fields}. Merge them."""
    out = []
    for el in elements:
        d = data[el.get("i", 0)] if data else {}
        merged = {"b": float(el.get("b", 0.0))}
        for k in fields:
            merged[k] = d.get(k, 0)
        out.append(merged)
    return out


def parse_v4(data: dict, bpm: float) -> dict:
    """Parse a V4.0.0 beatmap dict -> canonical {notes, bombs, walls}."""
    notes = []
    for n in _deref(data.get("colorNotes", []), data.get("colorNotesData", []),
                    ("x", "y", "c", "d")):
        notes.append({"b": n["b"], "t": _sec(n["b"], bpm), "x": int(n["x"]),
                      "y": int(n["y"]), "c": int(n["c"]), "d": int(n["d"])})
    bombs = []
    for n in _deref(data.get("bombNotes", []), data.get("bombNotesData", []), ("x", "y")):
        bombs.append({"b": n["b"], "t": _sec(n["b"], bpm), "x": int(n["x"]), "y": int(n["y"])})
    walls = []
    for o in _deref(data.get("obstacles", []), data.get("obstaclesData", []),
                    ("x", "y", "d", "w", "h")):
        walls.append({"b": o["b"], "t": _sec(o["b"], bpm), "dur": float(o["d"]),
                      "x": int(o["x"]), "y": int(o["y"]), "w": int(o["w"]), "h": int(o["h"])})
    return {"notes": notes, "bombs": bombs, "walls": walls}


def parse_v3(data: dict, bpm: float) -> dict:
    """Parse a V3.x.x beatmap dict (inline notes) -> canonical {notes, bombs, walls}."""
    notes = [{"b": float(n.get("b", 0.0)), "t": _sec(float(n.get("b", 0.0)), bpm),
              "x": int(n.get("x", 0)), "y": int(n.get("y", 0)),
              "c": int(n.get("c", 0)), "d": int(n.get("d", 8))}
             for n in data.get("colorNotes", [])]
    bombs = [{"b": float(n.get("b", 0.0)), "t": _sec(float(n.get("b", 0.0)), bpm),
              "x": int(n.get("x", 0)), "y": int(n.get("y", 0))}
             for n in data.get("bombNotes", [])]
    walls = [{"b": float(o.get("b", 0.0)), "t": _sec(float(o.get("b", 0.0)), bpm),
              "dur": float(o.get("d", 0.0)), "x": int(o.get("x", 0)),
              "y": int(o.get("y", 0)), "w": int(o.get("w", 1)), "h": int(o.get("h", 1))}
             for o in data.get("obstacles", [])]
    return {"notes": notes, "bombs": bombs, "walls": walls}


def parse_beatmap(data: dict, bpm: float) -> dict:
    """Dispatch on version string to the right parser (V2 plain / V3 inline / V4 indexed)."""
    ver = str(data.get("version", data.get("_version", "2.0.0")))
    if ver.startswith("4"):
        return parse_v4(data, bpm)
    if ver.startswith("3"):
        return parse_v3(data, bpm)
    return parse_v2(data, bpm)


def bpm_from_audio_data(audio: dict) -> float:
    """Derive a single representative BPM from a V4 *.audio.gz dict."""
    sc = float(audio.get("songSampleCount", 0))
    fr = float(audio.get("songFrequency", 44100)) or 44100.0
    bpmd = audio.get("bpmData", [])
    if sc and bpmd:
        eb = float(bpmd[-1].get("eb", 0.0))
        seconds = sc / fr
        if seconds > 0 and eb > 0:
            return eb / (seconds / 60.0)
    return 0.0
