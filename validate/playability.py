"""Lightweight playability pass over canonical notes (dedupe + basic parity)."""
from __future__ import annotations

# cut directions: 0 up,1 down,2 left,3 right,4 up-left,5 up-right,6 down-left,7 down-right,8 any
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4, 8: 8}


def clamp_notes(notes):
    out = []
    for n in notes:
        n = dict(n)
        n["x"] = min(3, max(0, int(n["x"])))
        n["y"] = min(2, max(0, int(n["y"])))
        n["d"] = min(8, max(0, int(n["d"])))
        n["c"] = 1 if int(n["c"]) else 0
        out.append(n)
    return out


def dedupe(notes):
    """Drop notes sharing the same time+cell (keep first)."""
    seen, out = set(), []
    for n in sorted(notes, key=lambda z: z["b"]):
        key = (round(float(n["b"]), 4), n["x"], n["y"])
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def fix_parity(notes):
    """Alternate cut direction when the same hand repeats an identical swing quickly."""
    last = {0: None, 1: None}  # color -> (beat, dir)
    out = []
    for n in sorted(notes, key=lambda z: z["b"]):
        c, d = n["c"], n["d"]
        prev = last[c]
        if prev is not None and d != 8:
            pb, pd = prev
            if d == pd and (n["b"] - pb) < 1.0:   # same swing within a beat -> flip
                d = OPPOSITE.get(pd, d)
                n = {**n, "d": d}
        last[c] = (n["b"], d)
        out.append(n)
    return out


def validate(canon: dict) -> dict:
    notes = fix_parity(dedupe(clamp_notes(canon.get("notes", []))))
    return {"notes": notes, "bombs": canon.get("bombs", []), "walls": canon.get("walls", [])}
