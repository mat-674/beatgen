"""Pack canonical notes -> a playable BeatSaver V2 map folder (Info.dat + .dat + song.egg)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# BeatSaber difficulty -> (_difficultyRank, default NJS)
DIFF_RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}


def canonical_to_v2(canon: dict) -> dict:
    notes = [{"_time": float(n["b"]), "_lineIndex": int(n["x"]), "_lineLayer": int(n["y"]),
              "_type": int(n["c"]), "_cutDirection": int(n["d"])} for n in canon.get("notes", [])]
    notes += [{"_time": float(b["b"]), "_lineIndex": int(b["x"]), "_lineLayer": int(b["y"]),
               "_type": 3, "_cutDirection": 0} for b in canon.get("bombs", [])]
    notes.sort(key=lambda z: z["_time"])
    obstacles = [{"_time": float(w["b"]), "_lineIndex": int(w["x"]),
                  "_type": 0 if int(w.get("y", 0)) == 0 else 1,
                  "_duration": float(w.get("dur", 0)), "_width": int(w.get("w", 1))}
                 for w in canon.get("walls", [])]
    return {"_version": "2.0.0", "_events": [], "_notes": notes, "_obstacles": obstacles}


def write_audio_egg(samples, sr: int, dst: Path) -> bool:
    """Encode mono float32 samples -> Ogg Vorbis song.egg via CHUNKED streaming.

    libsndfile's vorbis encoder segfaults on a single bulk write of large arrays,
    but streaming in ~1s blocks is stable.
    """
    import numpy as np
    import soundfile as sf
    y = np.asarray(samples, dtype="float32").reshape(-1)
    with sf.SoundFile(str(dst), "w", sr, 1, format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(y), sr):
            f.write(y[i:i + sr])
    return True


def write_map(out_dir: Path, canon: dict, *, song_name: str, bpm: float,
              difficulty: str = "Expert", audio=None, njs: float = 16.0):
    """audio = (mono_float32_samples, sample_rate) or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_file = f"{difficulty}Standard.dat"
    (out_dir / diff_file).write_text(json.dumps(canonical_to_v2(canon)), encoding="utf-8")

    if audio is not None:
        try:
            write_audio_egg(audio[0], audio[1], out_dir / "song.egg")
        except Exception as e:
            print(f"  [warn] ogg encode failed: {e}")

    info = {
        "_version": "2.0.0", "_songName": song_name, "_songSubName": "",
        "_songAuthorName": "Unknown", "_levelAuthorName": "beatgen-ai",
        "_beatsPerMinute": round(float(bpm), 3), "_songTimeOffset": 0,
        "_shuffle": 0, "_shufflePeriod": 0.5, "_previewStartTime": 12,
        "_previewDuration": 10, "_songFilename": "song.egg",
        "_coverImageFilename": "", "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": difficulty, "_difficultyRank": DIFF_RANK.get(difficulty, 7),
                "_beatmapFilename": diff_file, "_noteJumpMovementSpeed": njs,
                "_noteJumpStartBeatOffset": 0, "_customData": {},
            }],
        }],
    }
    (out_dir / "Info.dat").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return out_dir
