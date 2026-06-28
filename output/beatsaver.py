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
    but streaming in ~1s blocks is stable. Used only as a fallback when no original
    source file is available (CLI without --write-audio src).
    """
    import numpy as np
    import soundfile as sf
    y = np.asarray(samples, dtype="float32").reshape(-1)
    with sf.SoundFile(str(dst), "w", sr, 1, format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(y), sr):
            f.write(y[i:i + sr])
    return True


def write_song_egg(src: Path, dst: Path) -> bool:
    """Write the ORIGINAL uploaded audio to song.egg at native SR + channels.

    `.ogg`/`.egg` are copied byte-for-byte (lossless — Beat Saber reads ogg renamed
    to .egg). Other formats are decoded at their native sample rate / channel layout
    and re-encoded to Ogg Vorbis (chunked to dodge the libsndfile segfault). This is
    what fixes the "shakal" audio: the old path wrote the 22.05 kHz mono *analysis*
    signal instead of the real file.
    """
    import numpy as np
    src = Path(src)
    if src.suffix.lower() in (".ogg", ".egg"):
        shutil.copy(src, dst)
        return True

    import soundfile as sf
    try:
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)   # (frames, ch)
    except Exception:
        import librosa
        y, sr = librosa.load(str(src), sr=None, mono=False)             # native SR, keep stereo
        data = np.asarray(y, dtype="float32")
        data = data.T if data.ndim == 2 else data[:, None]              # -> (frames, ch)
    sr = int(sr)
    ch = data.shape[1]
    with sf.SoundFile(str(dst), "w", sr, ch, format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(data), sr):
            f.write(data[i:i + sr])
    return True


def write_cover(src: Path, out_dir: Path) -> str:
    """Copy/normalize a cover image into the map folder. Returns the filename used.

    Prefers a centre-cropped square JPEG (Beat Saber wants square art) when Pillow is
    available; otherwise copies the file as-is. Returns "" on failure.
    """
    src = Path(src)
    try:
        from PIL import Image
        im = Image.open(src).convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        im.save(out_dir / "cover.jpg", "JPEG", quality=90)
        return "cover.jpg"
    except Exception:
        try:
            dst = out_dir / ("cover" + (src.suffix.lower() or ".jpg"))
            shutil.copy(src, dst)
            return dst.name
        except Exception as e:
            print(f"  [warn] cover copy failed: {e}")
            return ""


def write_map(out_dir: Path, beatmaps, *, song_name: str, bpm: float,
              song_author: str = "Unknown", level_author: str = "beatgen-ai",
              song_sub_name: str = "", audio_src=None, audio=None,
              cover_src=None, njs: float = 16.0):
    """Pack one or more difficulties into a single BeatSaver V2 level folder.

    beatmaps : {difficulty_name: canon_dict}  (e.g. {"Expert": canon, "Hard": canon})
    audio_src: path to the original audio file (preferred -> full quality song.egg).
    audio    : (mono_float32_samples, sample_rate) fallback when no source file exists.
    cover_src: path to a cover image, or None.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # one .dat per difficulty, ordered by rank
    diff_beatmaps = []
    for difficulty in sorted(beatmaps, key=lambda d: DIFF_RANK.get(d, 7)):
        diff_file = f"{difficulty}Standard.dat"
        (out_dir / diff_file).write_text(
            json.dumps(canonical_to_v2(beatmaps[difficulty])), encoding="utf-8")
        diff_beatmaps.append({
            "_difficulty": difficulty, "_difficultyRank": DIFF_RANK.get(difficulty, 7),
            "_beatmapFilename": diff_file, "_noteJumpMovementSpeed": njs,
            "_noteJumpStartBeatOffset": 0, "_customData": {},
        })

    # audio: prefer the original file (native SR/stereo), fall back to raw samples
    if audio_src is not None:
        try:
            write_song_egg(audio_src, out_dir / "song.egg")
        except Exception as e:
            print(f"  [warn] song.egg from source failed: {e}")
            audio_src = None
    if audio_src is None and audio is not None:
        try:
            write_audio_egg(audio[0], audio[1], out_dir / "song.egg")
        except Exception as e:
            print(f"  [warn] ogg encode failed: {e}")

    cover_name = write_cover(cover_src, out_dir) if cover_src else ""

    info = {
        "_version": "2.0.0", "_songName": song_name, "_songSubName": song_sub_name,
        "_songAuthorName": song_author, "_levelAuthorName": level_author,
        "_beatsPerMinute": round(float(bpm), 3), "_songTimeOffset": 0,
        "_shuffle": 0, "_shufflePeriod": 0.5, "_previewStartTime": 12,
        "_previewDuration": 10, "_songFilename": "song.egg",
        "_coverImageFilename": cover_name, "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": diff_beatmaps,
        }],
    }
    (out_dir / "Info.dat").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return out_dir
