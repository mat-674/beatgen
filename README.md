# beatgen — AI Beat Saber map generator

Generate a playable Beat Saber map from any song. `beatgen` learns from the **65 official
OST/DLC levels** shipped in `BeatmapLevelsData/` and uses a two-stage neural pipeline
(the proven *Beat Sage / DanceDanceConvolution* design):

- **Stage 1 — *when*** a note happens: a dilated-conv TCN reads the mel-spectrogram and
  predicts a per-frame onset/density probability, conditioned on the target difficulty.
- **Stage 2 — *which*** note it is: a GRU walks the resulting action timeline and predicts,
  for each hand (red/blue), whether it fires and its lane / layer / cut-direction,
  conditioned on local audio, difficulty, and the previous note (for flow & parity).

Output is a standard **BeatSaver V2 map folder** (`Info.dat` + `<Difficulty>Standard.dat`
+ `song.egg`) that loads directly in Beat Saber or ChroMapper.

---

## 1. Requirements

- **Python 3.10+** (developed on 3.13).
- ~2 GB disk for the dataset/extracted audio, ~3 GB for the CUDA PyTorch wheel.
- Optional: an **NVIDIA GPU** for fast training (CPU works for inference and small training runs).
- The OST bundles already in `BeatmapLevelsData/` (65 files). Drop in more DLC bundles to scale up.

---

## 2. Install

The installer creates a `.venv` and installs everything, letting you pick the **CPU** or
**CUDA** PyTorch build.

**Windows (double-click or terminal):**
```bat
install.bat                 :: interactive — asks CPU or CUDA
```

**Any OS (Python):**
```bash
python install.py                         # interactive prompt
python install.py --runtime cpu           # CPU build
python install.py --runtime cuda          # CUDA build (default wheel cu124)
python install.py --runtime cuda --cuda cu126
python install.py --runtime cuda --no-venv  # into the current environment
```

CUDA wheels bundle their own CUDA runtime — you only need a recent NVIDIA **driver**, not a
matching system CUDA toolkit. Pick the `--cuda` tag closest to your driver
(`cu118 / cu121 / cu124 / cu126 / cu128`).

Everything below assumes the venv interpreter. On Windows that is
`.venv\Scripts\python.exe`; on Linux/macOS `.venv/bin/python`. The examples use the
Windows path.

---

## 3. Run the full pipeline yourself

Three steps turn raw content into trained models. Run them in order:

```bash
# 1) Build the dataset from ANY folder, recursively. Pass one or more sources.
#    Writes a compact, content-independent dataset (NO audio is copied):
#    dataset/<song>/{mel.npy, <Difficulty>.json, meta.json}  (+ dataset/index.json)
.venv\Scripts\python.exe extract/build_dataset.py BeatmapLevelsData
#    ...or point it anywhere — bundles AND standard custom-map folders are both eaten:
.venv\Scripts\python.exe extract/build_dataset.py "D:\CustomLevels" BeatmapLevelsData

# 2) Train Stage 1 (onsets). Saves models/_ckpt/stage1.pt (checkpoints every 5 epochs)
.venv\Scripts\python.exe models/stage1.py --epochs 30

# 3) Train Stage 2 (notes). Saves models/_ckpt/stage2.pt
.venv\Scripts\python.exe models/stage2.py --epochs 30
```

The dataset builder is **omnivorous and recursive** — it walks every subfolder and detects:
- **Unity asset bundles** by magic bytes (any/no extension) — the official OST/DLC.
- **standard BeatSaver map folders** (`Info.dat` + `.dat` + `.ogg/.egg`) — community maps.

Beatmaps in V2 / V3 / V4 schemas are all normalized. Mel-spectrograms are computed straight
from the source audio and cached as `mel.npy`; nothing else from the audio is stored, so the
`dataset/` stays small (~180 MB for 65 songs) and you can delete the sources afterward.
Re-runs skip songs that already have `mel.npy` (use `--force` to recompute).

Useful flags:
- `stage1.py`: `--epochs --steps --bs --lr --device {cpu,cuda}`
- `stage2.py`: `--epochs --lr --device {cpu,cuda}`

Both auto-use CUDA when available; force with `--device`. Training prints validation metrics
on three held-out songs (`beatsaber`, `crabrave`, `turnmeon` — see `models/common.py:VAL_SONGS`).

---

## 4. Generate maps

### Web UI (Gradio)
```bash
.venv\Scripts\python.exe app.py
```
Open the printed local URL. Upload a song (`.wav` / `.ogg` / `.flac`), choose the
**difficulty** and **density threshold**, optionally set the **BPM** (0 = auto-detect),
pick the **runtime**, and click *Generate map*. You get a downloadable `.zip` plus stats.

### Command line
```bash
.venv\Scripts\python.exe generate.py path\to\song.wav --difficulty Expert --thr 0.85 --out out\mysong
```
- `--difficulty` Easy / Normal / Hard / Expert / ExpertPlus
- `--thr` note-density threshold, 0.3–0.97 (**higher = fewer notes**; ~0.85 matches Expert density)
- `--bpm` override auto-detected tempo
- `--device cpu|cuda`

### Play the result
The output folder (or unzipped UI download) is a normal custom level:
- **Beat Saber:** copy the folder into `...\Beat Saber\Beat Saber_Data\CustomLevels\` (with a mod loader / SongCore).
- **ChroMapper:** open the folder to inspect/edit notes.

---

## 5. Repo layout

```
beatgen/
  BeatmapLevelsData/        # input: official OST/DLC Unity bundles (gitignored)
  extract/
    unpack_bundle.py        # spike: inspect a single bundle's assets
    loaders.py              # omnivorous recursive discovery (bundles + map folders)
    build_dataset.py        # any folder -> dataset/ (mel + canonical notes + bpm)
  schema/canonical.py       # unify BeatSaber V2 / V3 / V4 -> canonical notes
  features/
    audio.py                # load + log-mel + time<->frame helpers
  models/
    common.py               # difficulty ids, mel cache + normalization, dataset listing
    stage1.py               # onset TCN (model + train)
    stage2.py               # note GRU (model + train)
    _ckpt/                  # trained checkpoints (stage1.pt, stage2.pt)
  validate/playability.py   # dedupe + basic parity repair
  output/beatsaver.py       # canonical -> playable BeatSaver V2 folder (+ ogg song.egg)
  generate.py               # end-to-end inference (CLI + load_models/run API)
  app.py                    # Gradio UI
  install.py / install.bat  # CPU/CUDA installer
  dataset/                  # generated by step 1-2
  out/                      # generated maps
```

---

## 6. How it works (internals)

- **Extraction.** `extract/loaders.py` walks any folder. Unity bundles (LZ4 `UnityFS`,
  Unity 6000) are read with `UnityPy`: the song is an `AudioClip`, each beatmap is a
  `TextAsset` (plain **V2** `_notes` or gzipped **V4** `colorNotes`+`colorNotesData`), and a
  `MonoBehaviour` ties audio↔difficulties; BPM comes from the `*.audio` asset. Standard
  custom-map folders are read straight from `Info.dat` (+ **V2/V3** `.dat` files). Every
  schema is normalized to one canonical note format (`schema/canonical.py`).
- **Features.** Audio → 80-bin log-mel at 22.05 kHz, hop 512 (~23 ms / frame, ~43 fps).
- **Stage 1.** Mel + difficulty embedding → dilated 1D conv stack → per-frame onset logit.
  Trained with `BCEWithLogits` (positive-weighted, onsets are sparse). At inference, onsets
  are peak-picked above `--thr` with a minimum frame gap.
- **Stage 2.** For each onset, a GRU predicts red/blue presence + lane/layer/direction,
  conditioned on local mel context, difficulty, and a teacher-forced previous note.
- **Validate + pack.** Notes are de-duplicated and a light parity pass flips repeated swings.
  Beats are derived from onset time × BPM, written as V2, and the audio is encoded to
  `song.egg` (Ogg Vorbis, streamed in 1 s chunks — a single bulk write segfaults this
  libsndfile wheel).

---

## 7. Scaling up (more DLC + GPU)

1. Point `build_dataset.py` at more content — extra DLC bundles and/or folders of community
   maps (`python extract/build_dataset.py BeatmapLevelsData "D:\CustomLevels"`).
2. Re-run steps 1–3. The builder skips songs that already have `mel.npy` (use `--force` to redo).
3. For real training, install the CUDA build (`python install.py --runtime cuda`) and train
   with more epochs; the scripts pick up the GPU automatically.

---

## 8. Current status & limitations

Proof-of-life trained on the 65 OST (CPU). It works end-to-end and produces coherent,
onset-synced maps; the difficulty embedding learns density (held-out crabrave: Expert@0.85
→ 636 notes vs 643 original; Normal is much sparser). Known gaps to close with more data:

- Stage 1 is recall-biased — tune `--thr` (per-difficulty threshold calibration is a TODO).
- Stage 2 cut directions collapse mostly to up/down and skip the top layer — needs more data
  and a richer autoregressive Stage 2 with parity-aware decoding.
- Walls / bombs / lighting / arcs are not generated yet (notes only).

See the project memory `ost-bundle-format.md` for full bundle-format details.
