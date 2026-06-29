"""Build scripts/beatgen_kaggle.ipynb from scratch (single source of truth).

The previous notebook was hand-edited and ended up with stray `#` inside
JSON source arrays. This script regenerates it with json.dump so the
output is always valid nbformat=4.
"""
import json
import os
import sys
from pathlib import Path

# -------- builder helpers --------
def md(*lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in lines],
    }

def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in lines],
    }

cells = []

# -------- intro --------
cells.append(md(
    "# beatgen — hardware-adaptive training",
    "",
    "End-to-end notebook that **adapts to whatever GPU/RAM you have**",
    "(Kaggle T4/P100, a rented H200, a local 3060, or CPU-only) and pulls",
    "community Beat Saber maps from BeatLeader on its own if you don't have",
    "them pre-mounted.",
    "",
    "Pipeline:",
    "",
    "1. **Setup** — clone repo, install deps",
    "2. **Detect host** — GPU/VRAM/CC, CPU/RAM → pick a class (tiny/small/medium/large/huge)",
    "3. **Locate inputs** — find `BeatmapLevelsData` + `data/community_maps` (Kaggle datasets, paths, or fresh download)",
    "4. **(optional) download community maps** — BeatLeader, respects rate limits, resumable",
    "5. **Build dataset** — `extract/build_dataset.py`",
    "6. **Tune** — bs/steps/epochs/lr/workers/MelCache size from the host class",
    "7. **Train Stage 1 (TCN onsets)**",
    "8. **Train Stage 2 (GRU notes)**",
    "9. **Export checkpoints** as a single archive",
    "",
    "The same notebook runs on Kaggle (with the community/OST datasets attached),",
    "on a rented H200 (with local paths), or anywhere in between.",
))

# -------- 1. setup --------
cells.append(md(
    "## 1. Where am I running?",
    "",
    "We auto-detect Kaggle vs local, but you can override:",
    "",
    "```",
    "REPO_URL      # Git URL to clone if the repo isn't already here",
    "WORKDIR       # Where to clone (default: /kaggle/working/beatgen on Kaggle, $PWD otherwise)",
    "FETCH_MAPS    # 'auto' | 'force' | 'off' — whether to pull community maps if missing",
    "FETCH_MAX     # How many maps to download when fetching",
    "```",
))

cells.append(code(
    "import os, sys, subprocess, shutil, time, json",
    "from pathlib import Path",
    "",
    "# Override these before running if you want.",
    "REPO_URL   = os.environ.get(\"BEATGEN_REPO_URL\",   \"https://github.com/mat-674/beatgen.git\")",
    "FETCH_MAPS = os.environ.get(\"BEATGEN_FETCH_MAPS\", \"auto\")  # 'auto' | 'force' | 'off'",
    "FETCH_MAX  = int(os.environ.get(\"BEATGEN_FETCH_MAX\", \"5000\"))",
    "",
    "IS_KAGGLE = os.path.isdir(\"/kaggle\") and os.path.isdir(\"/kaggle/working\")",
    "WORKDIR = Path(\"/kaggle/working/beatgen\") if IS_KAGGLE else Path(os.environ.get(\"BEATGEN_WORKDIR\", os.getcwd())).resolve()",
    "print(f\"IS_KAGGLE: {IS_KAGGLE}\")",
    "print(f\"WORKDIR:   {WORKDIR}\")",
    "print(f\"FETCH_MAPS:{FETCH_MAPS}\")",
))

cells.append(code(
    "# Clone (or reuse) the repo.",
    "if not (WORKDIR / \".git\").exists():",
    "    WORKDIR.parent.mkdir(parents=True, exist_ok=True)",
    "    print(f\"Cloning {REPO_URL} -> {WORKDIR}\")",
    "    subprocess.check_call([\"git\", \"clone\", \"--depth\", \"1\", REPO_URL, str(WORKDIR)])",
    "else:",
    "    print(f\"Reusing existing repo at {WORKDIR}\")",
    "",
    "os.chdir(WORKDIR)",
    "sys.path.insert(0, str(WORKDIR))",
    "sys.path.insert(0, str(WORKDIR / \"scripts\"))",
    "print(\"cwd:\", os.getcwd())",
))

cells.append(code(
    "# 2. Install deps. Kaggle already has torch+CUDA+librosa; we only add what's missing.",
    "def pip(*args):",
    "    return subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"--quiet\", \"--no-input\", *args])",
    "",
    "missing = []",
    "for mod, pkg in [(\"librosa\", \"librosa\"), (\"soundfile\", \"soundfile\"),",
    "                 (\"UnityPy\", \"UnityPy\"), (\"gradio\", \"gradio\")]:",
    "    try:",
    "        __import__(mod)",
    "    except ImportError:",
    "        missing.append(pkg)",
    "if missing:",
    "    print(\"installing:\", missing)",
    "    pip(*missing)",
    "else:",
    "    print(\"all python deps already present\")",
    "",
    "import torch, importlib",
    "print(\"torch\", torch.__version__, \"cuda available:\", torch.cuda.is_available())",
    "if torch.cuda.is_available():",
    "    print(\"device:\", torch.cuda.get_device_name(0))",
))

# -------- 2. detect host --------
cells.append(md(
    "## 2. Detect the host",
    "",
    "We probe the GPU (vendor, VRAM, compute capability), CPU count, and",
    "system RAM, then assign a class. The class drives all downstream",
    "hyperparameters so the same notebook can train a 3060 and an H200",
    "without manual edits.",
    "",
    "Tiers (heuristic, not benchmarked — adjust if you see OOM/underutilisation):",
    "",
    "| class | VRAM | example |",
    "|---|---|---|",
    "| `cpu`   | 0 GB    | no GPU / fallback |",
    "| `tiny`  | ≤6 GB   | GTX 1060, P100-16 |",
    "| `small` | 8-12 GB | RTX 3060, T4 |",
    "| `medium`| 16-24 GB| RTX 4090, A10, L4 |",
    "| `large` | 40-80 GB| A100-40, A100-80 |",
    "| `huge`  | ≥80 GB  | H100/H200, B200, multi-GPU |",
))

cells.append(code(
    "import shutil",
    "try:",
    "    import psutil",
    "except ImportError:",
    "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"--quiet\", \"psutil\"])",
    "    import psutil",
    "",
    "RAM_GB = psutil.virtual_memory().total / 2**30",
    "CPU_COUNT = os.cpu_count() or 4",
    "",
    "def probe_gpu():",
    "    if not torch.cuda.is_available():",
    "        return None",
    "    p = torch.cuda.get_device_properties(0)",
    "    return {",
    "        \"name\": torch.cuda.get_device_name(0),",
    "        \"vram_gb\": round(p.total_memory / 2**30, 1),",
    "        \"cc\": (p.major, p.minor),",
    "        \"sm_count\": p.multi_processor_count,",
    "    }",
    "",
    "GPU = probe_gpu()",
    "",
    "if GPU is None:",
    "    CLS = \"cpu\"",
    "elif GPU[\"vram_gb\"] <= 6:",
    "    CLS = \"tiny\"",
    "elif GPU[\"vram_gb\"] <= 12:",
    "    CLS = \"small\"",
    "elif GPU[\"vram_gb\"] <= 24:",
    "    CLS = \"medium\"",
    "elif GPU[\"vram_gb\"] <= 80:",
    "    CLS = \"large\"",
    "else:",
    "    CLS = \"huge\"",
    "",
    "print(json.dumps({\"gpu\": GPU, \"ram_gb\": round(RAM_GB, 1),",
    "                  \"cpu\": CPU_COUNT, \"class\": CLS}, indent=2))",
))

cells.append(code(
    "# 3. Pick a config from the host class.",
    "# These are starting points — feel free to override TUNE before training.",
    "",
    "TUNE = {",
    "    # bs       : per-step batch size for stage1 (stage2 isn't bs-bound)",
    "    # steps    : batches per epoch for stage1",
    "    # epochs   : upper bound; both stages early-stop after 3 flat evals",
    "    # lr       : Adam lr",
    "    # workers  : build_dataset.py / dataloader CPU workers (matters for mel extraction)",
    "    # max_mels : MelCache._MAX_MELS — size of the in-RAM mel pool (~3 MB per mel)",
    "    # disk     : how much disk we expect the dataset to need (GB)",
    "    #   class      bs   steps epochs  lr     workers              max_mels disk",
    "    \"cpu\":    dict(bs=4,   steps=20,  epochs=20, lr=1e-3, workers=min(8,  CPU_COUNT), max_mels=512,   disk=20),",
    "    \"tiny\":   dict(bs=8,   steps=30,  epochs=30, lr=1e-3, workers=min(8,  CPU_COUNT), max_mels=1024,  disk=20),",
    "    \"small\":  dict(bs=16,  steps=40,  epochs=40, lr=1e-3, workers=min(8,  CPU_COUNT), max_mels=2048,  disk=40),",
    "    \"medium\": dict(bs=32,  steps=60,  epochs=40, lr=1e-3, workers=min(12, CPU_COUNT), max_mels=4096,  disk=80),",
    "    \"large\":  dict(bs=64,  steps=80,  epochs=40, lr=1e-3, workers=min(16, CPU_COUNT), max_mels=8192,  disk=160),",
    "    \"huge\":   dict(bs=128, steps=120, epochs=40, lr=1e-3, workers=min(32, CPU_COUNT), max_mels=16384, disk=320),",
    "}[CLS]",
    "",
    "# RAM guardrail: cap max_mels so the cache stays under ~50% of RAM.",
    "mel_bytes = TUNE[\"max_mels\"] * 80 * 2000 * 4   # very rough: 80 mels, 2000 frames, f32",
    "if mel_bytes > 0.5 * RAM_GB * 2**30:",
    "    TUNE[\"max_mels\"] = max(512, int(0.4 * RAM_GB * 2**30 // (80 * 2000 * 4)))",
    "    print(f\"[tune] capped max_mels -> {TUNE['max_mels']} to fit {RAM_GB:.0f} GB RAM\")",
    "",
    "# Disk guardrail: cap workers so we don't trip OOM during build_dataset.",
    "free_gb = shutil.disk_usage(\"/\").free / 2**30",
    "if free_gb < TUNE[\"disk\"] * 1.2:",
    "    TUNE[\"disk\"] = max(10, int(free_gb * 0.7))",
    "    print(f\"[tune] capped dataset budget -> {TUNE['disk']} GB to fit {free_gb:.0f} GB free\")",
    "",
    "print(json.dumps({\"class\": CLS, **TUNE}, indent=2))",
))

# -------- 3. inputs --------
cells.append(md(
    "## 3. Locate inputs",
    "",
    "We need two things:",
    "",
    "* `BeatmapLevelsData/` — the 326 OST/DLC Unity bundles (only needed for OST training)",
    "* `data/community_maps/` — the BeatLeader-fetched community maps (the bulk of the training set)",
    "",
    "Where we look:",
    "",
    "1. already at `WORKDIR/{BeatmapLevelsData,data/community_maps}`",
    "2. Kaggle datasets at `/kaggle/input/*` (Kaggle only)",
    "3. otherwise — if `FETCH_MAPS in {\"auto\",\"force\"}` — run `scripts/fetch_community_maps.py`",
    "   to pull from BeatLeader. Resumable, rate-limited, deduped per (artist|title,duration±2s).",
))

cells.append(code(
    "# 4. Hunt for inputs in the usual spots, then symlink them in.",
    "def find_first(paths):",
    "    for p in paths:",
    "        if os.path.isdir(p):",
    "            return p",
    "    return None",
    "",
    "kaggle_ost = \"/kaggle/input/beatgen-ost-bundles/BeatmapLevelsData\" if IS_KAGGLE else None",
    "kaggle_com = \"/kaggle/input/beatgen-community-maps\" if IS_KAGGLE else None",
    "",
    "OST_DIR = find_first([str(WORKDIR / \"BeatmapLevelsData\"), kaggle_ost])",
    "CM_DIR  = find_first([str(WORKDIR / \"data\" / \"community_maps\"), kaggle_com])",
    "",
    "print(\"OST:\", OST_DIR or \"(missing — train on community only)\")",
    "print(\"community_maps:\", CM_DIR or \"(missing — will fetch)\")",
    "",
    "def link(src, dst):",
    "    if os.path.islink(dst) or os.path.isdir(dst):",
    "        return False",
    "    os.symlink(src, dst)",
    "    print(f\"  linked {dst} -> {src}\")",
    "    return True",
    "",
    "if OST_DIR and OST_DIR != str(WORKDIR / \"BeatmapLevelsData\"):",
    "    link(OST_DIR, str(WORKDIR / \"BeatmapLevelsData\"))",
    "if CM_DIR and CM_DIR != str(WORKDIR / \"data\" / \"community_maps\"):",
    "    link(CM_DIR, str(WORKDIR / \"data\" / \"community_maps\"))",
    "",
    "print()",
    "ost_list = sorted(os.listdir(WORKDIR / \"BeatmapLevelsData\"))[:5] if (WORKDIR / \"BeatmapLevelsData\").is_dir() else []",
    "cm_list  = sorted(os.listdir(WORKDIR / \"data\" / \"community_maps\"))[:5] if (WORKDIR / \"data\" / \"community_maps\").is_dir() else []",
    "print(\"BeatmapLevelsData top:\", ost_list or \"—\")",
    "print(\"community_maps top:  \", cm_list or \"—\")",
))

# -------- 4. download --------
cells.append(md(
    "## 4. (optional) Download community maps",
    "",
    "Only runs if no `community_maps` is found and `FETCH_MAPS != 'off'`.",
    "Hits BeatLeader's `/leaderboards` endpoint, applies the same quality",
    "filter as `scripts/fetch_community_maps.py`, and uses the resumable",
    "manifest at `data/community_maps/_index.sqlite` so a re-run picks up",
    "where it left off.",
    "",
    "Tunables (env vars or just edit here):",
    "",
    "```",
    "FETCH_MAX        # hard cap on downloaded songs (default 5000)",
    "FETCH_TYPES      # 'ranked,qualified,nominated'",
    "FETCH_MIN_STARS  # drop maps with stars < this",
    "FETCH_PAGE_SIZE  # BL page size, default 100",
    "FETCH_RATE       # BL requests per window",
    "```",
))

cells.append(code(
    "FETCH_TYPES     = os.environ.get(\"BEATGEN_FETCH_TYPES\", \"ranked\")",
    "FETCH_MIN_STARS = float(os.environ.get(\"BEATGEN_FETCH_MIN_STARS\", \"0\"))",
    "FETCH_PAGE_SIZE = int(os.environ.get(\"BEATGEN_FETCH_PAGE_SIZE\", \"100\"))",
    "FETCH_RATE      = int(os.environ.get(\"BEATGEN_FETCH_RATE\", \"50\"))",
    "",
    "have_community = (WORKDIR / \"data\" / \"community_maps\").is_dir() and any(",
    "    (WORKDIR / \"data\" / \"community_maps\").iterdir())",
    "should_fetch = (FETCH_MAPS == \"force\") or (FETCH_MAPS == \"auto\" and not have_community)",
    "",
    "if should_fetch:",
    "    print(f\"[fetch] downloading up to {FETCH_MAX} maps (types={FETCH_TYPES}, min_stars={FETCH_MIN_STARS})\")",
    "    t0 = time.time()",
    "    cmd = [",
    "        sys.executable, \"scripts/fetch_community_maps.py\",",
    "        \"--out\", str(WORKDIR / \"data\" / \"community_maps\"),",
    "        \"--max-songs\", str(FETCH_MAX),",
    "        \"--types\", FETCH_TYPES,",
    "        \"--min-stars\", str(FETCH_MIN_STARS),",
    "        \"--page-size\", str(FETCH_PAGE_SIZE),",
    "        \"--rate\", str(FETCH_RATE),",
    "    ]",
    "    print(\" \", \" \".join(cmd))",
    "    subprocess.check_call(cmd)",
    "    print(f\"[fetch] done in {(time.time()-t0)/60:.1f} min\")",
    "else:",
    "    print(f\"[fetch] skipping (have_community={have_community}, FETCH_MAPS={FETCH_MAPS})\")",
    "",
    "n_songs = sum(1 for p in (WORKDIR / \"data\" / \"community_maps\").iterdir() if p.is_dir())",
    "print(f\"community_maps: {n_songs} songs\")",
))

# -------- 5. build dataset --------
cells.append(md(
    "## 5. Build the compact dataset",
    "",
    "`extract/build_dataset.py` walks the input folders, computes mel",
    "spectrograms once per song, and writes a compact `dataset/` tree",
    "(one dir per song, `mel.npy` + one `<Difficulty>.json` per diff).",
    "This is the only CPU-heavy step.",
    "",
    "We pass `--workers` based on the host class so we don't saturate a",
    "Kaggle 4-core box on a 64-core H200 node.",
))

cells.append(code(
    "have_ost = (WORKDIR / \"BeatmapLevelsData\").is_dir() and any(",
    "    (WORKDIR / \"BeatmapLevelsData\").iterdir())",
    "have_cm  = (WORKDIR / \"data\" / \"community_maps\").is_dir() and any(",
    "    (WORKDIR / \"data\" / \"community_maps\").iterdir())",
    "",
    "if not (have_ost or have_cm):",
    "    raise SystemExit(\"no input data: need BeatmapLevelsData/ and/or data/community_maps/\")",
    "",
    "inputs = []",
    "if have_cm:  inputs.append(str(WORKDIR / \"data\" / \"community_maps\"))",
    "if have_ost: inputs.append(str(WORKDIR / \"BeatmapLevelsData\"))",
    "",
    "t0 = time.time()",
    "cmd = [sys.executable, \"extract/build_dataset.py\", *inputs,",
    "       \"--out\", \"dataset\", \"--workers\", str(TUNE[\"workers\"])]",
    "print(\" \", \" \".join(cmd))",
    "subprocess.check_call(cmd)",
    "print(f\"build_dataset took {(time.time()-t0)/60:.1f} min\")",
))

cells.append(code(
    "# Sanity-check what we just built.",
    "ds = WORKDIR / \"dataset\"",
    "n_songs = sum(1 for p in ds.iterdir() if p.is_dir())",
    "n_diff = 0; total_notes = 0",
    "for jp in ds.rglob(\"*.json\"):",
    "    if jp.name == \"meta.json\": continue",
    "    n_diff += 1",
    "    total_notes += len(json.loads(jp.read_text()).get(\"notes\", []))",
    "size_gb = sum(p.stat().st_size for p in ds.rglob(\"*\")) / 1e9",
    "print(f\"songs:           {n_songs}\")",
    "print(f\"difficulties:    {n_diff}\")",
    "print(f\"total notes:     {total_notes:,}\")",
    "print(f\"dataset on disk: {size_gb:.2f} GB\")",
    "assert n_songs > 0, \"dataset is empty — check extract/build_dataset.py output above\"",
))

# -------- 6. apply tune --------
cells.append(md(
    "## 6. Apply the tune",
    "",
    "Patches `models/common.py:MelCache._MAX_MELS` to whatever the host",
    "class picked. Both `stage1.py` and `stage2.py` import this constant,",
    "so a single in-place bump is enough — no model code changes needed.",
    "",
    "If you re-run the notebook, the patch is idempotent.",
))

cells.append(code(
    "import re",
    "common_py = (WORKDIR / \"models\" / \"common.py\").read_text()",
    "new_max = TUNE[\"max_mels\"]",
    "patched = re.sub(r\"(_MAX_MELS\\s*=\\s*)\\d+\", rf\"\\g<1>{new_max}\", common_py)",
    "if patched != common_py:",
    "    (WORKDIR / \"models\" / \"common.py\").write_text(patched)",
    "    print(f\"bumped MelCache._MAX_MELS -> {new_max} (~{new_max * 80 * 2000 * 4 / 2**30:.1f} GB worst-case)\")",
    "else:",
    "    print(f\"MelCache._MAX_MELS already = {new_max}\")",
    "",
    "print()",
    "print(\"Final tune:\")",
    "print(json.dumps({\"class\": CLS, **TUNE}, indent=2))",
))

# -------- 7. stage 1 --------
cells.append(md(
    "## 7. Train Stage 1 — TCN onset predictor",
    "",
    "We pass the class-derived hyperparameters straight through:",
    "",
    "```",
    "--bs, --steps, --epochs, --lr, --device, --out-dir",
    "```",
    "",
    "Best F1 is mirrored to `models/_ckpt/stage1.{best,latest}.pt`; the",
    "`.best.pt` is what the inference app loads.",
))

cells.append(code(
    "ckpt_dir = WORKDIR / \"models\" / \"_ckpt\"",
    "ckpt_dir.mkdir(parents=True, exist_ok=True)",
    "device = \"cuda\" if GPU else \"cpu\"",
    "",
    "t0 = time.time()",
    "subprocess.check_call([",
    "    sys.executable, str(WORKDIR / \"models\" / \"stage1.py\"),",
    "    \"--data\", \"dataset\",",
    "    \"--epochs\", str(TUNE[\"epochs\"]),",
    "    \"--steps\",  str(TUNE[\"steps\"]),",
    "    \"--bs\",     str(TUNE[\"bs\"]),",
    "    \"--lr\",     str(TUNE[\"lr\"]),",
    "    \"--device\", device,",
    "    \"--out-dir\", str(ckpt_dir),",
    "])",
    "print(f\"\\nstage1 done in {(time.time()-t0)/60:.1f} min\")",
))

# -------- 8. stage 2 --------
cells.append(md(
    "## 8. Train Stage 2 — GRU note predictor",
    "",
    "Stage 2 isn't batch-bound (sequences are short, one per beatmap),",
    "so we keep `bs=1` and only inherit `--epochs` and `--lr` from the",
    "tune. `--out-dir` is shared so the best-model mirroring logic in",
    "`stage2.py` writes to the same `_ckpt/` dir.",
))

cells.append(code(
    "t0 = time.time()",
    "subprocess.check_call([",
    "    sys.executable, str(WORKDIR / \"models\" / \"stage2.py\"),",
    "    \"--data\", \"dataset\",",
    "    \"--epochs\", str(TUNE[\"epochs\"]),",
    "    \"--lr\",     str(TUNE[\"lr\"]),",
    "    \"--device\", device,",
    "    \"--out-dir\", str(ckpt_dir),",
    "])",
    "print(f\"\\nstage2 done in {(time.time()-t0)/60:.1f} min\")",
))

# -------- 9. export --------
cells.append(md(
    "## 9. Export checkpoints",
    "",
    "Packs `models/_ckpt/` into a single zip and drops it where the user",
    "can grab it:",
    "",
    "* Kaggle: `/kaggle/working/beatgen-ckpt.zip` (Kaggle preserves until session end)",
    "* local: `out/beatgen-ckpt.zip`",
    "",
    "Best-model checkpoints (`*.best.pt`) are kept; intermediate `*.bak-*.pt`",
    "and `*.latest.pt` mirrors are removed to keep the archive small.",
))

cells.append(code(
    "import zipfile",
    "",
    "out_dir = Path(\"/kaggle/working\") if IS_KAGGLE else (WORKDIR / \"out\")",
    "out_dir.mkdir(parents=True, exist_ok=True)",
    "ckpts = sorted(ckpt_dir.glob(\"*.pt\"))",
    "print(\"checkpoints found:\", [c.name for c in ckpts])",
    "",
    "for f in ckpt_dir.glob(\"*.bak-*.pt\"):",
    "    f.unlink()",
    "    print(\"  dropped backup\", f.name)",
    "for f in ckpt_dir.glob(\"*.latest.pt\"):",
    "    f.unlink()",
    "    print(\"  dropped .latest mirror\", f.name)",
    "",
    "archive = out_dir / \"beatgen-ckpt.zip\"",
    "with zipfile.ZipFile(archive, \"w\", compression=zipfile.ZIP_STORED) as zf:",
    "    for f in sorted(ckpt_dir.glob(\"*.pt\")):",
    "        zf.write(f, arcname=f\"_ckpt/{f.name}\")",
    "print(f\"\\n{archive}  ({archive.stat().st_size/1e6:.1f} MB)\")",
))

cells.append(md(
    "## What's next",
    "",
    "1. Download `beatgen-ckpt.zip` from the Output tab (Kaggle) or grab `out/beatgen-ckpt.zip` (local).",
    "2. Unzip into the repo's `models/_ckpt/` — this overwrites `stage1.pt` / `stage2.pt`",
    "   (the .best.pt mirrors the same names, so legacy callers keep working).",
    "3. Run the app locally:",
    "   ```bash",
    "   python app.py          # Gradio UI",
    "   python app_train.py    # training monitor",
    "   python generate.py --audio my_song.mp3 --difficulty Expert",
    "   ```",
    "",
    "### Tips for the next run",
    "",
    "- On the `huge` class you can crank `--bs 256 --steps 200` and finish in a fraction of the time — the heuristic stays conservative on purpose so first runs don't OOM.",
    "- `FETCH_MIN_STARS=3` is a good floor for the next fetch: drops one-hand warmups and meme maps that hurt Stage 2.",
    "- Resume an interrupted run with `--resume` pointing at `*.best.pt` — same checkpoint format as `save_with_backup`.",
    "- Re-running the notebook reuses the manifest at `data/community_maps/_index.sqlite`; delete it to start the download from scratch.",
))

# -------- metadata + assemble --------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "kaggle": {
            "accelerator": "gpu",
            "data_sources": ["beatgen-community-maps", "beatgen-ost-bundles"],
            "docker_image": "python:latest",
            "isKaggleNotebook": True,
            "language": "python",
            "title": "beatgen — hardware-adaptive training",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "beatgen_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size} bytes, {len(cells)} cells)")
