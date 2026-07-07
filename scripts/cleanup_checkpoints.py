"""Clean up stale ``.bak-*.pt`` files in ``models/_ckpt/``.

Each training run leaves behind a timestamped ``stage{N}.bak-<UTC>.pt`` from
``save_with_backup`` (see ``models/common.py``). After a few weeks of
iterating these accumulate to ~1.5 GB on the disk for no benefit — the
current ``stage{N}.pt`` / ``stage{N}.best.pt`` / ``stage{N}.latest.pt`` are
the only files inference actually loads (see ``generate._pick_ckpt``).

This script keeps the **3 most recent** bak files per stage and deletes the
rest. Safe by default — dry-run prints what would be removed without
touching anything. Pass ``--apply`` to actually delete.

Usage:
    python scripts/cleanup_checkpoints.py            # dry run
    python scripts/cleanup_checkpoints.py --apply    # delete for real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_KEEP = 3  # number of newest .bak-*.pt to retain per stage


def _collect(ckpt_dir: Path) -> dict[str, list[Path]]:
    """Group ``stage1.bak-*.pt`` / ``stage2.bak-*.pt`` files by stage name."""
    out: dict[str, list[Path]] = {}
    for p in sorted(ckpt_dir.glob("stage*.bak-*.pt")):
        # filename shape: stage1.bak-20260630T181218Z.pt
        stage = p.name.split(".", 1)[0]            # -> "stage1"
        out.setdefault(stage, []).append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "models" / "_ckpt")
    ap.add_argument("--keep", type=int, default=_KEEP,
                    help=f"newest bak files to keep per stage (default {_KEEP})")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run, just prints)")
    args = ap.parse_args()
    if not args.ckpt_dir.is_dir():
        print(f"checkpoint dir not found: {args.ckpt_dir}", file=sys.stderr)
        return 2

    groups = _collect(args.ckpt_dir)
    if not groups:
        print(f"no stage*.bak-*.pt files under {args.ckpt_dir}")
        return 0

    total_to_free = 0
    total_kept = 0
    for stage, files in sorted(groups.items()):
        # sort ascending by mtime so the newest are at the end
        files.sort(key=lambda p: p.stat().st_mtime)
        keep = files[-args.keep:] if len(files) > args.keep else files
        delete = [f for f in files if f not in keep]
        total_kept += len(keep)
        for f in delete:
            size_mb = f.stat().st_size / (1024 ** 2)
            total_to_free += size_mb
            tag = "DELETED" if args.apply else "would-delete"
            print(f"  [{tag:11s}] {f.name:50s} ({size_mb:7.1f} MB)")
            if args.apply:
                try:
                    f.unlink()
                except OSError as e:
                    print(f"  [ERROR] {f.name}: {e}", file=sys.stderr)
        print(f"  -> {stage}: keep {len(keep)}, delete {len(delete)}")
    print(f"\n{'total would free' if not args.apply else 'total freed'}: "
          f"{total_to_free:.1f} MB across {len(groups)} stages "
          f"(kept {total_kept} bak files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())