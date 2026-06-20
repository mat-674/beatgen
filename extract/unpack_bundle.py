"""Phase 0 spike: crack open a Beat Saber UnityFS bundle and see what's inside.

Goal: prove we can recover BOTH audio and NOTES. Key question is whether the
SerializedFile carries a type-tree (=> MonoBehaviour reads as a dict) or whether
the note data is stashed in a custom binary / TextAsset that needs reversing.

Usage:
    python extract/unpack_bundle.py <bundle_path> [--out out_dir]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import UnityPy


def short(value, n=400):
    s = repr(value)
    return s if len(s) <= n else s[:n] + f"... <+{len(s) - n} chars>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, default=Path("extract/_spike_out"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = UnityPy.load(str(args.bundle))

    objects = list(env.objects)
    type_counts = Counter(o.type.name for o in objects)
    print(f"== {args.bundle.name} ==")
    print(f"objects: {len(objects)}")
    print("type histogram:")
    for t, c in type_counts.most_common():
        print(f"  {t:20s} {c}")
    print()

    mono_index = []
    for i, obj in enumerate(objects):
        tname = obj.type.name

        if tname == "AudioClip":
            try:
                clip = obj.read()
                name = getattr(clip, "m_Name", f"clip{i}")
                print(f"[AudioClip] name={name!r}")
                # m_AudioData / samples; UnityPy exposes .samples dict {name: bytes(wav)}
                try:
                    samples = clip.samples
                    for sname, data in samples.items():
                        safe = "".join(c if c.isalnum() else "_" for c in sname) or f"clip{i}"
                        p = args.out / f"{safe}.wav"
                        p.write_bytes(data)
                        print(f"   -> wrote {p} ({len(data)} bytes)")
                except Exception as e:
                    print(f"   !! samples decode failed: {e}")
                    # dump raw fields for diagnosis
                    for attr in ("m_CompressionFormat", "m_Channels", "m_Frequency",
                                 "m_Length", "m_Source", "m_Size"):
                        print(f"      {attr} = {getattr(clip, attr, '<none>')!r}")
            except Exception as e:
                print(f"[AudioClip] read failed: {e}")
            print()

        elif tname == "MonoBehaviour":
            entry = {"index": i, "path_id": obj.path_id}
            has_tree = False
            try:
                has_tree = obj.serialized_type is not None and obj.serialized_type.node is not None
            except Exception:
                has_tree = False
            entry["has_typetree"] = bool(has_tree)
            try:
                if has_tree:
                    tree = obj.read_typetree()
                    entry["script"] = tree.get("m_Name") if isinstance(tree, dict) else None
                    entry["keys"] = list(tree.keys()) if isinstance(tree, dict) else None
                else:
                    mb = obj.read()
                    entry["name"] = getattr(mb, "m_Name", None)
                    raw = obj.get_raw_data()
                    entry["raw_len"] = len(raw)
            except Exception as e:
                entry["error"] = str(e)
            mono_index.append(entry)

        elif tname == "TextAsset":
            try:
                ta = obj.read()
                name = getattr(ta, "m_Name", f"text{i}")
                script = getattr(ta, "m_Script", b"")
                blob = script.encode("utf-8", "surrogateescape") if isinstance(script, str) else bytes(script)
                p = args.out / f"text_{name}.bin"
                p.write_bytes(blob)
                print(f"[TextAsset] name={name!r} len={len(blob)} -> {p}")
                print(f"   head: {short(blob[:200])}")
            except Exception as e:
                print(f"[TextAsset] read failed: {e}")

    # Summarize MonoBehaviours — this is where notes most likely live
    print("== MonoBehaviours ==")
    n_tree = sum(1 for m in mono_index if m.get("has_typetree"))
    print(f"total: {len(mono_index)}, with type-tree: {n_tree}")
    for m in mono_index[:40]:
        print(" ", short(m, 300))

    (args.out / "mono_index.json").write_text(
        json.dumps(mono_index, indent=2, default=str), encoding="utf-8")

    # Dump full typetrees of any MB that has one (likely the beatmap data)
    if n_tree:
        dump = []
        for obj in objects:
            if obj.type.name != "MonoBehaviour":
                continue
            try:
                if obj.serialized_type and obj.serialized_type.node:
                    dump.append(obj.read_typetree())
            except Exception:
                pass
        (args.out / "mono_typetrees.json").write_text(
            json.dumps(dump, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {len(dump)} typetrees -> {args.out / 'mono_typetrees.json'}")
    else:
        print("\n!! NO type-trees on any MonoBehaviour -> notes likely need reversing")


if __name__ == "__main__":
    sys.exit(main())
