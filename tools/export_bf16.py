#!/usr/bin/env python3
"""Export an inference-only bf16 checkpoint, optionally with an exact XOR delta.

Run in the verified OpenPI environment. This program does not load a model, use a
GPU, connect to a camera, or open a CAN interface. Existing paths are never
overwritten. ``--source`` is a checkpoint ROOT containing params/ and assets/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
import zipfile

import numpy as np


FORMAT = "openpi-bf16-xor-v1"


def flatten_tree(tree, path=()):
    result = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Parameter paths must have nonempty string keys")
            result.update(flatten_tree(value, path + (key,)))
    else:
        if not path:
            raise ValueError("Expected a nonempty parameter dictionary")
        result[path] = np.asarray(tree)
    return result


def unflatten_tree(leaves):
    result = {}
    for path, value in leaves.items():
        if not path or not all(isinstance(key, str) and key for key in path):
            raise ValueError("Invalid leaf path")
        node = result
        for key in path[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError("Conflicting leaf paths")
        if path[-1] in node:
            raise ValueError("Duplicate leaf path")
        node[path[-1]] = value
    return result


def array_hash(array):
    array = np.asarray(array, order="C")
    # ml_dtypes bfloat16 exposes buffer format 'E', unsupported by memoryview.
    # Flatten first so scalar leaves also support a uint8 view without copying.
    raw = array.reshape(-1).view(np.uint8)
    return hashlib.sha256(memoryview(raw)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_hash_manifest(root, name="SHA256SUMS", include=None):
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == root / name:
            continue
        rel = path.relative_to(root)
        if include is None or include(rel):
            records.append(f"{file_hash(path)}  {rel.as_posix()}\n")
    if not records:
        raise ValueError("Refusing to write an empty checksum manifest")
    (root / name).write_text("".join(records))


def verify_hash_manifest(root, name):
    root = Path(root).resolve()
    lines = (root / name).read_text().splitlines()
    if not lines:
        raise ValueError("Empty checksum manifest")
    checked = set()
    for line in lines:
        checksum, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if len(checksum) != 64 or root not in path.parents or relative in checked:
            raise ValueError("Invalid checksum entry")
        if not path.is_file() or file_hash(path) != checksum:
            raise ValueError(f"Checksum mismatch: {relative}")
        checked.add(relative)
    return checked


def restore_bf16(params_path):
    # Set before importing JAX/OpenPI. Export/apply must never allocate a GPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import ml_dtypes
    from openpi.models import model

    return model.restore_params(
        str(Path(params_path).resolve()),
        restore_type=np.ndarray,
        dtype=ml_dtypes.bfloat16,
    )


def save_params(params_path, params):
    import orbax.checkpoint as ocp

    params_path = Path(params_path).resolve()
    if params_path.exists():
        raise FileExistsError(f"Will not overwrite {params_path}")
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(
            str(params_path), args=ocp.args.PyTreeSave({"params": params})
        )


def verify_trees(expected, restored):
    left, right = flatten_tree(expected), flatten_tree(restored)
    if set(left) != set(right):
        raise ValueError("Restored parameter paths differ")
    total = 0
    for path, array in left.items():
        other = right[path]
        if (
            array.shape != other.shape
            or array.dtype != other.dtype
            or not np.array_equal(array, other)
        ):
            raise ValueError(f"Restored leaf mismatch: {'/'.join(path)}")
        total += array.nbytes
    return {"leaves": len(left), "parameter_bytes": total, "all_leaves_equal": True}


def resolve_base_params(base):
    base = Path(base).expanduser().resolve()
    params = base / "params" if (base / "params").is_dir() else base
    if not params.is_dir():
        raise FileNotFoundError(params)
    return params


def write_delta(base, target, output, *, source_name, base_name, compression_level=6):
    """Write a standard np.load-compatible compressed NPZ, one leaf at a time.

    Streaming the .npy entries is equivalent to np.savez_compressed's ZIP format
    but avoids materializing a second full tree of XOR tensors in host memory.
    """
    if sys.byteorder != "little":
        raise RuntimeError("The exact delta format requires a little-endian host")
    output = Path(output)
    base_flat, target_flat = flatten_tree(base), flatten_tree(target)
    if set(base_flat) != set(target_flat):
        raise ValueError("Base and target parameter paths differ")
    records = []
    changed = 0
    with zipfile.ZipFile(
        output / "bf16_delta.npz", "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level, allowZip64=True,
    ) as archive:
        for i, path in enumerate(sorted(target_flat)):
            a, b = base_flat[path], target_flat[path]
            if a.shape != b.shape or str(a.dtype) != "bfloat16" or a.dtype != b.dtype:
                raise ValueError(f"Base/target leaf mismatch: {'/'.join(path)}")
            a, b = np.asarray(a, order="C"), np.asarray(b, order="C")
            xor = np.asarray(np.bitwise_xor(a.view(np.uint16), b.view(np.uint16)), dtype=np.uint16).reshape(b.shape)
            unchanged = not bool(np.any(xor))
            key = f"leaf_{i:06d}"
            record = {
                "key": key, "path": list(path), "shape": list(b.shape),
                "dtype": "bfloat16", "unchanged": unchanged,
                "base_sha256": array_hash(a), "target_sha256": array_hash(b),
                "xor_sha256": array_hash(xor), "bytes": int(b.nbytes),
            }
            if not unchanged:
                # NPY headers + uint16 payload, no pickle/object arrays.
                with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, xor, allow_pickle=False)
                changed += 1
            records.append(record)
            del xor
    metadata = {
        "format": FORMAT, "byteorder": "little", "source": str(source_name),
        "base": str(base_name), "leaves": records, "leaf_count": len(records),
        "changed_leaves": changed, "parameter_bytes": sum(r["bytes"] for r in records),
        "delta_bytes": (output / "bf16_delta.npz").stat().st_size,
        "npz_sha256": file_hash(output / "bf16_delta.npz"),
        "exact_relative_to_default_bf16_inference": True,
    }
    (output / "delta_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def new_staging_directory(source, output):
    source, output = Path(source).resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must not overlap the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".partial-" + uuid.uuid4().hex[:12])
    stage.mkdir()
    return output, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", help="Optional base checkpoint root OR params directory")
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=6)
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    if not (source / "params").is_dir() or not (source / "assets").is_dir():
        raise ValueError("Source must contain params/ and assets/")
    output, stage = new_staging_directory(source, args.output)
    started = time.perf_counter()
    params = restore_bf16(source / "params")
    if any(str(a.dtype) != "bfloat16" for a in flatten_tree(params).values()):
        raise ValueError("Every exported parameter leaf must be bfloat16")
    save_params(stage / "params", params)
    restored = restore_bf16(stage / "params")
    verification = verify_trees(params, restored)
    del restored
    gc.collect()
    shutil.copytree(source / "assets", stage / "assets")
    if args.base:
        base_path = resolve_base_params(args.base)
        base = restore_bf16(base_path)
        delta = write_delta(
            base, params, stage, source_name=source, base_name=base_path,
            compression_level=args.compression_level,
        )
        verification["delta"] = {
            key: delta[key] for key in ["changed_leaves", "delta_bytes", "parameter_bytes"]
        }
        del base
        write_hash_manifest(
            stage, "DELTA_SHA256SUMS",
            include=lambda p: p.parts[0] == "assets" or p.name in {
                "bf16_delta.npz", "delta_metadata.json"
            },
        )
    verification.update(
        source=str(source), dtype="bfloat16", inference_only=True,
        identical_to_source_restored_as_bf16=True, devices_opened=False,
        elapsed_seconds=time.perf_counter() - started,
    )
    (stage / "export_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    write_hash_manifest(stage)
    if output.exists():
        raise FileExistsError(f"Output appeared during export: {output}")
    stage.rename(output)
    print(json.dumps(dict(verification, output=str(output)), indent=2), flush=True)


if __name__ == "__main__":
    main()
