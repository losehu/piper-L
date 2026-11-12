#!/usr/bin/env python3
"""Reconstruct an exact bf16 checkpoint from an existing base and XOR package.

CPU-only file processing. Never connects to robot, camera, or GPU devices.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

from export_bf16 import (
    FORMAT, array_hash, file_hash, flatten_tree, new_staging_directory,
    resolve_base_params, restore_bf16, save_params, unflatten_tree,
    verify_hash_manifest, verify_trees, write_hash_manifest,
)


def reconstruct_delta(base, patch):
    if sys.byteorder != "little":
        raise RuntimeError("The exact delta format requires a little-endian host")
    patch = Path(patch)
    metadata = json.loads((patch / "delta_metadata.json").read_text())
    if metadata.get("format") != FORMAT or metadata.get("byteorder") != "little":
        raise ValueError("Unsupported delta format")
    records = metadata["leaves"]
    if not 1 <= len(records) <= 10000 or metadata["leaf_count"] != len(records):
        raise ValueError("Invalid leaf count")
    if file_hash(patch / "bf16_delta.npz") != metadata["npz_sha256"]:
        raise ValueError("NPZ checksum mismatch")
    base_flat = flatten_tree(base)
    expected_paths = [tuple(r["path"]) for r in records]
    if len(set(expected_paths)) != len(expected_paths) or set(expected_paths) != set(base_flat):
        raise ValueError("Base paths do not match the delta")
    expected_keys = {r["key"] for r in records if not r["unchanged"]}
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate NPZ keys")
    # Check ZIP entry sizes before NumPy decompresses them.
    with zipfile.ZipFile(patch / "bf16_delta.npz") as archive:
        if len(archive.infolist()) != len(expected_keys):
            raise ValueError("Unexpected NPZ entry count")
        for record in records:
            if record["unchanged"]:
                continue
            entry = archive.getinfo(record["key"] + ".npy")
            if entry.file_size > int(record["bytes"]) + 4096:
                raise ValueError("Oversized NPZ leaf")
    target = {}
    with np.load(patch / "bf16_delta.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("Unexpected NPZ keys")
        for record in records:
            path = tuple(record["path"])
            array = np.asarray(base_flat.pop(path), order="C")
            if (
                str(array.dtype) != "bfloat16"
                or list(array.shape) != record["shape"]
                or array.nbytes != record["bytes"]
                or array_hash(array) != record["base_sha256"]
            ):
                raise ValueError(f"Wrong base leaf: {'/'.join(path)}")
            if record["unchanged"]:
                result = array
            else:
                xor = arrays[record["key"]]
                if (
                    xor.dtype != np.dtype("uint16") or xor.shape != array.shape
                    or array_hash(xor) != record["xor_sha256"]
                ):
                    raise ValueError(f"Wrong XOR leaf: {'/'.join(path)}")
                result = np.asarray(np.bitwise_xor(array.view(np.uint16), xor), dtype=np.uint16).reshape(array.shape).view(array.dtype)
            if array_hash(result) != record["target_sha256"]:
                raise ValueError(f"Reconstructed target mismatch: {'/'.join(path)}")
            target[path] = result
    return unflatten_tree(target), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Existing base ROOT or params directory")
    parser.add_argument("--patch", required=True, help="Directory containing delta, metadata and assets/")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    patch = Path(args.patch).expanduser().resolve()
    if not (patch / "assets").is_dir():
        raise ValueError("Missing checkpoint assets/")
    checked = verify_hash_manifest(patch, "DELTA_SHA256SUMS")
    if not {"bf16_delta.npz", "delta_metadata.json"}.issubset(checked):
        raise ValueError("Delta files are absent from the checksum manifest")
    for p in (patch / "assets").rglob("*"):
        if p.is_file() and p.relative_to(patch).as_posix() not in checked:
            raise ValueError("An asset is absent from the checksum manifest")
    base_path = resolve_base_params(args.base)
    output, stage = new_staging_directory(base_path, args.output)
    if patch == output or patch in output.parents or output in patch.parents:
        raise ValueError("Output must not overlap the delta package")
    started = time.perf_counter()
    base = restore_bf16(base_path)
    target, metadata = reconstruct_delta(base, patch)
    del base
    gc.collect()
    save_params(stage / "params", target)
    restored = restore_bf16(stage / "params")
    verification = verify_trees(target, restored)
    del restored
    shutil.copytree(patch / "assets", stage / "assets")
    verification.update(
        base=str(base_path), patch=str(patch), dtype="bfloat16",
        all_base_leaf_hashes_verified=True, all_target_leaf_hashes_verified=True,
        inference_only=True, exact_relative_to_default_bf16_inference=True,
        source_checkpoint=metadata["source"], devices_opened=False,
        elapsed_seconds=time.perf_counter() - started,
    )
    (stage / "apply_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    write_hash_manifest(stage)
    if output.exists():
        raise FileExistsError(f"Output appeared during reconstruction: {output}")
    stage.rename(output)
    print(json.dumps(dict(verification, output=str(output)), indent=2), flush=True)


if __name__ == "__main__":
    main()
