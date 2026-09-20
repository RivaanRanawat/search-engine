"""Convert a v1-format index dir into the block-compressed format (v2c).

Quantizes impacts to u8 (global linear scale — quality gate: full-dev MRR
delta measured before this design was adopted), then runs the native
converter for delta+bitpack blocks.

Usage: python -m searchengine.compress_index <src_dir> <dst_dir>
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "native", "compress.c")
_BIN = os.path.join(_DIR, "native", "compress")


def _npy_data_offset(path: str) -> int:
    """Byte offset where a .npy file's raw array data begins."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        np.lib.format._read_array_header(f, version)
        return f.tell()


def build(src: str, dst: str, quant_scale: float | None = None) -> dict:
    """quant_scale: force a specific impact-quantization scale instead of
    deriving it from this index's own maximum.

    Required for sharding: with per-index scales, two shards quantize scores
    onto DIFFERENT grids, so the broker merges numbers that are not strictly
    comparable. That alone cost ~2.7% of top-10 agreement (notes/17)."""
    if (not os.path.exists(_BIN)
            or os.path.getmtime(_BIN) < os.path.getmtime(_SRC)):
        subprocess.run(["clang", "-O2", "-o", _BIN, _SRC], check=True)
    os.makedirs(dst, exist_ok=True)
    t_all = time.perf_counter()

    with open(f"{src}/meta.json") as f:
        meta = json.load(f)
    offsets = np.load(f"{src}/offsets.u64.npy").astype(np.int64)
    imp = np.asarray(np.load(f"{src}/impacts.f32.npy", mmap_mode="r"))
    scale = quant_scale if quant_scale else float(imp.max()) / 255.0
    impq = np.rint(np.clip(imp / scale, 0, 255)).astype(np.uint8)
    del imp

    tmp = f"{dst}/_tmp"
    os.makedirs(tmp, exist_ok=True)
    # Read inputs in place from their .npy files (data offset from the npy
    # header) instead of copying to scratch: the copies cost ~1.8GB and
    # filled the disk once (notes/13).
    impq_path = f"{tmp}/impq.raw"
    impq.tofile(impq_path)
    del impq
    offs_path = f"{tmp}/offs.raw"
    offsets.tofile(offs_path)
    n_terms = len(offsets) - 1

    t0 = time.perf_counter()
    r = subprocess.run(
        [_BIN, f"{src}/docids.u32.npy", str(_npy_data_offset(
            f"{src}/docids.u32.npy")),
         impq_path, "0", offs_path, "0",
         str(n_terms), f"{dst}/blob.bin", f"{tmp}/blast.raw",
         f"{tmp}/bwidth.raw", f"{tmp}/bmaxq.raw"],
        check=True, capture_output=True)
    nblocks = int(r.stdout.split()[0])
    compress_s = time.perf_counter() - t0

    # hard integrity gates: a truncated write here once produced a silently
    # corrupt index (see notes/10) — never trust, always verify
    dfs_chk = np.diff(offsets)
    expect = int(((dfs_chk + 127) // 128).sum())
    assert nblocks == expect, (nblocks, expect)
    for f_, unit in (("blast.raw", 4), ("bwidth.raw", 1), ("bmaxq.raw", 1)):
        got = os.path.getsize(f"{tmp}/{f_}")
        assert got == expect * unit, (f_, got, expect * unit)

    np.save(f"{dst}/block_last.u32.npy",
            np.fromfile(f"{tmp}/blast.raw", np.uint32))
    np.save(f"{dst}/block_width.u8.npy",
            np.fromfile(f"{tmp}/bwidth.raw", np.uint8))
    np.save(f"{dst}/block_maxq.u8.npy",
            np.fromfile(f"{tmp}/bmaxq.raw", np.uint8))
    dfs = np.diff(offsets)
    np.save(f"{dst}/dfs.i64.npy", dfs)
    tnb = (dfs + 127) // 128
    tb0 = np.zeros(n_terms + 1, np.int64)
    np.cumsum(tnb, out=tb0[1:])
    assert tb0[-1] == nblocks, (tb0[-1], nblocks)
    np.save(f"{dst}/term_block_start.i64.npy", tb0)
    for f_ in ("terms.pkl", "doclens.u32.npy"):
        shutil.copy(f"{src}/{f_}", f"{dst}/{f_}")
    meta.update({"format": "c1", "quant_scale": scale, "block_size": 128,
                 "n_blocks": nblocks})
    with open(f"{dst}/meta.json", "w") as f:
        json.dump(meta, f)
    shutil.rmtree(tmp)
    sizes = {os.path.basename(p): os.path.getsize(f"{dst}/{p}")
             for p in os.listdir(dst)}
    return {"compress_s": compress_s, "total_s": time.perf_counter() - t_all,
            "n_blocks": nblocks, "blob_gb": sizes["blob.bin"] / 1e9,
            "hot_mb": (sizes["blob.bin"] + sizes["block_last.u32.npy"]
                       + sizes["block_width.u8.npy"]
                       + sizes["block_maxq.u8.npy"]) / 1e6}


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1], sys.argv[2]), indent=2))
