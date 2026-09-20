"""v3 searcher: block-compressed index + native MaxScore-with-block-skips.

Integer scoring domain (u8-quantized impacts); returned scores are rescaled
by quant_scale for display.
"""
import ctypes
import json
import os
import pickle
import subprocess
import threading

import numpy as np

from .porter import stem
from .tokenizer import tokenize

MAX_TERMS = 64
_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "native", "bmw.c")
_LIB = os.path.join(_DIR, "native", "libbmw.dylib")


def _lib():
    if (not os.path.exists(_LIB)
            or os.path.getmtime(_LIB) < os.path.getmtime(_SRC)):
        subprocess.run(["clang", "-O2", "-shared", "-o", _LIB, _SRC],
                       check=True)
    lib = ctypes.CDLL(_LIB)
    lib.bmw_query.restype = ctypes.c_int64
    lib.bmw_query.argtypes = [ctypes.c_void_p] * 5 + \
        [ctypes.c_void_p] * 4 + \
        [ctypes.c_int32, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
         ctypes.POINTER(ctypes.c_int64)]
    lib.bmw_intersect.restype = ctypes.c_int64
    lib.bmw_intersect.argtypes = [ctypes.c_void_p] * 5 + \
        [ctypes.c_void_p] * 3 + \
        [ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64]
    return lib


class _Scratch(threading.local):
    def __init__(self):
        self.tb0 = (ctypes.c_int64 * MAX_TERMS)()
        self.tnb = (ctypes.c_int64 * MAX_TERMS)()
        self.tdf = (ctypes.c_int64 * MAX_TERMS)()
        self.tmaxq = (ctypes.c_uint32 * MAX_TERMS)()
        self.out_ids = (ctypes.c_uint32 * 1024)()
        self.out_scores = (ctypes.c_uint32 * 1024)()
        self.stats = (ctypes.c_int64 * 2)()


class SearcherV3:
    def __init__(self, index_dir: str):
        with open(f"{index_dir}/meta.json") as f:
            self.meta = json.load(f)
        assert self.meta.get("format") == "c1"
        with open(f"{index_dir}/terms.pkl", "rb") as f:
            self.terms: dict[str, int] = pickle.load(f)
        self.scale = self.meta["quant_scale"]
        self.n_docs = self.meta["n_docs"]
        self._stemmed = bool(self.meta.get("stemmed"))
        self.blob = np.memmap(f"{index_dir}/blob.bin", np.uint8, "r")
        self.blast = np.load(f"{index_dir}/block_last.u32.npy")
        self.bwidth = np.load(f"{index_dir}/block_width.u8.npy")
        self.bmaxq = np.load(f"{index_dir}/block_maxq.u8.npy")
        self.tb0 = np.load(f"{index_dir}/term_block_start.i64.npy")
        self.dfs = np.load(f"{index_dir}/dfs.i64.npy")
        # block data offsets: prefix over ceil(c*w/8)+c
        nb = self.meta["n_blocks"]
        counts = np.full(nb, 128, np.int64)
        last_counts = self.dfs - (self.tb0[1:] - self.tb0[:-1] - 1) * 128
        counts[self.tb0[1:] - 1] = last_counts
        sizes = (counts * self.bwidth.astype(np.int64) + 7) // 8 + counts
        self.boff = np.zeros(nb, np.int64)  # same bytes as u64 for kernel
        np.cumsum(sizes[:-1], out=self.boff[1:])
        # per-term max quantized impact
        self.tmaxq_all = np.maximum.reduceat(
            self.bmaxq, self.tb0[:-1].astype(np.intp)).astype(np.uint32)
        self.lib = _lib()
        self._tl = _Scratch()
        self.last_stats = (0, 0)
        self._p = {n: a.ctypes.data for n, a in
                   (("blob", self.blob), ("blast", self.blast),
                    ("bwidth", self.bwidth), ("bmaxq", self.bmaxq),
                    ("boff", self.boff))}

    def _qtids(self, query: str) -> set[int]:
        toks = tokenize(query)
        if self._stemmed:
            toks = [stem(t) for t in toks]
        return {self.terms[t] for t in toks if t in self.terms}

    def intersect(self, terms: list[str], max_out: int = 200_000) -> np.ndarray:
        """All docids containing EVERY given term (already stemmed upstream
        if the index is stemmed). Complete, not top-k — this is the
        candidate set for exact phrase matching."""
        tids = []
        for t in terms:
            tid = self.terms.get(t)
            if tid is None:
                return np.empty(0, np.uint32)
            tids.append(tid)
        n = len(tids)
        if n == 0:
            return np.empty(0, np.uint32)
        tb0 = (ctypes.c_int64 * n)()
        tnb = (ctypes.c_int64 * n)()
        tdf = (ctypes.c_int64 * n)()
        for i, tid in enumerate(tids):
            tb0[i] = int(self.tb0[tid])
            tnb[i] = int(self.tb0[tid + 1]) - int(self.tb0[tid])
            tdf[i] = int(self.dfs[tid])
        out = (ctypes.c_uint32 * max_out)()
        cnt = self.lib.bmw_intersect(
            self._p["blob"], self._p["blast"], self._p["bwidth"],
            self._p["bmaxq"], self._p["boff"], tb0, tnb, tdf, n, out, max_out)
        return np.frombuffer(out, np.uint32, int(cnt)).copy()

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        tids = self._qtids(query)
        if not tids:
            return []
        entries = sorted((int(self.tmaxq_all[tid]), tid) for tid in tids)
        entries = entries[-MAX_TERMS:]
        n = len(entries)
        t = self._tl
        for i, (mq, tid) in enumerate(entries):
            t.tb0[i] = int(self.tb0[tid])
            t.tnb[i] = int(self.tb0[tid + 1]) - int(self.tb0[tid])
            t.tdf[i] = int(self.dfs[tid])
            t.tmaxq[i] = mq
        cnt = self.lib.bmw_query(
            self._p["blob"], self._p["blast"], self._p["bwidth"],
            self._p["bmaxq"], self._p["boff"],
            t.tb0, t.tnb, t.tdf, t.tmaxq,
            n, min(k, 1024), t.out_ids, t.out_scores, t.stats)
        self.last_stats = (int(t.stats[0]), int(t.stats[1]))
        return [(int(t.out_ids[i]), float(t.out_scores[i]) * self.scale)
                for i in range(cnt)]
