"""v2 searcher: exact top-k via native MaxScore kernel (C, ctypes).

Same v1 index format + one sidecar (max_impact.f32.npy, built on first use).
Exact results (modulo fp addition order) — numpy v1 remains the oracle.
"""
import ctypes
import os
import threading

import numpy as np

from .native import lib
from .search_v1 import SearcherV1
from .tokenizer import tokenize

MAX_TERMS = 64


class _Scratch(threading.local):
    """Per-thread ctypes buffers: the kernel call releases the GIL, so
    concurrent searches would race on shared scratch."""

    def __init__(self):
        self.ids_ptrs = (ctypes.c_void_p * MAX_TERMS)()
        self.imp_ptrs = (ctypes.c_void_p * MAX_TERMS)()
        self.lens = (ctypes.c_int64 * MAX_TERMS)()
        self.maxs = (ctypes.c_float * MAX_TERMS)()
        self.out_ids = (ctypes.c_uint32 * 1024)()
        self.out_scores = (ctypes.c_float * 1024)()
        self.stats = (ctypes.c_int64 * 2)()


class SearcherV2(SearcherV1):
    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        side = f"{index_dir}/max_impact.f32.npy"
        if not os.path.exists(side):
            offs = self.offsets.astype(np.intp)
            imp = np.load(f"{index_dir}/impacts.f32.npy", mmap_mode="r")
            mx = np.maximum.reduceat(imp, offs[:-1]).astype(np.float32)
            np.save(side, mx)
        self.max_impact = np.load(side)
        self._tl = _Scratch()
        self._ids_base = self.docids.ctypes.data
        self._imp_base = self.impacts.ctypes.data
        self.last_stats = (0, 0)

    def intersect(self, terms: list[str], max_out: int = 200_000) -> np.ndarray:
        """All docids containing EVERY given term (already stemmed upstream
        if the index is stemmed). Complete, not top-k — this is the
        candidate set for exact phrase matching. Same contract as
        SearcherV3.intersect; verified to return identical sets.

        Raw arrays make this simple: each term's docid slice is sorted, so
        drive from the smallest list and keep only candidates found in every
        other list via binary-search probes (np.searchsorted)."""
        spans = []
        for t in terms:
            tid = self.terms.get(t)
            if tid is None:
                return np.empty(0, np.uint32)
            s, e = int(self.offsets[tid]), int(self.offsets[tid + 1])
            spans.append((e - s, s, e))
        if not spans:
            return np.empty(0, np.uint32)
        spans.sort()  # smallest df first: the result can't be bigger than it
        _, s, e = spans[0]
        cand = np.asarray(self.docids[s:e])
        for _, s, e in spans[1:]:
            if cand.size == 0:
                break
            arr = self.docids[s:e]
            idx = np.searchsorted(arr, cand)
            idx[idx == len(arr)] = len(arr) - 1
            cand = cand[arr[idx] == cand]
        return cand[:max_out].astype(np.uint32, copy=False)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        tids = self._qtids(query)
        if not tids:
            return []
        entries = []
        for tid in tids:
            s, e = int(self.offsets[tid]), int(self.offsets[tid + 1])
            entries.append((float(self.max_impact[tid]), s, e))
        entries.sort()  # max impact ASCENDING (kernel contract)
        entries = entries[-MAX_TERMS:]
        n = len(entries)
        t = self._tl
        for i, (mx, s, e) in enumerate(entries):
            t.ids_ptrs[i] = self._ids_base + 4 * s
            t.imp_ptrs[i] = self._imp_base + 4 * s
            t.lens[i] = e - s
            t.maxs[i] = mx
        cnt = lib.maxscore_query(
            t.ids_ptrs, t.imp_ptrs, t.lens, t.maxs,
            n, min(k, 1024), t.out_ids, t.out_scores, t.stats)
        self.last_stats = (int(t.stats[0]), int(t.stats[1]))
        return [(int(t.out_ids[i]), float(t.out_scores[i]))
                for i in range(cnt)]
