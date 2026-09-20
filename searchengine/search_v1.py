"""v1 searcher: mmap'd arrays + precomputed impacts + adaptive top-k.

Design chosen by experiment (branch query-scoring/exp1-accumulation):
- accumulate: scores[docids_slice] += impacts_slice per term (docids are
  unique within a posting list, so fancy-index += is exact)
- top-k: argpartition over full score array when candidate postings > 1M
  (flat ~22ms), else over the touched docids only (~1-3ms). Crossover
  measured: np.unique on 5.2M ids costs 99ms vs 22ms flat scan.
"""
import json
import pickle

import numpy as np

from .tokenizer import tokenize

ADAPTIVE_THRESHOLD = 1_000_000


class SearcherV1:
    def __init__(self, index_dir: str):
        with open(f"{index_dir}/meta.json") as f:
            self.meta = json.load(f)
        with open(f"{index_dir}/terms.pkl", "rb") as f:
            self.terms: dict[str, int] = pickle.load(f)
        self.offsets = np.load(f"{index_dir}/offsets.u64.npy")
        self.docids = np.load(f"{index_dir}/docids.u32.npy", mmap_mode="r")
        self.impacts = np.load(f"{index_dir}/impacts.f32.npy", mmap_mode="r")
        self.n_docs = self.meta["n_docs"]
        self._scores = np.zeros(self.n_docs, np.float32)
        self._stemmed = bool(self.meta.get("stemmed"))

    def _qtids(self, query: str) -> set[int]:
        toks = tokenize(query)
        if self._stemmed:
            from .porter import stem
            toks = [stem(t) for t in toks]
        return {self.terms[t] for t in toks if t in self.terms}

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        tids = self._qtids(query)
        if not tids:
            return []
        slices = []
        total = 0
        for tid in tids:
            s, e = int(self.offsets[tid]), int(self.offsets[tid + 1])
            slices.append((s, e))
            total += e - s
        scores = self._scores
        scores.fill(0.0)
        for s, e in slices:
            scores[self.docids[s:e]] += self.impacts[s:e]
        if total > ADAPTIVE_THRESHOLD:
            k_ = min(k, self.n_docs)
            idx = np.argpartition(scores, -k_)[-k_:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            return [(int(d), float(scores[d])) for d in idx if scores[d] > 0]
        touched = np.unique(np.concatenate(
            [self.docids[s:e] for s, e in slices]))
        vals = scores[touched]
        k_ = min(k, len(vals))
        idx = np.argpartition(vals, -k_)[-k_:]
        idx = idx[np.argsort(vals[idx])[::-1]]
        return [(int(touched[i]), float(vals[i])) for i in idx]
