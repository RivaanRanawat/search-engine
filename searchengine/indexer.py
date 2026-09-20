"""v0 indexer: the simplest thing that is honestly a search index.

In-memory inverted index: term -> list[(docid, tf)]. Doc lengths for BM25.
Persistence: pickle (naive on purpose; v1 replaces this with a real format).
Single process, single core.
"""
from collections import defaultdict
import pickle
import time

from .tokenizer import tokenize


class IndexV0:
    def __init__(self):
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lens: list[int] = []
        self.doc_ids: list[int] = []  # external ids (MS MARCO pids)
        self.n_docs = 0
        self.total_len = 0

    def add(self, ext_id: int, text: str) -> None:
        toks = tokenize(text)
        docid = self.n_docs
        tf: dict[str, int] = {}
        get = tf.get
        for t in toks:
            tf[t] = get(t, 0) + 1
        for t, f in tf.items():
            self.postings[t].append((docid, f))
        self.doc_lens.append(len(toks))
        self.doc_ids.append(ext_id)
        self.n_docs += 1
        self.total_len += len(toks)


def build_from_tsv(path: str, max_docs: int | None = None) -> tuple[IndexV0, dict]:
    """Returns (index, stage timings dict)."""
    idx = IndexV0()
    t0 = time.perf_counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            idx.add(int(pid), text)
            if max_docs is not None and idx.n_docs >= max_docs:
                break
    build_s = time.perf_counter() - t0
    return idx, {"build_s": build_s, "n_docs": idx.n_docs,
                 "n_terms": len(idx.postings), "n_tokens": idx.total_len}


def save(idx: IndexV0, path: str) -> float:
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        pickle.dump(
            {"postings": dict(idx.postings), "doc_lens": idx.doc_lens,
             "doc_ids": idx.doc_ids, "n_docs": idx.n_docs,
             "total_len": idx.total_len},
            f, protocol=pickle.HIGHEST_PROTOCOL)
    return time.perf_counter() - t0


def load(path: str) -> tuple[IndexV0, float]:
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        d = pickle.load(f)
    idx = IndexV0()
    idx.postings = d["postings"]
    idx.doc_lens = d["doc_lens"]
    idx.doc_ids = d["doc_ids"]
    idx.n_docs = d["n_docs"]
    idx.total_len = d["total_len"]
    return idx, time.perf_counter() - t0
