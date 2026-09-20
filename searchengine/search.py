"""v0 query engine: BM25, term-at-a-time accumulation, top-k heap.

k1=0.9, b=0.4 (Anserini defaults for MS MARCO).
"""
import heapq
import math

from .indexer import IndexV0
from .tokenizer import tokenize

K1 = 0.9
B = 0.4


def search(idx: IndexV0, query: str, k: int = 10) -> list[tuple[int, float]]:
    """Returns [(external_doc_id, score)] best-first."""
    terms = tokenize(query)
    if not terms or idx.n_docs == 0:
        return []
    avgdl = idx.total_len / idx.n_docs
    scores: dict[int, float] = {}
    doc_lens = idx.doc_lens
    for t in set(terms):
        plist = idx.postings.get(t)
        if not plist:
            continue
        df = len(plist)
        idf = math.log(1.0 + (idx.n_docs - df + 0.5) / (df + 0.5))
        for docid, tf in plist:
            dl = doc_lens[docid]
            s = idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl))
            if docid in scores:
                scores[docid] += s
            else:
                scores[docid] = s
    top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
    return [(idx.doc_ids[d], s) for d, s in top]
