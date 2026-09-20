"""The real hybrid architecture: two INDEPENDENT retrievers, then fusion.

Until now the dense model only rescored BM25's candidates, so its recall was
BM25's recall. With an ANN index (searchengine/ann.py) the dense arm
retrieves from the whole corpus on its own, and the two arms fail
differently — the entire premise of hybrid search.

Measures, on the same queries:
  recall@k     BM25-only, dense-only, and their UNION (the new ceiling)
  MRR@10       for each fusion rule:
                 - linear blend of min-max normalised scores
                 - RRF (rank-based, k=60)
                 - dense-rescore-of-union (scores exist for every candidate
                   because we hold the codes)

Why re-test RRF: notes/13 found RRF *lost* to a linear blend, but that was a
RERANK setting where both scores existed for one shared candidate list. RRF
exists precisely for the case where lists differ and scores are not
comparable — which is only now true. Assuming the old result transfers would
be exactly the sort of unexamined constant that capped quality in notes/23.

Usage: python -m experiments.two_arm_fusion [n_queries] [split]
"""
import json
import random
import statistics
import sys
import time

import numpy as np

from searchengine.ann import IVFIndex
from searchengine.pq import PQ
from searchengine.search_hybrid import HybridSearcher

NPROBE = 32


def load(split: str):
    qf = ("data/queries.train.tsv" if split == "train"
          else "data/queries.dev.small.tsv")
    rf = ("data/qrels.train.tsv" if split == "train"
          else "data/qrels.dev.small.tsv")
    queries, qrels = {}, {}
    with open(qf, encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    with open(rf, encoding="utf-8") as f:
        for line in f:
            a = line.split()
            qrels.setdefault(int(a[0]), set()).add(int(a[2]))
    return queries, qrels


def mrr10(ranked, rel) -> float:
    for r, p in enumerate(ranked[:10], 1):
        if p in rel:
            return 1.0 / r
    return 0.0


def norm(x):
    return ((x - x.min()) / (x.max() - x.min())
            if len(x) and x.max() > x.min() else np.zeros_like(x))


def main() -> None:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    K = 1000

    queries, qrels = load(split)
    qids = sorted(q for q in qrels if q in queries)
    random.seed(1234)
    qids = random.sample(qids, min(n_q, len(qids)))

    h = HybridSearcher("indexes/v2c", "indexes/dense", alpha=0.1)
    ann = IVFIndex("indexes/ann", "indexes/dense")

    res = {"split": split, "n_queries": len(qids), "K": K, "nprobe": NPROBE}
    rec_b = rec_d = rec_u = 0.0
    mrr = {"bm25": 0.0, "dense_only": 0.0, "linear_rerank": 0.0,
           "union_dense": 0.0, "union_linear": 0.0, "rrf": 0.0}
    lat_b, lat_d = [], []

    for qid in qids:
        q, rel = queries[qid], qrels[qid]
        t0 = time.perf_counter()
        lex = h.bm25.search(q, K)
        lat_b.append((time.perf_counter() - t0) * 1e3)
        lex_ids = [p for p, _ in lex]
        lex_sc = np.array([s for _, s in lex], np.float32)

        qv = h.enc.encode([q], batch=1, is_query=True)[0]
        t0 = time.perf_counter()
        d_ids, d_sc = ann.search(qv, k=K, nprobe=NPROBE)
        lat_d.append((time.perf_counter() - t0) * 1e3)
        d_ids = d_ids.tolist()

        rec_b += any(p in rel for p in lex_ids)
        rec_d += any(p in rel for p in d_ids)
        union = list(dict.fromkeys(lex_ids + d_ids))
        rec_u += any(p in rel for p in union)

        mrr["bm25"] += mrr10(lex_ids, rel)
        mrr["dense_only"] += mrr10(d_ids, rel)

        # (a) rerank BM25 only — the current shipped pipeline
        ds_lex = PQ.adc(h.pq.lut(qv),
                        np.asarray(h.codes[np.array(lex_ids, np.int64)]))
        blend = 0.1 * norm(lex_sc) + 0.9 * norm(ds_lex)
        mrr["linear_rerank"] += mrr10(
            [lex_ids[i] for i in np.argsort(-blend)], rel)

        # (b) score the UNION densely (codes exist for every document)
        u = np.array(union, np.int64)
        ds_u = PQ.adc(h.pq.lut(qv), np.asarray(h.codes[u]))
        mrr["union_dense"] += mrr10(
            [union[i] for i in np.argsort(-ds_u)], rel)

        # (c) linear blend over the union; BM25 score is 0 where absent
        bmap = dict(zip(lex_ids, lex_sc.tolist()))
        bs_u = np.array([bmap.get(p, 0.0) for p in union], np.float32)
        blend_u = 0.1 * norm(bs_u) + 0.9 * norm(ds_u)
        mrr["union_linear"] += mrr10(
            [union[i] for i in np.argsort(-blend_u)], rel)

        # (d) RRF over the two ranked lists
        rr = {}
        for i, p in enumerate(lex_ids):
            rr[p] = rr.get(p, 0.0) + 1.0 / (60 + i + 1)
        for i, p in enumerate(d_ids):
            rr[p] = rr.get(p, 0.0) + 1.0 / (60 + i + 1)
        mrr["rrf"] += mrr10(sorted(rr, key=lambda p: -rr[p]), rel)

    n = len(qids)
    res["recall_at_K"] = {"bm25": round(rec_b / n, 4),
                          "dense": round(rec_d / n, 4),
                          "union": round(rec_u / n, 4)}
    res["mrr10"] = {k: round(v / n, 5) for k, v in mrr.items()}
    res["latency_ms"] = {"bm25_p50": round(statistics.median(lat_b), 2),
                         "ann_p50": round(statistics.median(lat_d), 2)}
    print(json.dumps(res, indent=2))
    with open(f"bench/results/two_arm_{split}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
