"""Is MRR@10 0.293 the model's ceiling, or the candidate list's?

The hybrid reranks BM25's top-K (K=50, inherited from the Phase-A study).
Any relevant passage BM25 misses at depth K is UNREACHABLE by reranking, so
the achievable MRR is bounded by BM25 recall@K. This measures both:

  recall@K  — fraction of queries whose relevant passage BM25 finds at all
              (the hard ceiling on any reranker over those candidates)
  MRR@10    — what the hybrid actually achieves at that K

Reranking deeper is nearly free: passage vectors are already stored, so the
added cost is ADC table lookups over more PQ codes plus a deeper BM25
traversal. If MRR climbs with K, the current number was a configuration
choice, not a model limit.

Usage: python -m experiments.rerank_depth [n_queries] [depths...]
"""
import json
import random
import statistics
import sys
import time

import numpy as np

from searchengine.pq import PQ
from searchengine.search_hybrid import HybridSearcher


def main() -> None:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    depths = [int(x) for x in sys.argv[2:]] or [10, 50, 100, 200, 500, 1000]

    queries, qrels = {}, {}
    with open("data/queries.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    with open("data/qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.split()
            qrels.setdefault(int(a[0]), set()).add(int(a[2]))
    qids = sorted(q for q in qrels if q in queries)
    random.seed(4242)
    qids = random.sample(qids, min(n_q, len(qids)))

    h = HybridSearcher("indexes/v2c", "indexes/dense", alpha=0.1)
    maxd = max(depths)

    def norm(x):
        return ((x - x.min()) / (x.max() - x.min())
                if x.max() > x.min() else np.zeros_like(x))

    # retrieve once at max depth; every smaller K is a prefix of it
    cand, bm, lat_bm = {}, {}, []
    for qid in qids:
        t0 = time.perf_counter()
        hits = h.bm25.search(queries[qid], maxd)
        lat_bm.append((time.perf_counter() - t0) * 1e3)
        cand[qid] = [p for p, _ in hits]
        bm[qid] = np.array([s for _, s in hits], np.float32)
    qv = {qid: h.enc.encode([queries[qid]], batch=1, is_query=True)[0]
          for qid in qids}

    out = {"n_queries": len(qids),
           "bm25_retrieval_p50_ms_at_maxdepth": round(
               statistics.median(lat_bm), 2), "depths": []}
    for K in depths:
        rec = mrr_h = mrr_b = 0.0
        lat = []
        for qid in qids:
            pids = cand[qid][:K]
            rel = qrels[qid]
            if any(p in rel for p in pids):
                rec += 1.0                       # ceiling: reachable at all
            for r, p in enumerate(pids[:10], 1):
                if p in rel:
                    mrr_b += 1.0 / r
                    break
            if not pids:
                continue
            t0 = time.perf_counter()
            ds = PQ.adc(h.pq.lut(qv[qid]),
                        np.asarray(h.codes[np.array(pids, np.int64)]))
            blend = 0.1 * norm(bm[qid][:K]) + 0.9 * norm(ds)
            ranked = [pids[i] for i in np.argsort(-blend)][:10]
            lat.append((time.perf_counter() - t0) * 1e3)
            for r, p in enumerate(ranked, 1):
                if p in rel:
                    mrr_h += 1.0 / r
                    break
        n = len(qids)
        row = {"K": K, "bm25_recall_at_K": round(rec / n, 4),
               "mrr10_bm25": round(mrr_b / n, 5),
               "mrr10_hybrid": round(mrr_h / n, 5),
               "rerank_p50_ms": round(statistics.median(lat), 3)}
        print(json.dumps(row), flush=True)
        out["depths"].append(row)

    with open("bench/results/rerank_depth.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
