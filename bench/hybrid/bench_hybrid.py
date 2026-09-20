"""End-to-end hybrid benchmark: quality AND latency, on the real system.

Honesty rules enforced here:
- Reports dense COVERAGE: the fraction of BM25 candidates whose embeddings
  actually exist in the dense index. While the corpus embedding job is
  still running only a prefix of docids is covered, and quality numbers on
  partial coverage are NOT comparable to full-corpus numbers. The harness
  refuses to print a headline MRR unless coverage is 100%, printing a
  clearly-labelled partial figure instead.
- Latency is measured on the full pipeline (BM25 + query encode + ADC),
  warm, with the phase breakdown that tells us what to optimize next.

Usage: python -m bench.hybrid.bench_hybrid <index_dir> <dense_dir> [n_queries]
"""
import json
import statistics
import sys
import time

import numpy as np

from searchengine.pq import PQ
from searchengine.search_hybrid import HybridSearcher


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main() -> None:
    index_dir, dense_dir = sys.argv[1], sys.argv[2]
    n_q = int(sys.argv[3]) if len(sys.argv) > 3 else 6980

    queries, qrels = {}, {}
    with open("data/queries.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    with open("data/qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.split()
            qrels.setdefault(int(a[0]), set()).add(int(a[2]))
    qids = sorted(q for q in qrels if q in queries)[:n_q]

    h = HybridSearcher(index_dir, dense_dir, alpha=0.1)  # depth = TOPK_RERANK
    codes = h.codes
    # coverage: an all-zero code row means "not yet written" (see
    # embed_corpus.verify) — treat those docs as uncovered
    lat, lat_bm25, lat_enc = [], [], []
    rr_hybrid = rr_bm25 = 0.0
    covered_hits = total_hits = 0

    for qid in qids:
        q = queries[qid]
        t0 = time.perf_counter()
        base = h.bm25.search(q, h.topk_bm25)
        t1 = time.perf_counter()
        qv = h.enc.encode([q], batch=1)[0]
        t2 = time.perf_counter()
        pids = np.array([p for p, _ in base], np.int64)
        bs = np.array([s for _, s in base], np.float32)
        if len(pids):
            c = np.asarray(codes[pids])
            covered = (c.max(axis=1) > 0)
            covered_hits += int(covered.sum())
            total_hits += len(pids)
            ds = PQ.adc(h.pq.lut(qv), c)
            ds[~covered] = ds.min() - 1.0 if covered.any() else 0.0
            n = lambda x: ((x - x.min()) / (x.max() - x.min())
                           if x.max() > x.min() else np.zeros_like(x))
            blend = 0.1 * n(bs) + 0.9 * n(ds)
            ranked = pids[np.argsort(-blend)][:10]
        else:
            ranked = np.array([], np.int64)
        t3 = time.perf_counter()
        lat_bm25.append((t1 - t0) * 1e3)
        lat_enc.append((t2 - t1) * 1e3)
        lat.append((t3 - t0) * 1e3)
        rel = qrels[qid]
        for r, p in enumerate(ranked, 1):
            if int(p) in rel:
                rr_hybrid += 1.0 / r
                break
        for r, (p, _) in enumerate(base[:10], 1):
            if p in rel:
                rr_bm25 += 1.0 / r
                break

    cov = covered_hits / max(1, total_hits)
    res = {"n_queries": len(qids), "dense_coverage": round(cov, 4),
           "mrr_bm25": round(rr_bm25 / len(qids), 5),
           "p50_ms": round(pct(lat, 50), 3), "p95_ms": round(pct(lat, 95), 3),
           "p99_ms": round(pct(lat, 99), 3),
           "mean_ms": round(statistics.fmean(lat), 3),
           "phase_bm25_ms": round(statistics.fmean(lat_bm25), 3),
           "phase_encode_ms": round(statistics.fmean(lat_enc), 3),
           "qps_1core": round(1000 / statistics.fmean(lat), 1)}
    key = ("mrr_hybrid" if cov >= 0.9999
           else f"mrr_hybrid_PARTIAL_coverage_{cov:.3f}")
    res[key] = round(rr_hybrid / len(qids), 5)
    print(json.dumps(res, indent=2))
    if cov >= 0.9999:
        with open("bench/results/hybrid_full.json", "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
