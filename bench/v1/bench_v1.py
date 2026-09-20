"""v1 full benchmark: FULL corpus, apples-to-apples vs Anserini BM25 0.184.

- MRR@10 over ALL 6980 dev-small queries (no subset caveats anymore).
- Latency: per-query, distinguishing first pass (cold-ish mmap pages) from
  second pass (warm). Percentiles over the full query set, plus breakdown
  by candidate-postings volume (the adaptive-top-k split).

Usage: python -m bench.v1.bench_v1 <data_dir> <index_dir> <out_json>
"""
import json
import statistics
import sys
import time

from searchengine.search_hybrid import open_lexical


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main() -> None:
    data_dir, index_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.perf_counter()
    s = open_lexical(index_dir)
    load_s = time.perf_counter() - t0

    queries: dict[int, str] = {}
    with open(f"{data_dir}/queries.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    qrels: dict[int, set[int]] = {}
    with open(f"{data_dir}/qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, _, pid, _ = line.split()
            qrels.setdefault(int(qid), set()).add(int(pid))

    qids = sorted(q for q in qrels if q in queries)
    r: dict = {"n_queries": len(qids), "load_s": load_s}

    # pass 1: cold-ish (fresh mmap; OS cache state as-is) — also MRR
    lat_cold, rr = [], 0.0
    t0 = time.perf_counter()
    for qid in qids:
        t1 = time.perf_counter()
        hits = s.search(queries[qid], k=10)
        lat_cold.append((time.perf_counter() - t1) * 1e3)
        for rank, (pid, _) in enumerate(hits, 1):
            if pid in qrels[qid]:
                rr += 1.0 / rank
                break
    r["pass1_total_s"] = time.perf_counter() - t0
    r["mrr_at_10_full"] = rr / len(qids)

    # pass 2: warm
    lat_warm = []
    for qid in qids:
        t1 = time.perf_counter()
        s.search(queries[qid], k=10)
        lat_warm.append((time.perf_counter() - t1) * 1e3)

    for name, lats in (("cold", lat_cold), ("warm", lat_warm)):
        r[f"{name}_p50_ms"] = pct(lats, 50)
        r[f"{name}_p95_ms"] = pct(lats, 95)
        r[f"{name}_p99_ms"] = pct(lats, 99)
        r[f"{name}_mean_ms"] = statistics.fmean(lats)
        r[f"{name}_max_ms"] = max(lats)
    r["warm_qps_1core"] = 1000.0 / r["warm_mean_ms"]

    print(json.dumps(r, indent=2))
    with open(out_json, "w") as f:
        json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
