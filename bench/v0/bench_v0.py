"""v0 full benchmark: build/save/load/query/quality at multiple corpus sizes.

Measures the scaling slope so we can extrapolate the full-corpus wall with
data instead of vibes.

Usage: python -m bench.v0.bench_v0 <data_dir> <out_json> [sizes...]

Honesty notes:
- MRR@10 is computed only over dev queries that have >=1 relevant pid within
  the indexed prefix. Fewer distractor docs = easier task, so subset MRR is
  an UPPER bound on full-corpus MRR. Recorded as `mrr_subset_upper_bound`.
- Latency measured on a fixed 300-query sample (seed 7) of those queries,
  warm (index in RAM). Cold load time reported separately.
"""
import json
import random
import resource
import statistics
import sys
import time

from searchengine.indexer import build_from_tsv, save, load
from searchengine.search import search


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_queries(path: str) -> dict[int, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            out[int(qid)] = text
    return out


def load_qrels(path: str) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, pid, _ = line.split()
            out.setdefault(int(qid), set()).add(int(pid))
    return out


def run_size(data_dir: str, n_docs: int, queries, qrels) -> dict:
    r: dict = {"n_docs": n_docs}
    idx, stats = build_from_tsv(f"{data_dir}/collection.tsv", n_docs)
    r.update(stats)
    r["build_docs_per_s"] = stats["n_docs"] / stats["build_s"]
    r["rss_after_build_gb"] = rss_gb()

    pkl = f"{data_dir}/v0_{n_docs}.pkl"
    r["save_s"] = save(idx, pkl)
    import os
    r["pickle_gb"] = os.path.getsize(pkl) / 1e9

    # relevant-in-subset queries only (see honesty note)
    eligible = [qid for qid, pids in qrels.items()
                if qid in queries and any(p < n_docs for p in pids)]
    eligible.sort()
    r["n_eligible_queries"] = len(eligible)

    # quality: MRR@10 (upper bound wrt full corpus)
    rr_total = 0.0
    t0 = time.perf_counter()
    for qid in eligible:
        hits = search(idx, queries[qid], k=10)
        for rank, (pid, _) in enumerate(hits, 1):
            if pid in qrels[qid]:
                rr_total += 1.0 / rank
                break
    r["mrr_eval_s"] = time.perf_counter() - t0
    r["mrr_subset_upper_bound"] = rr_total / len(eligible) if eligible else 0.0

    # latency distribution on fixed sample, warm
    random.seed(7)
    sample = random.sample(eligible, min(300, len(eligible)))
    lats = []
    for qid in sample:
        t0 = time.perf_counter()
        search(idx, queries[qid], k=10)
        lats.append((time.perf_counter() - t0) * 1e3)
    lats.sort()
    r["query_p50_ms"] = statistics.median(lats)
    r["query_p95_ms"] = lats[int(0.95 * len(lats))]
    r["query_p99_ms"] = lats[int(0.99 * len(lats))]
    r["query_mean_ms"] = statistics.fmean(lats)

    del idx
    _, r["pickle_load_s"] = load(pkl)
    return r


def main() -> None:
    data_dir, out_json = sys.argv[1], sys.argv[2]
    sizes = [int(s) for s in sys.argv[3:]] or [100_000, 500_000, 1_000_000]
    queries = load_queries(f"{data_dir}/queries.dev.small.tsv")
    qrels = load_qrels(f"{data_dir}/qrels.dev.small.tsv")
    results = []
    for n in sizes:
        print(f"=== size {n} ===", flush=True)
        r = run_size(data_dir, n, queries, qrels)
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
