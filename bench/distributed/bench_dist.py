"""Distributed benchmark: correctness first, then latency/throughput.

Three questions, in order of importance:

1. **Does sharding change the ANSWERS?** BM25's idf uses collection-wide
   df and N. Shards that only know their own slice compute different idf,
   so identical documents get different scores depending on how the corpus
   was split. We measure top-10 agreement and MRR@10 against the
   single-node index, for local-stats shards and for global-stats shards.
2. **What does fan-out cost?** A scatter-gather query costs the MAX over
   shards plus merge, not the mean — measured against single-node.
3. **What happens when a shard dies?** Results must degrade visibly, never
   silently.

Usage: python -m bench.distributed.bench_dist <broker_port> <mode>
"""
import json
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request


def fetch(port: int, path: str, timeout: float = 10.0):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    port = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "local"
    n_q = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

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
    random.seed(2024)
    qids = random.sample(qids, min(n_q, len(qids)))

    # single-node reference
    from searchengine.search_hybrid import open_lexical
    ref = open_lexical("indexes/v2c")

    agree_at10 = 0.0
    exact_top1 = 0
    mrr_dist = mrr_ref = 0.0
    lat, degraded = [], 0
    for qid in qids:
        q = queries[qid]
        r_ref = [p for p, _ in ref.search(q, 10)]
        t0 = time.perf_counter()
        d = fetch(port, f"/search?q={urllib.parse.quote(q)}&k=10")
        lat.append((time.perf_counter() - t0) * 1e3)
        r_dist = [h[0] for h in d["hits"]]
        degraded += bool(d.get("degraded"))
        if r_ref and r_dist:
            agree_at10 += len(set(r_ref) & set(r_dist)) / 10.0
            exact_top1 += int(r_ref[0] == r_dist[0])
        rel = qrels[qid]
        for r, p in enumerate(r_dist[:10], 1):
            if p in rel:
                mrr_dist += 1.0 / r
                break
        for r, p in enumerate(r_ref[:10], 1):
            if p in rel:
                mrr_ref += 1.0 / r
                break

    n = len(qids)
    lat.sort()
    out = {"mode": mode, "n_queries": n,
           "top10_overlap_vs_single_node": round(agree_at10 / n, 4),
           "top1_exact_match": round(exact_top1 / n, 4),
           "mrr_distributed": round(mrr_dist / n, 5),
           "mrr_single_node": round(mrr_ref / n, 5),
           "broker_p50_ms": round(lat[n // 2], 2),
           "broker_p95_ms": round(lat[int(0.95 * n)], 2),
           "broker_p99_ms": round(lat[int(0.99 * n)], 2),
           "broker_mean_ms": round(statistics.fmean(lat), 2),
           "degraded_responses": degraded}
    print(json.dumps(out, indent=2))
    with open(f"bench/results/dist_{mode}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
