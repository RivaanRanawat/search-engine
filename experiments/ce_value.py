"""Cross-encoder reranking: quality vs latency, measured.

The question is NOT "is a cross-encoder better" (it is, universally) but
"what does it cost us per point of MRR on THIS machine, and at what depth
does the curve flatten". Depth N is selected on TRAIN queries; dev-small is
reported once at the chosen N.

Usage: python -m experiments.ce_value <split:train|dev> <n_queries> <depths...>
"""
import json
import random
import statistics
import sys
import time

import numpy as np

from searchengine.cross_encoder import CrossEncoder
from searchengine.search_hybrid import open_lexical
from searchengine.server import DocStore


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


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    depths = [int(x) for x in sys.argv[3:]] or [10, 20, 50]

    queries, qrels = load(split)
    qids = sorted(q for q in qrels if q in queries)
    random.seed(31337)
    qids = random.sample(qids, min(n_q, len(qids)))

    s = open_lexical("indexes/v2c")
    store = DocStore("data/collection.tsv", "indexes/v2c")
    ce = CrossEncoder(threads=6)

    maxd = max(depths)
    cands, bm25_mrr = {}, 0.0
    for qid in qids:
        hits = s.search(queries[qid], maxd)
        cands[qid] = hits
        bm25_mrr += mrr10([p for p, _ in hits], qrels[qid])
    bm25_mrr /= len(qids)

    results = {"split": split, "n_queries": len(qids),
               "mrr_bm25": round(bm25_mrr, 5), "depths": []}
    print(json.dumps({"mrr_bm25": results["mrr_bm25"]}), flush=True)

    for d in depths:
        tot, lats = 0.0, []
        for qid in qids:
            hits = cands[qid][:d]
            if not hits:
                continue
            pids = [p for p, _ in hits]
            texts = [t.decode("utf-8", "replace")
                     for t in store.text_bytes_many(pids)]
            t0 = time.perf_counter()
            sc = ce.score(queries[qid], texts)
            lats.append((time.perf_counter() - t0) * 1e3)
            ranked = [pids[i] for i in np.argsort(-sc)]
            tot += mrr10(ranked, qrels[qid])
        lats.sort()
        r = {"depth": d, "mrr": round(tot / len(qids), 5),
             "rerank_p50_ms": round(lats[len(lats) // 2], 1),
             "rerank_p95_ms": round(lats[int(0.95 * len(lats))], 1),
             "rerank_mean_ms": round(statistics.fmean(lats), 1)}
        print(json.dumps(r), flush=True)
        results["depths"].append(r)

    with open(f"bench/results/ce_{split}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
