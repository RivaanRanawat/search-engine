"""Incremental indexing benchmark: what does 'live' actually cost?

Three questions:
1. **Write cost** — seconds per flushed segment, documents per second.
2. **Read cost vs segment count** — every extra segment is another posting
   traversal and another top-k merge, so query latency should grow roughly
   linearly in segment count. This is the entire reason merge policies
   exist, and the slope is the thing to measure.
3. **Quality drift** — a segment embeds the collection statistics known when
   it was written. Does incremental writing rank differently from a full
   rebuild of the same documents? Compared as top-10 agreement and MRR@10.

Usage: python -m bench.live.bench_live <n_docs> <batch_size>
"""
import json
import shutil
import statistics
import sys
import time

import numpy as np

from searchengine.live.reader import IndexMerger, LiveSearcher
from searchengine.live.writer import IndexWriter

LIVE = "/tmp/bench_live"
FULL = "/tmp/bench_full"


def load_docs(n: int) -> list[str]:
    out = []
    with open("data/collection.tsv", encoding="utf-8") as f:
        for line in f:
            out.append(line.split("\t", 1)[1].strip())
            if len(out) >= n:
                break
    return out


def eval_queries(n_q: int = 300):
    queries, qrels = {}, {}
    with open("data/queries.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    with open("data/qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.split()
            qrels.setdefault(int(a[0]), set()).add(int(a[2]))
    import random
    qids = sorted(q for q in qrels if q in queries)
    random.seed(808)
    return random.sample(qids, n_q), queries, qrels


def main() -> None:
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 25_000

    shutil.rmtree(LIVE, ignore_errors=True)
    shutil.rmtree(FULL, ignore_errors=True)
    docs = load_docs(n_docs)
    qids, queries, qrels = eval_queries()
    res = {"n_docs": n_docs, "batch": batch, "writes": [], "reads": []}

    w = IndexWriter(LIVE)
    for i in range(0, n_docs, batch):
        chunk = docs[i:i + batch]
        t0 = time.perf_counter()
        w.add_many(chunk)
        info = w.commit()
        dt = time.perf_counter() - t0
        res["writes"].append({"segment": info["flushed"], "docs": len(chunk),
                              "s": round(dt, 2),
                              "docs_per_s": round(len(chunk) / dt)})
        # read cost at this segment count
        s = LiveSearcher(LIVE)
        lat = []
        for qid in qids[:100]:
            t1 = time.perf_counter()
            s.search(queries[qid], 10)
            lat.append((time.perf_counter() - t1) * 1e3)
        res["reads"].append({"segments": s.n_segments,
                             "p50_ms": round(statistics.median(lat), 3),
                             "mean_ms": round(statistics.fmean(lat), 3)})
        print(json.dumps({**res["writes"][-1], **res["reads"][-1]}), flush=True)

    # full rebuild of the same documents, for the drift comparison
    from searchengine.compress_index import build as compress
    from searchengine.indexer_v4 import build as build_index
    with open("/tmp/bench_full.tsv", "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(f"{i}\t{' '.join(d.split())}\n")
    t0 = time.perf_counter()
    build_index("/tmp/bench_full.tsv", f"{FULL}_raw", workers=6)
    compress(f"{FULL}_raw", FULL)
    res["full_rebuild_s"] = round(time.perf_counter() - t0, 1)

    from searchengine.search_hybrid import open_lexical
    ref = open_lexical(FULL)
    live = LiveSearcher(LIVE)
    overlap = mrr_live = mrr_full = 0.0
    for qid in qids:
        a = [p for p, _ in ref.search(queries[qid], 10)]
        b = [p for p, _ in live.search(queries[qid], 10)]
        overlap += len(set(a) & set(b)) / 10.0
        rel = qrels[qid]
        for r, p in enumerate(b, 1):
            if p in rel:
                mrr_live += 1.0 / r
                break
        for r, p in enumerate(a, 1):
            if p in rel:
                mrr_full += 1.0 / r
                break
    n = len(qids)
    res["drift"] = {"top10_overlap_vs_full_rebuild": round(overlap / n, 4),
                    "mrr_live": round(mrr_live / n, 5),
                    "mrr_full_rebuild": round(mrr_full / n, 5),
                    "segments": live.n_segments}

    # merge everything down and re-measure
    m = IndexMerger(LIVE, max_segments=1, merge_factor=99)
    merges = []
    while True:
        r = m.maybe_merge()
        if not r:
            break
        merges.append(r)
    live.reopen()
    lat = []
    for qid in qids[:100]:
        t1 = time.perf_counter()
        live.search(queries[qid], 10)
        lat.append((time.perf_counter() - t1) * 1e3)
    ov2 = 0.0
    for qid in qids:
        a = [p for p, _ in ref.search(queries[qid], 10)]
        b = [p for p, _ in live.search(queries[qid], 10)]
        ov2 += len(set(a) & set(b)) / 10.0
    res["after_merge"] = {"merges": merges,
                          "segments": live.n_segments,
                          "p50_ms": round(statistics.median(lat), 3),
                          "top10_overlap_vs_full_rebuild": round(ov2 / n, 4)}

    print(json.dumps(res, indent=2))
    with open("bench/results/live_index.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
