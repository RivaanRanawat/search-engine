"""Build a reusable cache of Phase-A artifacts so fusion/PQ/quantization
experiments cost seconds instead of re-embedding every time.

Caches: candidate pids per query, BM25 scores, passage embeddings (int8 AND
fp32 encoders), query embeddings, qrels — for a fixed query sample.

Usage: python -m experiments.hybrid_cache <n_queries> <topK> <out.npz>
"""
import sys
import time

import numpy as np

from searchengine.encoder import Encoder
from searchengine.search_v2 import SearcherV2
from experiments.hybrid_value import load_eval

import random


def main() -> None:
    n_q, topk, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    queries, qrels = load_eval()
    qids = sorted(q for q in qrels if q in queries)
    random.seed(1234)
    qids = random.sample(qids, min(n_q, len(qids)))

    s = SearcherV2("indexes/v1s")
    cand, bm = [], []
    for qid in qids:
        hits = s.search(queries[qid], topk)
        pids = [p for p, _ in hits] + [-1] * (topk - len(hits))
        scores = [sc for _, sc in hits] + [0.0] * (topk - len(hits))
        cand.append(pids)
        bm.append(scores)
    cand = np.array(cand, np.int64)
    bm = np.array(bm, np.float32)

    need = sorted({int(p) for p in cand.ravel() if p >= 0})
    offs = np.load("indexes/v1s/lineoffsets.i64.npy")
    texts = []
    with open("data/collection.tsv", "rb") as f:
        for pid in need:
            f.seek(int(offs[pid]))
            texts.append(f.readline().decode("utf-8").split("\t", 1)[1].strip())
    print(f"{len(qids)} queries, {len(need)} unique passages", flush=True)

    payload = {"qids": np.array(qids, np.int64), "cand": cand, "bm25": bm,
               "need": np.array(need, np.int64)}
    for tag, quant in (("int8", True), ("fp32", False)):
        enc = Encoder(quantized=quant, threads=6)
        t0 = time.perf_counter()
        payload[f"demb_{tag}"] = enc.encode(texts, batch=32)
        payload[f"qemb_{tag}"] = enc.encode([queries[q] for q in qids],
                                            batch=32)
        print(f"{tag}: {len(texts)/(time.perf_counter()-t0):.0f} passages/s",
              flush=True)
        del enc
    # relevance as a padded matrix (max 5 rels per query in dev-small)
    maxr = max(len(qrels[q]) for q in qids)
    rel = np.full((len(qids), maxr), -1, np.int64)
    for i, q in enumerate(qids):
        for j, p in enumerate(sorted(qrels[q])):
            rel[i, j] = p
    payload["rel"] = rel
    np.savez(out, **payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
