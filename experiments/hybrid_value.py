"""PHASE A — what is dense reranking actually WORTH on this corpus?

Answer this BEFORE spending ~9h embedding 8.84M passages (notes/12).

Method (honest, not a shortcut): for a random sample of sealed-dev queries,
take the BM25 top-K candidates the production engine actually returns, embed
exactly those passages plus the queries, and rerank. The reranker sees
precisely what it would see in a full system, so the measured MRR@10 is the
real number — we simply avoid embedding 8.8M passages nobody retrieves.

Compares:
  BM25            — current v7 engine
  Dense rerank    — cosine(query, passage) over BM25 top-K
  RRF fusion      — 1/(60+rank_bm25) + 1/(60+rank_dense)
  Linear fusion   — normalized score blend, alpha swept
Also compares int8 vs fp32 encoders (int8 is 2.5x faster but cos 0.955).

Usage: python -m experiments.hybrid_value <n_queries> <topK> [fp32]
"""
import json
import random
import sys
import time

import numpy as np

from searchengine.encoder import Encoder
from searchengine.search_v2 import SearcherV2


def load_eval():
    queries, qrels = {}, {}
    with open("data/queries.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    with open("data/qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.split()
            qrels.setdefault(int(a[0]), set()).add(int(a[2]))
    return queries, qrels


def mrr_at_10(ranked_pids, relevant) -> float:
    for rank, pid in enumerate(ranked_pids[:10], 1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def main() -> None:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    quant = not (len(sys.argv) > 3 and sys.argv[3] == "fp32")

    queries, qrels = load_eval()
    qids = sorted(q for q in qrels if q in queries)
    random.seed(1234)
    qids = random.sample(qids, min(n_q, len(qids)))

    s = SearcherV2("indexes/v1s")
    t0 = time.perf_counter()
    cand = {qid: [pid for pid, _ in s.search(queries[qid], topk)]
            for qid in qids}
    bm25_scores = {qid: {pid: sc for pid, sc in s.search(queries[qid], topk)}
                   for qid in qids}
    print(f"BM25 candidates: {time.perf_counter()-t0:.1f}s", flush=True)

    need = sorted({pid for v in cand.values() for pid in v})
    print(f"unique passages to embed: {len(need)}", flush=True)

    # fetch texts by line offset
    import numpy as _np
    offs = _np.load("indexes/v1s/lineoffsets.i64.npy")
    texts = []
    with open("data/collection.tsv", "rb") as f:
        for pid in need:
            f.seek(int(offs[pid]))
            texts.append(f.readline().decode("utf-8").split("\t", 1)[1].strip())

    enc = Encoder(quantized=quant, threads=6)
    t0 = time.perf_counter()
    demb = enc.encode(texts, batch=32)
    dt = time.perf_counter() - t0
    print(f"embedded {len(texts)} passages in {dt:.0f}s "
          f"({len(texts)/dt:.0f}/s)", flush=True)
    pos = {pid: i for i, pid in enumerate(need)}
    qemb = enc.encode([queries[q] for q in qids], batch=32)
    qpos = {qid: i for i, qid in enumerate(qids)}

    res = {"n_queries": len(qids), "topk": topk,
           "encoder": "int8" if quant else "fp32"}
    mr_bm25 = mr_dense = mr_rrf = 0.0
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    mr_lin = {a: 0.0 for a in alphas}
    for qid in qids:
        pids = cand[qid]
        if not pids:
            continue
        rel = qrels[qid]
        mr_bm25 += mrr_at_10(pids, rel)
        qv = qemb[qpos[qid]]
        sims = np.array([float(qv @ demb[pos[p]]) for p in pids])
        order = np.argsort(-sims)
        dense_ranked = [pids[i] for i in order]
        mr_dense += mrr_at_10(dense_ranked, rel)
        # RRF (k=60, standard)
        rr_b = {p: 1.0 / (60 + i + 1) for i, p in enumerate(pids)}
        rr_d = {p: 1.0 / (60 + i + 1) for i, p in enumerate(dense_ranked)}
        fused = sorted(pids, key=lambda p: -(rr_b[p] + rr_d[p]))
        mr_rrf += mrr_at_10(fused, rel)
        # linear blend on min-max normalized scores
        bs = np.array([bm25_scores[qid][p] for p in pids])
        bs = (bs - bs.min()) / max(1e-9, bs.max() - bs.min())
        ds = (sims - sims.min()) / max(1e-9, sims.max() - sims.min())
        for a in alphas:
            blend = a * bs + (1 - a) * ds
            mr_lin[a] += mrr_at_10([pids[i] for i in np.argsort(-blend)], rel)

    n = len(qids)
    res.update({"mrr_bm25": round(mr_bm25 / n, 5),
                "mrr_dense_rerank": round(mr_dense / n, 5),
                "mrr_rrf": round(mr_rrf / n, 5),
                "mrr_linear": {str(a): round(v / n, 5)
                               for a, v in mr_lin.items()}})
    print(json.dumps(res, indent=2))
    tag = "int8" if quant else "fp32"
    with open(f"bench/results/hybrid_value_{tag}_{n_q}q_{topk}k.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
