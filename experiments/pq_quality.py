"""How much MRR does PQ compression cost, per byte spent?

Uses the cached Phase-A artifacts (no re-embedding). Trains PQ on a disjoint
half of the passages and evaluates rerank MRR@10 on the queries, sweeping
M (bytes per vector). Also reports what each option costs on disk for the
full 8.84M corpus — the number that actually decides the design.

Usage: python -m experiments.pq_quality <cache.npz>
"""
import json
import sys
import time

import numpy as np

from searchengine.pq import PQ

N_DOCS_FULL = 8_841_823


def mrr(cache, score_fn) -> float:
    qids, cand, rel = cache["qids"], cache["cand"], cache["rel"]
    total = 0.0
    for i in range(len(qids)):
        pids = cand[i]
        valid = pids >= 0
        if not valid.any():
            continue
        s = score_fn(i, pids, valid)
        order = np.argsort(-s)
        ranked = pids[valid][order][:10] if s.shape[0] == valid.sum() \
            else pids[order][:10]
        relset = set(int(p) for p in rel[i] if p >= 0)
        for r, p in enumerate(ranked, 1):
            if int(p) in relset:
                total += 1.0 / r
                break
    return total / len(qids)


def main() -> None:
    cache = np.load(sys.argv[1])
    tag = "int8"
    demb, qemb = cache[f"demb_{tag}"], cache[f"qemb_{tag}"]
    need = cache["need"]
    pos = {int(p): i for i, p in enumerate(need)}
    cand, bm25 = cache["cand"], cache["bm25"]

    def dense_exact(i, pids, valid):
        idx = [pos[int(p)] for p in pids[valid]]
        return demb[idx] @ qemb[i]

    res = {"mrr_bm25": round(mrr(cache, lambda i, p, v: bm25[i][v]), 5),
           "mrr_dense_exact": round(mrr(cache, dense_exact), 5),
           "pq": []}

    # train on a disjoint half of the passages (no leakage into eval scoring)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(demb))
    train = demb[perm[:len(demb) // 2]]

    for m in (24, 32, 48, 64, 96, 128):
        t0 = time.perf_counter()
        pq = PQ(m=m).train(train, iters=15)
        codes = pq.encode(demb)
        train_s = time.perf_counter() - t0

        def scorer(i, pids, valid, pq=pq, codes=codes):
            idx = [pos[int(p)] for p in pids[valid]]
            return PQ.adc(pq.lut(qemb[i]), codes[idx])

        recon = pq.decode(pq.encode(demb[:5000]))
        cos = float((recon * demb[:5000]).sum(1).mean()
                    / np.linalg.norm(recon, axis=1).mean())
        r = {"m_bytes": m, "mrr": round(mrr(cache, scorer), 5),
             "recon_cos": round(cos, 4),
             "corpus_gb": round(N_DOCS_FULL * m / 1e9, 3),
             "train_s": round(train_s, 1)}
        print(json.dumps(r), flush=True)
        res["pq"].append(r)

    print(json.dumps(res, indent=2))
    with open("bench/results/pq_quality.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
