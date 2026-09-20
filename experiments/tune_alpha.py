"""Tune the hybrid fusion weight α on TRAIN queries only.

Why this exists: the first α sweep (notes/13) was run on dev-small, which is
this project's sealed evaluation set. Selecting a hyperparameter there is
test-set peeking, even though the effect is small — so α is re-selected here
on held-out TRAIN queries and dev-small is touched exactly once, later, to
report the final number.

Production fidelity: candidate vectors are PQ-encoded with the real trained
codebook, so the scores swept here are the scores the server computes.

Usage: python -m experiments.tune_alpha [n_queries] [topk]
"""
import json
import random
import sys

import numpy as np

from searchengine.encoder import Encoder
from searchengine.pq import PQ
from searchengine.search_hybrid import open_lexical
from searchengine.server import DocStore


def main() -> None:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    queries, qrels = {}, {}
    with open("data/queries.train.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    with open("data/qrels.train.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.split()
            qrels.setdefault(int(a[0]), set()).add(int(a[2]))
    qids = sorted(q for q in qrels if q in queries)
    random.seed(4242)
    qids = random.sample(qids, n_q)

    s = open_lexical("indexes/v2c")
    store = DocStore("data/collection.tsv", "indexes/v2c")
    pq = PQ.load("indexes/dense/pq_centroids.npy")
    enc = Encoder(quantized=True, threads=4)

    cand, bm = [], []
    for qid in qids:
        hits = s.search(queries[qid], topk)
        cand.append([p for p, _ in hits])
        bm.append([sc for _, sc in hits])

    need = sorted({p for c in cand for p in c})
    texts = [t.decode("utf-8", "replace")
             for t in store.text_bytes_many(need)]
    print(f"{len(qids)} train queries, embedding {len(need)} passages ...",
          flush=True)
    demb = enc.encode(texts, batch=32)
    codes = pq.encode(demb)                      # exactly what the index holds
    pos = {p: i for i, p in enumerate(need)}
    qemb = enc.encode([queries[q] for q in qids], batch=32)

    def norm(x):
        return ((x - x.min()) / (x.max() - x.min())
                if x.max() > x.min() else np.zeros_like(x))

    results = {}
    for alpha in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
        tot = 0.0
        for i, qid in enumerate(qids):
            pids = cand[i]
            if not pids:
                continue
            ds = PQ.adc(pq.lut(qemb[i]),
                        codes[[pos[p] for p in pids]])
            blend = alpha * norm(np.array(bm[i], np.float32)) \
                + (1 - alpha) * norm(ds)
            ranked = [pids[j] for j in np.argsort(-blend)][:10]
            rel = qrels[qid]
            for r, p in enumerate(ranked, 1):
                if p in rel:
                    tot += 1.0 / r
                    break
        results[alpha] = round(tot / len(qids), 5)
        print(f"alpha={alpha:<5} train MRR@10={results[alpha]}", flush=True)

    best = max(results, key=results.get)
    print(json.dumps({"best_alpha": best, "train_mrr": results[best],
                      "grid": results}, indent=2))
    with open("bench/results/alpha_train.json", "w") as f:
        json.dump({"best_alpha": best, "grid": results,
                   "n_queries": n_q, "topk": topk}, f, indent=2)


if __name__ == "__main__":
    main()
