"""Does a cross-encoder add anything ON TOP of the bi-encoder hybrid?

The standalone sweep (bench/results/ce_train.json) showed the cross-encoder
reaching hybrid-level MRR only at depth 50, for ~270x the latency. That
makes CE pointless as a REPLACEMENT for hybrid. The remaining question —
the one worth compute — is whether it improves a hybrid-ranked shortlist,
which is the classic 3-stage cascade: BM25 -> bi-encoder -> cross-encoder.

Pipeline: BM25 top-50 -> hybrid blend (bi-encoder + PQ, alpha=0.1)
          -> take top-N -> cross-encoder rerank -> MRR@10.
TRAIN queries only.

Usage: python -m experiments.ce_on_hybrid [n_queries] [ce_depths...]
"""
import json
import random
import statistics
import sys
import time

import numpy as np

from searchengine.cross_encoder import CrossEncoder
from searchengine.encoder import Encoder
from searchengine.pq import PQ
from searchengine.search_hybrid import open_lexical
from searchengine.server import DocStore
from experiments.ce_value import load, mrr10


def main() -> None:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    depths = [int(x) for x in sys.argv[2:]] or [10, 20]

    queries, qrels = load("train")
    qids = sorted(q for q in qrels if q in queries)
    random.seed(31337)          # same sample as ce_value for comparability
    qids = random.sample(qids, min(n_q, len(qids)))

    s = open_lexical("indexes/v2c")
    store = DocStore("data/collection.tsv", "indexes/v2c")
    pq = PQ.load("indexes/dense/pq_centroids.npy")
    bi = Encoder(quantized=True, threads=6)
    ce = CrossEncoder(threads=6)

    def norm(x):
        return ((x - x.min()) / (x.max() - x.min())
                if x.max() > x.min() else np.zeros_like(x))

    res = {"n_queries": len(qids), "stages": {}}
    mrr_bm25 = mrr_hyb = 0.0
    ce_mrr = {d: 0.0 for d in depths}
    ce_lat = {d: [] for d in depths}
    hyb_lat = []

    for qid in qids:
        hits = s.search(queries[qid], 50)
        if not hits:
            continue
        pids = [p for p, _ in hits]
        bs = np.array([sc for _, sc in hits], np.float32)
        rel = qrels[qid]
        mrr_bm25 += mrr10(pids, rel)

        texts = [t.decode("utf-8", "replace")
                 for t in store.text_bytes_many(pids)]
        t0 = time.perf_counter()
        codes = pq.encode(bi.encode(texts, batch=32))     # index-equivalent
        qv = bi.encode([queries[qid]], batch=1)[0]
        ds = PQ.adc(pq.lut(qv), codes)
        blend = 0.1 * norm(bs) + 0.9 * norm(ds)
        hyb_order = np.argsort(-blend)
        hyb_lat.append((time.perf_counter() - t0) * 1e3)
        hyb_pids = [pids[i] for i in hyb_order]
        mrr_hyb += mrr10(hyb_pids, rel)

        for d in depths:
            short = hyb_pids[:d]
            stexts = [texts[pids.index(p)] for p in short]
            t0 = time.perf_counter()
            sc = ce.score(queries[qid], stexts)
            ce_lat[d].append((time.perf_counter() - t0) * 1e3)
            ce_mrr[d] += mrr10([short[i] for i in np.argsort(-sc)], rel)

    n = len(qids)
    res["stages"]["bm25"] = {"mrr": round(mrr_bm25 / n, 5)}
    res["stages"]["hybrid"] = {"mrr": round(mrr_hyb / n, 5),
                               "note": "includes on-the-fly passage encoding; "
                                       "served latency uses the prebuilt index"}
    for d in depths:
        lat = sorted(ce_lat[d])
        res["stages"][f"hybrid+ce@{d}"] = {
            "mrr": round(ce_mrr[d] / n, 5),
            "ce_p50_ms": round(lat[len(lat) // 2], 1),
            "ce_mean_ms": round(statistics.fmean(lat), 1)}
    print(json.dumps(res, indent=2))
    with open("bench/results/ce_on_hybrid.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
