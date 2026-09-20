"""Encoder bake-off: is our 2021 MS MARCO bi-encoder the weak link?

notes/18 isolated the bi-encoder as the cause of out-of-domain collapse
(BM25 0.688 vs dense 0.570 on SciFact) and exonerated PQ. Modern small
encoders are trained on diverse data rather than MS MARCO alone, at the
same 384 dimensions — so a swap costs nothing in storage and only what the
extra layers cost in time.

This is time-sensitive: the corpus embedding job is partway through with the
OLD model. Every completed shard raises the cost of switching, so the
decision is made now rather than after it finishes.

Measures, for each model, on the SAME candidates:
  - in-domain MS MARCO MRR@10 (rerank of BM25 top-50, TRAIN queries)
  - out-of-domain BEIR nDCG@10 (rerank of BM25 top-50)
  - encode throughput (the cost side of the trade)
  - best alpha per corpus (since notes/18 showed alpha is per-corpus)

Usage: python -m experiments.encoder_bakeoff [n_msmarco_queries]
"""
import json
import random
import sys
import time

import numpy as np

from bench.beir.run_beir import convert, download, load_eval, ndcg_at_k
from searchengine.encoder import Encoder
from searchengine.search_hybrid import open_lexical
from searchengine.server import DocStore

MODELS = ["models/minilm", "models/bge", "models/gte"]
ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def norm(x):
    return ((x - x.min()) / (x.max() - x.min())
            if x.max() > x.min() else np.zeros_like(x))


def msmarco_candidates(n_q: int):
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
    random.seed(555)
    qids = random.sample(qids, n_q)
    s = open_lexical("indexes/v2c")
    store = DocStore("data/collection.tsv", "indexes/v2c")
    cand, bm = {}, {}
    for qid in qids:
        hits = s.search(queries[qid], 50)
        cand[qid] = [p for p, _ in hits]
        bm[qid] = np.array([sc for _, sc in hits], np.float32)
    need = sorted({p for v in cand.values() for p in v})
    texts = [t.decode("utf-8", "replace") for t in store.text_bytes_many(need)]
    return qids, queries, qrels, cand, bm, need, texts


def beir_candidates(name: str):
    d = download(name)
    coll, ids = convert(d)
    idx = f"{d}/idx"
    import os
    if not os.path.exists(f"{idx}/meta.json"):
        from searchengine.compress_index import build as compress
        from searchengine.indexer_v4 import build as build_index
        build_index(coll, f"{d}/idx_raw", workers=4, k1=0.82, b=0.75)
        compress(f"{d}/idx_raw", idx)
    s = open_lexical(idx)
    store = DocStore(coll, idx)
    queries, qrels = load_eval(d, ids)
    qids = [q for q in qrels if q in queries]
    cand, bm = {}, {}
    for qid in qids:
        hits = s.search(queries[qid], 50)
        cand[qid] = [p for p, _ in hits]
        bm[qid] = np.array([sc for _, sc in hits], np.float32)
    need = sorted({p for v in cand.values() for p in v})
    texts = [t.decode("utf-8", "replace") for t in store.text_bytes_many(need)]
    return qids, queries, qrels, cand, bm, need, texts


def evaluate(model: str, data, metric: str) -> dict:
    qids, queries, qrels, cand, bm, need, texts = data
    enc = Encoder(model_dir=model, quantized=True, threads=6)
    t0 = time.perf_counter()
    demb = enc.encode(texts, batch=32)
    rate = len(texts) / (time.perf_counter() - t0)
    qemb = enc.encode([queries[q] for q in qids], batch=32, is_query=True)
    pos = {p: i for i, p in enumerate(need)}
    out = {"encode_per_s": round(rate, 1), "alphas": {}}
    for alpha in ALPHAS:
        tot = 0.0
        for i, qid in enumerate(qids):
            pids = cand[qid]
            if not pids:
                continue
            ds = demb[[pos[p] for p in pids]] @ qemb[i]
            blend = alpha * norm(bm[qid]) + (1 - alpha) * norm(ds)
            ranked = [pids[j] for j in np.argsort(-blend)]
            if metric == "mrr":
                rel = qrels[qid]
                for r, p in enumerate(ranked[:10], 1):
                    if p in rel:
                        tot += 1.0 / r
                        break
            else:
                tot += ndcg_at_k(ranked, qrels[qid])
        out["alphas"][str(alpha)] = round(tot / len(qids), 5)
    best = max(out["alphas"], key=lambda a: out["alphas"][a])
    out["best_alpha"] = float(best)
    out["best_score"] = out["alphas"][best]
    del enc
    return out


def main() -> None:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    results = {}

    print("=== MS MARCO (in-domain, MRR@10, TRAIN queries) ===", flush=True)
    ms = msmarco_candidates(n_q)
    for m in MODELS:
        r = evaluate(m, ms, "mrr")
        results.setdefault(m, {})["msmarco"] = r
        print(f"{m}: best_alpha={r['best_alpha']} mrr={r['best_score']} "
              f"({r['encode_per_s']}/s) {r['alphas']}", flush=True)
    del ms

    for ds in ("scifact", "nfcorpus"):
        print(f"=== BEIR {ds} (out-of-domain, nDCG@10) ===", flush=True)
        data = beir_candidates(ds)
        for m in MODELS:
            r = evaluate(m, data, "ndcg")
            results.setdefault(m, {})[ds] = r
            print(f"{m}: best_alpha={r['best_alpha']} ndcg={r['best_score']} "
                  f"{r['alphas']}", flush=True)
        del data

    with open("bench/results/encoder_bakeoff.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
