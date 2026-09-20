"""Why does dense reranking HURT out of domain, and by how much?

BEIR SciFact: BM25 nDCG@10 = 0.688, hybrid (alpha=0.1) = 0.611. The dense
stage that is worth +56% on MS MARCO costs 11% here. Two candidate causes,
separated by this experiment:

  A. DOMAIN MISMATCH — msmarco-MiniLM was trained on web/QA queries; SciFact
     is scientific claim verification. The BEIR paper's headline finding is
     precisely that BM25 beats many dense retrievers zero-shot.
  B. CODEBOOK MISMATCH — our PQ centroids were fit on MS MARCO embeddings
     and are being applied to a different embedding distribution.

Sweeping alpha with EXACT dense scores (no PQ) versus PQ scores separates
them: if exact-dense is also bad, it is the model (A); if only PQ is bad,
it is the codebook (B).

Usage: python -m experiments.ood_alpha <beir_dataset>
"""
import json
import sys

import numpy as np

from bench.beir.run_beir import convert, download, load_eval, ndcg_at_k
from searchengine.encoder import Encoder
from searchengine.pq import PQ
from searchengine.search_hybrid import open_lexical
from searchengine.server import DocStore


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    d = download(name)
    coll, ids = convert(d)
    idx = f"{d}/idx"
    s = open_lexical(idx)
    store = DocStore(coll, idx)
    queries, qrels = load_eval(d, ids)
    qids = [q for q in qrels if q in queries]

    enc = Encoder(quantized=True, threads=6)
    pq = PQ.load("indexes/dense/pq_centroids.npy")

    cand, bm = {}, {}
    for qid in qids:
        hits = s.search(queries[qid], 50)
        cand[qid] = [p for p, _ in hits]
        bm[qid] = np.array([sc for _, sc in hits], np.float32)

    need = sorted({p for v in cand.values() for p in v})
    texts = [t.decode("utf-8", "replace") for t in store.text_bytes_many(need)]
    print(f"{len(qids)} queries, embedding {len(need)} passages ...", flush=True)
    demb = enc.encode(texts, batch=32)
    pos = {p: i for i, p in enumerate(need)}
    qemb = enc.encode([queries[q] for q in qids], batch=32)
    codes = pq.encode(demb)

    # a codebook fitted on THIS corpus isolates the codebook effect
    pq_local = PQ(m=96).train(demb, iters=15)
    codes_local = pq_local.encode(demb)

    def norm(x):
        return ((x - x.min()) / (x.max() - x.min())
                if x.max() > x.min() else np.zeros_like(x))

    out = {"dataset": name, "n_queries": len(qids), "sweeps": {}}
    for mode in ("exact", "pq_msmarco_codebook", "pq_local_codebook"):
        row = {}
        for alpha in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            tot = 0.0
            for i, qid in enumerate(qids):
                pids = cand[qid]
                if not pids:
                    continue
                idxs = [pos[p] for p in pids]
                if mode == "exact":
                    ds = demb[idxs] @ qemb[i]
                elif mode == "pq_msmarco_codebook":
                    ds = PQ.adc(pq.lut(qemb[i]), codes[idxs])
                else:
                    ds = PQ.adc(pq_local.lut(qemb[i]), codes_local[idxs])
                blend = alpha * norm(bm[qid]) + (1 - alpha) * norm(ds)
                tot += ndcg_at_k([pids[j] for j in np.argsort(-blend)],
                                 qrels[qid])
            row[str(alpha)] = round(tot / len(qids), 5)
        out["sweeps"][mode] = row
        print(mode, json.dumps(row), flush=True)

    print(json.dumps(out, indent=2))
    with open(f"bench/results/ood_alpha_{name}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
