"""BEIR zero-shot evaluation — does this engine generalize, or did we tune
it into a MS MARCO-shaped overfit?

Everything in this project was chosen against MS MARCO: k1/b, the stemmer,
the fusion weight α, the bi-encoder (which is literally named
msmarco-MiniLM). BEIR is the standard out-of-domain check, and running it
is the difference between "our engine scores 0.19" and "our engine works".

Pipeline per dataset: download → convert to our TSV (sequential ids + id
map) → build index with the SAME code that serves MS MARCO → evaluate
nDCG@10 for BM25 and, optionally, hybrid (small corpora are cheap to embed,
so out-of-domain dense quality is measurable too).

Usage: python -m bench.beir.run_beir <dataset> [--hybrid]
  datasets: scifact, nfcorpus, arguana, scidocs, fiqa, trec-covid
"""
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
ROOT = "data/beir"


def download(name: str) -> str:
    d = f"{ROOT}/{name}"
    if os.path.exists(f"{d}/corpus.jsonl"):
        return d
    os.makedirs(ROOT, exist_ok=True)
    zpath = f"{ROOT}/{name}.zip"
    # download atomically: an interrupted curl used to leave a partial .zip
    # that was then cached forever and failed to unzip on every rerun
    if not os.path.exists(zpath) or not zipfile.is_zipfile(zpath):
        url = f"{BASE}/{name}.zip"
        print(f"downloading {url} ...", flush=True)
        tmp = zpath + ".part"
        subprocess.run(["curl", "-sL", "--fail", "-o", tmp, url], check=True)
        if not zipfile.is_zipfile(tmp):
            os.remove(tmp)
            raise RuntimeError(f"{name}: downloaded file is not a zip")
        os.replace(tmp, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(ROOT)
    os.remove(zpath)
    return d


def convert(d: str) -> tuple[str, list[str]]:
    """BEIR corpus.jsonl -> our collection.tsv (docid == line number)."""
    out = f"{d}/collection.tsv"
    ids: list[str] = []
    with open(f"{d}/corpus.jsonl", encoding="utf-8") as f, \
            open(out, "w", encoding="utf-8") as o:
        for i, line in enumerate(f):
            rec = json.loads(line)
            text = f"{rec.get('title', '')} {rec.get('text', '')}"
            text = " ".join(text.split())
            o.write(f"{i}\t{text}\n")
            ids.append(rec["_id"])
    with open(f"{d}/idmap.json", "w") as f:
        json.dump(ids, f)
    return out, ids


def load_eval(d: str, ids: list[str]):
    pos = {ext: i for i, ext in enumerate(ids)}
    queries = {}
    with open(f"{d}/queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r["text"]
    qrels: dict[str, dict[int, int]] = {}
    qf = f"{d}/qrels/test.tsv"
    with open(qf, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            a = line.split()
            if len(a) < 3:
                continue
            qid, did, rel = a[0], a[1], int(float(a[2]))
            if did in pos and rel > 0:
                qrels.setdefault(qid, {})[pos[did]] = rel
    return queries, qrels


def ndcg_at_k(ranked: list[int], rels: dict[int, int], k: int = 10) -> float:
    dcg = sum((2 ** rels.get(p, 0) - 1) / np.log2(r + 1)
              for r, p in enumerate(ranked[:k], 1))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(r + 1) for r, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    name = sys.argv[1]
    want_hybrid = "--hybrid" in sys.argv
    d = download(name)
    coll, ids = convert(d)
    idx_raw, idx = f"{d}/idx_raw", f"{d}/idx"

    from searchengine.indexer_v4 import build as build_index
    from searchengine.compress_index import build as compress
    t0 = time.perf_counter()
    if not os.path.exists(f"{idx}/meta.json"):
        build_index(coll, idx_raw, workers=4, k1=0.82, b=0.75)
        compress(idx_raw, idx)
    build_s = time.perf_counter() - t0

    from searchengine.search_hybrid import open_lexical
    s = open_lexical(idx)
    queries, qrels = load_eval(d, ids)
    qids = [q for q in qrels if q in queries]

    res = {"dataset": name, "n_docs": len(ids), "n_queries": len(qids),
           "build_s": round(build_s, 1)}

    tot, lat = 0.0, []
    ranked_cache = {}
    for qid in qids:
        t1 = time.perf_counter()
        hits = s.search(queries[qid], 100)
        lat.append((time.perf_counter() - t1) * 1e3)
        ranked_cache[qid] = hits
        tot += ndcg_at_k([p for p, _ in hits], qrels[qid])
    res["ndcg@10_bm25"] = round(tot / len(qids), 5)
    res["p50_ms"] = round(float(np.median(lat)), 2)

    if want_hybrid:
        from searchengine.encoder import Encoder
        from searchengine.pq import PQ
        from searchengine.server import DocStore
        store = DocStore(coll, idx)
        enc = Encoder(quantized=True, threads=6)
        pq = PQ.load("indexes/dense/pq_centroids.npy")   # MS MARCO codebook
        norm = lambda x: ((x - x.min()) / (x.max() - x.min())
                          if x.max() > x.min() else np.zeros_like(x))
        tot_h = 0.0
        for qid in qids:
            hits = ranked_cache[qid][:50]
            if not hits:
                continue
            pids = [p for p, _ in hits]
            bs = np.array([sc for _, sc in hits], np.float32)
            texts = [t.decode("utf-8", "replace")
                     for t in store.text_bytes_many(pids)]
            codes = pq.encode(enc.encode(texts, batch=32))
            qv = enc.encode([queries[qid]], batch=1)[0]
            ds = PQ.adc(pq.lut(qv), codes)
            blend = 0.1 * norm(bs) + 0.9 * norm(ds)
            tot_h += ndcg_at_k([pids[i] for i in np.argsort(-blend)],
                               qrels[qid])
        res["ndcg@10_hybrid"] = round(tot_h / len(qids), 5)

    print(json.dumps(res, indent=2))
    os.makedirs("bench/results", exist_ok=True)
    with open(f"bench/results/beir_{name}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
