"""k1/b grid tuning — ON TRAIN QUERIES ONLY (dev-small is the sealed eval).

For each (k1,b): recompute impacts + max-impact sidecar in-place on a copy
of the index's impact files (cheap: vectorized, ~10s), then MRR@10 over a
fixed 2000-query sample of TRAIN queries with qrels.

Usage: python -m bench.quality.tune_k1b <index_dir> <data_dir> <out_json>
"""
import json
import random
import sys
import time

import numpy as np


def recompute_impacts(index_dir: str, k1: float, b: float) -> None:
    offsets = np.load(f"{index_dir}/offsets.u64.npy").astype(np.int64)
    docids = np.load(f"{index_dir}/docids.u32.npy", mmap_mode="r")
    tfs = np.load(f"{index_dir}/tfs.u8.npy", mmap_mode="r")
    doclens = np.load(f"{index_dir}/doclens.u32.npy")
    with open(f"{index_dir}/meta.json") as f:
        meta = json.load(f)
    n_docs = meta["n_docs"]
    dfs = np.diff(offsets)
    dl = doclens.astype(np.float32)
    avgdl = float(dl.mean())
    idf_t = np.log(1.0 + (n_docs - dfs + 0.5) / (dfs + 0.5)).astype(np.float32)
    total = int(offsets[-1])
    impacts = np.lib.format.open_memmap(
        f"{index_dir}/impacts.f32.npy", mode="w+", dtype=np.float32,
        shape=(total,))
    idf_p = np.repeat(idf_t, dfs)
    CH = 50_000_000
    for s in range(0, total, CH):
        e = min(s + CH, total)
        tf_f = tfs[s:e].astype(np.float32)
        dlp = dl[docids[s:e]]
        impacts[s:e] = idf_p[s:e] * tf_f * (k1 + 1.0) / (
            tf_f + k1 * (1.0 - b + b * dlp / avgdl))
    impacts.flush()
    del impacts, idf_p
    # refresh sidecar
    imp = np.load(f"{index_dir}/impacts.f32.npy", mmap_mode="r")
    mx = np.maximum.reduceat(imp, offsets[:-1]).astype(np.float32)
    np.save(f"{index_dir}/max_impact.f32.npy", mx)
    meta.update({"k1": k1, "b": b})
    with open(f"{index_dir}/meta.json", "w") as f:
        json.dump(meta, f)


def mrr_on(searcher, queries, qrels, qids) -> float:
    rr = 0.0
    for qid in qids:
        for rank, (pid, _) in enumerate(searcher.search(queries[qid], 10), 1):
            if pid in qrels[qid]:
                rr += 1.0 / rank
                break
    return rr / len(qids)


def main() -> None:
    index_dir, data_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    queries = {}
    with open(f"{data_dir}/queries.train.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[int(qid)] = text
    qrels: dict[int, set[int]] = {}
    with open(f"{data_dir}/qrels.train.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.split()
            qrels.setdefault(int(a[0]), set()).add(int(a[2]))
    random.seed(99)
    qids = random.sample(sorted(q for q in qrels if q in queries), 2000)

    from searchengine.search_v2 import SearcherV2
    results = []
    grid = [(k1, b) for k1 in (0.6, 0.82, 0.9, 1.2)
            for b in (0.3, 0.4, 0.68, 0.75)]
    for k1, b in grid:
        t0 = time.perf_counter()
        recompute_impacts(index_dir, k1, b)
        s = SearcherV2(index_dir)
        m = mrr_on(s, queries, qrels, qids)
        results.append({"k1": k1, "b": b, "train_mrr": m,
                        "s": time.perf_counter() - t0})
        print(json.dumps(results[-1]), flush=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
    best = max(results, key=lambda r: r["train_mrr"])
    print("BEST", json.dumps(best))


if __name__ == "__main__":
    main()
