"""Rescore an index's impacts against externally supplied collection
statistics.

Shared by the distributed shard builder and the live segment writer: both
face the same problem — a partial index must score as if it knew the whole
collection, because BM25's idf and length normalisation are collection-wide
(notes/17, notes/19).
"""
import json
import pickle

import numpy as np


def rescore_with_global(index_dir: str, gdf: dict, n_docs: int,
                        avgdl: float) -> None:
    with open(f"{index_dir}/terms.pkl", "rb") as f:
        terms = pickle.load(f)
    with open(f"{index_dir}/meta.json") as f:
        meta = json.load(f)
    k1, b = meta["k1"], meta["b"]
    offsets = np.load(f"{index_dir}/offsets.u64.npy").astype(np.int64)
    docids = np.load(f"{index_dir}/docids.u32.npy")
    tfs = np.load(f"{index_dir}/tfs.u8.npy")
    doclens = np.load(f"{index_dir}/doclens.u32.npy").astype(np.float32)
    local_df = np.diff(offsets)

    gdf_arr = np.empty(len(local_df), np.float64)
    for t, tid in terms.items():
        gdf_arr[tid] = gdf.get(t, int(local_df[tid]))
    idf = np.log(1.0 + (n_docs - gdf_arr + 0.5) / (gdf_arr + 0.5))
    idf_p = np.repeat(idf.astype(np.float32), local_df)
    tf_f = tfs.astype(np.float32)
    dl = doclens[docids]
    impacts = (idf_p * tf_f * (k1 + 1.0)
               / (tf_f + k1 * (1.0 - b + b * dl / avgdl))).astype(np.float32)
    np.save(f"{index_dir}/impacts.f32.npy", impacts)
    np.save(f"{index_dir}/max_impact.f32.npy",
            np.maximum.reduceat(impacts, offsets[:-1].astype(np.intp)))
    meta.update({"global_stats": True, "avgdl": avgdl,
                 "n_docs_global": n_docs})
    with open(f"{index_dir}/meta.json", "w") as f:
        json.dump(meta, f)
