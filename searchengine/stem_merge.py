"""Stem-by-merge: derive a Porter-stemmed index from an unstemmed v1 index
WITHOUT re-tokenizing the corpus.

Insight: stemming only merges vocabulary terms; doc lengths and avgdl are
unchanged (token count is stem-invariant). So:
  1. stem the 1.47M vocab terms in Python (~seconds),
  2. merge posting lists of same-stem terms with ONE vectorized u64-key sort
     (key = new_tid<<24 | docid; docid < 2^24 since 8.84M, tid < 2^24),
  3. sum tfs of merged (term,doc) duplicates via reduceat on boundaries,
  4. recompute idf/impacts (df changed), write a normal v1-format index.
Corpus re-scan avoided: ~40 min of interpreted tokenize+stem → ~1.5 min of
numpy. meta.json records stemmed=true; searchers stem queries accordingly.

Usage: python -m searchengine.stem_merge <src_index_dir> <dst_index_dir> [k1] [b]
"""
import json
import pickle
import sys
import time

import numpy as np

from .porter import stem

TF_CLAMP = 63


def build_stemmed(src: str, dst: str, k1: float = 0.9, b: float = 0.4) -> dict:
    import os
    os.makedirs(dst, exist_ok=True)
    t_all = time.perf_counter()
    with open(f"{src}/meta.json") as f:
        meta = json.load(f)
    assert not meta.get("stemmed"), "source already stemmed"
    with open(f"{src}/terms.pkl", "rb") as f:
        terms: dict[str, int] = pickle.load(f)
    offsets = np.load(f"{src}/offsets.u64.npy").astype(np.int64)
    docids = np.load(f"{src}/docids.u32.npy")
    tfs = np.load(f"{src}/tfs.u8.npy")
    doclens = np.load(f"{src}/doclens.u32.npy")
    n_old = len(terms)

    # 1. stem vocabulary
    t0 = time.perf_counter()
    new_terms: dict[str, int] = {}
    old2new = np.empty(n_old, np.uint32)
    for term, tid in terms.items():
        s = stem(term)
        nid = new_terms.get(s)
        if nid is None:
            nid = new_terms[s] = len(new_terms)
        old2new[tid] = nid
    stem_s = time.perf_counter() - t0

    # 2. one big key sort
    t0 = time.perf_counter()
    dfs_old = np.diff(offsets)
    key = np.repeat(old2new.astype(np.uint64), dfs_old) << np.uint64(24)
    key |= docids.astype(np.uint64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    tf_sorted = tfs[order].astype(np.uint32)
    del order
    sort_s = time.perf_counter() - t0

    # 3. collapse duplicate (new_tid, docid) keys
    t0 = time.perf_counter()
    starts = np.empty(len(key), bool)
    starts[0] = True
    np.not_equal(key[1:], key[:-1], out=starts[1:])
    start_idx = np.flatnonzero(starts)
    uniq_key = key[start_idx]
    del key
    tf_new = np.minimum(np.add.reduceat(tf_sorted, start_idx),
                        TF_CLAMP).astype(np.uint8)
    del tf_sorted
    docids_new = (uniq_key & np.uint64((1 << 24) - 1)).astype(np.uint32)
    tid_per_posting = (uniq_key >> np.uint64(24)).astype(np.uint32)
    del uniq_key
    n_new = len(new_terms)
    dfs_new = np.bincount(tid_per_posting, minlength=n_new).astype(np.int64)
    del tid_per_posting
    offsets_new = np.zeros(n_new + 1, np.uint64)
    np.cumsum(dfs_new, out=offsets_new[1:])
    total = int(offsets_new[-1])
    collapse_s = time.perf_counter() - t0

    # 4. impacts with new dfs
    t0 = time.perf_counter()
    n_docs = meta["n_docs"]
    dl = doclens.astype(np.float32)
    avgdl = float(dl.mean())
    idf_t = np.log(1.0 + (n_docs - dfs_new + 0.5) / (dfs_new + 0.5)).astype(np.float32)
    impacts = np.empty(total, np.float32)
    idf_p = np.repeat(idf_t, dfs_new)
    CH = 50_000_000
    for s0 in range(0, total, CH):
        e0 = min(s0 + CH, total)
        tf_f = tf_new[s0:e0].astype(np.float32)
        dlp = dl[docids_new[s0:e0]]
        impacts[s0:e0] = idf_p[s0:e0] * tf_f * (k1 + 1.0) / (
            tf_f + k1 * (1.0 - b + b * dlp / avgdl))
    del idf_p
    impacts_s = time.perf_counter() - t0

    np.save(f"{dst}/offsets.u64.npy", offsets_new)
    np.save(f"{dst}/docids.u32.npy", docids_new)
    np.save(f"{dst}/tfs.u8.npy", tf_new)
    np.save(f"{dst}/impacts.f32.npy", impacts)
    np.save(f"{dst}/doclens.u32.npy", doclens)
    with open(f"{dst}/terms.pkl", "wb") as f:
        pickle.dump(new_terms, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_new = {"n_docs": n_docs, "n_terms": n_new, "total_postings": total,
                "avgdl": avgdl, "k1": k1, "b": b, "stemmed": True}
    with open(f"{dst}/meta.json", "w") as f:
        json.dump(meta_new, f)
    return {"stem_s": stem_s, "sort_s": sort_s, "collapse_s": collapse_s,
            "impacts_s": impacts_s, "total_s": time.perf_counter() - t_all,
            **meta_new}


if __name__ == "__main__":
    k1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    b = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
    print(json.dumps(build_stemmed(sys.argv[1], sys.argv[2], k1, b), indent=2))
