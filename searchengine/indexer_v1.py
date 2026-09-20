"""v1 indexer: full-corpus, compact on-disk binary index.

Why (measured, notes/03): v0's Python-object postings cost 77B/posting →
28GB extrapolated (> 16GB RAM) and cap scoring at ~10.8M postings/s.
v1 stores postings as contiguous typed arrays and precomputes BM25 impact
scores at build time (winner of query-scoring/exp1-accumulation).

Build strategy (napkin-math choice, notes/04): single streaming pass
appending packed (docid*64 + min(tf,63)) into a per-term array('I') buffer
(~150ns/posting interpreted), then vectorized assembly into final arrays.
The alternative (two-pass cursor scatter with numpy scalar writes) costs
~100ns *per numpy scalar op* × ~4 ops × 350M postings ≈ 30+ min — rejected.

On-disk layout (dir):
  meta.json          n_docs, avgdl, k1, b, n_terms, total_postings
  terms.pkl          dict term -> termid
  offsets.u64.npy    postings start per termid (len n_terms+1)
  docids.u32.npy     concatenated posting docids (doc-ordered per term)
  tfs.u8.npy         term freqs (clamped 63; BM25 tf saturation makes the
                     clamp irrelevant: score(63)≈score(∞) at k1=0.9)
  impacts.f32.npy    precomputed BM25 impact per posting
  doclens.u32.npy    tokens per doc

docid == line number == MS MARCO pid (asserted during build).
"""
import json
import os
import pickle
import time
from array import array

import numpy as np

from .tokenizer import tokenize

K1 = 0.9
B = 0.4
TF_CLAMP = 63


def build(collection_path: str, out_dir: str, max_docs: int | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    timings: dict = {}

    term_bufs: dict[str, array] = {}
    doc_lens = array("I")
    t0 = time.perf_counter()
    n_docs = 0
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            assert int(pid) == n_docs, f"pids not sequential at line {n_docs}"
            toks = tokenize(text)
            tf: dict[str, int] = {}
            get = tf.get
            for t in toks:
                tf[t] = get(t, 0) + 1
            packed_doc = n_docs << 6
            for t, c in tf.items():
                buf = term_bufs.get(t)
                if buf is None:
                    buf = term_bufs[t] = array("I")
                buf.append(packed_doc | min(c, TF_CLAMP))
            doc_lens.append(len(toks))
            n_docs += 1
            if max_docs is not None and n_docs >= max_docs:
                break
    timings["scan_s"] = time.perf_counter() - t0

    # assembly: term order = first-seen; termid assignment is arbitrary
    t0 = time.perf_counter()
    n_terms = len(term_bufs)
    dfs = np.empty(n_terms, np.int64)
    terms = {}
    for tid, (term, buf) in enumerate(term_bufs.items()):
        terms[term] = tid
        dfs[tid] = len(buf)
    offsets = np.zeros(n_terms + 1, np.uint64)
    np.cumsum(dfs, out=offsets[1:])
    total = int(offsets[-1])

    docids = np.empty(total, np.uint32)
    tfs = np.empty(total, np.uint8)
    pos = 0
    for term, buf in term_bufs.items():
        v = np.frombuffer(buf, dtype=np.uint32)
        n = len(v)
        docids[pos:pos + n] = v >> 6
        tfs[pos:pos + n] = (v & TF_CLAMP).astype(np.uint8)
        pos += n
    term_bufs.clear()
    timings["assemble_s"] = time.perf_counter() - t0

    # impacts, vectorized in chunks (idf repeated per posting is 2.8GB if
    # materialized at once; chunking caps temp memory)
    t0 = time.perf_counter()
    dl = np.asarray(doc_lens, dtype=np.float32)
    avgdl = float(dl.mean())
    idf_t = np.log(1.0 + (n_docs - dfs + 0.5) / (dfs + 0.5)).astype(np.float32)
    impacts = np.empty(total, np.float32)
    idf_p = np.repeat(idf_t, dfs)  # 4B * total; acceptable, freed below
    CH = 50_000_000
    for s in range(0, total, CH):
        e = min(s + CH, total)
        tf_f = tfs[s:e].astype(np.float32)
        dlp = dl[docids[s:e]]
        impacts[s:e] = idf_p[s:e] * tf_f * (K1 + 1.0) / (
            tf_f + K1 * (1.0 - B + B * dlp / avgdl))
    del idf_p
    timings["impacts_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.save(f"{out_dir}/offsets.u64.npy", offsets)
    np.save(f"{out_dir}/docids.u32.npy", docids)
    np.save(f"{out_dir}/tfs.u8.npy", tfs)
    np.save(f"{out_dir}/impacts.f32.npy", impacts)
    np.save(f"{out_dir}/doclens.u32.npy", np.asarray(doc_lens, np.uint32))
    with open(f"{out_dir}/terms.pkl", "wb") as f:
        pickle.dump(terms, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"n_docs": n_docs, "n_terms": n_terms, "total_postings": total,
            "avgdl": avgdl, "k1": K1, "b": B}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f)
    timings["write_s"] = time.perf_counter() - t0
    timings.update(meta)
    return timings


if __name__ == "__main__":
    import sys
    stats = build(sys.argv[1], sys.argv[2],
                  int(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(json.dumps(stats, indent=2))
