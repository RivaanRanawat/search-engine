"""IVF-ADC: approximate nearest-neighbour search over the stored PQ codes.

This is the missing arm of the standard hybrid architecture. Until now the
dense model could only *rescore* BM25's candidates, so anything BM25 missed
was unreachable — measured as a hard ceiling of recall@1000 = 0.87
(notes/23). A dense *retriever* searches the whole corpus independently, so
the two arms fail differently and their union beats either.

Why IVF rather than HNSW: it reuses machinery this project already has
(k-means from pq.py, ADC scoring) and its memory is trivial — an assignment
array (4 bytes/doc) plus centroids — whereas an HNSW graph over 8.84M nodes
would cost several GB of links on a 16GB machine that is already holding a
764MB lexical index and 849MB of codes. IVFPQ is also what FAISS uses at
exactly this scale.

Construction:
  1. k-means over a sample of PQ-reconstructed vectors -> coarse centroids
  2. assign every document to its nearest centroid (chunked matmul)
  3. store the assignment as CSR inverted lists (sorted docids per cluster)

Query: encode -> pick `nprobe` nearest centroids -> ADC-score only the
documents in those lists -> top-k. Scoring is unchanged from the reranker
(same codes, same ADC tables), so quality differences come purely from
which candidates get considered.
"""
import json
import os
import time

import numpy as np

from .pq import PQ, _kmeans


class IVFIndex:
    @staticmethod
    def build(dense_dir: str, out_dir: str, n_clusters: int = 4096,
              sample: int = 400_000, chunk: int = 200_000,
              seed: int = 0) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        t_all = time.perf_counter()
        pq = PQ.load(f"{dense_dir}/pq_centroids.npy")
        codes = np.load(f"{dense_dir}/codes.u8.npy", mmap_mode="r")
        n = len(codes)

        # 1. coarse centroids from a random sample of reconstructed vectors
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, min(sample, n), replace=False))
        vecs = pq.decode(np.asarray(codes[idx]))
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
        centroids = _kmeans(vecs, n_clusters, iters=12, seed=seed)
        centroids /= np.clip(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9, None)
        train_s = time.perf_counter() - t0
        del vecs

        # 2. assign every document (chunked: decoding all 8.8M at once would
        #    need 13.6GB; chunks keep peak memory at a few hundred MB)
        t0 = time.perf_counter()
        assign = np.empty(n, np.int32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            v = pq.decode(np.asarray(codes[s:e]))
            v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
            assign[s:e] = np.argmax(v @ centroids.T, axis=1).astype(np.int32)
        assign_s = time.perf_counter() - t0

        # 3. CSR inverted lists: docids grouped by cluster, each group sorted
        t0 = time.perf_counter()
        order = np.argsort(assign, kind="stable").astype(np.int32)
        counts = np.bincount(assign, minlength=n_clusters).astype(np.int64)
        offsets = np.zeros(n_clusters + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        csr_s = time.perf_counter() - t0

        np.save(f"{out_dir}/centroids.f32.npy", centroids.astype(np.float32))
        np.save(f"{out_dir}/postings.i32.npy", order)
        np.save(f"{out_dir}/offsets.i64.npy", offsets)
        meta = {"n_docs": int(n), "n_clusters": int(n_clusters),
                "dense_dir": dense_dir,
                "mean_list": float(n / n_clusters),
                "max_list": int(counts.max()), "empty_lists": int((counts == 0).sum()),
                "train_s": round(train_s, 1), "assign_s": round(assign_s, 1),
                "csr_s": round(csr_s, 1),
                "total_s": round(time.perf_counter() - t_all, 1)}
        with open(f"{out_dir}/meta.json", "w") as f:
            json.dump(meta, f)
        return meta

    def __init__(self, index_dir: str, dense_dir: str | None = None):
        with open(f"{index_dir}/meta.json") as f:
            self.meta = json.load(f)
        dense_dir = dense_dir or self.meta["dense_dir"]
        self.pq = PQ.load(f"{dense_dir}/pq_centroids.npy")
        self.codes = np.load(f"{dense_dir}/codes.u8.npy", mmap_mode="r")
        self.centroids = np.load(f"{index_dir}/centroids.f32.npy")
        self.postings = np.load(f"{index_dir}/postings.i32.npy")
        self.offsets = np.load(f"{index_dir}/offsets.i64.npy")

    def search(self, qv: np.ndarray, k: int = 100, nprobe: int = 32):
        """Returns (docids, scores) for the top-k by approximate cosine."""
        cs = self.centroids @ qv
        probe = np.argpartition(-cs, min(nprobe, len(cs) - 1))[:nprobe]
        parts = [self.postings[self.offsets[c]:self.offsets[c + 1]]
                 for c in probe]
        cand = np.concatenate(parts) if parts else np.empty(0, np.int32)
        if cand.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        cand = np.sort(cand)                      # sorted => sequential mmap
        scores = PQ.adc(self.pq.lut(qv), np.asarray(self.codes[cand]))
        kk = min(k, len(scores))
        top = np.argpartition(-scores, kk - 1)[:kk]
        top = top[np.argsort(-scores[top])]
        return cand[top].astype(np.int64), scores[top]
