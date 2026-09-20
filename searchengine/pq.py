"""Product quantization — from scratch (Lloyd's k-means, no ML deps).

Why this exists (notes/12): 8.84M x 384-dim f16 vectors = 6.8GB, and this
machine has ~1.9GB free. PQ is what FAISS/ScaNN/DiskANN do about that:
split each vector into M subvectors, k-means each subspace to 256 centroids,
store one byte per subspace. M=64 -> 64B/vector -> 566MB for the corpus,
a 24x compression over f32.

Scoring uses **asymmetric distance computation (ADC)**: the query stays in
full precision; for each subspace we precompute its dot product against all
256 centroids (M x 256 table), then a document's score is M table lookups
plus adds. Asymmetric beats symmetric because we never quantize the query.

Vectors here are L2-normalized, so inner product == cosine.
"""
import numpy as np


def _kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    """Lloyd's algorithm with k-means++-lite init (random distinct points)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n <= k:
        pad = np.repeat(x[-1:], k - n, axis=0) if n < k else x[:0]
        return np.vstack([x, pad]).astype(np.float32)
    cent = x[rng.choice(n, k, replace=False)].astype(np.float32).copy()
    for _ in range(iters):
        # assign: ||x||^2 is constant per point, so argmin over -2xc + ||c||^2
        cn = (cent * cent).sum(1)
        assign = np.argmin(-2.0 * (x @ cent.T) + cn[None, :], axis=1)
        # update
        sums = np.zeros_like(cent)
        counts = np.bincount(assign, minlength=k).astype(np.float32)
        np.add.at(sums, assign, x)
        nz = counts > 0
        cent[nz] = sums[nz] / counts[nz, None]
        if (~nz).any():  # re-seed dead centroids on random points
            cent[~nz] = x[rng.choice(n, int((~nz).sum()), replace=False)]
    return cent


class PQ:
    """M subspaces x 256 centroids. codes: uint8[n, M]."""

    def __init__(self, m: int = 64, dim: int = 384):
        assert dim % m == 0, (dim, m)
        self.m = m
        self.dim = dim
        self.dsub = dim // m
        self.centroids: np.ndarray | None = None  # (m, 256, dsub)

    def train(self, x: np.ndarray, iters: int = 20, seed: int = 0) -> "PQ":
        x = np.ascontiguousarray(x, np.float32)
        self.centroids = np.empty((self.m, 256, self.dsub), np.float32)
        for i in range(self.m):
            sub = x[:, i * self.dsub:(i + 1) * self.dsub]
            self.centroids[i] = _kmeans(sub, 256, iters, seed + i)
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, np.float32)
        out = np.empty((len(x), self.m), np.uint8)
        for i in range(self.m):
            sub = x[:, i * self.dsub:(i + 1) * self.dsub]
            c = self.centroids[i]
            d = -2.0 * (sub @ c.T) + (c * c).sum(1)[None, :]
            out[:, i] = np.argmin(d, axis=1).astype(np.uint8)
        return out

    def decode(self, codes: np.ndarray) -> np.ndarray:
        out = np.empty((len(codes), self.dim), np.float32)
        for i in range(self.m):
            out[:, i * self.dsub:(i + 1) * self.dsub] = \
                self.centroids[i][codes[:, i]]
        return out

    def lut(self, q: np.ndarray) -> np.ndarray:
        """(m, 256) table of inner products between query subvectors and
        centroids. Score of a code vector = sum_m lut[m, code[m]]."""
        t = np.empty((self.m, 256), np.float32)
        for i in range(self.m):
            t[i] = self.centroids[i] @ q[i * self.dsub:(i + 1) * self.dsub]
        return t

    @staticmethod
    def adc(lut: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Scores for codes (n, m) via table lookups — no decompression."""
        return lut[np.arange(lut.shape[0])[None, :], codes].sum(1)

    def save(self, path: str) -> None:
        np.save(path, self.centroids)

    @classmethod
    def load(cls, path: str, dim: int = 384) -> "PQ":
        c = np.load(path)
        pq = cls(m=c.shape[0], dim=dim)
        pq.centroids = c
        return pq
