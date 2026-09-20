"""Dense vector index with two storage modes, chosen by size — not habit.

Product quantization exists because 8.84M x 384 f32 vectors are 13.6GB and
do not fit this machine (notes/12). That reason evaporates for a small
corpus: the 7,177-page web index needs 11MB of exact float32, so quantizing
it would trade ~1% of ranking quality for nothing. The mode is therefore a
measured decision:

    exact  : vectors stored f32, scores are true cosine
    pq     : 96 bytes/vector, ~0.8% quality cost, needed above ~1M docs

Also records WHICH encoder built the index. The bake-off (notes/19) showed
in-domain and out-of-domain corpora want different models, so mixing an
index built by one encoder with queries embedded by another would silently
produce garbage. The searcher refuses to do that.
"""
import json
import os

import numpy as np

from .encoder import Encoder
from .pq import PQ

EXACT_MAX_DOCS = 1_000_000          # ~1.5GB f32 — above this, quantize


class DenseIndex:
    @staticmethod
    def build(texts: list[str], out_dir: str, model_dir: str,
              mode: str | None = None, threads: int = 6) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        enc = Encoder(model_dir=model_dir, quantized=True, threads=threads)
        emb = enc.encode(texts, batch=32)
        if mode is None:
            mode = "exact" if len(texts) <= EXACT_MAX_DOCS else "pq"
        meta = {"mode": mode, "model_dir": model_dir, "dim": int(emb.shape[1]),
                "n_docs": len(texts)}
        if mode == "exact":
            np.save(f"{out_dir}/vectors.f32.npy", emb)
        else:
            pq = PQ(m=96).train(emb[:min(len(emb), 200_000)], iters=20)
            pq.save(f"{out_dir}/pq_centroids.npy")
            np.save(f"{out_dir}/codes.u8.npy", pq.encode(emb))
        with open(f"{out_dir}/meta.json", "w") as f:
            json.dump(meta, f)
        return meta

    def __init__(self, index_dir: str, threads: int = 4):
        with open(f"{index_dir}/meta.json") as f:
            self.meta = json.load(f)
        self.mode = self.meta["mode"]
        self.model_dir = self.meta["model_dir"]
        self.enc = Encoder(model_dir=self.model_dir, quantized=True,
                           threads=threads)
        if self.mode == "exact":
            self.vectors = np.load(f"{index_dir}/vectors.f32.npy",
                                   mmap_mode="r")
        else:
            self.pq = PQ.load(f"{index_dir}/pq_centroids.npy")
            self.codes = np.load(f"{index_dir}/codes.u8.npy", mmap_mode="r")

    def encode_query(self, query: str) -> np.ndarray:
        return self.enc.encode([query], batch=1, is_query=True)[0]

    def score(self, qv: np.ndarray, docids: np.ndarray) -> np.ndarray:
        """Similarity of a query vector against specific documents."""
        if self.mode == "exact":
            return np.asarray(self.vectors[docids]) @ qv
        return PQ.adc(self.pq.lut(qv), np.asarray(self.codes[docids]))
