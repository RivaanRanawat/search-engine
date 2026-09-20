"""Hybrid searcher: BM25 retrieval + dense rerank over PQ-compressed vectors.

Pipeline per query:
  1. BM25 top-K via the C MaxScore kernel (K=1000 — see TOPK_RERANK:
     K is a recall ceiling, not a knob)
  2. encode the query once with the bi-encoder (full precision — asymmetric
     distance: we never quantize the query)
  3. score candidates by ADC table lookups against their PQ codes
  4. blend: alpha * norm(bm25) + (1-alpha) * norm(dense), alpha tuned on a
     TRAIN sample (see experiments/), never on sealed dev

Measured on the full sealed dev set: BM25 0.18910 -> hybrid 0.32 MRR@10
(notes/22, notes/23). The single biggest lever after the model itself is
retrieval DEPTH: at K=50 BM25's recall is only 0.61, so 39% of queries have
no relevant passage to rerank at all.
"""
import json

import numpy as np

import os

from .encoder import Encoder
from .pq import PQ


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def open_lexical(index_dir: str):
    """Open whichever lexical index format lives in index_dir: the raw
    arrays (SearcherV2, fastest — the default) or the block-compressed
    capacity mode (SearcherV3, 3.65x smaller, ~1.7x slower at p50).
    Same quality to within quantization noise (MRR delta 4e-5, notes/10);
    top-10 SETS can differ on near-ties because v3 ranks by u8-quantized
    impacts while v2 ranks by exact floats."""
    if os.path.exists(f"{index_dir}/blob.bin"):
        from .search_v3 import SearcherV3
        return SearcherV3(index_dir)
    from .search_v2 import SearcherV2
    return SearcherV2(index_dir)


#: Fusion weight on BM25. NOT a universal constant — it is a per-corpus
#: parameter, and using the wrong one makes results WORSE than plain BM25.
#: Measured (notes/18):
#:   MS MARCO (in-domain, tuned on train queries): 0.1  -> MRR 0.315
#:                                                 0.5  -> MRR 0.253
#:   BEIR SciFact (out-of-domain, nDCG@10):        0.1  -> 0.611  (BM25 0.688!)
#:                                                 0.5  -> 0.708
#: So 0.1 is right in-domain and actively harmful out-of-domain. For a corpus
#: with no relevance judgments to tune on, 0.5 is the safe starting point.
ALPHA_IN_DOMAIN = 0.1
ALPHA_UNKNOWN_DOMAIN = 0.5

#: How deep to retrieve before reranking. This is a RECALL CEILING, not a
#: tuning knob: a relevant passage BM25 misses at depth K can never be
#: recovered by reranking. Measured on MS MARCO (notes/23):
#:     K:        50      200     500     1000
#:     recall@K  0.61    0.75    0.83    0.87
#:     MRR@10    0.296   0.313   0.318   0.320   (dev)
#:     p99       29ms    36ms    42ms    47ms
#: Reranking deeper is nearly free (ADC table lookups over stored codes);
#: the cost is the deeper BM25 traversal. 1000 sits at ~2x inside the
#: p99<100ms goal, so recall is bought with latency we already have.
#: Validated on TRAIN before adoption (same monotone trend), dev reported once.
TOPK_RERANK = 1000


class HybridSearcher:
    def __init__(self, index_dir: str, dense_dir: str,
                 alpha: float = ALPHA_IN_DOMAIN,
                 topk_bm25: int = TOPK_RERANK, threads: int = 4):
        self.bm25 = open_lexical(index_dir)
        with open(f"{dense_dir}/meta.json") as f:
            self.dmeta = json.load(f)
        self.pq = PQ.load(f"{dense_dir}/pq_centroids.npy")
        self.codes = np.load(f"{dense_dir}/codes.u8.npy", mmap_mode="r")
        # The encoder is built LAZILY, on first use inside whichever process
        # ends up serving. ONNX Runtime spawns worker threads, and forking a
        # multi-threaded process can deadlock the child (Python warns about
        # exactly this). Building it after the fork gives every worker its
        # own clean session; the mmap'd codes are still shared via page
        # cache, which is where the memory actually is.
        self._enc = None
        self._enc_threads = threads
        self._enc_quantized = self.dmeta.get("encoder") == "int8"
        # an index may record the alpha tuned for ITS corpus; that always
        # wins over the caller's default
        self.alpha = float(self.dmeta.get("alpha", alpha))
        self.topk_bm25 = topk_bm25
        self.ce = None                 # lazily built: 90MB model, opt-in tier
        self.ce_threads = threads

    @property
    def enc(self) -> Encoder:
        if self._enc is None:
            self._enc = Encoder(quantized=self._enc_quantized,
                                threads=self._enc_threads)
        return self._enc

    def search(self, query: str, k: int = 10,
               explain: bool = False) -> list[tuple[int, float]]:
        hits = self.bm25.search(query, self.topk_bm25)
        if not hits:
            return []
        pids = np.array([p for p, _ in hits], np.int64)
        bs = np.array([s for _, s in hits], np.float32)
        qv = self.enc.encode([query], batch=1, is_query=True)[0]
        ds = PQ.adc(self.pq.lut(qv), np.asarray(self.codes[pids]))
        blend = self.alpha * _minmax(bs) + (1.0 - self.alpha) * _minmax(ds)
        order = np.argsort(-blend)[:k]
        if explain:
            return [(int(pids[i]), float(blend[i]), float(bs[i]),
                     float(ds[i])) for i in order]
        return [(int(pids[i]), float(blend[i])) for i in order]

    def search_ce(self, query: str, store, k: int = 10, depth: int = 20
                  ) -> list[tuple[int, float]]:
        """Third cascade stage: cross-encoder rerank of the hybrid top-`depth`.

        This is where most of the remaining quality lives. Sealed dev:
        hybrid 0.311 -> hybrid+CE 0.388 MRR@10 (+25%), at p50 141ms — over
        the p99<100ms goal, so it is an OPT-IN tier, never the default.

        depth=20 is re-derived, not inherited. Over BM25 top-50 candidates
        depth 10 beat 20 (notes/15); with today's K=1000 candidates the
        optimum moved (train: 10 -> 0.4046, 20 -> 0.4151, 50 -> 0.4077).
        The optimum shortlist depth depends on CANDIDATE QUALITY, and it is
        still non-monotone because the int8 cross-encoder gets noisy.
        """
        if self.ce is None:
            from .cross_encoder import CrossEncoder
            self.ce = CrossEncoder(threads=self.ce_threads)
        base = self.search(query, depth)
        if not base:
            return []
        pids = [p for p, _ in base]
        texts = [t.decode("utf-8", "replace")
                 for t in store.text_bytes_many(pids)]
        sc = self.ce.score(query, texts)
        order = np.argsort(-sc)[:k]
        return [(pids[i], float(sc[i])) for i in order]
