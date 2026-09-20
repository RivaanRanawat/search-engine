"""Bi-encoder wrapper with the optimizations this machine requires.

Measured constraints (notes/12): naive batching gives 57 passages/s → 42.8h
for the corpus. Two levers, both implemented here and benchmarked in
bench/hybrid/bench_encoder.py:

1. **Length-bucketed batching.** MS MARCO passages average ~58 tokens but
   naive batching pads every batch to the batch maximum (up to max_len).
   Sorting a large chunk by token length before batching removes most of
   that waste; results are restored to input order by the caller.
2. **int8 dynamic quantization** of the ONNX graph (weights int8, activations
   dynamic) — no calibration data needed, typically 2-3x on ARM NEON.

CoreML EP is deliberately NOT used: it partitions this graph into 50
sub-models, exhausted RAM and killed the machine once (notes/12).
"""
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = "models/minilm"

#: Different embedding families demand different pooling and prompt
#: conventions, and getting them wrong silently degrades quality rather than
#: erroring — BGE uses the CLS token and a query-side instruction; E5 uses
#: mean pooling with "query:"/"passage:" prefixes; MiniLM/GTE use mean
#: pooling with no prefix. Encoded here so a model swap is one config line.
MODEL_CONFIGS = {
    "models/minilm": {"pool": "mean", "q_prefix": "", "d_prefix": ""},
    "models/gte": {"pool": "mean", "q_prefix": "", "d_prefix": ""},
    "models/bge": {"pool": "cls", "d_prefix": "",
                   "q_prefix": "Represent this sentence for searching "
                               "relevant passages: "},
    "models/e5": {"pool": "mean", "q_prefix": "query: ",
                  "d_prefix": "passage: "},
}


def quantize_model(src: str, dst: str) -> str:
    """int8-dynamic-quantize the ONNX model (idempotent)."""
    if not os.path.exists(dst):
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    return dst


class Encoder:
    def __init__(self, model_dir: str = MODEL_DIR, threads: int = 6,
                 max_len: int = 256, quantized: bool = True):
        path = f"{model_dir}/model.onnx"
        if quantized:
            path = quantize_model(path, f"{model_dir}/model.int8.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU EP only — see notes/12 (CoreML EP OOM-killed this machine)
        self.sess = ort.InferenceSession(path, so,
                                         providers=["CPUExecutionProvider"])
        self.needs_tt = any(i.name == "token_type_ids"
                            for i in self.sess.get_inputs())
        self.tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tok.enable_truncation(max_length=max_len)
        self.tok.no_padding()
        self.dim = 384
        cfg = MODEL_CONFIGS.get(model_dir.rstrip("/"),
                                {"pool": "mean", "q_prefix": "",
                                 "d_prefix": ""})
        self.pool = cfg["pool"]
        self.q_prefix = cfg["q_prefix"]
        self.d_prefix = cfg["d_prefix"]

    def _forward(self, id_lists: list[list[int]]) -> np.ndarray:
        n = len(id_lists)
        t = max(len(x) for x in id_lists)
        ids = np.zeros((n, t), np.int64)
        am = np.zeros((n, t), np.int64)
        for i, x in enumerate(id_lists):
            ids[i, :len(x)] = x
            am[i, :len(x)] = 1
        feed = {"input_ids": ids, "attention_mask": am}
        if self.needs_tt:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]
        if self.pool == "cls":
            emb = out[:, 0, :]                      # BGE-family: CLS token
        else:
            m = am[:, :, None].astype(np.float32)
            emb = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.clip(norms, 1e-9, None)).astype(np.float32)

    def encode(self, texts: list[str], batch: int = 64, bucket: bool = True,
               is_query: bool = False) -> np.ndarray:
        """Returns L2-normalized float32 embeddings in INPUT order."""
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        prefix = self.q_prefix if is_query else self.d_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        id_lists = [e.ids for e in self.tok.encode_batch(texts)]
        order = (np.argsort([len(x) for x in id_lists], kind="stable")
                 if bucket else np.arange(len(id_lists)))
        out = np.empty((len(texts), self.dim), np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            out[idx] = self._forward([id_lists[i] for i in idx])
        return out
