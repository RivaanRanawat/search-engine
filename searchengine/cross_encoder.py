"""Cross-encoder reranker: scores (query, passage) jointly.

Unlike the bi-encoder (which compares two independently computed vectors),
a cross-encoder runs full attention over the concatenated pair, so it can
model term interactions directly. That is why it is the strongest reranker
in the literature — and why it costs a forward pass PER CANDIDATE, making
it a query-time-only, top-N-only tool. Nothing is precomputed or stored.

Same machine-specific engineering as the bi-encoder (notes/13): int8
dynamic quantization, length-bucketed batching, CPU EP only (the CoreML EP
OOM-killed this machine).
"""
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = "models/ce"


class CrossEncoder:
    def __init__(self, model_dir: str = MODEL_DIR, threads: int = 6,
                 max_len: int = 256, quantized: bool = True):
        path = f"{model_dir}/model.onnx"
        if quantized:
            q = f"{model_dir}/model.int8.onnx"
            if not os.path.exists(q):
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantize_dynamic(path, q, weight_type=QuantType.QInt8)
            path = q
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tok.enable_truncation(max_length=max_len)
        self.tok.no_padding()

    def _encode_pairs(self, query: str, passages: list[str]):
        encs = self.tok.encode_batch([(query, p) for p in passages])
        return ([e.ids for e in encs], [e.type_ids for e in encs])

    def score(self, query: str, passages: list[str],
              batch: int = 16) -> np.ndarray:
        """Relevance logits, one per passage, in input order."""
        if not passages:
            return np.zeros(0, np.float32)
        ids_all, tt_all = self._encode_pairs(query, passages)
        order = np.argsort([len(x) for x in ids_all], kind="stable")
        out = np.empty(len(passages), np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            chunk = [ids_all[i] for i in idx]
            tts = [tt_all[i] for i in idx]
            t = max(len(x) for x in chunk)
            ids = np.zeros((len(chunk), t), np.int64)
            am = np.zeros((len(chunk), t), np.int64)
            tt = np.zeros((len(chunk), t), np.int64)
            for j, (x, y) in enumerate(zip(chunk, tts)):
                ids[j, :len(x)] = x
                am[j, :len(x)] = 1
                tt[j, :len(y)] = y
            feed = {"input_ids": ids, "attention_mask": am}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = tt
            logits = self.sess.run(None, feed)[0]
            out[idx] = logits.reshape(len(chunk), -1)[:, 0].astype(np.float32)
        return out
