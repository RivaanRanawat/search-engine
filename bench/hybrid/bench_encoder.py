"""Measure each encoder optimization separately, with RSS tracking.

Configs: fp32/int8 x bucketed/naive x batch sizes. Same 4096 real passages
every time; correctness checked by cosine agreement against the fp32
unbucketed reference (quantization must not change semantics).

Usage: python -m bench.hybrid.bench_encoder
"""
import json
import resource
import time

import numpy as np

from searchengine.encoder import Encoder


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_texts(n: int) -> list[str]:
    out = []
    with open("data/collection.tsv", encoding="utf-8") as f:
        for line in f:
            out.append(line.split("\t", 1)[1].strip())
            if len(out) >= n:
                break
    return out


def main() -> None:
    N = 4096
    texts = load_texts(N)
    probe = Encoder(quantized=False, threads=1)
    toks = [len(e.ids) for e in probe.tok.encode_batch(texts)]
    print(f"token lengths: mean {np.mean(toks):.1f} p50 {np.median(toks):.0f} "
          f"p95 {np.percentile(toks, 95):.0f} max {max(toks)}", flush=True)
    del probe

    results = []
    ref = None
    for quant in (False, True):
        enc = Encoder(quantized=quant, threads=6)
        for bucket in (False, True):
            for batch in (32, 64, 128):
                t0 = time.perf_counter()
                emb = enc.encode(texts, batch=batch, bucket=bucket)
                dt = time.perf_counter() - t0
                if ref is None:
                    ref = emb
                agree = float((emb * ref).sum(1).mean())
                r = {"quant": "int8" if quant else "fp32", "bucket": bucket,
                     "batch": batch, "rate_per_s": round(N / dt, 1),
                     "full_corpus_h": round(8841823 / (N / dt) / 3600, 2),
                     "cos_vs_fp32_ref": round(agree, 4),
                     "rss_gb": round(rss_gb(), 2)}
                print(json.dumps(r), flush=True)
                results.append(r)
        del enc
    with open("bench/results/encoder_opt.json", "w") as f:
        json.dump(results, f, indent=2)
    best = max(results, key=lambda r: r["rate_per_s"])
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main()
