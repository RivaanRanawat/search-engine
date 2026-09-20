"""Two questions Part 4 answers about the (docid<<6)|tf build buffers.

1. What does the packing actually buy? Compare appending one packed integer
   per posting against keeping separate docid/tf buffers.
2. What would it cost to give the docid more bits by shrinking the tf field?
   Answered from the real index's term-frequency distribution.

Usage: python3 -m bench.build.bench_packing [index_dir] [out_json]
"""
import json
import random
import sys
import time
from array import array

import numpy as np

VOCAB, N, REPS = 200_000, 2_000_000, 3
K1 = 0.9


def _bench(fn) -> float:
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best / N * 1e9          # ns per posting


def packing_cost() -> dict:
    rng = random.Random(0)
    terms = [f"t{rng.randint(0, VOCAB - 1)}" for _ in range(N)]

    def packed():
        bufs: dict = {}
        for i, t in enumerate(terms):
            b = bufs.get(t)
            if b is None:
                b = bufs[t] = array("I")
            b.append((i << 6) | 3)

    def split():
        bufs: dict = {}
        for i, t in enumerate(terms):
            pair = bufs.get(t)
            if pair is None:
                pair = bufs[t] = (array("I"), array("B"))
            pair[0].append(i)
            pair[1].append(3)

    p, s = _bench(packed), _bench(split)
    return {"packed_ns_per_posting": round(p, 1),
            "separate_arrays_ns_per_posting": round(s, 1),
            "packing_speedup": round(s / p, 2)}


def tf_headroom(index_dir: str) -> dict:
    """How many postings would a smaller tf field actually clamp?"""
    tfs = np.load(f"{index_dir}/tfs.u8.npy", mmap_mode="r")
    sample = np.asarray(tfs[::7])
    sat = lambda tf: tf * (K1 + 1) / (tf + K1)
    return {"total_postings": int(len(tfs)),
            "pct_above": {str(t): round(float((sample > t).mean()) * 100, 4)
                          for t in (3, 7, 15, 31, 63)},
            "saturated_value": {str(t): round(sat(t), 4) for t in (15, 31, 63)},
            "splits": {f"{b}+{32-b}": {"max_docs": 2 ** b - 1,
                                       "tf_clamp": 2 ** (32 - b) - 1}
                       for b in (26, 28, 30)}}


def main() -> None:
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "indexes/v1s"
    out = sys.argv[2] if len(sys.argv) > 2 else "bench/results/build_packing.json"
    res = {"packing": packing_cost(), "tf_headroom": tf_headroom(index_dir)}
    print(json.dumps(res, indent=1))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
