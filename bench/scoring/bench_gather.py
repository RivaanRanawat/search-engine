"""Where does BM25-at-query-time actually spend its time?

Part 4 justifies impact precomputation. This measures what precomputation
actually removes, on this machine, across the df spectrum.

Without precomputation, per posting: read the doc length via the posting's
docid (a SCATTERED read into a 35 MB array), then evaluate BM25.
With precomputation: read one float from a CONTIGUOUS per-term slice.
Both paths then accumulate identically, so accumulation is excluded.
"""
import json
import sys
import time

import numpy as np

N = 8_841_823
K1, B, AVGDL, IDF = 0.9, 0.4, 57.85, 3.0
REPS = 9


def best_ms(fn) -> float:
    """Fastest of REPS runs: least contaminated by scheduling noise."""
    out = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        out = min(out, (time.perf_counter() - t0) * 1000)
    return out


def measure(doc_lens: np.ndarray, nposts: int, rng) -> dict:
    docids = np.sort(rng.choice(N, size=nposts, replace=False)).astype(np.uint32)
    tf = rng.integers(1, 10, size=nposts).astype(np.float32)
    dl = doc_lens[docids].astype(np.float32)
    # what the index would store: one impact per posting, contiguous per term
    impacts = (IDF * tf * (K1 + 1.0) /
               (tf + K1 * (1.0 - B + B * dl / AVGDL))).astype(np.float32)

    gather = best_ms(lambda: doc_lens[docids])
    arith = best_ms(lambda: IDF * tf * (K1 + 1.0) /
                            (tf + K1 * (1.0 - B + B * dl / AVGDL)))
    seq = best_ms(lambda: impacts.copy())          # contiguous read, precomputed
    return {"postings": nposts,
            "avg_docid_gap": round(N / nposts, 1),
            "gather_ms": round(gather, 3),
            "arithmetic_ms": round(arith, 3),
            "precomputed_read_ms": round(seq, 3),
            "speedup": round((gather + arith) / seq, 1)}


def main() -> None:
    rng = np.random.default_rng(0)
    doc_lens = rng.integers(10, 200, size=N).astype(np.uint32)
    results = [measure(doc_lens, n, rng)
               for n in (4_000_000, 500_000, 50_000, 2_000)]
    for r in results:
        print(f"  {r['postings']:>9,} postings (gap {r['avg_docid_gap']:>6}): "
              f"gather {r['gather_ms']:>6} | arith {r['arithmetic_ms']:>6} | "
              f"precomputed read {r['precomputed_read_ms']:>6} ms "
              f"→ {r['speedup']}x")
    out = sys.argv[1] if len(sys.argv) > 1 else "bench/results/scoring_gather.json"
    with open(out, "w") as f:
        json.dump({"n_docs": N, "reps": REPS, "results": results}, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
