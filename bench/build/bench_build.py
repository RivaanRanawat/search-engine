"""Honest end-to-end build benchmark — the number a skeptic should attack.

The previously quoted "8.84M docs in 33s" measured `indexer_v4` alone, with a
warm page cache, and excluded everything needed to actually SERVE:
compression to the served format, the doc-store offset sidecar, and the
surface-form vocabulary. It also never built positions at all.

This measures every stage separately, cold and warm, and reports the fair
denominators (tokens/s and MB/s) alongside docs/s, because docs/s is
meaningless across corpora with different document lengths — MS MARCO
passages average 58 tokens, and a "document" collection averages hundreds.

Usage: python -m bench.build.bench_build <collection.tsv> <workdir> [workers]
       (set EVICT=1 to flush the page cache first for a cold measurement)
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def evict_page_cache(scratch: str, gb: int = 20) -> float:
    """Flush the corpus out of the page cache by streaming a larger file
    through it. `sudo purge` would be cleaner but needs privileges."""
    t0 = time.perf_counter()
    path = f"{scratch}/evict.bin"
    chunk = os.urandom(1 << 20)
    with open(path, "wb") as f:
        for _ in range(gb * 1024):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    with open(path, "rb") as f:
        while f.read(1 << 24):
            pass
    os.remove(path)
    return time.perf_counter() - t0


def main() -> None:
    coll = sys.argv[1]
    work = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    cold = os.environ.get("EVICT") == "1"

    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    raw, comp = f"{work}/raw", f"{work}/idx"

    corpus_bytes = os.path.getsize(coll)
    res = {"workers": workers, "cache": "cold" if cold else "warm",
           "corpus_gb": round(corpus_bytes / 1e9, 3)}

    if cold:
        res["evict_s"] = round(evict_page_cache(work), 1)

    from searchengine.compress_index import build as compress
    from searchengine.indexer_v4 import build as build_index

    t0 = time.perf_counter()
    stats = build_index(coll, raw, workers=workers)
    res["stage_index_s"] = round(time.perf_counter() - t0, 2)
    res.update({k: stats[k] for k in
                ("n_docs", "n_terms", "total_postings")})
    res["phases"] = {k: round(v, 2) for k, v in stats.items()
                     if k.endswith("_s")}

    t0 = time.perf_counter()
    compress(raw, comp)
    res["stage_compress_s"] = round(time.perf_counter() - t0, 2)

    # doc-store offsets: required to fetch any document text (snippets)
    t0 = time.perf_counter()
    offs = [0]
    with open(coll, "rb") as f:
        for line in f:
            offs.append(offs[-1] + len(line))
    np.save(f"{comp}/lineoffsets.i64.npy", np.array(offs[:-1], np.int64))
    res["stage_docstore_s"] = round(time.perf_counter() - t0, 2)

    total = (res["stage_index_s"] + res["stage_compress_s"]
             + res["stage_docstore_s"])
    res["TOTAL_servable_s"] = round(total, 2)

    tokens = int(np.load(f"{comp}/doclens.u32.npy").sum())
    res["tokens"] = tokens
    res["rates"] = {
        "docs_per_s_indexer_only": round(res["n_docs"] / res["stage_index_s"]),
        "docs_per_s_servable": round(res["n_docs"] / total),
        "tokens_per_s_servable": round(tokens / total),
        "MB_per_s_servable": round(corpus_bytes / 1e6 / total, 1),
    }
    res["caveats"] = {
        "positions_built": False,
        "stored_fields": "none (doc text read from the original TSV via mmap)",
        "note": "docs/s is corpus-dependent; tokens/s and MB/s are the "
                "comparable units.",
    }
    print(json.dumps(res, indent=2))
    tag = "cold" if cold else "warm"
    with open(f"bench/results/build_{tag}_{workers}w.json", "w") as f:
        json.dump(res, f, indent=2)
    shutil.rmtree(raw, ignore_errors=True)


if __name__ == "__main__":
    main()
