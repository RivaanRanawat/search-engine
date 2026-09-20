"""Interactive CLI for v1: python -m searchengine.cli_v1 <index_dir> <collection.tsv>

Snippets are fetched by seeking directly into collection.tsv via a byte-offset
array built on first use (pids are line numbers).
"""
import sys
import time

import numpy as np

from .search_hybrid import open_lexical


def build_line_offsets(path: str) -> np.ndarray:
    offs = [0]
    with open(path, "rb") as f:
        for line in f:
            offs.append(offs[-1] + len(line))
    return np.array(offs[:-1], dtype=np.int64)


def main() -> None:
    index_dir, collection = sys.argv[1], sys.argv[2]
    print("loading index ...", flush=True)
    t0 = time.perf_counter()
    s = open_lexical(index_dir)
    print(f"loaded {s.n_docs} docs in {time.perf_counter()-t0:.1f}s; "
          "building snippet offsets ...", flush=True)
    offs = build_line_offsets(collection)
    fh = open(collection, "rb")
    while True:
        try:
            q = input("query> ").strip()
        except EOFError:
            break
        if not q:
            continue
        t0 = time.perf_counter()
        hits = s.search(q, k=10)
        ms = (time.perf_counter() - t0) * 1e3
        for rank, (pid, score) in enumerate(hits, 1):
            fh.seek(offs[pid])
            text = fh.readline().decode("utf-8").split("\t", 1)[1]
            print(f"{rank:2d}. [{score:6.2f}] pid={pid} {text[:110].strip()}")
        print(f"({ms:.1f} ms)")


if __name__ == "__main__":
    main()
