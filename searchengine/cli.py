"""Interactive CLI: python -m searchengine.cli <index.pkl> <collection.tsv>

Loads the pickled v0 index and serves queries; prints top-10 with snippets.
"""
import sys
import time

from .indexer import load
from .search import search


def load_texts(collection_path: str, wanted: set[int]) -> dict[int, str]:
    texts = {}
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            if int(pid) in wanted:
                texts[int(pid)] = text
                if len(texts) == len(wanted):
                    break
    return texts


def main() -> None:
    index_path, collection_path = sys.argv[1], sys.argv[2]
    print(f"loading {index_path} ...", flush=True)
    idx, load_s = load(index_path)
    print(f"loaded {idx.n_docs} docs in {load_s:.1f}s")
    while True:
        try:
            q = input("query> ").strip()
        except EOFError:
            break
        if not q:
            continue
        t0 = time.perf_counter()
        hits = search(idx, q, k=10)
        ms = (time.perf_counter() - t0) * 1e3
        texts = load_texts(collection_path, {pid for pid, _ in hits})
        for rank, (pid, score) in enumerate(hits, 1):
            snippet = texts.get(pid, "")[:120]
            print(f"{rank:2d}. [{score:6.2f}] pid={pid} {snippet}")
        print(f"({ms:.1f} ms)")


if __name__ == "__main__":
    main()
