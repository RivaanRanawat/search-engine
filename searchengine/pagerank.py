"""PageRank over the crawled link graph — the query-independent prior that
MS MARCO structurally cannot provide (a passage collection has no links).

Power iteration with proper dangling-node handling:
    PR = (1-d)/N + d * (A^T (PR/outdeg) + dangling_mass/N)
Sparse matvec is done with np.bincount over the edge arrays — the same
"vectorize the scatter, never loop per element" technique the indexer uses
(notes/03). 1M edges per iteration costs milliseconds.

Usage: python -m searchengine.pagerank <crawl_dir> [damping] [iters]
"""
import json
import sys
import time

import numpy as np


def load_graph(crawl_dir: str):
    """Map crawled URLs to ids, then edges to (src_id, dst_id). Links to
    pages we never fetched are dropped: they have no text to rank, and
    keeping them would let uncrawled hubs absorb rank we cannot serve."""
    url_to_id, n = {}, 0
    with open(f"{crawl_dir}/meta.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            url_to_id[m["url"]] = m["id"]
            n = max(n, m["id"] + 1)
    src, dst = [], []
    with open(f"{crawl_dir}/links.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            a, b = url_to_id.get(parts[0]), url_to_id.get(parts[1])
            if a is not None and b is not None and a != b:
                src.append(a)
                dst.append(b)
    e = np.array([src, dst], np.int64) if src else np.zeros((2, 0), np.int64)
    # collapse duplicate edges (many pages link the same target repeatedly)
    if e.shape[1]:
        key = e[0] * np.int64(n) + e[1]
        _, keep = np.unique(key, return_index=True)
        e = e[:, np.sort(keep)]
    return e, n, url_to_id


def pagerank(edges: np.ndarray, n: int, damping: float = 0.85,
             iters: int = 50, tol: float = 1e-8) -> np.ndarray:
    if n == 0:
        return np.zeros(0, np.float64)
    src, dst = edges[0], edges[1]
    outdeg = np.bincount(src, minlength=n).astype(np.float64)
    dangling = outdeg == 0
    pr = np.full(n, 1.0 / n, np.float64)
    for it in range(iters):
        contrib = np.zeros(n, np.float64)
        if len(src):
            share = pr[src] / outdeg[src]
            contrib = np.bincount(dst, weights=share, minlength=n)
        dangle = pr[dangling].sum() / n
        new = (1.0 - damping) / n + damping * (contrib + dangle)
        delta = np.abs(new - pr).sum()
        pr = new
        if delta < tol:
            break
    return pr / pr.sum()


def main() -> None:
    crawl_dir = sys.argv[1]
    damping = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    t0 = time.perf_counter()
    edges, n, url_to_id = load_graph(crawl_dir)
    t1 = time.perf_counter()
    pr = pagerank(edges, n, damping, iters)
    t2 = time.perf_counter()
    np.save(f"{crawl_dir}/pagerank.f64.npy", pr)

    id_to_url = {v: k for k, v in url_to_id.items()}
    top = np.argsort(-pr)[:15]
    print(json.dumps({"nodes": n, "edges": int(edges.shape[1]),
                      "load_s": round(t1 - t0, 2),
                      "pagerank_s": round(t2 - t1, 3),
                      "dangling": int((np.bincount(edges[0], minlength=n) == 0).sum())},
                     indent=2))
    print("\ntop pages by PageRank:")
    for i in top:
        print(f"  {pr[i]:.6f}  {id_to_url.get(int(i), '?')[:95]}")


if __name__ == "__main__":
    main()
