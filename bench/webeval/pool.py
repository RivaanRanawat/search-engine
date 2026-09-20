"""Build a TREC-style judgment pool for the crawled web corpus.

Web ranking has been unmeasurable because no relevance judgments exist for a
self-crawled corpus (notes/16, notes/21) — so β (PageRank weight) and α
(dense weight) were exposed as parameters rather than tuned. This creates
the missing ingredient.

Method, following TREC:
1. Take a fixed query set covering the crawl's actual topics.
2. **Pool** the top-k of SEVERAL different ranking configurations. Pooling
   matters: judging only one system's output biases evaluation toward that
   system, because documents it never returns are never judged and so score
   zero for everyone by construction.
3. Judge the pool once. Every configuration is then scored on the same
   judged set.
4. The pool is emitted in SHUFFLED order with no system labels, so a
   document cannot be judged more kindly for having come from a favoured
   ranker.

Usage: python -m bench.webeval.pool <out.jsonl>
"""
import json
import random
import sys

from searchengine.build_web_index import WebSearcher

# Queries chosen to match what the crawl actually covers (Wikipedia topics,
# IR/search blogs, programming and database writing) and to span the query
# types a web engine sees: navigational, informational, and natural-language.
QUERIES = [
    "information retrieval",
    "how do search engines rank pages",
    "bm25 ranking function",
    "inverted index data structure",
    "why is my database slow",
    "postgres performance tuning",
    "sql indexing",
    "quantum computing explained",
    "manhattan project history",
    "machine learning basics",
    "how to write clearly",
    "creative commons license terms",
    "wikipedia main page",
    "arxiv information retrieval papers",
    "python programming language",
    "distributed systems consistency",
    "text tokenization and stemming",
    "vector embeddings for search",
    "web crawler politeness robots txt",
    "open source search software",
]

POOL_DEPTH = 5


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "bench/webeval/pool.jsonl"
    configs = {
        "bm25":        dict(beta=0.0, dense_dir=None),
        "bm25_pr":     dict(beta=0.15, dense_dir=None),
        "hybrid":      dict(beta=0.0, dense_dir="indexes/web_dense"),
        "hybrid_pr":   dict(beta=0.15, dense_dir="indexes/web_dense"),
    }
    searchers = {name: WebSearcher("indexes/web", beta=c["beta"],
                                   dense_dir=c["dense_dir"], alpha=0.3)
                 for name, c in configs.items()}

    pool: dict[str, dict[int, dict]] = {}
    for q in QUERIES:
        seen: dict[int, dict] = {}
        for name, s in searchers.items():
            for h in s.search(q, POOL_DEPTH):
                seen.setdefault(h["id"], {"id": h["id"], "url": h["url"],
                                          "title": h["title"]})
        pool[q] = seen

    rng = random.Random(20260809)
    rows = []
    for q, docs in pool.items():
        ids = list(docs)
        rng.shuffle(ids)                       # strip any system ordering
        for did in ids:
            rows.append({"query": q, **docs[did]})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"queries": len(QUERIES), "pool_size": len(rows),
                      "mean_per_query": round(len(rows) / len(QUERIES), 1),
                      "out": out_path}, indent=2))


if __name__ == "__main__":
    main()
