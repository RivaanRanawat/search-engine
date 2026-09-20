"""Evaluate web-corpus ranking configurations against the pooled judgments.

This is the measurement that β (PageRank weight) and α (dense weight) were
missing. Metric is nDCG@10 with graded relevance (0/1/2), matching BEIR.

Queries with NO relevant document in the pool are reported separately rather
than scored: nDCG is undefined when the ideal DCG is zero, and silently
counting them as 0.0 would drag every configuration down equally while
hiding a corpus-coverage problem behind a ranking metric.

Usage: python -m bench.webeval.eval_web
"""
import json

import numpy as np

from searchengine.build_web_index import WebSearcher

QRELS = "bench/webeval/qrels.json"


def ndcg_at_k(ranked_ids, rels: dict, k: int = 10) -> float:
    gains = [rels.get(str(d), 0) for d in ranked_ids[:k]]
    dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else float("nan")


def main() -> None:
    with open(QRELS) as f:
        qrels = {k: v for k, v in json.load(f).items()
                 if not k.startswith("_")}
    judged = {q: r for q, r in qrels.items() if max(r.values()) > 0}
    empty = [q for q in qrels if q not in judged]

    configs = []
    for beta in (0.0, 0.05, 0.15, 0.3):
        configs.append((f"bm25+pr(beta={beta})", dict(beta=beta, dense_dir=None,
                                                      alpha=1.0)))
    for alpha in (0.1, 0.3, 0.5):
        configs.append((f"hybrid(alpha={alpha})",
                        dict(beta=0.0, dense_dir="indexes/web_dense",
                             alpha=alpha)))
    for beta in (0.05, 0.15, 0.3):
        configs.append((f"hybrid(a=0.3)+pr(beta={beta})",
                        dict(beta=beta, dense_dir="indexes/web_dense",
                             alpha=0.3)))

    rows = []
    for name, cfg in configs:
        s = WebSearcher("indexes/web", beta=cfg["beta"],
                        dense_dir=cfg["dense_dir"], alpha=cfg["alpha"])
        scores = [ndcg_at_k([h["id"] for h in s.search(q, 10)], rels)
                  for q, rels in judged.items()]
        rows.append((name, float(np.nanmean(scores))))
        del s

    rows.sort(key=lambda r: -r[1])
    out = {"n_queries_judged": len(judged),
           "queries_with_no_relevant_doc": empty,
           "metric": "nDCG@10 (graded 0/1/2, pooled LLM judgments)",
           "results": [{"config": n, "ndcg@10": round(v, 4)} for n, v in rows]}
    print(json.dumps(out, indent=2))
    with open("bench/results/web_ndcg.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
