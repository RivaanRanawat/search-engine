"""Turn a crawl into a served web index: BM25 + PageRank prior + metadata.

The crawler emits pages.tsv in exactly the format the MS MARCO pipeline
eats (docid == line number), so indexing is the SAME code path — no
web-specific indexer. What the web adds that a passage collection cannot
provide is the **link graph**, and therefore a query-independent quality
prior (PageRank).

Ranking: score = BM25 + w * log1p(PageRank * N)
PageRank spans orders of magnitude, so it enters in log space; w controls
how much a well-linked page can outrank a better text match. Without
relevance judgments for this corpus we cannot tune w — it is exposed and
documented rather than silently baked in (notes/16).

Usage: python -m searchengine.build_web_index <crawl_dir> <index_dir>
"""
import json
import os
import shutil
import sys
import time

import numpy as np


ENGLISH_STOPWORDS = frozenset(
    "the of and to in a is that for it as was with be by on not he this are "
    "or from at his an they which one you were her all she there been their "
    "has more when who will no if out so what up its about into them can "
    "only other new some could time these two may then do first any my".split())


def looks_english(text: str) -> bool:
    """Cheap language identification.

    Necessary because our tokenizer keeps only [a-z0-9]: a Mongolian or
    Ukrainian page yields almost no tokens, so its doc length collapses and
    BM25's length normalisation then REWARDS it — non-English pages were
    outranking real English matches for English queries (notes/16). Two
    signals, both robust on short text: the share of letters that are ASCII,
    and the share of tokens that are common English function words.
    """
    letters = [c for c in text[:4000] if c.isalpha()]
    if len(letters) < 50:
        return False
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
    if ascii_ratio < 0.85:
        return False
    words = text.lower().split()[:400]
    if len(words) < 20:
        return False
    stop_ratio = sum(w.strip(".,;:!?()[]\"'") in ENGLISH_STOPWORDS
                     for w in words) / len(words)
    return stop_ratio >= 0.10


def _filter_pages(crawl_dir: str, work_dir: str) -> tuple[str, str, dict]:
    """Drop non-English pages and renumber ids contiguously (the indexer
    requires docid == line number). Returns paths + filter stats."""
    os.makedirs(work_dir, exist_ok=True)
    keep_old_ids: dict[int, int] = {}
    kept = dropped = 0
    with open(f"{crawl_dir}/pages.tsv", encoding="utf-8", errors="replace") as f, \
            open(f"{work_dir}/pages.tsv", "w", encoding="utf-8") as o:
        for line in f:
            if not line.endswith("\n"):
                break                       # partial final line (live crawl)
            pid, _, text = line.partition("\t")
            if not pid.isdigit():
                continue
            if looks_english(text):
                o.write(f"{kept}\t{text}")
                keep_old_ids[int(pid)] = kept
                kept += 1
            else:
                dropped += 1
    with open(f"{crawl_dir}/meta.jsonl", encoding="utf-8", errors="replace") as f, \
            open(f"{work_dir}/meta.jsonl", "w", encoding="utf-8") as o:
        for line in f:
            if not line.endswith("\n"):
                break
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m["id"] in keep_old_ids:
                m["id"] = keep_old_ids[m["id"]]
                o.write(json.dumps(m) + "\n")
    shutil.copy(f"{crawl_dir}/links.tsv", f"{work_dir}/links.tsv")
    return work_dir, work_dir, {"kept": kept, "dropped_non_english": dropped}


def build(crawl_dir: str, index_dir: str) -> dict:
    from .indexer_v4 import build as build_index
    from .compress_index import build as compress
    from .pagerank import load_graph, pagerank

    work = f"{index_dir}_filtered"
    _, _, fstats = _filter_pages(crawl_dir, work)
    crawl_dir = work

    t0 = time.perf_counter()
    raw = f"{index_dir}_raw"
    stats = build_index(f"{crawl_dir}/pages.tsv", raw, workers=6,
                        k1=0.82, b=0.75)
    compress(raw, index_dir)
    index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges, n, url_to_id = load_graph(crawl_dir)
    pr = pagerank(edges, n)
    # pad/truncate to the indexed doc count (a page can be crawled but have
    # been dropped by the indexer only if empty — keep the arrays aligned)
    n_docs = stats["n_docs"]
    if len(pr) < n_docs:
        pr = np.concatenate([pr, np.full(n_docs - len(pr), 1.0 / max(1, n))])
    np.save(f"{index_dir}/pagerank.f64.npy", pr[:n_docs])
    pr_s = time.perf_counter() - t0

    # url/title sidecar for result display
    urls, titles = [""] * n_docs, [""] * n_docs
    with open(f"{crawl_dir}/meta.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m["id"] < n_docs:
                urls[m["id"]] = m["url"]
                titles[m["id"]] = m.get("title", "")
    with open(f"{index_dir}/urls.json", "w") as f:
        json.dump({"urls": urls, "titles": titles}, f)
    shutil.copy(f"{crawl_dir}/pages.tsv", f"{index_dir}/pages.tsv")
    shutil.rmtree(raw, ignore_errors=True)

    shutil.rmtree(work, ignore_errors=True)
    out = {"n_docs": n_docs, "n_terms": stats["n_terms"],
           "postings": stats["total_postings"], "index_s": round(index_s, 1),
           "pagerank_s": round(pr_s, 2), "edges": int(edges.shape[1]),
           **fstats,
           "index_mb": round(sum(
               os.path.getsize(f"{index_dir}/{f}")
               for f in os.listdir(index_dir)) / 1e6, 1)}
    with open(f"{index_dir}/web_meta.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


class WebSearcher:
    """BM25 + a BOUNDED PageRank prior over a crawled corpus.

    First attempt added `w * log1p(pr*N)` directly to the BM25 score. That is
    unbounded relative to the score spread: with w=2 the prior contributed up
    to ~6 points against BM25 scores of ~10, and the query "how does bm25
    work" returned Creative Commons licence pages — the most-linked documents
    in the crawl, and completely irrelevant.

    Fix: blend min-max normalised signals, exactly as the dense hybrid does.
    The prior can then shift ranking within the candidate set but can never
    dominate it, which is how production engines bound static priors.
    beta is exposed, not silently baked in: tuning it honestly needs
    relevance judgments for this corpus, which we do not have (notes/16).
    """

    def __init__(self, index_dir: str, beta: float = 0.0,
                 dense_dir: str | None = None, alpha: float = 0.3):
        """alpha weights BM25 against dense similarity; beta weights the
        PageRank prior.

        Both are now MEASURED rather than guessed (notes/25, 19 pooled and
        judged queries, nDCG@10):

            hybrid a=0.3, beta=0     0.7921   <- default
            hybrid a=0.3, beta=0.05  0.7897
            hybrid a=0.3, beta=0.15  0.7707   <- the old, eyeballed default
            hybrid a=0.3, beta=0.3   0.6238
            BM25 alone               0.6464

        alpha=0.3 independently reproduces the out-of-domain optimum found on
        BEIR (notes/19). **beta defaults to 0**: PageRank is monotonically
        harmful on this corpus. A 7,177-page crawl has only ~4 intra-corpus
        links per page, so the graph is too thin to encode topical authority
        — its top-ranked pages are generic hubs (Wikimedia, Creative Commons,
        university home pages) that are rarely what a query wants. PageRank
        earns its keep on a web-scale graph, not a laptop-scale one; the
        machinery is kept and can be re-enabled by passing beta.
        """
        from .search_hybrid import open_lexical
        self.s = open_lexical(index_dir)
        self.pr = np.load(f"{index_dir}/pagerank.f64.npy")
        with open(f"{index_dir}/urls.json") as f:
            meta = json.load(f)
        self.urls, self.titles = meta["urls"], meta["titles"]
        self.beta = beta
        self.alpha = alpha
        self.n = len(self.pr)
        self.dense = None
        if dense_dir and os.path.exists(f"{dense_dir}/meta.json"):
            from .dense import DenseIndex
            d = DenseIndex(dense_dir)
            if d.meta["n_docs"] != self.n:
                # a stale dense index would score the wrong documents
                raise ValueError(
                    f"dense index has {d.meta['n_docs']} docs but lexical "
                    f"index has {self.n}; rebuild the dense index")
            self.dense = d

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    def search(self, query: str, k: int = 10, depth: int = 100):
        hits = self.s.search(query, max(k, depth))
        if not hits:
            return []
        pids = np.array([p for p, _ in hits], np.int64)
        bs = np.array([sc for _, sc in hits], np.float32)
        prior = np.log1p(self.pr[pids] * self.n).astype(np.float32)

        # relevance = lexical, optionally blended with semantic similarity
        if self.dense is not None:
            ds = self.dense.score(self.dense.encode_query(query), pids)
            relevance = (self.alpha * self._norm(bs)
                         + (1.0 - self.alpha) * self._norm(ds))
        else:
            ds = None
            relevance = self._norm(bs)

        final = ((1.0 - self.beta) * relevance
                 + self.beta * self._norm(prior))
        order = np.argsort(-final)[:k]
        return [{"id": int(pids[i]), "score": float(final[i]),
                 "bm25": float(bs[i]),
                 "dense": (float(ds[i]) if ds is not None else None),
                 "pagerank": float(self.pr[pids[i]]),
                 "url": self.urls[pids[i]], "title": self.titles[pids[i]]}
                for i in order]


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1], sys.argv[2]), indent=2))
