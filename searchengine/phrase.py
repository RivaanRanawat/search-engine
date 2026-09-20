"""Exact phrase search: `"quoted phrases"` in the query, Google-style.

Design (measured trade, notes/14): the textbook solution is a positions
index, which would add ~500MB-1GB, a full corpus re-scan, and positional
decode in the query kernel. Instead:

  1. A document containing the phrase MUST contain every phrase term, so a
     conjunctive intersection over those terms gives a **provably complete**
     candidate set — no recall loss, unlike top-K post-filtering.
  2. Verify each candidate against the stored document text (already mmap'd
     for snippets) with a whitespace/punctuation-insensitive matcher.

Exact results, zero index growth. The cost is proportional to the number of
documents containing all the terms, which for a real phrase is small; the
pathological case ("the of and") is bounded by `max_candidates` and
reported honestly via `truncated`. Positions would beat this on those
pathological phrases only — deferred until a benchmark shows they matter.
"""
import re

from .porter import stem
from .tokenizer import tokenize

_PHRASE_RE = re.compile(r'"([^"]*)"')


def parse_query(q: str) -> tuple[list[list[str]], list[str]]:
    """Returns (phrases as token lists, loose terms)."""
    phrases = []
    for m in _PHRASE_RE.finditer(q):
        toks = tokenize(m.group(1))
        if toks:
            phrases.append(toks)
    loose = tokenize(_PHRASE_RE.sub(" ", q))
    return phrases, loose


def phrase_regex(phrase_tokens: list[str]) -> "re.Pattern[bytes]":
    """Compile a phrase into a bytes regex that is EXACTLY equivalent to
    token-level adjacency under our tokenizer ([a-z0-9]+ runs, lowercased).

    Tokens must be separated by one or more non-alphanumeric characters, and
    guarded at both ends so a phrase term cannot be a fragment of a longer
    token ('cost of living' must not match inside 'accost of living').

    Doing this with one C-level regex instead of tokenizing each candidate
    document in Python is what makes phrase verification affordable: it cut
    "cost of living" from 2008ms to a few tens of ms (notes/14).
    """
    body = b"[^a-z0-9]+".join(re.escape(t.encode()) for t in phrase_tokens)
    return re.compile(b"(?<![a-z0-9])" + body + b"(?![a-z0-9])",
                      re.IGNORECASE)


def text_contains_phrase(text: str, phrase_tokens: list[str]) -> bool:
    """Reference implementation (token-level). Kept as the correctness
    oracle the fast regex path is tested against."""
    toks = tokenize(text)
    n, m = len(toks), len(phrase_tokens)
    if m == 0 or n < m:
        return False
    first = phrase_tokens[0]
    for i in range(n - m + 1):
        if toks[i] == first and toks[i:i + m] == phrase_tokens:
            return True
    return False


class PhraseSearcher:
    """Wraps a lexical searcher + doc store to answer phrase queries."""

    def __init__(self, searcher, store, max_candidates: int = 50_000):
        self.s = searcher
        self.store = store
        self.max_candidates = max_candidates
        self.stemmed = getattr(searcher, "_stemmed", False)

    def _stems(self, toks: list[str]) -> list[str]:
        return [stem(t) for t in toks] if self.stemmed else list(toks)

    def search(self, query: str, k: int = 10, exact_count: bool = False
               ) -> dict:
        """Top-k documents containing every quoted phrase, ranked by BM25.

        Fast path: walk the BM25 ranking in descending score and verify each
        document, stopping once k are verified. Because the walk is in score
        order, the first k verified ARE the k highest-scoring matches — the
        result is exact, not an approximation, and it typically verifies a
        few dozen documents instead of every co-occurrence (which for a
        phrase like "cost of living" is 14k documents).

        Slow path: if the ranked prefix runs out before k matches are found
        (terms that co-occur constantly but rarely adjacently), fall back to
        the complete conjunctive intersection so the answer stays exact.
        """
        phrases, loose = parse_query(query)
        if not phrases:
            return {"hits": self.s.search(query, k), "phrases": [],
                    "verified_scanned": 0, "path": "no-phrase"}

        regexes = [phrase_regex(p) for p in phrases]
        full_query = " ".join([t for p in phrases for t in p] + loose)
        scanned = 0
        for depth in (100, 500, 2500, 10_000):
            ranked = self.s.search(full_query, depth)
            if not ranked:
                break
            pids = [p for p, _ in ranked]
            texts = self.store.text_bytes_many(pids)
            hits = []
            for (pid, sc), t in zip(ranked, texts):
                if all(r.search(t) for r in regexes):
                    hits.append((pid, sc))
                    if len(hits) >= k:
                        break
            scanned = len(ranked)
            if len(hits) >= k or len(ranked) < depth:
                return {"hits": hits, "phrases": phrases,
                        "verified_scanned": scanned, "path": "ranked-walk"}

        # exhaustive fallback — complete by construction
        all_terms = self._stems([t for p in phrases for t in p])
        cands = self.s.intersect(sorted(set(all_terms)),
                                 max_out=self.max_candidates)
        pid_list = cands.tolist()
        texts = self.store.text_bytes_many(pid_list)
        verified = [p for p, t in zip(pid_list, texts)
                    if all(r.search(t) for r in regexes)]
        return {"hits": self._score(verified, phrases, loose, k),
                "phrases": phrases, "verified_scanned": len(pid_list),
                "candidates": int(len(cands)), "verified": len(verified),
                "truncated": len(cands) >= self.max_candidates,
                "path": "exhaustive"}

    def _score(self, pids: list[int], phrases, loose, k: int
               ) -> list[tuple[int, float]]:
        """Rank verified docs by the engine's own BM25 for the full query.

        Retrieval depth grows until the top-k of the verified set is
        provably settled: we need k verified docs inside the ranked prefix,
        because any verified doc outside it scores below every doc inside.
        Starting shallow keeps the common case cheap (a top-10000 query cost
        14ms; top-100 costs 7ms) without ever guessing.
        """
        if not pids:
            return []
        query = " ".join([t for p in phrases for t in p] + loose)
        want = set(pids)
        depth = 100
        while True:
            ranked = self.s.search(query, depth)
            found = [(p, sc) for p, sc in ranked if p in want]
            if len(found) >= k or depth >= 10_000 or len(ranked) < depth:
                break
            depth *= 5
        if len(found) >= k:
            return found[:k]
        # exhausted retrieval depth: report what ranked, then the remainder
        # (still exact membership, just unscored tail ordering)
        rest = [p for p in pids if p not in {q for q, _ in found}]
        return found + [(p, 0.0) for p in rest[:k - len(found)]]
