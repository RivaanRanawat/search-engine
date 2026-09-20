"""'Did you mean' spell correction — Norvig-style with corpus-df prior.

Per query term: if it's OOV or very rare in the corpus, generate edit
candidates (distance 1 directly; distance 2 only if no d1 hit) and pick the
candidate maximizing document frequency. Edit-distance-1 is preferred over
2 regardless of df (P(1 typo) >> P(2 typos) dominates any frequency ratio
we can observe).

Trigger threshold df<RARE: a term that appears in <5 of 8.8M docs is more
likely a typo than intent; correction is only *suggested*, never silently
applied (Google shows "did you mean" — so do we).
"""
import pickle

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
RARE = 5


class SpellCorrector:
    def __init__(self, surface_df_path: str):
        with open(surface_df_path, "rb") as f:
            self.df: dict[str, int] = pickle.load(f)

    def _edits1(self, w: str):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        for a, b in splits:
            if b:
                yield a + b[1:]                       # delete
                if len(b) > 1:
                    yield a + b[1] + b[0] + b[2:]     # transpose
            for c in ALPHABET:
                if b:
                    yield a + c + b[1:]               # replace
                yield a + c + b                       # insert

    def _best(self, cands) -> tuple[str, int]:
        best, bdf = None, 0
        for c in cands:
            d = self.df.get(c, 0)
            if d > bdf:
                best, bdf = c, d
        return best, bdf

    def correct_term(self, w: str) -> str | None:
        """Returns a correction, or None if w looks fine / nothing better."""
        wdf = self.df.get(w, 0)
        if wdf >= RARE or len(w) < 3:
            return None
        e1 = set(self._edits1(w))
        best, bdf = self._best(e1)
        if best is not None and bdf > max(wdf * 100, RARE):
            return best
        # distance 2 (bounded: only from the ~450 d1 strings, dedup'd)
        seen = set()
        for x in e1:
            for y in self._edits1(x):
                if y not in seen:
                    seen.add(y)
        best2, bdf2 = self._best(seen)
        if best2 is not None and bdf2 > max(wdf * 1000, RARE * 20):
            return best2
        return None

    def did_you_mean(self, query_terms: list[str]) -> list[str] | None:
        """Corrected term list, or None if no term needed correction."""
        out, changed = [], False
        for t in query_terms:
            c = self.correct_term(t)
            out.append(c if c else t)
            changed = changed or c is not None
        return out if changed else None
