"""Query autocomplete from the MS MARCO train-query log (502K real queries).

Structure: unique queries sorted lexicographically; prefix lookup is a
binary-search range (np.searchsorted on a sorted list of strings via
bisect), ranked by (shorter first) — with no popularity signal in the log,
generality is the best proxy Google-style logs would provide counts for.
"""
import bisect


class Suggester:
    def __init__(self, train_queries_path: str):
        seen = set()
        qs = []
        with open(train_queries_path, encoding="utf-8") as f:
            for line in f:
                q = line.rstrip("\n").split("\t", 1)[1].strip().lower()
                if q and q not in seen:
                    seen.add(q)
                    qs.append(q)
        qs.sort()
        self.qs = qs

    def suggest(self, prefix: str, k: int = 8) -> list[str]:
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        lo = bisect.bisect_left(self.qs, prefix)
        hi = bisect.bisect_right(self.qs, prefix + "￿")
        cands = self.qs[lo:min(hi, lo + 200)]  # cap scan window
        cands.sort(key=len)
        return cands[:k]
