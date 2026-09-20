"""Porter stemmer (original 1980 algorithm), faithful implementation.

Used at VOCABULARY level (1.47M terms) and query time — never in the
corpus-scan hot loop, so pure Python is fine (see notes/07: stem-by-merge).
Matches Lucene's PorterStemFilter behavior (classic Porter1), which is what
Anserini's 0.184 baseline uses.
"""


def _cons(w: str, i: int) -> bool:
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(w: str) -> int:
    """number of VC sequences in w."""
    n = 0
    i = 0
    ln = len(w)
    while i < ln and _cons(w, i):
        i += 1
    while i < ln:
        while i < ln and not _cons(w, i):
            i += 1
        if i == ln:
            break
        n += 1
        while i < ln and _cons(w, i):
            i += 1
    return n


def _vowel_in_stem(w: str) -> bool:
    return any(not _cons(w, i) for i in range(len(w)))


def _doublec(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    if len(w) < 3:
        return False
    if not (_cons(w, len(w) - 3) and not _cons(w, len(w) - 2)
            and _cons(w, len(w) - 1)):
        return False
    return w[-1] not in "wxy"


_STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
          ("anci", "ance"), ("izer", "ize"), ("bli", "ble"), ("alli", "al"),
          ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"),
          ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
          ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
          ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log")]
_STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
          ("ical", "ic"), ("ful", ""), ("ness", "")]
_STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
          "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]


def stem(w: str) -> str:
    if len(w) <= 2:
        return w

    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]

    # step 1b
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    elif ((w.endswith("ed") and _vowel_in_stem(w[:-2]))
          or (w.endswith("ing") and _vowel_in_stem(w[:-3]))):
        w = w[:-2] if w.endswith("ed") else w[:-3]
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _doublec(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"

    # step 1c
    if w.endswith("y") and _vowel_in_stem(w[:-1]):
        w = w[:-1] + "i"

    # step 2
    for suf, rep in _STEP2:
        if w.endswith(suf):
            if _m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break

    # step 3
    for suf, rep in _STEP3:
        if w.endswith(suf):
            if _m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break

    # step 4
    for suf in _STEP4:
        if w.endswith(suf):
            stem_ = w[:-len(suf)]
            if _m(stem_) > 1:
                if suf == "ion" and (not stem_ or stem_[-1] not in "st"):
                    pass
                else:
                    w = stem_
            break

    # step 5a
    if w.endswith("e"):
        a = w[:-1]
        if _m(a) > 1 or (_m(a) == 1 and not _cvc(a)):
            w = a
    # step 5b
    if _m(w) > 1 and _doublec(w) and w.endswith("l"):
        w = w[:-1]
    return w
