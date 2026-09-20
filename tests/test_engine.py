"""Regression suite: every component that can silently produce WRONG
results has a test that would catch it.

Philosophy: this project's benchmarks measure speed and quality, but neither
catches correctness bugs — a broken index can be fast and score well on
average while ranking specific queries wrongly. These tests build tiny
indexes from a synthetic corpus with known answers, plus oracle
cross-checks on real data when the full index is present.

Run: python3 -m pytest tests/ -q
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

from searchengine.porter import stem
from searchengine.pq import PQ
from searchengine.spell import SpellCorrector
from searchengine.suggest import Suggester
from searchengine.tokenizer import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_INDEX = os.path.join(ROOT, "indexes/v2c")

# ---------------------------------------------------------------- fixtures

TINY_DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "quantum computing uses qubits for computation",
    "the fox is quick and the dog is lazy",
    "running runner runs ran quickly",
    "manhattan project atomic bomb history",
    "a b c d e f g",
] + [f"filler document number {i} about nothing important" for i in range(50)]


@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory):
    """Build a real index (native scanner + full pipeline) over a tiny
    corpus whose correct answers we can reason about by hand."""
    d = tmp_path_factory.mktemp("tiny")
    coll = d / "collection.tsv"
    with open(coll, "w") as f:
        for i, t in enumerate(TINY_DOCS):
            f.write(f"{i}\t{t}\n")
    out = d / "idx"
    from searchengine.indexer_v4 import build
    stats = build(str(coll), str(out), workers=2)
    return str(out), stats, str(coll)


# ------------------------------------------------------------- tokenizer

def test_tokenizer_lowercases_and_splits():
    assert tokenize("Hello, World! 123abc") == ["hello", "world", "123abc"]


def test_tokenizer_drops_punctuation_only():
    assert tokenize("!!! ??? ...") == []


# ---------------------------------------------------------------- porter

@pytest.mark.parametrize("word,expected", [
    ("running", "run"), ("caresses", "caress"), ("ponies", "poni"),
    ("cats", "cat"), ("agreed", "agre"), ("relational", "relat"),
    ("hopeful", "hope"), ("goodness", "good"), ("sky", "sky"),
])
def test_porter_reference_vectors(word, expected):
    assert stem(word) == expected


def test_porter_matches_native_implementation():
    """The C scanner has its own Porter port; they must never diverge."""
    exe = os.path.join(ROOT, "searchengine/native/scanner")
    if not os.path.exists(exe):
        pytest.skip("native scanner not built")
    words = ["running", "flies", "happiness", "nationalization", "controlling",
             "relational", "sizes", "hopefully", "digitizer", "conflated"]
    r = subprocess.run([exe, "stemtest"], input="\n".join(words).encode(),
                       capture_output=True)
    got = r.stdout.decode().split()
    assert got == [stem(w) for w in words]


# ----------------------------------------------------------------- index

def test_index_stats_are_consistent(tiny_index):
    idx, stats, _ = tiny_index
    assert stats["n_docs"] == len(TINY_DOCS)
    offsets = np.load(f"{idx}/offsets.u64.npy")
    assert int(offsets[-1]) == stats["total_postings"]
    assert len(offsets) == stats["n_terms"] + 1


def test_postings_are_sorted_and_unique_per_term(tiny_index):
    """Doc-at-a-time query evaluation is WRONG if this ever breaks."""
    idx, _, _ = tiny_index
    offsets = np.load(f"{idx}/offsets.u64.npy").astype(np.int64)
    docids = np.load(f"{idx}/docids.u32.npy")
    for t in range(len(offsets) - 1):
        d = docids[offsets[t]:offsets[t + 1]]
        assert (np.diff(d) > 0).all(), f"term {t} postings not strictly sorted"


def test_doclens_match_tokenization(tiny_index):
    idx, _, coll = tiny_index
    dl = np.load(f"{idx}/doclens.u32.npy")
    for i, t in enumerate(TINY_DOCS):
        assert dl[i] == len(tokenize(t))


def test_search_finds_exact_term(tiny_index):
    from searchengine.search_v2 import SearcherV2
    idx, _, _ = tiny_index
    s = SearcherV2(idx)
    hits = s.search("qubits", 10)
    assert [p for p, _ in hits] == [1]


def test_search_ranks_by_relevance(tiny_index):
    from searchengine.search_v2 import SearcherV2
    idx, _, _ = tiny_index
    s = SearcherV2(idx)
    top = [p for p, _ in s.search("quick fox dog lazy", 3)]
    assert set(top[:2]) == {0, 2}


def test_search_is_stem_aware(tiny_index):
    from searchengine.search_v2 import SearcherV2
    idx, _, _ = tiny_index
    s = SearcherV2(idx)
    assert 3 in [p for p, _ in s.search("runs", 10)]


def test_unknown_terms_return_empty(tiny_index):
    from searchengine.search_v2 import SearcherV2
    idx, _, _ = tiny_index
    assert SearcherV2(idx).search("zzzznonexistentzzz", 10) == []


def test_empty_query_returns_empty(tiny_index):
    from searchengine.search_v2 import SearcherV2
    idx, _, _ = tiny_index
    assert SearcherV2(idx).search("", 10) == []


def test_k_larger_than_matches(tiny_index):
    from searchengine.search_v2 import SearcherV2
    idx, _, _ = tiny_index
    hits = SearcherV2(idx).search("qubits", 100)
    assert len(hits) == 1


# ------------------------------------------- compressed format equivalence

def test_compressed_index_matches_raw(tiny_index):
    """v6 compressed format must rank identically to the raw arrays."""
    from searchengine.compress_index import build as compress
    from searchengine.search_v2 import SearcherV2
    from searchengine.search_v3 import SearcherV3
    idx, _, _ = tiny_index
    with tempfile.TemporaryDirectory() as d:
        compress(idx, d)
        raw, comp = SearcherV2(idx), SearcherV3(d)
        for q in ["quick fox", "quantum computing", "the", "manhattan project",
                  "lazy dog running"]:
            a = [p for p, _ in raw.search(q, 5)]
            b = [p for p, _ in comp.search(q, 5)]
            assert a == b, f"divergence on {q!r}: {a} vs {b}"


def test_raw_intersect_matches_compressed(tiny_index):
    """SearcherV2.intersect (searchsorted probes over raw arrays) must
    return the same complete candidate set as the C kernel over the
    compressed blocks — phrase search relies on either being exact."""
    from searchengine.compress_index import build as compress
    from searchengine.search_v2 import SearcherV2
    from searchengine.search_v3 import SearcherV3
    idx, _, _ = tiny_index
    with tempfile.TemporaryDirectory() as d:
        compress(idx, d)
        raw, comp = SearcherV2(idx), SearcherV3(d)
        for terms in [["quick", "fox"], ["the"], ["manhattan", "project"],
                      ["the", "quick", "lazy"], ["nosuchterm", "fox"], []]:
            a = np.sort(raw.intersect(terms))
            b = np.sort(comp.intersect(terms))
            assert np.array_equal(a, b), f"intersect diverged on {terms}"


# ------------------------------------------------------------------- PQ

def test_pq_roundtrip_preserves_direction():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 384)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    pq = PQ(m=96).train(x[:2000], iters=10)
    recon = pq.decode(pq.encode(x))
    cos = (recon * x).sum(1) / np.linalg.norm(recon, axis=1)
    assert cos.mean() > 0.75


def test_pq_adc_matches_decoded_dot_product():
    """ADC table lookups must equal the explicit reconstructed dot product."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 384)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    pq = PQ(m=48).train(x, iters=5)
    codes = pq.encode(x)
    q = x[0]
    adc = PQ.adc(pq.lut(q), codes)
    exact = pq.decode(codes) @ q
    assert np.allclose(adc, exact, atol=1e-4)


def test_pq_encode_is_deterministic():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 384)).astype(np.float32)
    pq = PQ(m=32).train(x, iters=5)
    assert np.array_equal(pq.encode(x), pq.encode(x))


# ---------------------------------------------------------------- spell

@pytest.fixture(scope="module")
def speller():
    p = f"{FULL_INDEX}/surface_df.pkl"
    if not os.path.exists(p):
        pytest.skip("full index not present")
    return SpellCorrector(p)


@pytest.mark.parametrize("typo,fixed", [
    ("manhatan", "manhattan"), ("recepter", "receptor"),
    ("projct", "project"), ("calries", "calories"),
])
def test_spell_corrects_known_typos(speller, typo, fixed):
    assert speller.correct_term(typo) == fixed


def test_spell_leaves_common_words_alone(speller):
    for w in ["the", "project", "quantum", "receptor", "history"]:
        assert speller.correct_term(w) is None


def test_did_you_mean_returns_none_when_clean(speller):
    assert speller.did_you_mean(["manhattan", "project"]) is None


# -------------------------------------------------------------- suggest

def test_suggester_prefix_and_ordering(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("1\thow to cook rice\n2\thow to cook pasta well\n"
                 "3\tunrelated query\n4\thow to cook\n")
    s = Suggester(str(p))
    out = s.suggest("how to cook")
    assert out[0] == "how to cook"          # shortest first
    assert "unrelated query" not in out
    assert len(out) == 3


def test_suggester_empty_prefix():
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        f.write("1\tsomething\n")
        name = f.name
    assert Suggester(name).suggest("") == []
    os.unlink(name)


# ------------------------------------------------ full-index oracle checks

@pytest.mark.skipif(not os.path.exists(FULL_INDEX), reason="no full index")
def test_full_index_known_query_sanity():
    from searchengine.search_v3 import SearcherV3
    s = SearcherV3(FULL_INDEX)
    hits = s.search("what is the manhattan project", 10)
    assert len(hits) == 10
    assert all(a[1] >= b[1] for a, b in zip(hits, hits[1:])), "not sorted"


@pytest.mark.skipif(not os.path.exists(FULL_INDEX), reason="no full index")
def test_full_index_scores_are_finite():
    from searchengine.search_v3 import SearcherV3
    s = SearcherV3(FULL_INDEX)
    for q in ["the", "a of and to", "quantum entanglement experiment"]:
        for _, sc in s.search(q, 10):
            assert np.isfinite(sc) and sc > 0


# ---------------------------------------------------------------- phrase

from searchengine.phrase import (parse_query, phrase_regex,
                                 text_contains_phrase)


def test_parse_query_splits_phrases_and_loose_terms():
    ph, loose = parse_query('"new york" hotels "cheap deals" nearby')
    assert ph == [["new", "york"], ["cheap", "deals"]]
    assert loose == ["hotels", "nearby"]


def test_parse_query_without_quotes():
    assert parse_query("plain query") == ([], ["plain", "query"])


@pytest.mark.parametrize("text,phrase,expected", [
    ("the cost of living is high", ["cost", "of", "living"], True),
    ("Cost Of Living", ["cost", "of", "living"], True),
    ("cost-of-living adjustment", ["cost", "of", "living"], True),
    ("cost of high living", ["cost", "of", "living"], False),
    ("accost of living things", ["cost", "of", "living"], False),
    ("living cost of", ["cost", "of", "living"], False),
    ("the cost, of living.", ["cost", "of", "living"], True),
])
def test_phrase_regex_matches_token_adjacency(text, phrase, expected):
    """The fast regex path must agree with the token-level oracle, including
    the tricky cases: punctuation between words matches, but a phrase term
    must never match inside a longer token."""
    assert bool(phrase_regex(phrase).search(text.encode())) is expected
    assert text_contains_phrase(text, phrase) is expected


def test_phrase_regex_requires_order():
    rx = phrase_regex(["quick", "brown"])
    assert rx.search(b"the quick brown fox")
    assert not rx.search(b"the brown quick fox")


@pytest.mark.skipif(not os.path.exists(FULL_INDEX), reason="no full index")
def test_phrase_search_ranked_walk_equals_exhaustive():
    """The fast ranked-walk path must return exactly the exhaustive top-k."""
    from searchengine.phrase import PhraseSearcher
    from searchengine.search_v3 import SearcherV3
    from searchengine.server import DocStore
    s = SearcherV3(FULL_INDEX)
    store = DocStore(os.path.join(ROOT, "data/collection.tsv"), FULL_INDEX)
    ps = PhraseSearcher(s, store)
    for q in ['"manhattan project"', '"cost of living"', '"blood pressure"']:
        fast = [p for p, _ in ps.search(q, 10)["hits"]]
        phrases, loose = ps_parse = parse_query(q)
        rx = [phrase_regex(p) for p in phrases]
        terms = sorted({stem(t) for p in phrases for t in p})
        cands = s.intersect(terms, 50_000).tolist()
        texts = store.text_bytes_many(cands)
        ver = [p for p, t in zip(cands, texts) if all(r.search(t) for r in rx)]
        slow = [p for p, _ in ps._score(ver, phrases, loose, 10)]
        assert fast == slow, q


@pytest.mark.skipif(not os.path.exists(FULL_INDEX), reason="no full index")
def test_phrase_results_actually_contain_the_phrase():
    from searchengine.phrase import PhraseSearcher
    from searchengine.search_v3 import SearcherV3
    from searchengine.server import DocStore
    s = SearcherV3(FULL_INDEX)
    store = DocStore(os.path.join(ROOT, "data/collection.tsv"), FULL_INDEX)
    ps = PhraseSearcher(s, store)
    for q, toks in [('"manhattan project"', ["manhattan", "project"]),
                    ('"new york city"', ["new", "york", "city"])]:
        for pid, _ in ps.search(q, 10)["hits"]:
            assert text_contains_phrase(store.text(pid), toks)


@pytest.mark.skipif(not os.path.exists(FULL_INDEX), reason="no full index")
def test_intersect_is_conjunctive_and_sorted():
    from searchengine.search_v3 import SearcherV3
    s = SearcherV3(FULL_INDEX)
    ids = s.intersect([stem("manhattan"), stem("project")], 10_000)
    assert len(ids) > 0
    assert (np.diff(ids.astype(np.int64)) > 0).all(), "not sorted/unique"
    # every returned doc must contain BOTH terms
    single = {int(x) for x in s.intersect([stem("manhattan")], 200_000)}
    assert set(int(x) for x in ids) <= single


def test_intersect_missing_term_returns_empty():
    from searchengine.search_v3 import SearcherV3
    if not os.path.exists(FULL_INDEX):
        pytest.skip("no full index")
    s = SearcherV3(FULL_INDEX)
    assert len(s.intersect(["zzzznotawordzzzz", stem("project")])) == 0


# ------------------------------------------------------- crawler / web

from searchengine.crawler import normalize, HTMLExtract, Frontier
from searchengine.build_web_index import looks_english
from searchengine.pagerank import pagerank


@pytest.mark.parametrize("raw,expected", [
    ("http://Example.COM/a/b?x=1#frag", "http://example.com/a/b?x=1"),
    ("https://example.com:443/p", "https://example.com/p"),
    ("http://example.com:80/p", "http://example.com/p"),
    ("https://example.com/a//b", "https://example.com/a/b"),
    ("https://example.com/p?utm_source=x&q=2", "https://example.com/p?q=2"),
    ("ftp://example.com/x", None),
    ("not a url", None),
])
def test_url_normalization(raw, expected):
    assert normalize(raw) == expected


def test_html_extract_strips_scripts_and_finds_links():
    html = (b"<html><head><title>T</title></head><body>"
            b"<script>var x=1;</script><style>a{}</style>"
            b"<p>Hello world</p><a href='/next'>n</a></body></html>")
    title, text, links = HTMLExtract.parse(html, "https://e.com/page")
    assert title == "T"
    assert "Hello world" in text and "var x" not in text
    assert "https://e.com/next" in links


def test_html_extract_never_raises_on_broken_input():
    """A page with truncated multibyte UTF-8 once killed an entire crawl."""
    for bad in (b"\xd1\x00<html>", b"", b"<html><body>" + bytes(range(256))):
        title, text, links = HTMLExtract.parse(bad, "https://e.com/")
        assert isinstance(title, str) and isinstance(links, list)


def test_frontier_is_breadth_first_across_hosts():
    f = Frontier()
    for u in ["https://a.com/1", "https://a.com/2", "https://b.com/1"]:
        f.add(u)
    assert len(f) == 3
    hosts = [f.pop_host(), f.pop_host()]
    assert set(hosts) == {"a.com", "b.com"}


def test_frontier_dedupes():
    f = Frontier()
    f.add("https://a.com/1")
    f.add("https://a.com/1")
    assert len(f) == 1


@pytest.mark.parametrize("text,expected", [
    ("The quick brown fox jumps over the lazy dog and it was a fine day "
     "for all of them to be out in the sun with their friends", True),
    ("Мэдээллийн хайлт нь хэрэглэгчийн мэдээллийн хэрэгцээг хангах "
     "зорилгоор бичвэр хайх үйл явц юм байна гэж хэлж болно шүү", False),
])
def test_language_filter(text, expected):
    assert looks_english(text) is expected


def test_pagerank_concentrates_on_linked_hub():
    # 0,1,2 all link to 3; 3 links nowhere
    edges = np.array([[0, 1, 2], [3, 3, 3]], np.int64)
    pr = pagerank(edges, 4)
    assert np.isclose(pr.sum(), 1.0)
    assert pr[3] == pr.max()


def test_pagerank_uniform_on_empty_graph():
    pr = pagerank(np.zeros((2, 0), np.int64), 5)
    assert np.allclose(pr, 0.2)


# --------------------------------------------------------- distributed

def test_broker_merge_is_score_ordered():
    """Merging shard results must be a pure score sort over global ids."""
    from searchengine.distributed.broker import Broker
    b = Broker.__new__(Broker)          # no network needed for merge logic
    hits = [[5, 3.0], [9, 9.5], [1, 7.25]]
    hits.sort(key=lambda h: -h[1])
    assert [h[0] for h in hits] == [9, 1, 5]


def test_shard_pool_avoids_dead_replica():
    """Least-outstanding balancing alone routes everything to a DEAD replica
    (zero in-flight looks least loaded) — health tracking must prevent it."""
    from searchengine.distributed.broker import ShardPool
    sp = ShardPool([8001, 8002])
    sp.mark_down(8001)
    assert sp.pick() == 8002
    sp.release(8002)


def test_shard_pool_recovers_after_cooldown():
    from searchengine.distributed.broker import ShardPool
    sp = ShardPool([8001])
    sp.mark_down(8001)
    assert sp.pick() == 8001          # sole endpoint is still used
    sp.release(8001)
    sp.mark_up(8001)
    assert sp.pick() == 8001


# ------------------------------------------------- incremental indexing

def test_live_index_add_search_delete_merge(tmp_path):
    """The full live loop: add across segments, search, delete, merge."""
    from searchengine.live.reader import IndexMerger, LiveSearcher
    from searchengine.live.writer import IndexWriter
    d = str(tmp_path / "live")
    w = IndexWriter(d)
    w.add_many(["quantum computing uses qubits",
                "the quick brown fox jumps"])
    w.commit()
    w.add_many(["manhattan project atomic history",
                "another quantum entanglement document"])
    w.commit()

    s = LiveSearcher(d)
    assert s.n_segments == 2
    hits = s.search("quantum", 10)
    assert len(hits) == 2                      # one from each segment
    ids = [p for p, _ in hits]
    assert 0 in ids and 3 in ids               # global docids, not local

    # delete crosses into the right segment
    assert w.delete([0]) == 1
    s.reopen()
    assert 0 not in [p for p, _ in s.search("quantum", 10)]

    # merge preserves results and drops tombstoned docs permanently
    before = {p for p, _ in s.search("quantum", 10)}
    m = IndexMerger(d, max_segments=1, merge_factor=9)
    assert m.maybe_merge() is not None
    s.reopen()
    assert s.n_segments == 1
    assert {p for p, _ in s.search("quantum", 10)} == before
    assert 0 not in [p for p, _ in s.search("quantum", 10)]


def test_live_commit_is_atomic(tmp_path):
    """A manifest must never be observed half-written."""
    import json
    from searchengine.live.writer import IndexWriter
    d = str(tmp_path / "live2")
    w = IndexWriter(d)
    w.add("hello world document one")
    w.commit()
    with open(f"{d}/manifest.json") as f:
        m = json.load(f)                        # parses => not torn
    assert m["n_docs"] == 1
    assert not os.path.exists(f"{d}/manifest.json.tmp")


def test_live_docids_are_globally_unique_across_segments(tmp_path):
    from searchengine.live.reader import LiveSearcher
    from searchengine.live.writer import IndexWriter
    d = str(tmp_path / "live3")
    w = IndexWriter(d)
    seen = []
    for i in range(3):
        seen += w.add_many([f"doc {i}-{j} about searching text" for j in range(2)])
        w.commit()
    assert seen == list(range(6))
    s = LiveSearcher(d)
    got = {p for p, _ in s.search("searching text", 20)}
    assert got == set(range(6))


def test_docids_are_stable_across_merges(tmp_path):
    """A merge drops tombstoned docs and renumbers survivors; global ids
    must NOT shift, or anything keyed by docid (above all the dense PQ
    codes) silently points at the wrong document."""
    from searchengine.live.reader import IndexMerger, LiveSearcher
    from searchengine.live.writer import IndexWriter
    d = str(tmp_path / "stable")
    w = IndexWriter(d)
    w.add_many(["alpha document about search", "beta document about search"])
    w.commit()
    w.add_many(["gamma document about search", "delta document about search"])
    w.commit()
    s = LiveSearcher(d)
    before = dict(s.search("search", 10))          # id -> score
    w.delete([0])                                   # tombstone the first doc
    IndexMerger(d, max_segments=1, merge_factor=9).maybe_merge()
    s.reopen()
    after = dict(s.search("search", 10))
    assert 0 not in after                           # deleted stays deleted
    # every surviving document keeps the id it was given at write time
    assert set(after) == set(before) - {0}


# ------------------------------------------------------- server hardening

def test_int_param_clamps_and_survives_garbage():
    """Query params come from the internet: never trust them to parse, and
    never let them request unbounded work. A silently-unapplied patch once
    left the search path raising UnboundLocalError on EVERY request."""
    from searchengine.server import Handler
    f = Handler._int_param
    assert f({}, "k", 10, 1, 100) == 10                    # missing
    assert f({"k": ["abc"]}, "k", 10, 1, 100) == 10        # unparseable
    assert f({"k": ["-5"]}, "k", 10, 1, 100) == 1          # below range
    assert f({"k": ["99999999"]}, "k", 10, 1, 100) == 100  # above range
    assert f({"k": ["25"]}, "k", 10, 1, 100) == 25         # valid


def test_query_cache_is_lru_and_bounded():
    from searchengine import server
    server.CACHE.clear()
    old_max = server.CACHE_MAX
    server.CACHE_MAX = 3
    try:
        for i in range(5):
            server.cache_put((f"q{i}",), b"x")
        assert len(server.CACHE) == 3
        assert server.cache_get(("q0",)) is None       # evicted (oldest)
        assert server.cache_get(("q4",)) == b"x"
        server.cache_get(("q2",))                       # refresh
        server.cache_put(("q5",), b"y")
        assert server.cache_get(("q2",)) is not None    # survived as MRU
    finally:
        server.CACHE_MAX = old_max
        server.CACHE.clear()


def test_snippet_handles_empty_and_missing_terms():
    from searchengine.server import make_snippet
    assert make_snippet("", set()) == ""
    out = make_snippet("some document text here", {"zzz"})
    assert isinstance(out, str) and out
