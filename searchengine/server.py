"""v5 HTTP search server (threaded).

GET /search?q=<query>&k=10 -> JSON {hits:[{pid,score,snippet}],took_ms}
GET /healthz -> ok

Snippets: collection.tsv is mmap'd; line offsets cached as a sidecar .npy
next to the index (built once, ~9s). Snippet = window around the first
query-term hit, terms upper-cased... no, marked with **term** markers.

Threading model: ThreadingHTTPServer. The MaxScore kernel releases the GIL
(ctypes), so C work runs in parallel; Python glue serializes. Measured
scaling lives in bench/serving/.

Usage: python -m searchengine.server <index_dir> <collection.tsv> [port] [threads-note]
"""
import json
import mmap
import os
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .porter import stem
from .search_hybrid import open_lexical
from .tokenizer import tokenize


#: Query result cache. Real query traffic is Zipfian — a small head of
#: queries dominates — so an LRU in front of the engine converts repeats
#: into dictionary lookups. Keyed by everything that changes the answer.
STATS = {"queries": 0, "cache_hits": 0, "errors": 0, "_lat": []}
STATS_LOCK = threading.Lock()
CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
CACHE_LOCK = threading.Lock()
CACHE_MAX = 4096


def cache_get(key):
    with CACHE_LOCK:
        v = CACHE.get(key)
        if v is not None:
            CACHE.move_to_end(key)
        return v


def cache_put(key, value) -> None:
    with CACHE_LOCK:
        CACHE[key] = value
        CACHE.move_to_end(key)
        while len(CACHE) > CACHE_MAX:
            CACHE.popitem(last=False)


class DocStore:
    def __init__(self, collection: str, index_dir: str):
        side = f"{index_dir}/lineoffsets.i64.npy"
        if not os.path.exists(side):
            offs = [0]
            with open(collection, "rb") as f:
                for line in f:
                    offs.append(offs[-1] + len(line))
            np.save(side, np.array(offs[:-1], dtype=np.int64))
        self.offs = np.load(side)
        f = open(collection, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def text_bytes(self, pid: int) -> bytes:
        """Raw UTF-8 bytes of the document body — no decode. Phrase
        verification scans thousands of documents per query, where the
        str decode alone dominates."""
        s = int(self.offs[pid])
        e = int(self.offs[pid + 1]) - 1 if pid + 1 < len(self.offs) else -1
        line = self.mm[s:e if e != -1 else None]
        tab = line.find(b"\t")
        return line[tab + 1:] if tab != -1 else line

    def text_bytes_many(self, pids) -> list[bytes]:
        """Bulk fetch. Offsets are gathered with one vectorized numpy op and
        line ends come from the offsets array itself, so there is no
        per-document scalar indexing and no newline scan — that overhead was
        most of phrase verification's cost (notes/14)."""
        pids = np.asarray(pids, np.int64)
        starts = self.offs[pids]
        nxt = np.where(pids + 1 < len(self.offs), pids + 1, pids)
        ends = np.where(pids + 1 < len(self.offs), self.offs[nxt] - 1, -1)
        mm = self.mm
        out = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            line = mm[s:e if e != -1 else None]
            tab = line.find(b"\t")
            out.append(line[tab + 1:] if tab != -1 else line)
        return out

    def text(self, pid: int) -> str:
        return self.text_bytes(pid).decode("utf-8", "replace")


def make_snippet(text: str, qstems: set[str], width: int = 240) -> str:
    toks = tokenize(text)
    hit = next((i for i, t in enumerate(toks) if stem(t) in qstems), 0)
    # locate char position of the hit token (approx: rebuild via find)
    pos = text.lower().find(toks[hit]) if toks else 0
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    out = text[start:end]
    if start > 0:
        out = "…" + out
    if end < len(text):
        out += "…"
    return out


class Handler(BaseHTTPRequestHandler):
    searcher = None
    hybrid = None
    web = None
    phraser = None
    store: DocStore = None
    speller = None
    suggester = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence per-request logging
        pass

    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _int_param(q, name: str, default: int, lo: int, hi: int) -> int:
        """Query parameters come from the internet: never trust them to
        parse, and never let them request unbounded work."""
        try:
            v = int(q.get(name, [str(default)])[0])
        except (ValueError, TypeError):
            return default
        return max(lo, min(hi, v))

    def do_GET(self):
        """Any unhandled exception must become a JSON 500, not a dropped
        connection: a single malformed request should never take down a
        worker or leave a client hanging."""
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            pass                                  # client hung up; not ours
        except Exception as e:
            with STATS_LOCK:
                STATS["errors"] += 1
            try:
                body = json.dumps({"error": type(e).__name__,
                                   "detail": str(e)[:200]}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass

    def _route(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            from .ui import PAGE
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/healthz":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/metrics":
            with STATS_LOCK:
                snap = dict(STATS)
            lat = sorted(snap.pop("_lat", []) or [0.0])
            snap.update({
                "p50_ms": round(lat[len(lat) // 2], 3),
                "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 3),
                "p99_ms": round(lat[int(0.99 * (len(lat) - 1))], 3),
                "cache_hit_rate": round(
                    snap["cache_hits"] / max(1, snap["queries"]), 4),
                # Pre-forked workers do not share memory, so BOTH the cache
                # and these counters are per-process: a scrape hits one
                # random worker and sees ~1/Nth of traffic. Reported
                # explicitly rather than silently understating totals.
                "worker_pid": os.getpid(),
                "scope": "per-worker",
                "cache_entries": len(CACHE),
            })
            self._json(snap)
            return
        q = parse_qs(u.query)
        if u.path == "/suggest":
            if self.suggester is None:
                self._json({"suggestions": [], "took_ms": 0.0})
                return
            prefix = q.get("q", [""])[0]
            t0 = time.perf_counter()
            s = self.suggester.suggest(prefix, self._int_param(q, "k", 8, 1, 50))
            self._json({"suggestions": s,
                        "took_ms": round((time.perf_counter() - t0) * 1e3, 3)})
            return
        if u.path != "/search":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        query = q.get("q", [""])[0][:1000]      # cap absurd query lengths
        k = self._int_param(q, "k", 10, 1, 100)
        want_snippets = q.get("snippets", ["1"])[0] != "0"
        # dense reranking: on by default when a dense index is loaded
        rerank = (self.hybrid is not None
                  and q.get("rerank", ["1"])[0] != "0")
        t0 = time.perf_counter()

        ckey = (query, k, want_snippets, q.get("rerank", ["1"])[0])
        cached = cache_get(ckey)
        if cached is not None:
            body = cached
            with STATS_LOCK:
                STATS["queries"] += 1
                STATS["cache_hits"] += 1
                STATS["_lat"].append((time.perf_counter() - t0) * 1e3)
                del STATS["_lat"][:-5000]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        phrase_info = None
        ce_used = False
        web_meta = {}
        if '"' in query and self.phraser is not None:
            # exact phrase queries bypass dense reranking: the user asked for
            # a literal match, not a semantic neighbourhood
            pr = self.phraser.search(query, k)
            hits = pr["hits"]
            phrase_info = {"phrases": [" ".join(p) for p in pr["phrases"]],
                           "scanned": pr["verified_scanned"],
                           "path": pr["path"]}
        elif self.web is not None:
            web_hits = self.web.search(query, k)
            hits = [(h["id"], h["score"]) for h in web_hits]
            web_meta = {h["id"]: h for h in web_hits}
        elif rerank and q.get("rerank", ["1"])[0] == "ce":
            # opt-in precision tier: ~350ms for +0.044 MRR (notes/15)
            hits = self.hybrid.search_ce(query, self.store, k)
            ce_used = True
        else:
            hits = (self.hybrid.search(query, k) if rerank
                    else self.searcher.search(query, k))
        qtoks = tokenize(query)
        qstems = {stem(t) for t in qtoks}
        out = []
        for pid, score in hits:
            h = {"pid": pid, "score": round(score, 4)}
            if want_snippets:
                h["snippet"] = make_snippet(self.store.text(pid), qstems)
            if pid in web_meta:
                h["url"] = web_meta[pid]["url"]
                h["title"] = web_meta[pid]["title"]
            out.append(h)
        ranking = ("phrase" if phrase_info else
                   ("hybrid+ce" if ce_used else
                    ("web" if web_meta else ("hybrid" if rerank else "bm25"))))
        resp = {"hits": out, "ranking": ranking,
                "took_ms": round((time.perf_counter() - t0) * 1e3, 3)}
        if phrase_info:
            resp["phrase"] = phrase_info
        corrected = (self.speller.did_you_mean(qtoks)
                     if self.speller is not None else None)
        if corrected:
            resp["did_you_mean"] = " ".join(corrected)
        body = json.dumps(resp).encode()
        cache_put(ckey, body)
        with STATS_LOCK:
            STATS["queries"] += 1
            STATS["_lat"].append((time.perf_counter() - t0) * 1e3)
            del STATS["_lat"][:-5000]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    index_dir, collection = sys.argv[1], sys.argv[2]
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    # load BEFORE forking: index arrays are mmap'd (physically shared via
    # page cache); the vocab dict is COW-shared and mostly survives thanks
    # to gc.freeze()
    # A web index (crawled pages) carries urls.json and a PageRank prior, so
    # it is served through WebSearcher; a passage index is served directly.
    if os.path.exists(f"{index_dir}/urls.json"):
        from .build_web_index import WebSearcher
        dense = f"{index_dir}_dense"
        Handler.web = WebSearcher(
            index_dir, beta=0.15,
            dense_dir=dense if os.path.exists(f"{dense}/meta.json") else None)
        Handler.searcher = Handler.web.s
        print(f"web index: {Handler.web.n} pages, dense="
              f"{Handler.web.dense is not None}", flush=True)
    else:
        Handler.searcher = open_lexical(index_dir)
    Handler.store = DocStore(collection, index_dir)
    # Optional components degrade gracefully: an index built from a crawl
    # has no query log, and may not have a surface-form vocabulary. Missing
    # extras must not stop the engine from serving search.
    from .spell import SpellCorrector
    from .suggest import Suggester
    sdf = f"{index_dir}/surface_df.pkl"
    if os.path.exists(sdf):
        Handler.speller = SpellCorrector(sdf)
    else:
        print("spell correction: DISABLED (no surface_df.pkl)", flush=True)
    qlog = os.environ.get("QUERY_LOG", "data/queries.train.tsv")
    if os.path.exists(qlog):
        Handler.suggester = Suggester(qlog)
    else:
        print("suggestions: DISABLED (no query log)", flush=True)
    if hasattr(Handler.searcher, "intersect"):
        from .phrase import PhraseSearcher
        Handler.phraser = PhraseSearcher(Handler.searcher, Handler.store)

    # Dense reranking is enabled only if the dense index is COMPLETE. A
    # partially-built index would silently rank some documents with zeroed
    # vectors — wrong results are worse than no feature.
    dense_dir = os.environ.get("DENSE_DIR", "indexes/dense")
    if os.path.exists(f"{dense_dir}/codes.u8.npy"):
        with open(f"{dense_dir}/meta.json") as f:
            dmeta = json.load(f)
        n_shards = (dmeta["n_docs"] + 100_000 - 1) // 100_000
        done = sum(1 for i in range(n_shards)
                   if os.path.exists(f"{dense_dir}/done/{i:04d}"))
        if done == n_shards:
            from .search_hybrid import HybridSearcher
            Handler.hybrid = HybridSearcher(
                index_dir, dense_dir, alpha=0.1,
                threads=int(os.environ.get("ENC_THREADS", 2)))
            print("dense reranking: ENABLED", flush=True)
        else:
            print(f"dense reranking: DISABLED (index {done}/{n_shards} "
                  "shards complete)", flush=True)

    import socket
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", port))
    lsock.listen(1024)

    import gc
    gc.freeze()
    for _ in range(max(0, workers - 1)):
        if os.fork() == 0:
            break  # child

    # Warm up per-worker lazy state (the encoder is built after fork to avoid
    # forking a multi-threaded ONNX process). Without this the FIRST query a
    # worker sees pays ~130ms of model initialisation — measured, and paid by
    # a real user rather than by startup.
    if Handler.hybrid is not None:
        try:
            Handler.hybrid.search("warmup query", 1)
        except Exception as e:
            print(f"warmup failed (continuing): {e}", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler,
                              bind_and_activate=False)
    srv.socket = lsock
    print(f"serving on :{port} pid={os.getpid()} warm", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
