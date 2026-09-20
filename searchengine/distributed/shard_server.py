"""A single shard server: the normal engine, exposed over HTTP, plus the
global-docid translation the broker needs.

Deliberately the SAME searcher the single-node engine uses — a shard is not
a special kind of index, it is an index that happens to hold a slice.

Usage: python -m searchengine.distributed.shard_server <shard_idx_dir> <port>
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class Handler(BaseHTTPRequestHandler):
    searcher = None
    base = 0
    shard_id = 0
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/healthz":
            body = b"ok"
        elif u.path == "/search":
            t0 = time.perf_counter()
            hits = self.searcher.search(q.get("q", [""])[0],
                                        int(q.get("k", ["10"])[0]))
            # translate to GLOBAL docids so the broker can merge blindly
            body = json.dumps({
                "shard": self.shard_id,
                "hits": [[p + self.base, s] for p, s in hits],
                "took_ms": round((time.perf_counter() - t0) * 1e3, 3),
            }).encode()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    idx_dir, port = sys.argv[1], int(sys.argv[2])
    from ..search_hybrid import open_lexical
    Handler.searcher = open_lexical(idx_dir)
    with open(f"{idx_dir}/shard_meta.json") as f:
        m = json.load(f)
    Handler.base = m["base_docid"]
    Handler.shard_id = m["shard"]
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"shard {m['shard']} on :{port} "
          f"docs={m['n_docs']} base={m['base_docid']}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
