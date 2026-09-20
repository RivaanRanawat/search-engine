"""Scatter-gather broker: fan a query out to all shards, merge the top-k.

Production concerns implemented here, because they are the actual content
of the distributed problem:

- **Fan-out with a deadline.** Every shard is queried concurrently and the
  broker waits at most `timeout_ms`. Tail latency of a scatter-gather query
  is the MAX over shards, not the mean — this is the classic "tail at scale"
  problem, and it is why a deadline is not optional.
- **Partial results are surfaced, never hidden.** If a shard times out or
  dies, the response is still returned but flagged `degraded` with the
  missing shard listed. Silently returning fewer results is how distributed
  search engines lie to their users.
- **Replication with per-request choice.** When a shard has replicas, the
  broker picks the one with the fewest in-flight requests (a cheap
  least-outstanding-requests policy, which beats round-robin whenever
  replicas are unevenly loaded).
- **Merging is trivial only because shards return GLOBAL docids and
  comparable scores** — see shard_build.py for why comparable scores require
  global collection statistics.

Usage:
  python -m searchengine.distributed.broker <ports...> [--port 9000]
"""
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import http.client


COOLDOWN_S = 2.0


class ShardPool:
    """One logical shard, one or more replica endpoints.

    Load balancing is least-outstanding-requests WITH health tracking. The
    health part is not optional: a dead replica has zero in-flight requests,
    so pure least-outstanding balancing considers it the *best* choice and
    sends it every query. That is exactly what happened in testing — killing
    one replica of a 2-replica shard made the shard permanently degraded even
    though its sibling was healthy (notes/17). Failures put an endpoint in a
    short cooldown; a success clears it, so recovery needs no coordination.
    """

    def __init__(self, endpoints: list[int]):
        self.endpoints = endpoints
        self.inflight = defaultdict(int)
        self.down_until = defaultdict(float)
        self.lock = threading.Lock()
        self.local = threading.local()

    def pick(self, exclude: set[int] | None = None) -> int:
        exclude = exclude or set()
        now = time.monotonic()
        with self.lock:
            healthy = [e for e in self.endpoints
                       if e not in exclude and self.down_until[e] <= now]
            pool = healthy or [e for e in self.endpoints if e not in exclude] \
                or self.endpoints
            p = min(pool, key=lambda e: self.inflight[e])
            self.inflight[p] += 1
        return p

    def release(self, port: int) -> None:
        with self.lock:
            self.inflight[port] -= 1

    def mark_down(self, port: int) -> None:
        with self.lock:
            self.down_until[port] = time.monotonic() + COOLDOWN_S

    def mark_up(self, port: int) -> None:
        with self.lock:
            self.down_until[port] = 0.0

    def conn(self, port: int) -> http.client.HTTPConnection:
        pool = getattr(self.local, "conns", None)
        if pool is None:
            pool = self.local.conns = {}
        if port not in pool:
            pool[port] = http.client.HTTPConnection("127.0.0.1", port)
        return pool[port]


class Broker:
    def __init__(self, shard_ports: list[list[int]], timeout_ms: int = 250):
        self.shards = [ShardPool(p) for p in shard_ports]
        self.timeout = timeout_ms / 1000.0
        self.pool = ThreadPoolExecutor(max_workers=max(8, len(self.shards) * 4))

    def _query_shard(self, sp: ShardPool, query: str, k: int):
        """One retry on a connection-level failure.

        Without it, every pooled connection that was open when a shard
        restarted burns exactly one user-visible request: measured as
        intermittent `degraded` responses for several requests after a
        restart, one per worker thread holding a stale socket. A GET is
        idempotent, so retrying once on a fresh connection is safe and turns
        a restart into a non-event (notes/17)."""
        tried: set[int] = set()
        last_err = "unreachable"
        for attempt in (0, 1):
            port = sp.pick(exclude=tried)   # retry lands on a DIFFERENT replica
            tried.add(port)
            try:
                c = sp.conn(port)
                c.request("GET", f"/search?q={quote(query)}&k={k}")
                r = c.getresponse()
                data = json.loads(r.read())
                sp.mark_up(port)
                return data["hits"], data.get("shard"), None
            except Exception as e:
                last_err = str(e)
                sp.mark_down(port)
                try:
                    conns = getattr(sp.local, "conns", {})
                    dead = conns.pop(port, None)
                    if dead is not None:
                        dead.close()
                except Exception:
                    pass
            finally:
                sp.release(port)
        return [], None, last_err

    def search(self, query: str, k: int = 10) -> dict:
        t0 = time.perf_counter()
        futures = [(i, self.pool.submit(self._query_shard, sp, query, k))
                   for i, sp in enumerate(self.shards)]
        merged, failed = [], []
        deadline = time.perf_counter() + self.timeout
        for i, fut in futures:
            remaining = max(0.0, deadline - time.perf_counter())
            try:
                hits, _, err = fut.result(timeout=remaining)
                if err:
                    failed.append({"shard": i, "error": err[:120]})
                else:
                    merged.extend(hits)
            except Exception:
                failed.append({"shard": i, "error": "timeout"})
        merged.sort(key=lambda h: -h[1])
        return {"hits": [[int(p), float(s)] for p, s in merged[:k]],
                "shards_total": len(self.shards),
                "shards_failed": failed,
                "degraded": bool(failed),
                "took_ms": round((time.perf_counter() - t0) * 1e3, 3)}


class Handler(BaseHTTPRequestHandler):
    broker: Broker = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/healthz":
            body = b"ok"
        elif u.path == "/search":
            q = parse_qs(u.query)
            body = json.dumps(self.broker.search(
                q.get("q", [""])[0], int(q.get("k", ["10"])[0]))).encode()
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
    argv = sys.argv[1:]
    port, timeout_ms = 9000, 250
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--port":
            port = int(argv[i + 1]); i += 2
        elif argv[i] == "--timeout-ms":
            timeout_ms = int(argv[i + 1]); i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            args.append(argv[i]); i += 1
    # each arg is a shard: "8101" or "8101,8102" for replicas
    shard_ports = [[int(x) for x in a.split(",")] for a in args]
    Handler.broker = Broker(shard_ports, timeout_ms=timeout_ms)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"broker on :{port} shards={shard_ports}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
