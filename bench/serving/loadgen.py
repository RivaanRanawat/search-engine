"""Closed-loop HTTP load generator: N worker processes, each with C
persistent connections, real dev queries, fixed duration. Reports QPS and
client-side latency percentiles (aggregated across processes).

Usage: python -m bench.serving.loadgen <port> <nprocs> <conns_per_proc>
           <seconds> [snippets:0|1]
"""
import http.client
import json
import multiprocessing as mp
import random
import sys
import threading
import time


def load_queries() -> list[str]:
    qs = []
    with open("data/queries.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qs.append(line.rstrip("\n").split("\t", 1)[1])
    return qs


def conn_loop(port, queries, deadline, lats, snippets):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    rng = random.Random(threading.get_ident())
    from urllib.parse import quote
    while time.perf_counter() < deadline:
        q = quote(rng.choice(queries))
        t0 = time.perf_counter()
        conn.request("GET", f"/search?q={q}&snippets={snippets}")
        r = conn.getresponse()
        r.read()
        lats.append((time.perf_counter() - t0) * 1e3)
        assert r.status == 200


def proc_main(port, conns, seconds, snippets, out_q):
    queries = load_queries()
    deadline = time.perf_counter() + seconds
    lats: list[float] = []
    threads = [threading.Thread(target=conn_loop,
                                args=(port, queries, deadline, lats, snippets))
               for _ in range(conns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out_q.put(lats)


def main() -> None:
    port, nprocs, conns, seconds = (int(a) for a in sys.argv[1:5])
    snippets = sys.argv[5] if len(sys.argv) > 5 else "1"
    out_q = mp.Queue()
    procs = [mp.Process(target=proc_main,
                        args=(port, conns, seconds, snippets, out_q))
             for _ in range(nprocs)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    lats: list[float] = []
    for _ in procs:
        lats.extend(out_q.get())
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    lats.sort()
    n = len(lats)
    res = {"clients": nprocs * conns, "n_requests": n,
           "qps": round(n / wall, 1),
           "p50_ms": round(lats[n // 2], 2),
           "p95_ms": round(lats[int(.95 * n)], 2),
           "p99_ms": round(lats[int(.99 * n)], 2),
           "max_ms": round(lats[-1], 2), "snippets": snippets}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
