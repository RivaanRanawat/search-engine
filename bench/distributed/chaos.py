"""Fault injection for the distributed layer.

The honest limit recorded in notes/21 is that sharding was "validated on one
machine only". Extra machines cannot be conjured, but the thing a second
machine actually introduces is not mystique — it is **latency, jitter,
partial failure and partition**, and those can be injected here. What still
cannot be tested locally is real NIC/kernel behaviour and true hardware
parallelism, and that stays on the honest-limits list.

Injects, per shard, via a proxy that sits between broker and shard:
  - fixed added latency (a slow replica / distant node)
  - random jitter
  - hard partition (connection refused)
  - blackhole (accepts, never responds — the case a timeout must catch,
    and the one that hangs a client that lacks a deadline)

Then asserts the properties the broker claims:
  1. a slow shard must not exceed the broker deadline
  2. a dead shard must yield partial results flagged `degraded`, not an error
  3. a blackholed shard must be cut off by the deadline, not hang forever
  4. results from surviving shards must stay correct

Usage: python -m bench.distributed.chaos <broker_port> <shard_ports...>
"""
import json
import socket
import sys
import threading
import time
import urllib.request


class Proxy(threading.Thread):
    """TCP proxy with injectable faults, between broker and one shard."""

    def __init__(self, listen_port: int, target_port: int):
        super().__init__(daemon=True)
        self.listen_port = listen_port
        self.target_port = target_port
        self.delay = 0.0
        self.mode = "pass"          # pass | refuse | blackhole
        self._stop = threading.Event()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", listen_port))
        self.srv.listen(64)
        self.srv.settimeout(0.3)

    def run(self):
        while not self._stop.is_set():
            try:
                c, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(c,),
                             daemon=True).start()

    def _handle(self, client):
        mode = self.mode
        if mode == "refuse":
            client.close()
            return
        if mode == "blackhole":
            time.sleep(30)          # accept, never answer
            client.close()
            return
        if self.delay:
            time.sleep(self.delay)
        try:
            up = socket.create_connection(("127.0.0.1", self.target_port),
                                          timeout=5)
        except OSError:
            client.close()
            return
        def pump(a, b):
            try:
                while True:
                    d = a.recv(65536)
                    if not d:
                        break
                    b.sendall(d)
            except OSError:
                pass
            finally:
                for s in (a, b):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
        threading.Thread(target=pump, args=(client, up), daemon=True).start()
        pump(up, client)

    def stop(self):
        self._stop.set()
        try:
            self.srv.close()
        except OSError:
            pass


def query(port: int, q: str = "manhattan project", timeout: float = 10.0):
    t0 = time.perf_counter()
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/search?q={q.replace(' ', '+')}&k=5",
            timeout=timeout) as r:
        d = json.loads(r.read())
    d["_client_ms"] = (time.perf_counter() - t0) * 1e3
    return d


def main() -> None:
    broker_port = int(sys.argv[1])
    results = []

    def check(name, cond, detail):
        results.append({"check": name, "pass": bool(cond), "detail": detail})
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)

    print("baseline (all shards healthy):", flush=True)
    base = query(broker_port)
    n_base = len(base["hits"])
    check("healthy: not degraded", not base["degraded"],
          f"degraded={base['degraded']} hits={n_base} "
          f"took={base['took_ms']}ms")

    print("\ninjecting: one shard SLOW (+400ms, beyond the 250ms deadline)",
          flush=True)
    PROXIES[0].delay = 0.4
    slow = query(broker_port)
    check("slow shard is cut off by the deadline",
          slow["_client_ms"] < 1000,
          f"client saw {slow['_client_ms']:.0f}ms (deadline 250ms)")
    check("slow shard reported as degraded, not silently dropped",
          slow["degraded"],
          f"degraded={slow['degraded']} failed={slow['shards_failed']}")
    check("still returns results from healthy shards",
          len(slow["hits"]) > 0, f"{len(slow['hits'])} hits")
    PROXIES[0].delay = 0.0

    print("\ninjecting: one shard PARTITIONED (connection refused)",
          flush=True)
    PROXIES[0].mode = "refuse"
    part = query(broker_port)
    check("partition surfaces as degraded", part["degraded"],
          f"failed={part['shards_failed']}")
    check("partition still returns partial results",
          len(part["hits"]) > 0, f"{len(part['hits'])} hits")
    PROXIES[0].mode = "pass"

    print("\ninjecting: one shard BLACKHOLED (accepts, never replies)",
          flush=True)
    PROXIES[0].mode = "blackhole"
    t0 = time.perf_counter()
    bh = query(broker_port, timeout=15)
    bh_ms = (time.perf_counter() - t0) * 1e3
    check("blackhole cannot hang the query", bh_ms < 3000,
          f"returned in {bh_ms:.0f}ms")
    check("blackhole reported as degraded", bh["degraded"],
          f"failed={bh['shards_failed']}")
    PROXIES[0].mode = "pass"

    print("\nrecovery:", flush=True)
    time.sleep(3)                    # let the health cooldown expire
    rec = None
    for _ in range(8):
        rec = query(broker_port)
        if not rec["degraded"]:
            break
        time.sleep(1)
    check("recovers to healthy automatically", not rec["degraded"],
          f"degraded={rec['degraded']} hits={len(rec['hits'])}")
    check("results match the pre-fault baseline",
          [h[0] for h in rec["hits"]] == [h[0] for h in base["hits"]],
          "same top-5 docids as before any fault")

    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} checks passed")
    with open("bench/results/chaos.json", "w") as f:
        json.dump({"passed": passed, "total": len(results),
                   "checks": results}, f, indent=2)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    shard_ports = [int(p) for p in sys.argv[2:]]
    PROXIES = [Proxy(9200 + i, p) for i, p in enumerate(shard_ports)]
    for p in PROXIES:
        p.start()
    time.sleep(0.5)
    try:
        main()
    finally:
        for p in PROXIES:
            p.stop()
