"""Ground-truth Python-level benchmarks for this machine.

The MVP is Python; these numbers are the denominators of every napkin-math
estimate for v0: tokenization MB/s, dict ops/s, list append, heap push, etc.
"""
import time, re, random, heapq, sys

def timeit(fn, *args):
    t0 = time.perf_counter()
    r = fn(*args)
    return time.perf_counter() - t0, r

# realistic passage-like text, ~1MB
WORDS = ("the search engine index compression retrieval quantum manhattan "
         "project communication amplitude hotel presence ranking neural "
         "system performance latency throughput").split()
random.seed(42)
text = " ".join(random.choice(WORDS) for _ in range(180_000))
mb = len(text) / 1e6

tok_re = re.compile(r"[a-z0-9]+")

def bench_regex_tokenize():
    n = 0
    for _ in range(20):
        n += len(tok_re.findall(text))
    return n

def bench_split_tokenize():
    n = 0
    for _ in range(20):
        n += len(text.split())
    return n

def bench_dict_insert():
    d = {}
    for i in range(5_000_000):
        d[i * 2654435761 % 8_000_000] = i
    return len(d)

def bench_dict_lookup():
    d = {i: i for i in range(1_000_000)}
    s = 0
    for _ in range(5):
        for i in range(1_000_000):
            s += d[i]
    return s

def bench_list_append():
    for _ in range(5):
        l = []
        ap = l.append
        for i in range(1_000_000):
            ap(i)
    return len(l)

def bench_heap_topk():
    random.seed(1)
    scores = [random.random() for _ in range(1_000_000)]
    h = []
    for i, s in enumerate(scores):
        if len(h) < 10:
            heapq.heappush(h, (s, i))
        elif s > h[0][0]:
            heapq.heapreplace(h, (s, i))
    return len(h)

def bench_str_intern_count():
    # counting term freqs like an indexer inner loop
    toks = text.split()
    for _ in range(5):
        d = {}
        get = d.get
        for t in toks:
            d[t] = get(t, 0) + 1
    return len(d)

if __name__ == "__main__":
    t, n = timeit(bench_regex_tokenize)
    print(f"regex_tokenize_MBps {20*mb/t:.1f}")
    t, n = timeit(bench_split_tokenize)
    print(f"split_tokenize_MBps {20*mb/t:.1f}")
    t, n = timeit(bench_dict_insert)
    print(f"dict_insert_Mops {5/t:.2f}")
    t, n = timeit(bench_dict_lookup)
    print(f"dict_lookup_Mops {5/t:.2f}")
    t, n = timeit(bench_list_append)
    print(f"list_append_Mops {5/t:.2f}")
    t, n = timeit(bench_heap_topk)
    print(f"heap_top10_scan_Mops {1/t:.2f}")
    t, n = timeit(bench_str_intern_count)
    toks = len(text.split()) * 5
    print(f"termcount_Mtoks {toks/t/1e6:.2f}")
    print(f"python {sys.version.split()[0]}")
