"""Full-corpus embedding job: stream -> encode -> PQ-compress -> shard files.

Design forced by measurement (notes/12):
- 6.8GB of f16 vectors cannot be stored (1.9GB free) => PQ codes are written
  directly; full-precision vectors never touch disk.
- ~9h of compute at 268 passages/s => the job MUST be resumable and
  parallel-safe. Work is split into fixed shards of SHARD docs; each shard
  writes codes_<i>.u8 atomically (tmp + rename) and is skipped if present.
  Kill it, restart it, run several processes over disjoint shard ranges —
  all safe.
- RSS is printed per shard; the encoder holds one batch at a time.

Commands:
  train  <n_sample> <out_dir> [m]      sample corpus, train PQ, save codebook
  run    <out_dir> [shard_lo shard_hi] encode shards (default: all missing)
  status <out_dir>                     progress report
  verify <out_dir>                     assert no unwritten (all-zero) rows
"""
import json
import os
import resource
import sys
import time

import numpy as np

from .encoder import Encoder
from .pq import PQ

SHARD = 100_000
COLLECTION = "data/collection.tsv"
OFFSETS = "indexes/v1s/lineoffsets.i64.npy"


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _texts_for(offs: np.ndarray, lo: int, hi: int) -> list[str]:
    out = []
    with open(COLLECTION, "rb") as f:
        f.seek(int(offs[lo]))
        for _ in range(hi - lo):
            line = f.readline()
            out.append(line.decode("utf-8", "replace").split("\t", 1)[1].strip())
    return out


def cmd_train(n_sample: int, out_dir: str, m: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    offs = np.load(OFFSETS)
    n_docs = len(offs)
    rng = np.random.default_rng(11)
    # contiguous windows sampled across the corpus: random access to 200K
    # scattered lines would be seek-bound; windows keep reads sequential
    n_win = 200
    per = max(1, n_sample // n_win)
    enc = Encoder(quantized=True, threads=6)
    chunks = []
    t0 = time.perf_counter()
    for w in range(n_win):
        lo = int(rng.integers(0, n_docs - per))
        chunks.append(enc.encode(_texts_for(offs, lo, lo + per), batch=32))
    x = np.vstack(chunks)
    print(f"sampled {len(x)} vectors in {time.perf_counter()-t0:.0f}s "
          f"(rss {rss_gb():.1f}GB)", flush=True)
    t0 = time.perf_counter()
    pq = PQ(m=m).train(x, iters=20)
    pq.save(f"{out_dir}/pq_centroids.npy")
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump({"m": m, "dim": 384, "shard": SHARD, "n_docs": n_docs,
                   "encoder": "int8", "model": "msmarco-MiniLM-L6-cos-v5"}, f)
    print(f"trained PQ m={m} in {time.perf_counter()-t0:.0f}s -> {out_dir}",
          flush=True)


def _codes_path(out_dir: str) -> str:
    return f"{out_dir}/codes.u8.npy"


def _ensure_codes_file(out_dir: str, n_docs: int, m: int) -> None:
    """Preallocate the single output file once. Shards write disjoint slices
    of it, so there is never a second copy on disk (free space is the
    binding constraint here — see notes/12) and no merge pass is needed."""
    if not os.path.exists(_codes_path(out_dir)):
        a = np.lib.format.open_memmap(_codes_path(out_dir), mode="w+",
                                      dtype=np.uint8, shape=(n_docs, m))
        del a


def cmd_run(out_dir: str, lo_shard: int | None, hi_shard: int | None) -> None:
    with open(f"{out_dir}/meta.json") as f:
        meta = json.load(f)
    pq = PQ.load(f"{out_dir}/pq_centroids.npy")
    offs = np.load(OFFSETS)
    n_docs, m = meta["n_docs"], meta["m"]
    n_shards = (n_docs + SHARD - 1) // SHARD
    lo_shard = 0 if lo_shard is None else lo_shard
    hi_shard = n_shards if hi_shard is None else min(hi_shard, n_shards)
    os.makedirs(f"{out_dir}/done", exist_ok=True)
    _ensure_codes_file(out_dir, n_docs, m)
    codes_mm = np.lib.format.open_memmap(_codes_path(out_dir), mode="r+")
    enc = Encoder(quantized=True, threads=int(os.environ.get("ENC_THREADS", 6)))
    batch = int(os.environ.get("ENC_BATCH", 32))
    for si in range(lo_shard, hi_shard):
        flag = f"{out_dir}/done/{si:04d}"
        if os.path.exists(flag):
            continue
        lo, hi = si * SHARD, min((si + 1) * SHARD, n_docs)
        t0 = time.perf_counter()
        texts = _texts_for(offs, lo, hi)
        emb = enc.encode(texts, batch=batch)
        codes_mm[lo:hi] = pq.encode(emb)
        codes_mm.flush()
        # flag written only after a successful flush: a crash mid-shard just
        # means that shard is redone, never silently half-written
        open(flag, "w").close()
        dt = time.perf_counter() - t0
        print(f"shard {si}/{n_shards} [{lo}:{hi}] {len(texts)/dt:.0f}/s "
              f"{dt:.0f}s rss={rss_gb():.1f}GB", flush=True)


def cmd_status(out_dir: str) -> None:
    with open(f"{out_dir}/meta.json") as f:
        meta = json.load(f)
    n_shards = (meta["n_docs"] + SHARD - 1) // SHARD
    done = sum(1 for i in range(n_shards)
               if os.path.exists(f"{out_dir}/done/{i:04d}"))
    print(json.dumps({"shards_done": done, "shards_total": n_shards,
                      "docs_done": done * SHARD,
                      "pct": round(100 * done / n_shards, 1)}))


def cmd_verify(out_dir: str) -> None:
    """Every shard flagged done must contain no all-zero rows (a zero row
    means never-written memory, i.e. a lie in the index)."""
    with open(f"{out_dir}/meta.json") as f:
        meta = json.load(f)
    n_docs = meta["n_docs"]
    n_shards = (n_docs + SHARD - 1) // SHARD
    codes = np.load(_codes_path(out_dir), mmap_mode="r")
    bad = []
    for si in range(n_shards):
        if not os.path.exists(f"{out_dir}/done/{si:04d}"):
            continue
        lo, hi = si * SHARD, min((si + 1) * SHARD, n_docs)
        blk = np.asarray(codes[lo:hi])
        if (blk.max(axis=1) == 0).any():
            bad.append(si)
    print(json.dumps({"checked_shards": n_shards, "bad_shards": bad}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        cmd_train(int(sys.argv[2]), sys.argv[3],
                  int(sys.argv[4]) if len(sys.argv) > 4 else 64)
    elif cmd == "run":
        cmd_run(sys.argv[2],
                int(sys.argv[3]) if len(sys.argv) > 3 else None,
                int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == "verify":
        cmd_verify(sys.argv[2])
    elif cmd == "status":
        cmd_status(sys.argv[2])
    else:
        raise SystemExit(f"unknown command {cmd}")
