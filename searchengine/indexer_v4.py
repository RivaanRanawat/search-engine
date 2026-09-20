"""v4 indexer: parallel native scan + vectorized shard merge.

Pipeline (all timings printed):
  1. split collection.tsv into W contiguous byte shards (line-aligned);
  2. run W native scanner processes (tokenize+Porter+tf-count) in parallel;
  3. merge: global vocab union, then a vectorized COUNTING-SORT placement —
     per shard, every posting's final position is
       global_offset[gid] + already_written[gid] + rank_within_local_list
     computed with repeat/arange (no per-posting Python, no giant argsort);
  4. impacts + max-impact sidecar; write v1-format index dir.

Shards are contiguous pid ranges processed in order, so per-term docids
remain globally ascending — merge is placement, never a sort.

Usage: python -m searchengine.indexer_v4 <collection.tsv> <out_dir>
           [workers] [k1] [b]
"""
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

TF_CLAMP = 63
_DIR = os.path.dirname(os.path.abspath(__file__))
_SCANNER_SRC = os.path.join(_DIR, "native", "scanner.c")
_SCANNER = os.path.join(_DIR, "native", "scanner")
MAGIC = 0x53484152445F3032


def _ensure_scanner() -> None:
    if (not os.path.exists(_SCANNER) or
            os.path.getmtime(_SCANNER) < os.path.getmtime(_SCANNER_SRC)):
        subprocess.run(["clang", "-O2", "-o", _SCANNER, _SCANNER_SRC],
                       check=True)


def _split_offsets(path: str, w: int) -> list[tuple[int, int]]:
    size = os.path.getsize(path)
    cuts = [0]
    with open(path, "rb") as f:
        for i in range(1, w):
            f.seek(size * i // w)
            f.readline()  # advance to next line start
            cuts.append(f.tell())
    cuts.append(size)
    return [(cuts[i], cuts[i + 1]) for i in range(w)]


def _read_shard(path: str):
    with open(path, "rb") as f:
        head = np.frombuffer(f.read(40), np.uint8)
        magic = head[:8].view(np.uint64)[0]
        assert magic == MAGIC, hex(int(magic))
        stem_flag, min_pid, n_docs, n_terms = head[8:24].view(np.uint32)
        total = int(head[24:32].view(np.uint64)[0])
        names_bytes = int(head[32:40].view(np.uint64)[0])
        doclens = np.frombuffer(f.read(2 * int(n_docs)), np.uint16)
        lens = np.frombuffer(f.read(2 * int(n_terms)), np.uint16)
        blob = f.read(names_bytes)
        dfs = np.frombuffer(f.read(4 * int(n_terms)), np.uint32)
        packed = np.fromfile(f, np.uint32, total)
    ends = np.cumsum(lens.astype(np.int64))
    names = [blob[e - l:e].decode("latin1")
             for l, e in zip(lens.tolist(), ends.tolist())]
    return int(min_pid), doclens, names, dfs.astype(np.int64), packed


def build(collection: str, out_dir: str, workers: int = 4,
          k1: float = 0.82, b: float = 0.75) -> dict:
    _ensure_scanner()
    os.makedirs(out_dir, exist_ok=True)
    T = {}
    t_all = time.perf_counter()

    # 1+2. parallel scan
    t0 = time.perf_counter()
    spans = _split_offsets(collection, workers)
    shard_paths = [f"{out_dir}/shard{i}.bin" for i in range(workers)]
    procs = [subprocess.Popen(
        [_SCANNER, "scan", collection, str(s), str(e), shard_paths[i], "1"],
        stderr=subprocess.DEVNULL)
        for i, (s, e) in enumerate(spans)]
    for p in procs:
        assert p.wait() == 0
    T["scan_s"] = time.perf_counter() - t0

    # 3a. read shards + vocab union
    t0 = time.perf_counter()
    shards = [_read_shard(p) for p in shard_paths]
    shards.sort(key=lambda s: s[0])  # pid order
    vocab: dict[str, int] = {}
    gid_maps = []
    for _, _, names, dfs, _ in shards:
        gm = np.empty(len(names), np.uint32)
        for i, nm in enumerate(names):
            g = vocab.get(nm)
            if g is None:
                g = vocab[nm] = len(vocab)
            gm[i] = g
        gid_maps.append(gm)
    n_terms = len(vocab)
    T["vocab_s"] = time.perf_counter() - t0

    # 3b. counting-sort placement
    t0 = time.perf_counter()
    gdfs = np.zeros(n_terms, np.int64)
    for (_, _, _, dfs, _), gm in zip(shards, gid_maps):
        np.add.at(gdfs, gm, dfs)
    offsets = np.zeros(n_terms + 1, np.uint64)
    np.cumsum(gdfs, out=offsets[1:])
    total = int(offsets[-1])
    packed_all = np.empty(total, np.uint32)
    written = np.zeros(n_terms, np.int64)
    for (_, _, _, dfs, packed), gm in zip(shards, gid_maps):
        local_offs = np.zeros(len(dfs) + 1, np.int64)
        np.cumsum(dfs, out=local_offs[1:])
        starts = offsets[gm].astype(np.int64) + written[gm]
        tgt = np.repeat(starts, dfs)
        tgt += np.arange(len(packed), dtype=np.int64) - np.repeat(
            local_offs[:-1], dfs)
        packed_all[tgt] = packed
        written[gm] += dfs
    doclens = np.concatenate([s[1] for s in shards]).astype(np.uint32)
    n_docs = len(doclens)
    docids = (packed_all >> 6).astype(np.uint32)
    tfs = (packed_all & TF_CLAMP).astype(np.uint8)
    del packed_all
    T["place_s"] = time.perf_counter() - t0

    # 4. impacts + sidecar
    t0 = time.perf_counter()
    dl = doclens.astype(np.float32)
    avgdl = float(dl.mean())
    idf_t = np.log(1.0 + (n_docs - gdfs + 0.5) / (gdfs + 0.5)).astype(np.float32)
    impacts = np.empty(total, np.float32)
    idf_p = np.repeat(idf_t, gdfs)
    CH = 50_000_000
    for s0 in range(0, total, CH):
        e0 = min(s0 + CH, total)
        tf_f = tfs[s0:e0].astype(np.float32)
        impacts[s0:e0] = idf_p[s0:e0] * tf_f * (k1 + 1.0) / (
            tf_f + k1 * (1.0 - b + b * dl[docids[s0:e0]] / avgdl))
    del idf_p
    mx = np.maximum.reduceat(impacts, offsets[:-1].astype(np.intp)).astype(np.float32)
    T["impacts_s"] = time.perf_counter() - t0

    # 5. write
    t0 = time.perf_counter()
    np.save(f"{out_dir}/offsets.u64.npy", offsets)
    np.save(f"{out_dir}/docids.u32.npy", docids)
    np.save(f"{out_dir}/tfs.u8.npy", tfs)
    np.save(f"{out_dir}/impacts.f32.npy", impacts)
    np.save(f"{out_dir}/doclens.u32.npy", doclens)
    np.save(f"{out_dir}/max_impact.f32.npy", mx)
    with open(f"{out_dir}/terms.pkl", "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"n_docs": n_docs, "n_terms": n_terms, "total_postings": total,
            "avgdl": avgdl, "k1": k1, "b": b, "stemmed": True}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f)
    for p in shard_paths:
        os.remove(p)
    T["write_s"] = time.perf_counter() - t0
    T["total_s"] = time.perf_counter() - t_all
    T.update(meta)
    return T


if __name__ == "__main__":
    out = build(sys.argv[1], sys.argv[2],
                int(sys.argv[3]) if len(sys.argv) > 3 else 4,
                float(sys.argv[4]) if len(sys.argv) > 4 else 0.82,
                float(sys.argv[5]) if len(sys.argv) > 5 else 0.75)
    print(json.dumps(out, indent=2))
