"""Split a corpus into N document-partitioned shards.

Document partitioning (each shard holds a docid range, every shard sees
every term) is the standard web-search layout: query latency stays flat as
the corpus grows because shards work in parallel, and each shard is an
independent, self-contained index — the same format the single-node engine
serves, so shards need no special code.

**The subtlety that makes this interesting**: BM25 is not shard-local.
    idf(t) = log(1 + (N - df + 0.5)/(df + 0.5))
N and df are *collection* statistics. A shard that only knows its own slice
computes different idf values, and the score of the same document changes
depending on how the corpus was split. Real systems either accept that drift
or ship global statistics to the shards. We build BOTH so the difference can
be measured (bench/distributed):

  --global-stats : compute df/N/avgdl over the whole corpus first, then
                   force every shard to score with them (exact agreement
                   with single-node ranking)
  default        : each shard uses its own local statistics (the naive
                   split most people ship)

Usage:
  python -m searchengine.distributed.shard_build <collection.tsv> <out_dir>
         <n_shards> [--global-stats]
"""
import json
import os
import shutil
import sys
import time

import numpy as np


def split_corpus(collection: str, out_dir: str, n_shards: int) -> list[dict]:
    """Contiguous docid ranges; each shard file is renumbered from 0 because
    the index format requires docid == line number. The base offset is kept
    in meta so the broker can restore global ids."""
    os.makedirs(out_dir, exist_ok=True)
    size = os.path.getsize(collection)
    bounds = []
    with open(collection, "rb") as f:
        cuts = [0]
        for i in range(1, n_shards):
            f.seek(size * i // n_shards)
            f.readline()
            cuts.append(f.tell())
        cuts.append(size)
    shards = []
    with open(collection, "rb") as f:
        for i in range(n_shards):
            f.seek(cuts[i])
            path = f"{out_dir}/shard{i}"
            os.makedirs(path, exist_ok=True)
            base = None
            n = 0
            with open(f"{path}/pages.tsv", "wb") as o:
                while f.tell() < cuts[i + 1]:
                    line = f.readline()
                    if not line:
                        break
                    pid, _, text = line.partition(b"\t")
                    if base is None:
                        base = int(pid)
                    o.write(str(n).encode() + b"\t" + text)
                    n += 1
            shards.append({"shard": i, "base_docid": base or 0, "n_docs": n,
                           "dir": path})
    with open(f"{out_dir}/shards.json", "w") as f:
        json.dump(shards, f, indent=2)
    return shards


def global_stats(collection: str, out_dir: str,
                 existing_index: str | None = None) -> str:
    """df per term + N + avgdl over the WHOLE corpus, so shards can score
    identically to a single-node index.

    If a full index already exists its term dictionary and offsets ARE the
    global statistics (df = offsets[t+1]-offsets[t]) — no need to re-scan
    3GB of corpus to recompute what we already stored.
    """
    from ..indexer_v4 import build as build_index
    tmp = existing_index or f"{out_dir}/_global"
    if not os.path.exists(f"{tmp}/meta.json"):
        build_index(collection, tmp, workers=6)
    path = f"{out_dir}/global_stats.npz"
    import pickle
    with open(f"{tmp}/terms.pkl", "rb") as f:
        terms = pickle.load(f)
    offsets = np.load(f"{tmp}/offsets.u64.npy").astype(np.int64)
    doclens = np.load(f"{tmp}/doclens.u32.npy")
    dfs = np.diff(offsets)
    with open(f"{out_dir}/global_terms.pkl", "wb") as f:
        pickle.dump({t: int(dfs[i]) for t, i in terms.items()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(path, n_docs=len(doclens), avgdl=float(doclens.mean()))
    if existing_index is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return path


def build_shards(collection: str, out_dir: str, n_shards: int,
                 use_global: bool, existing_index: str | None = None,
                 quant_scale: float | None = None) -> dict:
    from ..indexer_v4 import build as build_index
    from ..compress_index import build as compress

    t0 = time.perf_counter()
    shards = split_corpus(collection, out_dir, n_shards)
    split_s = time.perf_counter() - t0

    gstats = None
    gs_s = 0.0
    if use_global:
        t0 = time.perf_counter()
        global_stats(collection, out_dir, existing_index)
        import pickle
        with open(f"{out_dir}/global_terms.pkl", "rb") as f:
            gdf = pickle.load(f)
        z = np.load(f"{out_dir}/global_stats.npz")
        gstats = (gdf, int(z["n_docs"]), float(z["avgdl"]))
        gs_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in shards:
        raw = f"{s['dir']}/raw"
        build_index(f"{s['dir']}/pages.tsv", raw, workers=2)
        if gstats is not None:
            _rescore_with_global(raw, *gstats)
        compress(raw, f"{s['dir']}/idx", quant_scale=quant_scale)
        with open(f"{s['dir']}/idx/shard_meta.json", "w") as f:
            json.dump(s, f)
        shutil.rmtree(raw, ignore_errors=True)
        os.remove(f"{s['dir']}/pages.tsv")
    index_s = time.perf_counter() - t0

    out = {"n_shards": n_shards, "global_stats": use_global,
           "split_s": round(split_s, 1), "global_stats_s": round(gs_s, 1),
           "index_s": round(index_s, 1),
           "docs": sum(s["n_docs"] for s in shards)}
    with open(f"{out_dir}/build_meta.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def _rescore_with_global(index_dir: str, gdf: dict, n_docs: int,
                         avgdl: float) -> None:
    """Recompute impacts using COLLECTION-wide df/N/avgdl instead of the
    shard's own. This is what makes distributed scores identical to
    single-node scores."""
    import pickle
    with open(f"{index_dir}/terms.pkl", "rb") as f:
        terms = pickle.load(f)
    with open(f"{index_dir}/meta.json") as f:
        meta = json.load(f)
    k1, b = meta["k1"], meta["b"]
    offsets = np.load(f"{index_dir}/offsets.u64.npy").astype(np.int64)
    docids = np.load(f"{index_dir}/docids.u32.npy")
    tfs = np.load(f"{index_dir}/tfs.u8.npy")
    doclens = np.load(f"{index_dir}/doclens.u32.npy").astype(np.float32)
    local_df = np.diff(offsets)
    gdf_arr = np.empty(len(local_df), np.float64)
    for t, tid in terms.items():
        gdf_arr[tid] = gdf.get(t, int(local_df[tid]))
    idf = np.log(1.0 + (n_docs - gdf_arr + 0.5) / (gdf_arr + 0.5))
    idf_p = np.repeat(idf.astype(np.float32), local_df)
    tf_f = tfs.astype(np.float32)
    dl = doclens[docids]
    impacts = idf_p * tf_f * (k1 + 1.0) / (tf_f + k1 * (1.0 - b + b * dl / avgdl))
    np.save(f"{index_dir}/impacts.f32.npy", impacts.astype(np.float32))
    mx = np.maximum.reduceat(impacts.astype(np.float32),
                             offsets[:-1].astype(np.intp))
    np.save(f"{index_dir}/max_impact.f32.npy", mx.astype(np.float32))
    meta.update({"global_stats": True, "avgdl": avgdl, "n_docs_global": n_docs})
    with open(f"{index_dir}/meta.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    print(json.dumps(build_shards(sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                  "--global-stats" in sys.argv), indent=2))
