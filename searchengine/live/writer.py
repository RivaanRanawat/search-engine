"""Incremental indexing: add documents without rebuilding the corpus.

Everything so far has been batch: 33s to rebuild 8.84M passages is fast
enough that incremental updates were an explicit non-goal (notes/00). That
stops being true the moment documents arrive continuously — a crawler
fetching 13 pages/s cannot wait for a full rebuild, and rebuilding to add
one document is O(corpus) work for O(1) new information.

Architecture (Lucene's, for the reasons Lucene has it — notes/01):
- An index is a set of **immutable segments** plus a manifest.
- New documents buffer in memory and flush to a NEW segment; nothing
  existing is rewritten, so writers never block readers.
- `commit()` swaps the manifest atomically (tmp + rename), so a reader sees
  either the old set of segments or the new one, never a torn state.
- Deletes are **tombstones** (a per-segment bitset), because segments are
  immutable; the space is reclaimed at merge.
- Query cost grows with segment count, so a **tiered merge policy**
  consolidates segments in the background.

### The scoring subtlety (measured, not assumed)
BM25's idf is a collection statistic, exactly as in the distributed case
(notes/17), and here the collection *changes over time*. A segment written
today embeds today's idf into its precomputed impacts. As the corpus grows,
those impacts drift from what a full rebuild would produce.

We handle it the way the measurements justify: each new segment is scored
with the **global df known at write time** (summed over all live segments
plus the new one), and merges rewrite impacts with fresh statistics —
bounding drift rather than pretending it does not exist. `bench/live/`
measures the actual drift against a full rebuild.
"""
import json
import os
import pickle
import shutil
import time

import numpy as np

MANIFEST = "manifest.json"


def _atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)          # readers see old or new, never partial


class IndexWriter:
    """Buffers documents, flushes them as immutable segments."""

    def __init__(self, index_dir: str, buffer_docs: int = 50_000):
        self.dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.manifest_path = f"{index_dir}/{MANIFEST}"
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"segments": [], "next_base": 0, "next_id": 0,
                             "n_docs": 0}
        self.buffer: list[tuple[int, str]] = []
        self.buffer_docs = buffer_docs

    # ---------------------------------------------------------- writing

    def add(self, text: str) -> int:
        """Buffer one document; returns its global docid."""
        docid = self.manifest["next_base"] + len(self.buffer)
        self.buffer.append((docid, text))
        return docid

    def add_many(self, texts: list[str]) -> list[int]:
        return [self.add(t) for t in texts]

    def _global_df(self) -> tuple[dict, int, float]:
        """df summed over live segments, plus N and total length — the
        statistics a new segment must be scored against."""
        df: dict[str, int] = {}
        n_docs = 0
        total_len = 0.0
        for seg in self.manifest["segments"]:
            if not seg.get("live", True):
                continue
            with open(f"{seg['dir']}/seg_df.pkl", "rb") as f:
                d = pickle.load(f)
            for t, c in d.items():
                df[t] = df.get(t, 0) + c
            n_docs += seg["n_docs"]
            total_len += seg["total_len"]
        return df, n_docs, total_len

    def flush(self) -> dict | None:
        """Write buffered documents as a new immutable segment."""
        if not self.buffer:
            return None
        from ..compress_index import build as compress
        from ..indexer_v4 import build as build_index
        from .stats import rescore_with_global

        t0 = time.perf_counter()
        seg_id = self.manifest["next_id"]
        seg_dir = f"{self.dir}/seg{seg_id:05d}"
        os.makedirs(seg_dir, exist_ok=True)
        tsv = f"{seg_dir}/docs.tsv"
        with open(tsv, "w", encoding="utf-8") as f:
            for i, (_, text) in enumerate(self.buffer):
                clean = " ".join(text.split())
                f.write(f"{i}\t{clean}\n")

        raw = f"{seg_dir}/raw"
        stats = build_index(tsv, raw, workers=2)

        # segment-local df, kept so future writers can sum global statistics
        with open(f"{raw}/terms.pkl", "rb") as f:
            terms = pickle.load(f)
        offsets = np.load(f"{raw}/offsets.u64.npy").astype(np.int64)
        dfs = np.diff(offsets)
        with open(f"{seg_dir}/seg_df.pkl", "wb") as f:
            pickle.dump({t: int(dfs[i]) for t, i in terms.items()}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        doclens = np.load(f"{raw}/doclens.u32.npy")
        total_len = float(doclens.sum())

        # score against the collection as it exists NOW (this segment
        # included), so a new segment is consistent with a full rebuild at
        # this instant
        gdf, gn, gtotal = self._global_df()
        for t, tid in terms.items():
            gdf[t] = gdf.get(t, 0) + int(dfs[tid])
        gn += stats["n_docs"]
        gtotal += total_len
        rescore_with_global(raw, gdf, gn, gtotal / max(1, gn))
        compress(raw, f"{seg_dir}/idx", quant_scale=None)
        shutil.copy(f"{seg_dir}/seg_df.pkl", f"{seg_dir}/idx/seg_df.pkl")
        shutil.rmtree(raw, ignore_errors=True)
        # the segment keeps its own document text: a merge rewrites segments
        # from source, and snippets need it anyway
        os.replace(tsv, f"{seg_dir}/idx/docs_store.tsv")

        # STABLE global docids: stored explicitly per segment rather than
        # derived as base+local. A merge drops tombstoned documents and
        # renumbers the survivors, so base+local silently reassigns ids —
        # which would corrupt anything keyed by docid, above all the dense
        # PQ codes. Caught by a test (notes/20); the map costs 8B/doc.
        base = self.manifest["next_base"]
        np.save(f"{seg_dir}/docids.u64.npy",
                np.arange(base, base + stats["n_docs"], dtype=np.uint64))
        seg = {"id": seg_id, "dir": seg_dir, "base_docid": base,
               "n_docs": stats["n_docs"],
               "total_len": total_len, "live": True,
               "created": time.time()}
        self.manifest["segments"].append(seg)
        self.manifest["next_base"] += stats["n_docs"]
        self.manifest["next_id"] += 1
        self.manifest["n_docs"] += stats["n_docs"]
        self.buffer.clear()
        seg["flush_s"] = round(time.perf_counter() - t0, 2)
        return seg

    def commit(self) -> dict:
        """Flush pending documents and publish the manifest atomically."""
        seg = self.flush()
        _atomic_write_json(self.manifest_path, self.manifest)
        return {"segments": len(self.manifest["segments"]),
                "n_docs": self.manifest["n_docs"],
                "flushed": seg["id"] if seg else None,
                "flush_s": seg["flush_s"] if seg else 0.0}

    # ---------------------------------------------------------- deleting

    def delete(self, docids: list[int]) -> int:
        """Tombstone documents. Segments are immutable, so deletion marks a
        bitset; space returns at merge."""
        n = 0
        for seg in self.manifest["segments"]:
            if not seg.get("live", True):
                continue
            lo, hi = seg["base_docid"], seg["base_docid"] + seg["n_docs"]
            local = [d - lo for d in docids if lo <= d < hi]
            if not local:
                continue
            path = f"{seg['dir']}/deletes.npy"
            mask = (np.load(path) if os.path.exists(path)
                    else np.zeros(seg["n_docs"], bool))
            mask[local] = True
            np.save(path, mask)
            seg["deleted"] = int(mask.sum())
            n += len(local)
        _atomic_write_json(self.manifest_path, self.manifest)
        return n
