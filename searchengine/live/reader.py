"""Multi-segment searcher + tiered merge policy.

Reading: a query runs against every live segment and the results are merged
by score — structurally the same scatter-gather the distributed broker does
(notes/17), just in-process. Segments return local docids; the reader adds
each segment's base to produce global ids, and drops tombstoned documents.

Merging: query cost grows with segment count, so segments are consolidated.
The policy is Lucene's TieredMergePolicy in miniature: merge groups of
similarly-sized segments, so a big segment is not rewritten every time a
small one arrives (which is what makes a naive "merge everything" policy
quadratic). Merging also refreshes impacts with current collection
statistics, bounding the idf drift that incremental writes introduce.
"""
import json
import os
import pickle
import shutil
import time

import numpy as np

from ..search_hybrid import open_lexical
from .writer import MANIFEST, _atomic_write_json


class LiveSearcher:
    """Point-in-time view over the segments named in the manifest."""

    def __init__(self, index_dir: str):
        self.dir = index_dir
        self.reopen()

    def reopen(self) -> None:
        with open(f"{self.dir}/{MANIFEST}") as f:
            self.manifest = json.load(f)
        self.segs = []
        for seg in self.manifest["segments"]:
            if not seg.get("live", True):
                continue
            s = open_lexical(f"{seg['dir']}/idx")
            dpath = f"{seg['dir']}/deletes.npy"
            deletes = np.load(dpath) if os.path.exists(dpath) else None
            # explicit local->global map (stable across merges)
            ipath = f"{seg['dir']}/docids.u64.npy"
            ids = (np.load(ipath) if os.path.exists(ipath)
                   else np.arange(seg["base_docid"],
                                  seg["base_docid"] + seg["n_docs"],
                                  dtype=np.uint64))
            self.segs.append((seg, s, deletes, ids))

    @property
    def n_segments(self) -> int:
        return len(self.segs)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        merged: list[tuple[int, float]] = []
        for seg, s, deletes, ids in self.segs:
            # over-fetch when a segment has tombstones, so deletions cannot
            # shrink the final result set below k
            want = k if deletes is None else min(k * 4, k + int(deletes.sum()))
            for pid, score in s.search(query, want):
                if deletes is not None and deletes[pid]:
                    continue
                merged.append((int(ids[pid]), score))
        merged.sort(key=lambda h: -h[1])
        return merged[:k]


def _tiered_candidates(segs: list[dict], max_segments: int,
                       merge_factor: int) -> list[dict]:
    """Pick a group of similarly-sized segments to merge, smallest first.
    Returns [] when the index is already tidy."""
    live = [s for s in segs if s.get("live", True)]
    if len(live) <= max_segments:
        return []
    live.sort(key=lambda s: s["n_docs"])
    return live[:merge_factor]


class IndexMerger:
    def __init__(self, index_dir: str, max_segments: int = 8,
                 merge_factor: int = 4):
        self.dir = index_dir
        self.max_segments = max_segments
        self.merge_factor = merge_factor

    def maybe_merge(self) -> dict | None:
        with open(f"{self.dir}/{MANIFEST}") as f:
            manifest = json.load(f)
        group = _tiered_candidates(manifest["segments"], self.max_segments,
                                   self.merge_factor)
        if not group:
            return None
        return self._merge(manifest, group)

    def _merge(self, manifest: dict, group: list[dict]) -> dict:
        """Rewrite a group of segments as one, dropping tombstoned docs and
        recomputing impacts with current collection statistics."""
        from ..compress_index import build as compress
        from ..indexer_v4 import build as build_index
        from .stats import rescore_with_global

        t0 = time.perf_counter()
        ids = {g["id"] for g in group}
        seg_id = manifest["next_id"]
        seg_dir = f"{self.dir}/seg{seg_id:05d}"
        os.makedirs(seg_dir, exist_ok=True)
        tsv = f"{seg_dir}/docs.tsv"

        # stored text is the source of truth for a rewrite; surviving
        # documents CARRY THEIR GLOBAL IDS so a merge never reassigns them
        n = 0
        kept_ids: list[int] = []
        with open(tsv, "w", encoding="utf-8") as out:
            for g in sorted(group, key=lambda s: s["base_docid"]):
                dpath = f"{g['dir']}/deletes.npy"
                deletes = np.load(dpath) if os.path.exists(dpath) else None
                ipath = f"{g['dir']}/docids.u64.npy"
                gids = (np.load(ipath) if os.path.exists(ipath)
                        else np.arange(g["base_docid"],
                                       g["base_docid"] + g["n_docs"],
                                       dtype=np.uint64))
                with open(f"{g['dir']}/idx/docs_store.tsv",
                          encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if deletes is not None and deletes[i]:
                            continue
                        _, _, text = line.partition("\t")
                        out.write(f"{n}\t{text}")
                        kept_ids.append(int(gids[i]))
                        n += 1
        raw = f"{seg_dir}/raw"
        stats = build_index(tsv, raw, workers=2)

        with open(f"{raw}/terms.pkl", "rb") as f:
            terms = pickle.load(f)
        offsets = np.load(f"{raw}/offsets.u64.npy").astype(np.int64)
        dfs = np.diff(offsets)
        with open(f"{seg_dir}/seg_df.pkl", "wb") as f:
            pickle.dump({t: int(dfs[i]) for t, i in terms.items()}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        doclens = np.load(f"{raw}/doclens.u32.npy")
        total_len = float(doclens.sum())

        # fresh global statistics over segments that will survive
        gdf: dict[str, int] = {}
        gn, gtotal = 0, 0.0
        for s in manifest["segments"]:
            if not s.get("live", True) or s["id"] in ids:
                continue
            with open(f"{s['dir']}/seg_df.pkl", "rb") as f:
                for t, c in pickle.load(f).items():
                    gdf[t] = gdf.get(t, 0) + c
            gn += s["n_docs"]
            gtotal += s["total_len"]
        for t, tid in terms.items():
            gdf[t] = gdf.get(t, 0) + int(dfs[tid])
        gn += stats["n_docs"]
        gtotal += total_len
        rescore_with_global(raw, gdf, gn, gtotal / max(1, gn))
        compress(raw, f"{seg_dir}/idx")
        shutil.copy(tsv, f"{seg_dir}/idx/docs_store.tsv")
        np.save(f"{seg_dir}/docids.u64.npy",
                np.array(kept_ids, dtype=np.uint64))
        shutil.rmtree(raw, ignore_errors=True)
        os.remove(tsv)

        new_seg = {"id": seg_id, "dir": seg_dir,
                   "base_docid": min(g["base_docid"] for g in group),
                   "n_docs": stats["n_docs"], "total_len": total_len,
                   "live": True, "created": time.time(),
                   "merged_from": sorted(ids)}
        for s in manifest["segments"]:
            if s["id"] in ids:
                s["live"] = False
        manifest["segments"].append(new_seg)
        manifest["next_id"] += 1
        manifest["n_docs"] = sum(s["n_docs"] for s in manifest["segments"]
                                 if s.get("live", True))
        _atomic_write_json(f"{self.dir}/{MANIFEST}", manifest)

        # only now, after the manifest no longer references them, is it safe
        # to delete the old segment directories
        for g in group:
            shutil.rmtree(g["dir"], ignore_errors=True)
        return {"merged": sorted(ids), "into": seg_id, "docs": stats["n_docs"],
                "merge_s": round(time.perf_counter() - t0, 2),
                "segments_now": sum(1 for s in manifest["segments"]
                                    if s.get("live", True))}
