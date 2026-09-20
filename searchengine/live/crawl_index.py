"""Continuous pipeline: crawl → index → searchable, with no rebuild.

This is the loop a real search engine runs, and it is only possible now that
both halves exist: the polite crawler (notes/16) and incremental segments
(notes/20). Pages are handed to an IndexWriter as they are fetched, so a
document becomes searchable seconds after it is crawled rather than after
the next full rebuild.

Merging runs opportunistically between commits: query cost grows ~0.15ms per
live segment, so the policy caps segment count instead of letting a long
crawl slowly degrade search.

Usage:
  python -m searchengine.live.crawl_index <seeds.txt> <index_dir>
         [max_pages] [commit_every]
"""
import asyncio
import json
import sys
import time

from ..build_web_index import looks_english
from ..crawler import Crawler
from .reader import IndexMerger, LiveSearcher
from .writer import IndexWriter


class LiveCrawlIndexer:
    """Buffers crawled pages and commits them as segments."""

    def __init__(self, index_dir: str, commit_every: int = 400):
        self.writer = IndexWriter(index_dir)
        self.merger = IndexMerger(index_dir, max_segments=6, merge_factor=4)
        self.commit_every = commit_every
        self.buf: list[str] = []
        self.urls: list[str] = []
        self.indexed = 0
        self.skipped_lang = 0
        self.events: list[dict] = []

    def on_page(self, url: str, title: str, text: str) -> None:
        doc = f"{title} {text}".strip()
        if not looks_english(doc):
            self.skipped_lang += 1
            return
        self.buf.append(doc)
        self.urls.append(url)
        if len(self.buf) >= self.commit_every:
            self.commit()

    def commit(self) -> None:
        if not self.buf:
            return
        t0 = time.perf_counter()
        self.writer.add_many(self.buf)
        info = self.writer.commit()
        self.indexed += len(self.buf)
        self.buf.clear()
        ev = {"committed": info["flushed"], "docs_total": self.indexed,
              "commit_s": round(time.perf_counter() - t0, 2)}
        m = self.merger.maybe_merge()
        if m:
            ev["merged"] = {"into": m["into"], "segments_now":
                            m["segments_now"], "merge_s": m["merge_s"]}
        self.events.append(ev)
        print(json.dumps(ev), flush=True)


async def run(seeds_file: str, index_dir: str, max_pages: int,
              commit_every: int) -> dict:
    seeds = [l.strip() for l in open(seeds_file)
             if l.strip() and not l.startswith("#")]
    idx = LiveCrawlIndexer(index_dir, commit_every)
    c = Crawler(f"{index_dir}/_crawl", max_pages, 24, on_page=idx.on_page)
    t0 = time.time()
    await c.run(seeds)
    idx.commit()                       # flush the tail
    s = LiveSearcher(index_dir)
    return {"crawled": c.n_pages, "indexed": idx.indexed,
            "skipped_non_english": idx.skipped_lang,
            "segments": s.n_segments, "commits": len(idx.events),
            "elapsed_s": round(time.time() - t0)}


def main() -> None:
    seeds, index_dir = sys.argv[1], sys.argv[2]
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    commit_every = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    print(json.dumps(asyncio.run(run(seeds, index_dir, max_pages,
                                     commit_every)), indent=2))


if __name__ == "__main__":
    main()
