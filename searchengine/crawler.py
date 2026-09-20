"""Polite async web crawler → corpus in the same TSV format the indexer eats.

Deliberately conservative, because this runs from a personal machine on a
home connection:
- robots.txt is fetched and obeyed per host (cached), including Crawl-delay
- at most ONE in-flight request per host, with a minimum delay between them
- global concurrency cap, request timeout, response size cap
- honest User-Agent with a contact URL, so operators can identify us
- only text/html, only http(s), no query-string explosion (configurable)

Design notes:
- Frontier is a per-host FIFO plus a global round-robin over hosts, which
  gives breadth-first coverage across sites instead of hammering one host
  (the failure mode of a naive global queue).
- Output is written incrementally so a crash keeps everything fetched so
  far: pages.tsv (docid \\t text), meta.jsonl (url/title/hash), links.tsv
  (src_url \\t dst_url). The link graph is resolved to ids afterwards, since
  a page's outlinks are usually discovered before their targets are crawled.
- Exact-duplicate suppression by content hash; near-duplicate handling is a
  separate offline step (dedup.py) because it needs the whole corpus.

Usage:
  python -m searchengine.crawler <seeds.txt> <out_dir> [max_pages] [concurrency]
"""
import asyncio
import hashlib
import json
import re
import sys
import time
import urllib.parse as up
import urllib.robotparser as urobot
from collections import defaultdict, deque

import aiohttp

UA = ("Mozilla/5.0 (compatible; SearchEngineLab/0.1; "
      "+https://example.invalid/bot; single-node research crawler)")
MAX_BYTES = 2_000_000
TIMEOUT = 15
DEFAULT_DELAY = 1.0          # seconds between requests to the same host
SKIP_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|svg|webp|ico|css|js|pdf|zip|gz|tgz|mp4|mp3|avi|"
    r"woff2?|ttf|eot|exe|dmg|iso|xml|rss|atom)(\?|$)", re.I)


def normalize(url: str) -> str | None:
    """Canonical form: drop fragments, lowercase host, strip default ports,
    drop obvious tracking params. Reduces the frontier and duplicate fetches."""
    try:
        u = up.urlsplit(url)
    except ValueError:
        return None
    if u.scheme not in ("http", "https"):
        return None
    host = u.hostname
    if not host:
        return None
    host = host.lower()
    if (u.scheme == "http" and u.port == 80) or \
       (u.scheme == "https" and u.port == 443):
        netloc = host
    else:
        netloc = f"{host}:{u.port}" if u.port else host
    q = up.parse_qsl(u.query, keep_blank_values=True)
    q = [(k, v) for k, v in q
         if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref"))]
    path = re.sub(r"/{2,}", "/", u.path) or "/"
    return up.urlunsplit((u.scheme, netloc, path, up.urlencode(q), ""))


class HTMLExtract:
    """Title, visible text and outlinks from HTML using lxml, with script/
    style/nav chrome removed. lxml is a parser, not a search engine — it is
    the one heavy lift we do not want to reimplement."""

    @staticmethod
    def parse(html: bytes, base_url: str) -> tuple[str, str, list[str]]:
        """Never raises. Real-world HTML lies about its encoding, mixes
        charsets mid-document and truncates multibyte sequences; one such
        page killed an entire crawl before this was hardened. We decode
        defensively ourselves and re-encode clean UTF-8 before parsing."""
        from lxml import html as lhtml
        try:
            text_html = html.decode("utf-8", "replace")
        except Exception:
            try:
                text_html = html.decode("latin-1", "replace")
            except Exception:
                return "", "", []
        try:
            doc = lhtml.fromstring(text_html)
        except Exception:
            return "", "", []
        try:
            for bad in doc.xpath("//script|//style|//noscript|//template|//svg"):
                parent = bad.getparent()
                if parent is not None:
                    parent.remove(bad)
        except Exception:
            pass
        try:
            title = (doc.findtext(".//title") or "").strip()
        except Exception:
            title = ""
        try:
            body_text = " ".join(doc.text_content().split())
        except Exception:
            body_text = ""
        links = []
        try:
            for href in doc.xpath("//a/@href"):
                absolute = up.urljoin(base_url, href)
                n = normalize(absolute)
                if n and not SKIP_EXT.search(n):
                    links.append(n)
        except Exception:
            pass
        return title, body_text, links


class Frontier:
    """Per-host FIFOs + round-robin across hosts (breadth across sites)."""

    def __init__(self, max_per_host: int = 2000):
        self.queues: dict[str, deque] = defaultdict(deque)
        self.hosts: deque[str] = deque()
        self.seen: set[str] = set()
        self.max_per_host = max_per_host

    def add(self, url: str) -> None:
        if url in self.seen:
            return
        host = up.urlsplit(url).hostname or ""
        if len(self.queues[host]) >= self.max_per_host:
            return
        self.seen.add(url)
        if not self.queues[host]:
            self.hosts.append(host)
        self.queues[host].append(url)

    def pop_host(self) -> str | None:
        return self.hosts.popleft() if self.hosts else None

    def push_host(self, host: str) -> None:
        if self.queues[host]:
            self.hosts.append(host)

    def __len__(self) -> int:
        return sum(len(q) for q in self.queues.values())


class Crawler:
    def __init__(self, out_dir: str, max_pages: int = 5000,
                 concurrency: int = 16, delay: float = DEFAULT_DELAY,
                 on_page=None):
        """on_page(url, title, text) is called for every accepted page,
        which is what lets a consumer index pages *during* the crawl rather
        than after it (searchengine/live/crawl_index.py)."""
        self.on_page = on_page
        self.out_dir = out_dir
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.delay = delay
        self.frontier = Frontier()
        self.robots: dict[str, tuple[urobot.RobotFileParser, float]] = {}
        self.last_hit: dict[str, float] = {}
        self.host_delay: dict[str, float] = {}   # raised by 429/503 backoff
        self.hashes: set[str] = set()
        self.n_pages = 0
        self.stats = defaultdict(int)
        self.url_to_id: dict[str, int] = {}

    async def _robots(self, session, host_scheme: str, host: str):
        if host in self.robots:
            return self.robots[host]
        rp = urobot.RobotFileParser()
        delay = self.delay
        try:
            url = f"{host_scheme}://{host}/robots.txt"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    body = (await r.content.read(MAX_BYTES)).decode(
                        "utf-8", "replace")
                    rp.parse(body.splitlines())
                else:
                    rp.parse([])          # no robots.txt => allowed
        except Exception:
            rp.parse([])
        try:
            cd = rp.crawl_delay(UA)
            if cd:
                delay = max(delay, float(cd))
        except Exception:
            pass
        self.robots[host] = (rp, delay)
        return self.robots[host]

    async def _fetch(self, session, url: str, host: str):
        """Returns (body, final_url, retry). retry=True means the server told
        us to slow down (429/503) — the URL is requeued and the host's delay
        is increased. Ignoring 429 is how a crawler gets an IP banned; we
        honour Retry-After when present and back off exponentially otherwise."""
        try:
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=TIMEOUT),
                    allow_redirects=True, max_redirects=5) as r:
                if r.status in (429, 503):
                    ra = r.headers.get("retry-after")
                    try:
                        wait = float(ra) if ra else 0.0
                    except ValueError:
                        wait = 0.0
                    cur = self.host_delay.get(host, self.delay)
                    self.host_delay[host] = min(60.0, max(cur * 2.0, wait, 2.0))
                    self.stats[f"backoff_{r.status}"] += 1
                    return None, None, True
                ctype = r.headers.get("content-type", "")
                if r.status != 200 or "text/html" not in ctype.lower():
                    self.stats[f"skip_status_{r.status}" if r.status != 200
                               else "skip_ctype"] += 1
                    return None, None, False
                body = await r.content.read(MAX_BYTES)
                # success on a previously throttled host: decay the penalty
                if host in self.host_delay and self.host_delay[host] > self.delay:
                    self.host_delay[host] = max(self.delay,
                                                self.host_delay[host] * 0.9)
                return body, str(r.url), False
        except asyncio.TimeoutError:
            self.stats["timeout"] += 1
        except Exception:
            self.stats["error"] += 1
        return None, None, False

    async def _worker(self, session, files):
        """One bad page must never kill the crawl: every iteration is
        isolated, so failures cost one URL and are counted, not fatal."""
        while self.n_pages < self.max_pages:
            try:
                done = await self._step(session, files)
            except Exception:
                self.stats["worker_error"] += 1
                continue
            if done:
                return

    async def _step(self, session, files) -> bool:
        """Crawl one URL. Returns True when the frontier is exhausted."""
        pages_f, meta_f, links_f = files
        host = self.frontier.pop_host()
        if host is None:
            await asyncio.sleep(0.2)
            return len(self.frontier) == 0
        q = self.frontier.queues[host]
        if not q:
            return False
        url = q.popleft()
        scheme = up.urlsplit(url).scheme
        rp, delay = await self._robots(session, scheme, host)
        try:
            allowed = rp.can_fetch(UA, url)
        except Exception:
            allowed = True
        if not allowed:
            self.stats["robots_denied"] += 1
            self.frontier.push_host(host)
            return False
        # per-host rate limit (robots Crawl-delay, raised by any 429/503)
        delay = max(delay, self.host_delay.get(host, 0.0))
        wait = self.last_hit.get(host, 0) + delay - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_hit[host] = time.monotonic()

        body, final_url, retry = await self._fetch(session, url, host)
        if retry:
            q.appendleft(url)        # requeue: we were throttled, not refused
            self.frontier.push_host(host)
            return False
        self.frontier.push_host(host)
        if not body:
            return False
        title, text, links = HTMLExtract.parse(body, final_url or url)
        if len(text) < 200:
            self.stats["skip_thin"] += 1
            return False
        h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        if h in self.hashes:
            self.stats["dup_exact"] += 1
            return False
        self.hashes.add(h)

        did = self.n_pages
        self.n_pages += 1
        self.url_to_id[normalize(final_url or url) or url] = did
        clean = f"{title} {text}".replace("\t", " ").replace("\r", " ")
        pages_f.write(f"{did}\t{clean}\n")
        meta_f.write(json.dumps({"id": did, "url": final_url or url,
                                 "title": title, "sha1": h,
                                 "len": len(text)}) + "\n")
        for l in links[:200]:
            links_f.write(f"{final_url or url}\t{l}\n")
            self.frontier.add(l)
        self.stats["fetched"] += 1
        if self.on_page is not None:
            try:
                self.on_page(final_url or url, title, text)
            except Exception:
                self.stats["on_page_error"] += 1   # never fail the crawl
        if self.n_pages % 100 == 0:
            pages_f.flush(); meta_f.flush(); links_f.flush()
            print(f"{self.n_pages} pages | frontier {len(self.frontier)} "
                  f"| {dict(self.stats)}", flush=True)
        return False

    async def run(self, seeds: list[str]):
        import os
        os.makedirs(self.out_dir, exist_ok=True)
        for s in seeds:
            n = normalize(s)
            if n:
                self.frontier.add(n)
        conn = aiohttp.TCPConnector(limit=self.concurrency, ttl_dns_cache=300)
        t0 = time.time()
        with open(f"{self.out_dir}/pages.tsv", "w") as pf, \
             open(f"{self.out_dir}/meta.jsonl", "w") as mf, \
             open(f"{self.out_dir}/links.tsv", "w") as lf:
            async with aiohttp.ClientSession(
                    connector=conn, headers={"User-Agent": UA}) as session:
                await asyncio.gather(*[
                    self._worker(session, (pf, mf, lf))
                    for _ in range(self.concurrency)])
        with open(f"{self.out_dir}/crawl_stats.json", "w") as f:
            json.dump({"pages": self.n_pages, "elapsed_s": time.time() - t0,
                       "pages_per_s": self.n_pages / max(1e-9, time.time() - t0),
                       "hosts": len(self.robots), **self.stats}, f, indent=2)
        print(json.dumps({"pages": self.n_pages,
                          "elapsed_s": round(time.time() - t0, 1),
                          **self.stats}, indent=2))


def main() -> None:
    seeds_file, out_dir = sys.argv[1], sys.argv[2]
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    seeds = [l.strip() for l in open(seeds_file) if l.strip()
             and not l.startswith("#")]
    c = Crawler(out_dir, max_pages, conc)
    asyncio.run(c.run(seeds))


if __name__ == "__main__":
    main()
