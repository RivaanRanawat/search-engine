#!/usr/bin/env bash
# Completes the hybrid (v10) loop unattended once the corpus embedding job
# finishes: verifies the dense index, evaluates it on sealed dev, and
# re-measures BM25/phrase latency on a now-quiet machine (earlier numbers
# were taken while the embedding job thrashed the page cache).
#
# Safe to run at any time: it waits for completion, and every step is
# idempotent. If the embedding job is not running, it starts it (the job
# itself skips finished shards).
#
#   nohup bash scripts/finish_hybrid.sh > /tmp/finish_hybrid.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

DENSE=indexes/dense
IDX=indexes/v1s

echo "[$(date)] waiting for the corpus embedding job ..."
while true; do
  pct=$(python3 -m searchengine.embed_corpus status "$DENSE" \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['pct'])")
  done_n=$(python3 -m searchengine.embed_corpus status "$DENSE" \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['shards_done'])")
  total=$(python3 -m searchengine.embed_corpus status "$DENSE" \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['shards_total'])")
  echo "[$(date)] $done_n/$total shards ($pct%)"
  [ "$done_n" = "$total" ] && break
  # restart the encoder if it died (resumable by design: finished shards are
  # skipped, an interrupted shard is simply redone)
  if ! pgrep -f "embed_corpus run" > /dev/null; then
    echo "[$(date)] encoder not running — restarting"
    ENC_THREADS="${ENC_THREADS:-4}" nohup python3 -m searchengine.embed_corpus run "$DENSE" \
      >> /tmp/embed_job.log 2>&1 &
  fi
  sleep 300
done

echo "[$(date)] verifying no unwritten rows ..."
python3 -m searchengine.embed_corpus verify "$DENSE" | tee /tmp/dense_verify.json

echo "[$(date)] full sealed-dev hybrid evaluation ..."
python3 -m bench.hybrid.bench_hybrid "$IDX" "$DENSE" 2>&1 | tail -30

echo "[$(date)] re-measuring lexical latency on a quiet machine ..."
python3 -m bench.v1.bench_v1 data "$IDX" bench/results/v11_quiet_bm25.json \
  2>&1 | tail -20

echo "[$(date)] regression suite ..."
python3 -m pytest tests/ -q 2>&1 | tail -3
echo "[$(date)] done. Results in bench/results/."
