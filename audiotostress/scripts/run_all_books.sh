#!/usr/bin/env bash
# Anchor and mine every book, in order of length, resumably.
#
# Every step reads from /dev/null. Without that, ffmpeg inside the miner eats
# the `while read` loop's own input and books are skipped silently — the run
# jumped from book one to book six and the log still counted them 1, 2, 3.
#
# Each book is two steps and each writes one file, so a run that stops leaves
# finished books finished: the loop skips any book whose outputs already exist.
# Shortest first, so the corpus starts growing early and a mistake shows up on
# a ten-hour book rather than a twenty-three-hour one.
set -u
cd /home/devops/audiotostress
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
# faster-whisper links against CUDA 12; this torch build ships CUDA 13 but the
# 12 libraries it needs are inside the wheel.
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
PY=.venv/bin/python
OUT=artifacts/books

$PY - <<'PLAN' > /tmp/booklist.tsv
import json
for b in json.load(open("artifacts/books/plan.json")):
    print(f"{b['slug']}\t{b['audio']}\t{b['text']}\t{b['hours']}\t{b['name']}")
PLAN

total=$(wc -l < /tmp/booklist.tsv); done_n=0
while IFS=$'\t' read -r slug audio text hours name; do
  done_n=$((done_n+1))
  echo "=== [$done_n/$total] $name (${hours}h) ==="
  if [ ! -s "$OUT/$slug.anchors.jsonl" ]; then
    $PY -u scripts/run_anchor_book.py --audio "$audio" --text "$text" \
        --out "$OUT/$slug.anchors.jsonl" < /dev/null 2>&1 \
        | stdbuf -oL grep -viE "warning|torch\._|output =|consider"
  else
    echo "  anchors present, skipping"
  fi
  if [ -s "$OUT/$slug.anchors.jsonl" ] && [ ! -s "$OUT/$slug.rows.jsonl" ]; then
    $PY -u scripts/run_mine_book.py --audio "$audio" --anchors "$OUT/$slug.anchors.jsonl" \
        --book "$slug" --out "$OUT/$slug.rows.jsonl" < /dev/null 2>&1 \
        | stdbuf -oL grep -viE "warning|torch\._|output =|consider|audio = torch"
  elif [ -s "$OUT/$slug.rows.jsonl" ]; then
    echo "  rows present, skipping"
  fi
done < /tmp/booklist.tsv
echo "=== ALL BOOKS DONE ==="
