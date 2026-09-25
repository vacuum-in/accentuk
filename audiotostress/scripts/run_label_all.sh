#!/usr/bin/env bash
# Label every book's ambiguous and OOV words with the audio ranker.
# Resumable: a book whose .labels.jsonl exists is skipped.
set -u
cd /home/devops/audiotostress
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
.venv/bin/python - <<'PLAN' > /tmp/labellist.tsv
import json
for b in json.load(open("artifacts/books/plan.json")):
    print(f"{b['slug']}\t{b['audio']}")
PLAN
total=$(wc -l < /tmp/labellist.tsv); i=0
while IFS=$'\t' read -r slug audio; do
  i=$((i+1))
  rows="artifacts/books/$slug.rows.jsonl"
  out="artifacts/books/$slug.labels.jsonl"
  [ -s "$rows" ] || { echo "=== [$i/$total] $slug — no rows, skipping"; continue; }
  [ -s "$out" ] && { echo "=== [$i/$total] $slug — labels present, skipping"; continue; }
  echo "=== [$i/$total] $slug ==="
  .venv/bin/python -u scripts/run_label_books.py --rows "$rows" --audio "$audio" \
      --anchors "artifacts/books/$slug.anchors.jsonl" --out "$out" < /dev/null 2>&1 \
      | stdbuf -oL grep -viE "warning|torch\._|output =|consider"
done < /tmp/labellist.tsv
echo "=== ALL LABELS DONE ==="
