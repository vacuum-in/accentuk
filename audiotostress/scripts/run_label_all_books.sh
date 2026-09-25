#!/usr/bin/env bash
# Ask the ranker about every ambiguous and out-of-vocabulary word in the books.
#
# Runs only after run_audit_books.py has said the rows are measurements. The
# first time this was skipped, 348,835 labels were produced from vowel
# boundaries that scored at chance.
set -eu
cd /home/devops/audiotostress
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
# Same reason as the miner: on eight gigabytes the default allocator fragments
# what little the models leave, and every stage slows by the same factor.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=artifacts/books

.venv/bin/python - <<'PLAN' > /tmp/label_all.tsv
import json, pathlib
for b in json.load(open("artifacts/books/plan.json")):
    if pathlib.Path(f"artifacts/books/{b['slug']}.rows2.jsonl").exists():
        print(f"{b['slug']}\t{b['audio']}")
PLAN

total=$(wc -l < /tmp/label_all.tsv)
index=0
while IFS=$'\t' read -r slug audio; do
  index=$((index + 1))
  out="$OUT/$slug.labels2.jsonl"
  if [ -s "$out" ]; then
    echo "=== $index/$total $slug — already labelled, skipping"
    continue
  fi
  echo "=== $index/$total $slug $(date +%H:%M)"
  timeout 4h .venv/bin/python -u scripts/run_label_books2.py \
      --rows "$OUT/$slug.rows2.jsonl" --audio "$audio" --out "$out" \
      < /dev/null 2>&1 \
    | stdbuf -oL grep -viE "warning|torch\._|consider|^ *$"
done < /tmp/label_all.tsv
echo "=== LABEL ALL DONE $(date +%H:%M) ==="
