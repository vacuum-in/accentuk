#!/usr/bin/env bash
# Prove every book before mining any book.
#
# The first book run mined 152 hours, labelled 348,835 words, and was then
# found to be noise. The check that would have caught it costs two windows per
# book. This runs that check on all twenty-seven, from the middle of each book
# rather than the opening, because openings are publisher boilerplate that the
# narrator may not read at all.
#
# Two numbers come out per book:
#   * the longest vowel against the lexicon — no model, no labels. Real
#     boundaries score near 60%; the void run scored 36-39% against 35-38%
#     chance.
#   * the ranker against the lexicon — what the labels will actually be worth.
set -eu
cd /home/devops/audiotostress
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
OUT=artifacts/books
WINDOWS="${WINDOWS:-2}"
SKIP="${SKIP:-5}"

.venv/bin/python - <<'PLAN' > /tmp/pilot.tsv
import json, pathlib
for b in json.load(open("artifacts/books/plan.json")):
    if pathlib.Path(f"artifacts/books/{b['slug']}.anchors.jsonl").exists():
        print(f"{b['slug']}\t{b['audio']}\t{b['hours']}")
PLAN

total=$(wc -l < /tmp/pilot.tsv)
index=0
while IFS=$'\t' read -r slug audio hours; do
  index=$((index + 1))
  rows="$OUT/$slug.pilot.jsonl"
  echo "=== $index/$total $slug ($hours h) $(date +%H:%M)"
  if [ ! -s "$rows" ]; then
    rm -f "$rows"
    timeout 40m .venv/bin/python -u scripts/run_mine_book_asr.py \
        --book "$slug" \
        --anchors "$OUT/$slug.anchors.jsonl" \
        --text "$OUT/$slug.jsonl" \
        --audio "$audio" \
        --out "$rows" \
        --skip "$SKIP" --windows "$WINDOWS" \
        --asr-model large-v3-turbo < /dev/null 2>&1 \
      | stdbuf -oL grep -E "kept|chunks|counts|Error|error" || true
  fi
  if [ -s "$rows" ] && [ ! -s "$OUT/$slug.pilot.calib.jsonl" ]; then
    timeout 20m .venv/bin/python -u scripts/run_label_books2.py \
        --rows "$rows" --audio "$audio" \
        --out "$OUT/$slug.pilot.calib.jsonl" --calibrate 4000 \
        < /dev/null 2>&1 | stdbuf -oL grep -E "overall|rows$" || true
  fi
done < /tmp/pilot.tsv
echo "=== PILOT DONE $(date +%H:%M) ==="
