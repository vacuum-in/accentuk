#!/usr/bin/env bash
# Re-mine every book with the ASR-first miner.
#
# The first run is void: its vowel boundaries scored at chance on the
# longest-vowel baseline, because the anchors drift by hundreds of words and
# forced alignment placed whatever word list it was handed. See PLAN_BOOKS.md.
#
# Every step reads /dev/null: ffmpeg consumes stdin and will eat this loop's
# own input, which once made the driver skip from book 1 to book 6 while the
# counter still said 3.
set -eu
cd /home/devops/audiotostress
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
# Eight gigabytes hold all three models with almost nothing to spare, and the
# default allocator fragments that remainder into uselessness within one
# window. Expandable segments keep it usable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=artifacts/books
STRIDE="${STRIDE:-1}"
SKIP="${SKIP:-0}"
SUFFIX="${SUFFIX:-rows2}"

# Only the books the pilot proved. Three of the twenty-seven produce no rows
# at all — the Machiavelli has no matching text, the Hugo and the Norwood have
# the wrong edition — and mining them would spend fifty hours of audio to
# discover that a second time.
.venv/bin/python - <<'PLAN' > /tmp/mine_all.tsv
import json
keep = set(json.load(open("artifacts/books/pilot_keep.json")))
for b in json.load(open("artifacts/books/plan.json")):
    if b["slug"] in keep:
        print(f"{b['slug']}\t{b['audio']}\t{b['hours']}")
PLAN

total=$(wc -l < /tmp/mine_all.tsv)
index=0
while IFS=$'\t' read -r slug audio hours; do
  index=$((index + 1))
  rows="$OUT/$slug.$SUFFIX.jsonl"
  if [ -s "$rows" ]; then
    echo "=== $index/$total $slug — already mined, skipping"
    continue
  fi
  if [ ! -s "$OUT/$slug.anchors.jsonl" ]; then
    echo "=== $index/$total $slug — no anchors, skipping"
    continue
  fi
  echo "=== $index/$total $slug ($hours h) $(date +%H:%M)"
  rm -f "$rows"
  # A book that stalls must not hold the other twenty-six.
  timeout 6h .venv/bin/python -u scripts/run_mine_book_asr.py \
      --book "$slug" \
      --anchors "$OUT/$slug.anchors.jsonl" \
      --text "$OUT/$slug.jsonl" \
      --audio "$audio" \
      --out "$rows" \
      --stride "$STRIDE" --skip "$SKIP" \
      --asr-model large-v3-turbo < /dev/null 2>&1 \
    | stdbuf -oL grep -viE "warning|lightning|pyannote|allow_tf32|^ *>>>|re-enabled|^See |torch.from_numpy|^ *$"
done < /tmp/mine_all.tsv
echo "=== MINE ALL DONE $(date +%H:%M) ==="
