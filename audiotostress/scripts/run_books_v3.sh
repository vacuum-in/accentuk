#!/usr/bin/env bash
# The whole book pipeline, v3: pilot every book, keep what passes, mine it.
#   run_books_v3.sh <only.json>     slugs to process (a JSON list)
set -eu
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ONLY="${1:?a JSON list of slugs}"
QUIET='warning|lightning|pyannote|allow_tf32|^ *>>>|re-enabled|^See |torch.from_numpy|nested|NLTK|incr_download|libtorchcodec|^ *$'

echo "=== PILOT $(date +%H:%M): two windows from the middle of every book"
.venv/bin/python -u scripts/run_mine_book_v3.py --plan artifacts/books/plan.json \
    --only "$ONLY" --suffix pilot3 --skip 5 --windows 2 < /dev/null 2>&1 \
  | stdbuf -oL grep -vE "$QUIET"

echo "=== REPORT $(date +%H:%M)"
.venv/bin/python scripts/run_pilot_report_v3.py --suffix pilot3 \
    --out artifacts/books/pilot_keep_v3.json | tee artifacts/books/pilot_v3.md

echo "=== MINE $(date +%H:%M): every book that passed, in full"
.venv/bin/python -u scripts/run_mine_book_v3.py --plan artifacts/books/plan.json \
    --only artifacts/books/pilot_keep_v3.json --suffix rows3 < /dev/null 2>&1 \
  | stdbuf -oL grep -vE "$QUIET"
echo "=== BOOKS V3 DONE $(date +%H:%M)"
