#!/usr/bin/env bash
# YouTube -> audio -> stress rows, end to end.
#   run_youtube_pipeline.sh <sources.txt> [--pilot]
# Fetches what the list names (URLs, playlists, channels, ytsearchN:query),
# then mines every fetched folder with the book-less miner. --pilot mines two
# windows from the middle of each source only, prints the per-source table,
# and stops: the rule is the same as for the books — prove a source before
# spending hours on it.
set -eu
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LIST="${1:?a file of sources}"; MODE="${2:-}"
QUIET='warning|lightning|pyannote|allow_tf32|^ *>>>|re-enabled|^See |torch.from_numpy|nested|NLTK|incr_download|libtorchcodec|^ *$'

echo "=== FETCH $(date +%H:%M)"
.venv/bin/python scripts/run_fetch_youtube.py --list "$LIST" --out youtube ${PER_SOURCE:+--per-source $PER_SOURCE} ${COOKIES:+--cookies $COOKIES} ${BROWSER:+--cookies-from-browser $BROWSER} --sleep "${SLEEP:-3}"

if [ "$MODE" = "--pilot" ]; then
  echo "=== PILOT $(date +%H:%M): two windows from the middle of each source"
  .venv/bin/python -u scripts/run_mine_audio_v4.py --folder youtube --suffix pilot4 \
      --skip 3 --windows 2 < /dev/null 2>&1 | stdbuf -oL grep -vE "$QUIET"
  .venv/bin/python scripts/run_pilot_report_v3.py --books youtube --suffix pilot4 \
      --out youtube/pilot_keep.json | tee youtube/pilot.md
  exit 0
fi

echo "=== MINE $(date +%H:%M)"
.venv/bin/python -u scripts/run_mine_audio_v4.py --folder youtube --suffix rows4 \
    < /dev/null 2>&1 | stdbuf -oL grep -vE "$QUIET"
echo "=== YOUTUBE DONE $(date +%H:%M)"
