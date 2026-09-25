#!/usr/bin/env bash
# The laptop's part of the YouTube mining: sweep the folder from the front in
# batches while any fetch (here or on the desktop) is still adding audio, then
# a final sweep. Existing rows are skipped, so what the desktop pushed back is
# not redone; the desktop sweeps from the back, and the two meet in the middle.
#   run_youtube_shard_laptop.sh [batch]   default 20 sources per invocation
set -u
cd "$(dirname "$0")/.."
BATCH="${1:-20}"
export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q='warning|lightning|pyannote|allow_tf32|^ *>>>|re-enabled|^See |torch.from_numpy|nested|NLTK|incr_download|libtorchcodec|^ *$'
mine() { .venv/bin/python -u scripts/run_mine_audio_v4.py --folder youtube --suffix rows4 --shard 0/1 \
           --max-sources "$BATCH" < /dev/null 2>&1 | stdbuf -oL grep -vE "$Q" | tee /tmp/laptop_pass.log; }
# The desktop (behind WSL's NAT) and the Macs announce their fetch by touching
# a marker here every pass; a marker older than 15 min means that fetch is done.
fetching() { pgrep -f run_fetch_youtube.py >/dev/null || [ -n "$(find youtube -maxdepth 1 -name '.*_fetching' -mmin -15 2>/dev/null)" ]; }
while fetching; do
  echo "=== pass $(date +%H:%M): $(ls youtube/*/meta.json 2>/dev/null | wc -l) fetched, $(ls youtube/*.rows4.jsonl 2>/dev/null | wc -l) mined"
  mine
  grep -q "mined 0$" /tmp/laptop_pass.log && sleep 120
done
echo "=== fetch over, final sweep $(date +%H:%M)"
BATCH=0 mine
echo "=== LAPTOP DONE $(date +%H:%M)"
