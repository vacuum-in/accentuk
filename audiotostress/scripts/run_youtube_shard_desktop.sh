#!/usr/bin/env bash
# The desktop's share of the YouTube mining: pull audio folders from the
# laptop as they land, mine its shard on the 3070, push the rows back.
#   run_youtube_shard_desktop.sh <k/n or a-b/n>   default 10/16
# Eight times realtime against the 5090's sixty-five, but it runs alongside.
set -u
SHARD="${1:-10/16}"
cd /home/devops/audiotostress
export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LAPTOP=user@gpu-host
Q='warning|lightning|pyannote|allow_tf32|^ *>>>|re-enabled|^See |torch.from_numpy|nested|NLTK|incr_download|libtorchcodec|^ *$'
mkdir -p youtube
while true; do
  # Only complete folders (meta.json is written last), audio + meta only.
  rsync -a --ignore-existing --include='*/' --include='audio.*' --include='meta.json' --exclude='*' \
      "$LAPTOP:audiotostress/youtube/" youtube/ 2>/dev/null
  echo "=== pass $(date +%H:%M): $(ls youtube | wc -l) folders local"
  .venv/bin/python -u scripts/run_mine_audio_v4.py --folder youtube --suffix rows4 --shard "$SHARD" \
      --batch 8 < /dev/null 2>&1 | stdbuf -oL grep -vE "$Q"
  rsync -a --include='*.rows4.jsonl' --exclude='*' youtube/ "$LAPTOP:audiotostress/youtube/" 2>/dev/null
  ssh "$LAPTOP" 'pgrep -f run_fetch_youtube >/dev/null' || { rsync -a --ignore-existing --include='*/' --include='audio.*' --include='meta.json' --exclude='*' "$LAPTOP:audiotostress/youtube/" youtube/ 2>/dev/null; [ "$(ls youtube | wc -l)" -le "$(ssh $LAPTOP 'ls audiotostress/youtube | wc -l')" ] && break; }
  sleep 300
done
# one last pass over anything that arrived at the end
.venv/bin/python -u scripts/run_mine_audio_v4.py --folder youtube --suffix rows4 --shard "$SHARD" --batch 8 < /dev/null 2>&1 | stdbuf -oL grep -vE "$Q"
rsync -a --include='*.rows4.jsonl' --exclude='*' youtube/ "$LAPTOP:audiotostress/youtube/"
echo "=== DESKTOP SHARD DONE $(date +%H:%M)"
