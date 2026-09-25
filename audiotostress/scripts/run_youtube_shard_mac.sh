#!/usr/bin/env bash
# A Mac's share of the YouTube mining: pull audio folders from the laptop,
# mine its shard with mlx-whisper + torch on mps, push the rows back.
#   run_youtube_shard_mac.sh <k/n or a-b/n>       e.g. 5/8
# Same loop as the desktop's; the machines differ only in --asr and --device.
set -u
cd "$(dirname "$0")/.."
SHARD="${1:?shard, e.g. 5/8}"
LAPTOP="${LAPTOP:-user@gpu-host}"
Q='warning|^\s*$'
mkdir -p youtube
pull() { rsync -a --ignore-existing --include='*/' --include='audio.*' --include='meta.json' --exclude='*' "$LAPTOP:audiotostress/youtube/" youtube/ 2>/dev/null; }
push() { rsync -a --include='*.rows4.jsonl' --exclude='*' youtube/ "$LAPTOP:audiotostress/youtube/" 2>/dev/null; }
mine() { .venv/bin/python -u scripts/run_mine_audio_v4.py --folder youtube --suffix rows4 --shard "$SHARD" \
           --asr mlx --device mps < /dev/null 2>&1 | grep -vE "$Q"; }
while true; do
  pull
  echo "=== pass $(date +%H:%M): $(ls youtube | wc -l | tr -d ' ') folders local, shard $SHARD"
  mine; push
  ssh "$LAPTOP" 'pgrep -f run_fetch_youtube >/dev/null' || { pull; break; }
  sleep 300
done
mine; push
echo "=== MAC SHARD $SHARD DONE $(date +%H:%M)"
