#!/usr/bin/env bash
# Promote a training run to a released, LFS-versioned model.
#
# `output/ml/models/` holds every training run and is gitignored. `models/`
# holds only released artifacts and is tracked with Git LFS. Promotion is an
# explicit step so experiments never enter history by accident.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${1:-output/ml/models/v3-xenc}"
NAME="${2:-$(basename "$SRC")}"
DEST="models/$NAME"

cd "$ROOT"

if [ ! -d "$SRC/checkpoint" ]; then
  echo "no checkpoint at $SRC/checkpoint" >&2
  exit 1
fi
if [ -e "$DEST" ]; then
  echo "$DEST already exists; remove it or choose another name" >&2
  exit 1
fi

mkdir -p "$DEST"
# Move rather than copy: the checkpoint is ~1.1 GB and duplicating it wastes
# disk for no benefit — the release copy is the one that matters.
mv "$SRC/checkpoint" "$DEST/checkpoint"
for meta in training_run.json evaluation.json release_report.json benchmark.json; do
  [ -f "$SRC/$meta" ] && cp "$SRC/$meta" "$DEST/$meta"
done

git add "$DEST" >/dev/null
echo "promoted $SRC -> $DEST"
echo
echo "LFS-tracked files staged:"
git lfs ls-files --name-only 2>/dev/null | sed 's/^/  /' || true
echo
du -sh "$DEST" | awk '{print "size on disk: " $1}'
