#!/usr/bin/env bash
# Package a trained model so work can continue on another machine.
#
# Separate from make_gpu_bundle.sh on purpose: that bundle is ~3 MB and travels
# anywhere, while this one carries 1.1 GB of weights. Keep them apart so moving
# the code and corpus never means moving a gigabyte.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:-models/v3-xenc}"
OUT="${2:-$ROOT/$(basename "$MODEL")-model.tar.gz}"

cd "$ROOT"

if [ ! -d "$MODEL/checkpoint" ]; then
  echo "no checkpoint at $MODEL/checkpoint" >&2
  exit 1
fi

# Provenance files travel with the weights: without them a checkpoint cannot be
# tied back to the corpus version and inventory hash it was trained against.
FILES=("$MODEL/checkpoint")
for meta in training_run.json evaluation.json release_report.json benchmark.json; do
  [ -f "$MODEL/$meta" ] && FILES+=("$MODEL/$meta")
done

tar -czf "$OUT" \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  "${FILES[@]}"

echo "bundle:  $OUT"
ls -lh "$OUT" | awk '{print "size:    " $5}'
echo "sha256:"
shasum -a 256 "$OUT" | awk '{print "  " $1}'
cat <<EOF

Restore on the other machine, from the repository root:

  tar -xzf $(basename "$OUT")

That recreates $MODEL/ with its checkpoint and provenance.

  # score with it
  ./ml/.venv/bin/python ml/scripts/run_final_report.py $MODEL output/ml/corpus/v3

  # continue training from these weights into a NEW directory
  ./ml/.venv/bin/python ml/scripts/run_crossencoder.py v3 $MODEL/checkpoint xenc2
EOF
