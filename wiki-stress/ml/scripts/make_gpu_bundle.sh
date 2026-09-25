#!/usr/bin/env bash
# Package everything needed to continue training on a CUDA machine.
#
# Excludes the 2.7 GB Wikipedia dump and the trained checkpoints: mining and
# annotation are already done and their outputs are in the corpus, and the GPU
# run starts from the pretrained base model rather than from these checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/ukstress-gpu-bundle.tar.gz}"

cd "$ROOT"

# BSD tar applies --exclude only to paths listed after it, so the flags come first.
tar -czf "$OUT" \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='*.pyc' \
  ml/pyproject.toml \
  ml/README.md \
  ml/CUDA.md \
  ml/src \
  ml/scripts \
  ml/tests \
  etl/pyproject.toml \
  etl/src \
  output/ml/corpus/v3 \
  output/ml/inventory_v1.jsonl \
  output/ml/inventory_v1.manifest.json \
  output/ml/ambiguous_forms.jsonl \
  output/ml/ambiguous_forms.report.json \
  output/ml/generated/generation_report.json \
  output/ml/mined/ukwiki/coverage.json \
  RESULTS.md \
  ANALYSIS_TRIAGE.md \
  PLAN_DATASET.md \
  PLAN_MARIAN_HOMOGRAPHS.md

echo "bundle: $OUT"
ls -lh "$OUT" | awk '{print "size:", $5}'
echo
echo "sha256:"
shasum -a 256 "$OUT" | awk '{print "  " $1}'
