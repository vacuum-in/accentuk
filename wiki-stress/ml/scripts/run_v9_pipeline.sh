#!/usr/bin/env bash
# Corpus -> model -> manifest -> gold measurement, in one pass.
#
# The steps have to happen in this order and each one has been got wrong
# separately before:
#
#   1. harvest raw generations (no re-spend; the validator may have changed)
#   2. merge into the training corpus, keeping provenance
#   3. train -- corpus and inventory are the ONLY things that differ from v8,
#      so the comparison isolates the balanced data
#   4. rebuild the serving manifest against the new model version
#   5. re-collect gold evidence, then sweep policy over it
#
# Step 3 must precede step 4: the manifest names the model version, and a
# manifest naming a checkpoint that does not exist yet is how a serving run
# ends up pinned to the wrong inventory.
set -euo pipefail
cd /home/devops/wiki-stress
PY=ml/.venv/bin/python
GOLD_SENTENCES=/mnt/c/lmfiles/ukrainian_homographs_1000.csv
GOLD_LABELS=/mnt/c/lmfiles/ukrainian_homographs_1000_gold.csv
INVENTORY=output/ml/ambiguous_forms_glossed.jsonl
VERSION=v9-balanced

echo "=== 1. harvest generated sentences ==="
$PY ml/scripts/run_generate_balanced.py --rebuild-only

echo "=== 2. merge corpus ==="
$PY ml/scripts/merge_corpus.py --out output/ml/silver_balanced_v3.json

echo "=== 3. train $VERSION ==="
$PY ml/scripts/run_finetune_inflected.py \
    --silver output/ml/silver_balanced_v3.json \
    --inventory "$INVENTORY" \
    --base models/v3-xenc/checkpoint \
    --holdout-by form \
    --output "output/ml/models/$VERSION"

echo "=== 4. rebuild serving manifest ==="
$PY ml/scripts/build_expanded_manifest.py \
    --inventory "$INVENTORY" \
    --inventory output/ml/uncovered_glossed.jsonl \
    --model-version "$VERSION" \
    --threshold 2.0 \
    --output output/ml/serving_manifest_v9.json

echo "=== 5. gold evaluation ==="
$PY ml/scripts/run_gold_eval.py \
    --sentences "$GOLD_SENTENCES" --gold "$GOLD_LABELS" \
    --model "output/ml/models/$VERSION" \
    --manifest output/ml/serving_manifest_v9.json \
    --extra-dataset 4 \
    --out output/ml/gold_eval_v9

echo "=== 6. policy sweep ==="
$PY ml/scripts/score_gold.py --decisions output/ml/gold_eval_v9/decisions.json --sweep
