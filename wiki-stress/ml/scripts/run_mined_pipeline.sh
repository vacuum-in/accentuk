#!/usr/bin/env bash
# Mined corpus -> labelled -> merged -> trained -> measured, unattended.
#
# Ordering matters and each step has been got wrong separately:
#   * label the subtitle mine and the Malyuk mine into SEPARATE raw files, so a
#     resume never has to guess which corpus a batch key belongs to;
#   * assemble through corpus.build_rows so mined-from-anywhere is admitted on
#     the same terms as mined-from-Wikipedia;
#   * train before rebuilding the manifest -- the manifest names the model
#     version, and one naming a checkpoint that does not exist yet is how a
#     serving run gets pinned to the wrong inventory.
set -euo pipefail
cd /home/devops/wiki-stress
PY=ml/.venv/bin/python
INV=output/ml/ambiguous_forms_glossed.jsonl
GS=/mnt/c/lmfiles/ukrainian_homographs_1000.csv
GL=/mnt/c/lmfiles/ukrainian_homographs_1000_gold.csv
VERSION=v9-mined

echo "=== 1. wait for the subtitle labelling to finish ==="
while pgrep -f "scripts/run_label.py" >/dev/null; do sleep 60; done

echo "=== 2. wait for the Malyuk mine to finish ==="
while pgrep -f "scripts/run_mine_corpus.py" >/dev/null; do sleep 60; done
cat output/ml/mined_malyuk/report.json

echo "=== 3. label the Malyuk mine ==="
$PY ml/scripts/run_label.py \
    --mined output/ml/mined_malyuk/candidates.jsonl \
    --forms "$INV" --tag malyuk --cap-per-form 16 --floor-per-form 4

echo "=== 4. assemble every mined corpus ==="
$PY ml/scripts/assemble_mined.py \
    --candidates output/ml/mined_subs/candidates.jsonl \
    --candidates output/ml/mined_malyuk/candidates.jsonl \
    --label output/ml/raw/label_subs.jsonl \
    --label output/ml/raw/label_malyuk.jsonl \
    --verify output/ml/raw/verify_subs.jsonl \
    --verify output/ml/raw/verify_malyuk.jsonl \
    --inventory "$INV" \
    --out output/ml/mined_labelled.json

echo "=== 5. merge into the training corpus ==="
$PY ml/scripts/run_generate_balanced.py --rebuild-only || true
$PY ml/scripts/merge_corpus.py \
    --add output/ml/mined_labelled.json \
    --add output/ml/generated_balanced.json \
    --inventory "$INV" \
    --out output/ml/silver_mined_v4.json

echo "=== 6. train $VERSION ==="
$PY ml/scripts/run_finetune_inflected.py \
    --silver output/ml/silver_mined_v4.json \
    --inventory "$INV" \
    --base models/v3-xenc/checkpoint \
    --holdout-by form \
    --output "output/ml/models/$VERSION"

echo "=== 7. rebuild the serving manifest ==="
$PY ml/scripts/build_expanded_manifest.py \
    --inventory "$INV" \
    --inventory output/ml/uncovered_glossed.jsonl \
    --model-version "$VERSION" --threshold 2.0 \
    --output output/ml/serving_manifest_v9.json

echo "=== 8. gold evaluation ==="
$PY ml/scripts/run_gold_eval.py --sentences "$GS" --gold "$GL" \
    --model "output/ml/models/$VERSION" \
    --manifest output/ml/serving_manifest_v9.json \
    --extra-dataset 4 --out output/ml/gold_eval_v9

echo "=== 9. policy sweep ==="
$PY ml/scripts/score_gold.py --decisions output/ml/gold_eval_v9/decisions.json --sweep
