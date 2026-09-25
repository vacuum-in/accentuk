#!/usr/bin/env bash
# Corpus from all three passes, then the token classifier.
set -eu
cd /home/devops/wiki-stress
until grep -q "PASS FINISH DONE" /home/devops/audiotostress/artifacts/books/pass2_finish.log; do sleep 120; done
# Pass 3 has no chained finish; audit and label it here.
cd /home/devops/audiotostress
bash scripts/run_pass_finish.sh rows22
cd /home/devops/wiki-stress
echo "=== build tok-v2 $(date +%H:%M)"
ml/.venv/bin/python ml/scripts/build_book_corpus.py --out output/ml/corpus/tok-v2 2>&1 | tail -12
echo "=== train tok-v2 $(date +%H:%M)"
NV="ml/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$PWD/$NV/cublas/lib:$PWD/$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml/.venv/bin/python -u ml/scripts/run_train_token_resolver.py \
  --corpus output/ml/corpus/tok-v2 --out models/tok-v2 --epochs 4 2>&1 \
  | grep -viE "warning|torch\._|consider"
echo "=== OVERNIGHT DONE $(date +%H:%M)"
