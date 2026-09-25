#!/usr/bin/env bash
# After a mining pass: audit it, label it, start the next.
set -eu
cd /home/devops/audiotostress
NV="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SUFFIX="$1"          # rows2b
NEXT_SKIP="${2:-}"   # 2 -> start pass 3; empty -> nothing after
OUT=artifacts/books

echo "=== audit $SUFFIX $(date +%H:%M)"
.venv/bin/python scripts/run_audit_books.py --rows $OUT/*.$SUFFIX.jsonl \
  | tee "$OUT/audit_$SUFFIX.md" | tail -8
if grep -q "below the floor" "$OUT/audit_$SUFFIX.md"; then
  echo "AUDIT FAILED for $SUFFIX — not labelling"; exit 1
fi

echo "=== label $SUFFIX $(date +%H:%M)"
.venv/bin/python - "$SUFFIX" <<'PLAN' > /tmp/label_pass.tsv
import json, pathlib, sys
suffix = sys.argv[1]
for b in json.load(open("artifacts/books/plan.json")):
    if pathlib.Path(f"artifacts/books/{b['slug']}.{suffix}.jsonl").exists():
        print(f"{b['slug']}\t{b['audio']}")
PLAN
while IFS=$'\t' read -r slug audio; do
  out="$OUT/$slug.labels_$SUFFIX.jsonl"
  [ -s "$out" ] && continue
  echo "--- $slug $(date +%H:%M)"
  timeout 4h .venv/bin/python -u scripts/run_label_books2.py \
      --rows "$OUT/$slug.$SUFFIX.jsonl" --audio "$audio" --out "$out" \
      < /dev/null 2>&1 | stdbuf -oL grep -E "rows$" || true
done < /tmp/label_pass.tsv
echo "=== label $SUFFIX done $(date +%H:%M)"

if [ -n "$NEXT_SKIP" ]; then
  echo "=== pass with skip $NEXT_SKIP $(date +%H:%M)"
  SKIP="$NEXT_SKIP" STRIDE=3 SUFFIX="rows2${NEXT_SKIP}" bash scripts/run_mine_all_books.sh
fi
echo "=== PASS FINISH DONE $(date +%H:%M)"
