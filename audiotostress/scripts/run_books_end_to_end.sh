#!/usr/bin/env bash
# Mine, audit, label. The audit is not optional and not advisory: a book whose
# longest-vowel baseline is near chance has boundaries that are not
# measurements, and labelling it produces confident nonsense.
set -eu
cd /home/devops/audiotostress
bash scripts/run_mine_all_books.sh
echo
.venv/bin/python scripts/run_audit_books.py --rows artifacts/books/*.rows2.jsonl \
  | tee artifacts/books/audit.md
echo
bash scripts/run_label_all_books.sh
