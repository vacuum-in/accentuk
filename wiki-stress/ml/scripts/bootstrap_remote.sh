#!/usr/bin/env bash
# Bring a freshly-synced checkout to a runnable state.
#
# The host may ship a Python older than this project supports (>=3.12), so uv
# provisions the interpreter rather than relying on the system one.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "uv not on PATH; add \$HOME/.local/bin" >&2; exit 1; }

uv venv --python 3.12 ml/.venv

# CUDA wheel must land before the project, or the CPU wheel is resolved as a
# dependency and silently wins.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected — installing CUDA torch"
  VIRTUAL_ENV=ml/.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "no GPU detected — installing CPU torch"
  VIRTUAL_ENV=ml/.venv uv pip install torch
fi

VIRTUAL_ENV=ml/.venv uv pip install -e etl -e ml

echo
echo "=== verification ==="
./ml/.venv/bin/python - <<'PY'
import torch, transformers
from ukstress.normalizer import lookup_key, stress_signature
from ukstress_ml import ambiguity, corpus
from pathlib import Path
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("canonicalization reuse:", lookup_key("За́мок"), stress_signature("замо́к"))
for split in ("train", "dev", "test_natural"):
    p = Path(f"output/ml/corpus/v3/{split}.jsonl")
    print(f"corpus {split}: {len(corpus.load_split(p)):,}" if p.exists() else f"corpus {split}: MISSING")
print("ambiguous forms:", len(ambiguity.load(Path("output/ml/ambiguous_forms.jsonl"))))
PY

./ml/.venv/bin/python -m pytest ml/tests -q

cat <<'EOF'

Ready. Note that annotation and generation need Azure credentials, which were
deliberately NOT transferred. Copy .env.example to .env and fill it in on this
host if you intend to run `run_label.py` or `run_generate.py`.

  score the released model:
    ./ml/.venv/bin/python ml/scripts/run_final_report.py models/v3-xenc output/ml/corpus/v3
  continue training into a new run directory:
    ./ml/.venv/bin/python ml/scripts/run_crossencoder.py v3 models/v3-xenc/checkpoint xenc2
EOF
