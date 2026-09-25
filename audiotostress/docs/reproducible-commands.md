# Reproducible commands

All commands use the locked `uv.lock` environment and should be run from the repository root.

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy src
openspec validate build-ukrainian-audio-stress-mining --strict --no-interactive
```

Typical pipeline stages are exposed as Python APIs so jobs can pin their input and output
fingerprints:

```bash
uv run python -c 'from ukstress.pipeline import build_bootstrap_examples'
uv run python -c 'from ukstress.training import train_ranker_step'
uv run python -c 'from ukstress.evaluation import evaluate_rows, write_evaluation_report'
uv run python -c 'from ukstress.calibration import precision_coverage_curve'
uv run python -c 'from ukstress.mining import discover_ambiguous_candidates, write_mining_outputs'
uv run python -c 'from ukstress.export import export_text_resolver'
```

Model-dependent WhisperX/SSL/Canary integration is opt-in through the corresponding extras and
uses the configured Hugging Face cache. Keep `model_cache_only: true` for offline reruns.
