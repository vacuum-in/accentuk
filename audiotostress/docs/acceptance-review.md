# OpenSpec acceptance review

The implementation was reviewed against all scenarios in the four change specifications:

```bash
openspec validate build-ukrainian-audio-stress-mining --strict --no-interactive
uv run pytest
```

The review covers canonical schemas and migrations, Ukrainian normalization/lexicon candidate
handling, safe audio and alignment rejection, masked/unmasked ranker evaluation, manual-gold
calibration and precision gating, evidence-preserving mining records, license-aware export,
deterministic fingerprints, and resumable shard processing. Large-model paths remain optional;
the default test suite uses mocks and tiny fixtures.
