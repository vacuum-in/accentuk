# ukstress

`ukstress` is a Python 3.11 pipeline for mining high-precision Ukrainian lexical-stress
examples from speech. Development follows the OpenSpec change in
`openspec/changes/build-ukrainian-audio-stress-mining/`.

Install the locked development environment and run checks with:

```bash
uv sync --dev --extra alignment
uv run pytest
uv run ruff check .
uv run mypy src
```

Word and vowel alignment default to WhisperX with the Ukrainian
`Yehor/wav2vec2-xls-r-300m-uk-with-small-lm` character-CTC checkpoint. WhisperX emits timed
characters; the pipeline retains timed vowel characters for acoustic-stress features. The model is downloaded to
the Hugging Face cache on first use; set `model_cache_only: true` after prefetching for offline,
reproducible runs.

The checked-in `uv.lock` is the authoritative dependency lock. Large models, speech corpora,
and the versioned stress lexicon remain external inputs and are never downloaded by default.
