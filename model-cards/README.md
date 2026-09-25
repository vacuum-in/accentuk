# Model cards: Ukrainian stress pipeline

Three trained models serve the stress pipeline in `wiki-stress`. Each card
describes one model. How they fit together is in `wiki-stress/docs/architecture.md`.

| card | model | role |
| --- | --- | --- |
| [`uk-stress-crossencoder`](uk-stress-crossencoder/README.md) | XLM-R cross-encoder `v19-v10` | picks the sense of a homograph from its gloss |
| [`uk-stress-token-classifier`](uk-stress-token-classifier/README.md) | XLM-R-uk token classifier `tok-v5` | picks the stress of a frequent ambiguous form from its sentence |
| [`uk-stress-combiner`](uk-stress-combiner/README.md) | learned combiner `combiner-v1` | weighs every tier's opinion and may overrule the tier order |

## Whole-pipeline results

Measured on 2026-09-25 against the live API. These are the pipeline's
numbers, not any one model's.

| test | combiner off (default) | combiner on |
| --- | ---: | ---: |
| top-200 ambiguous forms of Common Voice uk, audio gold at ranker confidence ≥ 0.99 (13,201 tokens) | 95.88% | 97.27% |
| lang-uk stress benchmark, words / heteronyms (1,026 sentences, human gold) | 95.92% / 84.62% | 96.02% / 85.28% |
| modern-text gold, 3,887 words (306 sentences; 42 reviewer-corrected) | 98.07% | 97.27% |

The datasets are not published. The cards say what each model was trained on
so that its behaviour can be understood.
