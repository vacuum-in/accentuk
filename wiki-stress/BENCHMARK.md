# Benchmark: lang-uk lexical stress

Scored by the maintainers' own harness from
[`lang-uk/ukrainian-tts-preprocessing`](https://github.com/lang-uk/ukrainian-tts-preprocessing),
on its 1,026-sentence dataset, against the baseline it ships an evaluator for.

Reproduce the headline against the running service:

```
ml/.venv/bin/python ml/scripts/run_live_bench.py \
    --benchmark <clone of ukrainian-tts-preprocessing> --callers 8
```

Reproduce an ablation through the offline harness, which reimplements the tier
order in Python and so can try a change the deployment does not yet have:

```
ml/.venv/bin/python ml/scripts/run_langukbench.py \
    --benchmark <clone of ukrainian-tts-preprocessing> \
    --model output/ml/models/v19-v10 \
    --manifest output/ml/serving_manifest_v26.json \
    --extra-dataset 3 --extra-dataset 5 --extra-dataset 7 \
    --extra-dataset 9 --extra-dataset 10 --extra-dataset 12 \
    --spacy-model models/uk_core_news_sm --device cpu
```

The two do not agree, and when they disagree the live run is the one that
counts. Giving morphology every ambiguous token was worth +8.29 heteronym
points in the harness and −0.54 against the deployment: the harness ran a
different model over a reduced manifest and agreed with itself.

## Headline

Live HTTP stack, eight concurrent callers — the product, not the harness.

| metric | baseline `ukrainian-word-stress` | this pipeline | Δ |
| --- | ---: | ---: | ---: |
| Word accuracy | 88.67% | **95.77%** | **+7.10** |
| Heteronym accuracy | 64.34% | **83.10%** | **+18.76** |
| Macro-F1 (heteronyms) | 47.26% | **64.24%** | **+16.98** |
| Sentence accuracy | 41.52% | **69.30%** | **+27.78** |
| Unambiguous words | 98.60% | **99.46%** | **+0.86** |

Ahead on all five. The unambiguous-words deficit is closed: it was never the
ETL losing accuracy, it was generated datasets outranking the curated wordlist
and a defect that wrote the acute inside decomposed letters.

**Ceiling of this benchmark: 99.76%** — 1,232 of 1,235 heteronym tokens have
their gold reading somewhere in the lexicon, so the remaining 20 points are
resolution quality, not missing data.

## Baseline

```python
# lexical_stress_benchmark/examples/evaluate_ukrainian_word_stress.py
from ukrainian_word_stress import OnAmbiguity, Stressifier
stressify = Stressifier(stress_symbol="+", on_ambiguity=OnAmbiguity.First)
```

`ukrainian-word-stress` is a dictionary library with Stanza disambiguation and
"take the first reading" on ambiguity. It is also the source this project's
lexicon was built from, which makes it the fairest possible comparison: any
gain has to come from what the pipeline adds, not from better raw data.

## How it got there

Every row is the live API unless noted. Each line changes one thing.

| # | change | word | heteronym | macro-F1 |
| ---: | --- | ---: | ---: | ---: |
| 0 | session start | 76.16% | 55.29% | 31.52% |
| 1 | + supplementary datasets 3–6 | 84.61% | 53.98% | 31.42% |
| 2 | + lang-uk heteronym dictionary (ds 7) | 84.80% | 55.07% | 35.09% |
| 3 | + monosyllables, compounds, proper-noun demotion | 85.87% | 55.18% | 33.59% |
| 4 | + morphology timeout fixed (tier was silently dead) | 87.89% | 71.65% | 56.15% |
| 5 | + Stanza → spaCy `uk_core_news_sm` | 88.10% | 72.85% | 58.87% |
| 6 | + `upos=PRON` ≡ `DET` | 88.71% | 78.52% | 59.11% |
| 7 | + `й` counted as a vowel ordinal | 90.36% | 79.17% | 60.11% |
| 8 | + suffix fallback, trie defaults, `uk_core_news_trf`, inventory gap closed | 94.98% | 82.55% | 63.89% |
| 9 | + morphology fills the model's low-margin abstentions | 94.98% | 82.66% | 64.24% |
| 10 | + four reviewed readings, generated data no longer outranks them | 95.58% | 83.10% | 64.24% |
| 11 | + prepositional stress shift on governed pronouns | **95.77%** | **83.10%** | **64.24%** |

Row 8 collects the work between the two measurement sessions and is the only
line here that changes more than one thing; rows 9–11 are each a single change
with a paired per-token test behind it.

Sentence accuracy moves further than word accuracy on rows 10 and 11 — 65.20%
to 69.30% — because a corrected form was often the only error in its sentence,
and sentence accuracy is all-or-nothing.

## Ablations

Measured through the evaluation harness, one variable at a time.

| configuration | word | heteronym | macro-F1 |
| --- | ---: | ---: | ---: |
| no morphology tier at all | 87.59% | 62.38% | 44.53% |
| Stanza | 89.05% | 72.96% | 56.91% |
| spaCy `uk_core_news_sm` (15 MB) | 89.19% | 73.50% | 60.05% |
| spaCy `uk_core_news_lg` (231 MB) | 88.99% | 72.08% | 57.73% |
| spaCy `sm` + `PRON`≡`DET` | 89.83% | 79.28% | 60.29% |

Three results worth keeping:

* **Without a morphological tier the pipeline scores 62.38% on heteronyms —
  below the 64.34% baseline.** The contextual model alone does not beat a
  dictionary; the parser is what makes it worthwhile.
* **The large spaCy model is worse than the small one.** 231 MB bought −1.4
  heteronym points. Tagger capacity was never the constraint.
* **A tag-vocabulary mismatch was worth 5.8 points.** The trie writes
  `upos=PRON` for `цьому`, `всього`, `усі`; UD taggers write `DET`. Case,
  gender and number agreed exactly and the match was refused on the label
  alone.

## Concurrency

Stanza holds one parser and is not thread-safe. Under load the tier timed out
and degraded to dictionary defaults while the API still returned 200.

| tagger | 8 concurrent | serial |
| --- | ---: | ---: |
| Stanza | 72.16% word / 78.46% unambiguous | 87.89% / 95.64% |
| spaCy (pool of 4) | **88.10% / 95.64%** | — |

The fix was only possible because the model is 15 MB: a pool of four costs
~60 MB against ~2 GB.

## Where the remaining 20 points are

Of heteronym tokens the pipeline still gets wrong:

| owner | tokens | current accuracy |
| --- | ---: | ---: |
| morphology (grammatical alternations) | ~94 | 85.4% resolved, 74% correct |
| cross-encoder (semantic homographs) | 121 | ~90% |
| free variation — either answer is correct | 13 | — |
| genuinely unreachable | 3 | — |

The model is consulted on **1.47%** of running-text tokens. Adding training
data is the lowest-yield lever available: a retrain on 85k rows — mining 240k
sentences, generating 12.5k, labelling 42k — scored **−0.3** on this benchmark.
