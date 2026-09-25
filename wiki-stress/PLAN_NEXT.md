# Next steps — 2026-08-27

Written after the pipeline was first measured on an external benchmark
(`lang-uk/ukrainian-tts-preprocessing`) against a published baseline. That
comparison reorders the priorities, and the reordering is the point of this
document.

## Where things stand

| | word acc | heteronym | macro-F1 |
| --- | --- | --- | --- |
| `ukrainian-word-stress` (baseline) | 88.67% | 64.34% | 47.26% |
| this pipeline | **89.83%** | **79.28%** | **60.29%** |
| ceiling of this benchmark | — | 99.76% | — |

20 points of heteronym headroom remain, and the data to close them is already
in the lexicon — this is resolution quality, not coverage.

## The lesson that should shape what comes next

Ranked by measured effect, the session's gains were: five bug fixes, two data
imports, one component swap, and one retrain that scored **−0.3**.

The retrain consumed hours of GPU and thousands of API calls across mining,
generation and labelling. The model is consulted on **1.47%** of running-text
tokens and is right about 90% of the time on them. **Adding training data is
the lowest-yield lever available and should not be the default response to a
low number.**

Before any further model work, measure which tier owns the loss.

## 1. Publish the supplementary datasets

20,684 rows across datasets 3–7 (gap-fill, trie recovery, homograph audit,
trie corrections, lang-uk heteronyms) are `building`. They reach the API only
through `SUPPLEMENTARY_DATASETS`, which is a deployment flag, not a published
lexicon.

Merge them into a published dataset through the ETL's normal path, so the
active dataset is self-sufficient and the flag becomes unnecessary.

## 2. Make the spaCy dependency real

`uk_core_news_sm` is currently loaded from a directory extracted by hand,
because its wheel pins spaCy 3.7.5, which will not build on Python 3.14 (blis).
It runs on 3.8.16 and warns about version mismatch.

Either pin spaCy to a version the wheel accepts, or vendor the model
directory and add it to `ml/pyproject.toml`. As it stands the morphology tier
depends on a path in a scratch directory.

## 3. Close the 44 forms whose tags cannot separate their readings

These are the ambiguities where the trie's tag sets are identical across
readings, so no parser can decide them — `за́мок`/`замо́к` is the type. They
need manifest entries and glosses, which routes them to the cross-encoder.

This is the one place where model coverage is the right answer, and it is
small and enumerable.

## 4. Work the remaining morphology errors

| | tokens |
| --- | --- |
| genuine tagger errors | 28 |
| parse satisfies several readings | 21 |

The first needs a better tagger or a rule; the second needs a tie-break that is
better than declining. Both are worth measuring before either is built —
`resolve()` currently refuses to guess, which is defensible and may already be
the right trade.

## 5. Only then, the model

121 tokens reached the cross-encoder and were answered wrong. That is the
entire remaining model-quality deficit on this benchmark. The mechanism for
improving it is known and measured: a group whose weakest sense reaches ~20% of
its training rows scores near 1.00, and below that it degrades in proportion.

Mining supplies senses that are rare in encyclopedias but ordinary in speech;
generation supplies senses that are rare everywhere. Both exist and work
(`run_mine_corpus.py`, `run_generate_balanced.py` with `--min-share`).

## Infrastructure, not optional

* **Put the repository under git.** A day of changes across `stress.go`,
  `serving.py`, `morphology.py`, the repository layer and a dozen scripts has
  no history and no way back.
* **Install Go.** The API was built and tested with a toolchain fetched into a
  session scratchpad.
* **Rotate `HF_TOKEN`** — it was pasted into a transcript.

## Not to repeat

* **Do not report a figure from the evaluation harness as the product's.** The
  harness reimplements the serving path; for most of this session it scored
  17.7 points higher than the live API on heteronyms, and the difference was a
  timeout that silently disabled a whole tier.
* **Do not trust a ceiling estimate until the scorer has been checked.** This
  benchmark's ceiling was reported as 95.06%, 72.79% and 98.30% before settling
  at 99.76%, each earlier figure wrong for a different defect in the measuring
  script.
* **Do not diagnose from aggregates.** The residual was attributed to coverage,
  then archaic orthography, then proper nouns — three plausible readings of the
  same summary statistic, all wrong. The per-token error list settled it in one
  pass and was available the whole time.
* **Establish the baseline first.** `ukrainian-word-stress` scores 88.67% word
  accuracy with no model at all. Knowing that on day one would have reframed
  every question that followed.
