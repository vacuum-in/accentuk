# Design: Homograph Stress Disambiguation

## 1. System overview

```text
Homograph inventory (SUM-derived)      External morphological dictionary
              |                                       |
              +-------------------+-------------------+
                                  v
              +--------------------------------------------+
              | Python: inventory                          |
              | complete senses -> collapse duplicates     |
              | -> expand paradigms -> transfer stress     |
              | -> compute ambiguous form surface          |
              +--------------------------------------------+
                                  |
              ambiguous_forms.jsonl (the crawl target set)
                                  |
              +--------------------------------------------+
              | Python: acquisition                        |
              | citations / dumps / bulk corpora / crawl   |
              | -> single-pass matcher -> per-sense sample |
              +--------------------------------------------+
                                  |
                    mined candidates + coverage report
                                  |
              +--------------------------------------------+
              | Python: annotation (Azure AI Foundry)      |
              | L0 complete -> L1 label -> L2 generate     |
              | -> L3 blind verify -> quarantine           |
              +--------------------------------------------+
                                  |
              +--------------------------------------------+
              | Python: corpus assembly                    |
              | validate -> dedup -> balance -> split      |
              | -> encode -> version                       |
              +--------------------------------------------+
                                  |
              +--------------------------------------------+
              | Python: training and evaluation            |
              | tokenizer -> Marian -> eval -> convert     |
              +--------------------------------------------+
                                  |
                          model artifact (CT2)
                                  |
              +--------------------------------------------+
              | Python: internal inference service         |
              | constrained candidate scoring only         |
              +--------------------------------------------+
                                  ^
                                  | internal network only
              +--------------------------------------------+
              | Go: public /v1 API                         |
              | lookup -> ambiguous spans -> batched call  |
              | -> merge -> serialize                      |
              +--------------------------------------------+
                                  ^
                              PostgreSQL
```

PostgreSQL remains the dictionary integration boundary. The inference service is
the second integration boundary and is internal. Go never touches the corpus,
the tokenizer, or the model lifecycle.

## 2. Why the existing lexicon cannot supply paradigms

Measured against `output/9dacc408065a68ee-forms-v4/` (29 579 lexemes, 363 272
word forms) and the working homograph inventory (4 829 senses, 2 250 groups):

| Check | Result |
| --- | --- |
| Homograph spellings present as a Wiktionary lemma | 264 / 2 250 (12 %) |
| Homograph spellings present as any Wiktionary form | 282 / 2 250 (13 %) |
| Sense-level stressed forms matching a parsed lexeme | 250 / 4 578 (5.5 %) |
| Matched lexemes that carry a paradigm | 324 / 324, median 14 forms |

The inventory derives from SUM; Ukrainian Wiktionary does not contain 88 % of
it. Where the two do overlap, Wiktionary paradigms are good. That is not enough
to build a corpus: without inflected forms there is nothing to search a corpus
for, and lemma-only matching discards most usable text.

The design therefore vendors an external morphological dictionary and treats the
PostgreSQL lexicon as a consumer of the result, not a supplier of its input.

## 3. Data ownership

| Artifact | Owner | Mutability |
| --- | --- | --- |
| Homograph inventory version | Python inventory stage | Immutable once frozen; identified by content hash |
| Morphological dictionary release | Vendored under `data/` | Pinned by checksum |
| Mined candidates, coverage report | Python acquisition stage | Immutable per corpus checksum |
| Raw annotation responses | Python annotation stage | Append-only, written before parsing |
| Quarantine | Python assembly stage | Append-only, never deleted |
| Corpus version | Python assembly stage | Immutable; records every input hash |
| Model artifact and evaluation report | Python training stage | Immutable per training run |
| Lexicon dataset | Unchanged, prior change | Unchanged |

Nothing in this change writes to lexicon tables. The corpus and the model
reference the lexicon dataset key they were built against; they do not modify
it, and a lexicon republication does not invalidate a model.

## 4. Boundaries and failure isolation

There is no transactional coupling between this change and the lexicon: the
corpus pipeline is file-based, and the serving path is read-only.

The two boundaries that matter are failure boundaries:

- **Annotation boundary.** Raw responses land on disk before parsing, so a
  parser defect costs a re-parse rather than a re-spend. Runs checkpoint per
  group, so an interruption costs one group.
- **Serving boundary.** The Go API calls the inference service only for spans it
  has already resolved to a candidate set. A failure there degrades the answer to
  the pre-change ambiguous response. The model can never make an unambiguous
  lookup wrong, because it is never consulted for one.

## 5. Model input format

Chosen: **marked span, word output.**

```text
src: Він відчинив ⟦ замок ⟧ старим ключем .
tgt: за́мок
```

Rationale: the target is one word, so the decoder has almost nothing to copy and
almost nothing to corrupt; the dictionary already resolves the rest of the
sentence correctly, and a model that re-emits it can only make it worse.

### Rejected: full-sentence accentor

`Він відчинив замок ключем.` → `Він відчи́нив за́мок ключе́м.`

Rejected because it spends the entire model budget re-deriving output the
dictionary already produces exactly, and introduces insertion and deletion
errors on words that were never ambiguous. It also makes evaluation dishonest:
sentence-level accuracy would be dominated by unambiguous tokens.

### Rejected as the primary model: encoder classifier

An encoder with a per-group softmax head is likely stronger per parameter, but
it needs a head per group and cannot answer for a homograph outside the
inventory. It is retained as a **baseline** the Marian model must beat, because
without it the Marian accuracy figures have no scale.

## 6. Constrained candidate scoring

At inference the candidate set is already known from `stress_lookup`. Rather
than generating, the service force-decodes each candidate and compares sequence
log-probabilities.

Consequences, all of which are the point:

- The output is always a form that exists in the lexicon. Emitting an invented
  stress is structurally impossible, not merely unlikely.
- The margin between the top two candidates is a usable confidence signal, so
  abstention has a threshold rather than a guess.
- Cost is linear in candidate count, which is 2 for 1 982 of the 2 250 groups.

## 7. Corpus strategy

Three tiers, descending label quality, ascending volume:

1. **Sense-bound citations.** Dictionary quotations already attached to a
   numbered sense — the only source where the label is free and human-authored.
   Small, and reserved primarily for evaluation gold and as style anchors for
   generation.
2. **Bulk natural text.** MediaWiki dumps and Ukrainian corpora, labelled by
   model. Wikisource matters disproportionately: obsolete, poetic and dialectal
   senses appear there and essentially nowhere else.
3. **Targeted crawl.** Only for senses still starved after tiers 1–2, bounded by
   robots.txt, per-host rate, and a page cap.

Generation fills what mining cannot reach. The blend target is roughly 60 %
mined, and the reported metric is accuracy on natural text, so a model that has
learned the generator's register cannot hide behind an aggregate number.

The order matters: **coverage is measured before the generation budget is set.**
Any estimate of how many senses mining will reach is a guess until the scan has
run, so the pipeline makes the scan a gate rather than an assumption.

### Matching approach

One automaton over all ambiguous surface forms — on the order of tens of
thousands of patterns — and one streaming pass per corpus, reusing the existing
bounded-memory page streaming. Per-sense reservoir sampling happens during the
pass, so a form occurring 90 000 times never reaches disk 90 000 times.

Matching normalizes with the existing canonical lookup key but stores the
original surface string, because the model must see text as it occurs, while
matching must agree with the dictionary.

## 8. Annotation strategy

Four jobs, deliberately not one prompt:

| Job | Purpose | Failure it guards against |
| --- | --- | --- |
| L0 | Complete missing definitions and contrasts | An unusable label definition, which corrupts every sentence built from it |
| L1 | Label mined sentences | — |
| L2 | Generate for starved senses | — |
| L3 | Blind verify with the target masked | A sentence whose context does not actually disambiguate |

Two rules do most of the work:

- **Contrastive prompting.** Every request shows all sibling senses. A model
  asked to illustrate one sense in isolation writes text that fits several; a
  model shown the alternatives writes text that separates them.
- **Blind verification by a different deployment.** If a masked-target reader
  cannot recover the intended sense, the context did not disambiguate — whoever
  wrote it. A group with a systematically low agreement rate is evidence that
  the *sense boundary* is wrong, which is a finding about the inventory, not
  about the sentences.

L0 output is never ground truth. A wrong definition does not produce a few bad
rows; it produces a consistent, plausible, wrong label class.

## 9. Performance rationale

The serving cost is bounded by design rather than by optimization: the model is
consulted only for ambiguous spans, which are a small fraction of tokens in
ordinary text, and each consultation scores 2–9 short candidates. The dominant
per-request cost stays the PostgreSQL lookup that already exists.

Batching all ambiguous spans of a request into one call keeps the added latency
to one round trip regardless of how many homographs a sentence contains.

An int8-quantized CPU artifact is the target; a GPU serving path is not part of
this change. No latency figure appears in this document, in the specs, or in
documentation until a benchmark has produced it on named hardware.

## 10. Repository layout

```text
ml/
├── pyproject.toml
├── src/ukstress_ml/
│   ├── cli.py
│   ├── inventory.py        # sense completion, duplicate collapse
│   ├── paradigms.py        # dictionary match, stress transfer
│   ├── ambiguity.py        # ambiguous form surface
│   ├── sources.py          # source registry, licences
│   ├── mine.py             # dump/corpus scan, automaton, sampling
│   ├── crawl.py            # bounded targeted crawl
│   ├── annotate/
│   │   ├── client.py       # Azure client, batch, resume, cost
│   │   ├── complete.py     # L0
│   │   ├── label.py        # L1
│   │   ├── generate.py     # L2
│   │   └── verify.py       # L3
│   ├── corpus.py           # validate, dedup, balance, split, encode
│   ├── tokenizer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── convert.py
│   └── serve.py            # internal inference service
├── tests/
└── Dockerfile
```

`ml/` depends on `etl/` for canonicalization and stable natural keys. It does
not re-implement them: a second normalization path would let corpus matching and
dictionary lookup disagree, which is the failure mode most likely to be silent.

Reused directly:

| Need | Existing code |
| --- | --- |
| Resumable download, checksum, manifest | `etl/src/ukstress/downloader.py:35` |
| Bounded-memory MediaWiki streaming | `etl/src/ukstress/dump_reader.py:39` |
| Ordered parallel page parsing | `etl/src/ukstress/dump_reader.py:86` |
| Canonical form and lookup key | `etl/src/ukstress/normalizer.py:33` |
| Stress validation | `etl/src/ukstress/normalizer.py:68` |
| Stress signature by vowel ordinal | `etl/src/ukstress/normalizer.py:81` |
| Stable content-hash keys | `etl/src/ukstress/deduplicator.py:12` |
| Run manifest shape | `etl/src/ukstress/pipeline.py` |

## 11. Artifacts

```text
data/dict_uk/                      pinned morphological dictionary + manifest
data/ukwiki-*.xml.bz2              corpus dumps, downloaded like the Wiktionary dump
output/ml/
  inventory_v1.jsonl               frozen senses, definitions, contrasts, paradigms
  ambiguous_forms.jsonl            the ambiguous surface
  mined/{corpus}/candidates.jsonl.zst
  mined_coverage.json              per-sense counts -> annotation budget
  raw/{job}/*.jsonl.zst            unparsed annotation responses
  labelled.jsonl.zst
  quarantine.jsonl.zst
  corpus/{version}/{train,dev,test_natural,test_hard,test_unseen}.jsonl.zst
  corpus/{version}/manifest.json
  models/{version}/                checkpoint, tokenizer, ct2, evaluation report
```

## 12. Rejected alternatives

**Deriving paradigms from PostgreSQL.** Measured at 12 % coverage of the
inventory. Rejected on evidence, not preference.

**Rule-based disambiguation from grammatical tags.** Resolves the 536 groups
whose senses differ by part of speech and none of the 1 714 that do not. Useful
as a feature, insufficient as an approach.

**Generating the whole corpus with a language model.** Cheaper and far faster,
and it produces a model that has learned one generator's register. Mining is
kept as the majority source specifically so the reported metric measures
Ukrainian rather than the annotator.

**Fine-tuning a pretrained translation checkpoint.** The label space is unrelated
to any translation task, and the embedding table would be discarded anyway.
Training from scratch on a small configuration is simpler and faster to iterate.

**Serving the model from Go.** Would require model runtime bindings in the
component whose entire specification is that it stays a thin read-only API.
Rejected in favour of an internal Python service and an explicit, recorded
amendment to the architecture boundary.

**Letting the model answer unambiguous tokens.** Would place a learned component
in front of exact data that is already correct. The model is consulted only
where the dictionary declines to choose.
