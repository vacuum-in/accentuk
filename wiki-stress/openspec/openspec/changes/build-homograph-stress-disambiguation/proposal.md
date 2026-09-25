# Proposal: Build Homograph Stress Disambiguation

## Intent

Add contextual homograph stress disambiguation to the Ukrainian stress lexicon:
a corpus pipeline that crawls and annotates sentences containing ambiguous word
forms, and a Marian sequence-to-sequence model that selects the correct stressed
form of a token from its sentence context.

The dictionary answers "which stress does this word have". This change answers
"which stress does this word have *here*".

`build-ukrainian-stress-lexicon` listed "contextual disambiguation of homographs
inside sentences" and "machine-learning stress prediction" as out of scope. This
change takes both into scope and supersedes those exclusions. It does not modify
the extraction, normalization, or publication behaviour of that change.

## Problem

`stress_lookup` returns every valid candidate for an ambiguous form and, by
design, never chooses one. For 2 250 Ukrainian spellings that is the whole
answer a consumer receives, and a text-to-speech or NLP consumer cannot use it.

Measured against `output/homographs_precess_pilot_enriched2.csv` (the working
homograph inventory) and the parsed lexicon in
`output/9dacc408065a68ee-forms-v4/`:

| Fact | Value |
| --- | --- |
| Homograph groups (distinct ambiguous spellings) | 2 250 |
| Sense rows across those groups | 4 829 |
| Groups whose senses share a part of speech | 1 714 (76 %) |
| Groups whose senses differ by part of speech | 536 |
| Senses with a filled definition | 2 811 (2 018 missing) |
| Groups with at least one sense missing a definition | 1 200 |
| Groups where two sense rows carry the same stressed form | 201 |
| Homograph spellings present as a Wiktionary lemma | **264 / 2 250 (12 %)** |
| Sense-level stressed forms matching a parsed lexeme | **250 / 4 578 (5.5 %)** |
| `reports/ambiguous_forms.json` produced by the current parser | empty (`[]`) |

Three consequences drive this change:

- **1 714 same-part-of-speech groups** mean grammatical tagging cannot resolve
  the majority of cases. Meaning must be read from context.
- **The existing lexicon covers 12 % of the inventory.** Paradigm expansion
  cannot be sourced from PostgreSQL and requires an external Ukrainian
  morphological dictionary.
- **Without paradigms there is nothing to crawl for.** Corpus acquisition
  depends on knowing the inflected surface forms of every sense, so paradigm
  expansion is a prerequisite capability rather than a parallel one.

## Scope

### In scope

- A complete, reviewed homograph sense inventory with definitions and
  sense-contrast statements.
- Paradigm expansion for every sense from a vendored external morphological
  dictionary, and computation of the ambiguous-form surface.
- Corpus acquisition: dictionary citations, MediaWiki dumps, bulk Ukrainian
  corpora, and a bounded targeted crawl for starved senses.
- LLM-assisted annotation through Azure AI Foundry: inventory completion,
  sense labelling of mined sentences, generation for starved senses, and blind
  verification.
- A versioned, provenance-tagged training corpus with reproducible splits.
- A Marian model trained to select a stressed form from context, evaluated with
  constrained candidate scoring.
- An internal inference service and its integration behind the existing Go
  lookup API.

### Out of scope

- Full-sentence accentuation of non-ambiguous words. The dictionary already
  resolves those, and this change does not replace it.
- Expanding dictionary coverage, changing extraction handlers, or altering the
  published lexicon schema.
- Text-to-speech synthesis, phonemization, or audio.
- Grapheme-to-phoneme conversion for out-of-vocabulary words.
- Training on any corpus whose licence has not been resolved.
- A public HTTP interface other than the existing Go `/v1` API.
- Automatic retraining, online learning, or user-feedback loops.

## Proposed capabilities

1. **Homograph inventory** — complete sense records, collapsed duplicate-stress
   senses, expanded paradigms, and the computed ambiguous-form surface.
2. **Corpus acquisition** — licence-gated, resumable, rate-limited crawling with
   a single-pass matcher and per-sense sampling.
3. **LLM annotation** — four separated Azure AI Foundry jobs with persisted raw
   responses, blind verification, and quarantine.
4. **Training corpus** — validation filters, balanced provenance-tagged splits,
   and a versioned manifest.
5. **Stress model** — Marian training, constrained candidate scoring, and
   measured evaluation against declared gates.
6. **Disambiguation API** — internal inference service and Go integration that
   preserves the existing ambiguity contract.

## Key decisions

- **Marked-span, word-output model format.** The source is the sentence with the
  ambiguous token delimited; the target is the stressed token alone. A
  full-sentence accentor was rejected: it forces the decoder to copy text the
  dictionary already resolves correctly and introduces insertion and deletion
  errors on solved input.
- **Constrained candidate scoring at inference.** The candidate set comes from
  `stress_lookup`. Each candidate is force-decoded and scored; the model returns
  the highest-scoring candidate. Free generation is never used, so emitting a
  form absent from the lexicon is structurally impossible.
- **An external morphological dictionary is vendored and pinned.** The 12 %
  lexicon overlap makes PostgreSQL unusable as a paradigm source.
- **Only the ambiguous form surface is trained.** Forms that separate
  orthographically are already resolved by exact lookup and are excluded.
- **Mined natural text is the primary source; generation fills starvation.**
  A model trained only on generated text learns the generator's register.
- **LLM output is never ground truth for a sense boundary.** Proposed
  definitions require human confirmation before any sentence is generated from
  them.
- **Blind verification uses a different deployment than labelling and
  generation**, so a model cannot confirm its own output.
- **PostgreSQL remains the integration boundary for the dictionary.** The model
  is consulted only for spans that exact lookup reports as ambiguous.

## Architecture boundary

`openspec/project.md` assigns all offline processing to Python and restricts Go
to the read-only lookup API, and `design.md` of the prior change states that
there is no HTTP service in the Python component.

This change amends that boundary explicitly rather than silently:

- Python gains an **internal** inference service. It is not published, not
  exposed outside the deployment network, and serves no external client.
- **Go remains the only public HTTP API.** It continues to own request parsing,
  validation, lookup, and serialization, and gains only the ability to call the
  internal service for spans it has already identified as ambiguous.
- Go SHALL NOT perform tokenization for training, model loading, corpus
  processing, annotation, or any part of the model lifecycle.
- If the internal service is unreachable, Go SHALL return the existing ambiguous
  response rather than failing the request.

The amendment is recorded as a MODIFIED requirement in
`specs/disambiguation-api/spec.md`.

## Assumptions

- The homograph inventory is authoritative for which spellings are ambiguous;
  its sense boundaries are subject to human review, not assumed correct.
- An external Ukrainian morphological dictionary with accented paradigms is
  obtainable under a licence permitting derived training data.
- A single GPU host is available for training; inference targets CPU.
- Azure AI Foundry batch endpoints are available, and spend is capped.
- Corpus size estimates in `design.md` are approximate until measured at
  acquisition; the plan records actual bytes and counts.
- Accuracy targets are acceptance targets, not guaranteed results. The
  implementation reports actual measurements.

## Impact

### New components

- `ml/`: Python package and CLI for inventory, mining, annotation, corpus
  assembly, training, evaluation, and inference.
- `data/dict_uk/`: pinned external morphological dictionary release.
- `output/ml/`: inventory, mined candidates, raw LLM responses, quarantine,
  corpus versions, model artifacts, and manifests.
- `deploy/`: an additional internal inference service container.

### Changed components

- `api/`: one new optional downstream call and a fallback path. No change to the
  existing request or response contract for unambiguous lookups.

### Data impact

No change to the published lexicon schema, migrations, or dataset publication.
The corpus and model artifacts are versioned independently of the lexicon
dataset and record the lexicon dataset key they were built against.

### API impact

The `/v1` contract gains an optional field indicating that a token was resolved
by the model rather than by exact lookup. Existing fields and their meanings are
unchanged. Clients that ignore the new field continue to work.

## Risks

- Sense boundaries in the inventory may be wrong or may overlap, producing
  labels no model can learn.
- The morphological dictionary may miss proper nouns and toponymic adjectives,
  a large share of this inventory.
- Mobile-stress paradigms can be transferred wrongly across inflected forms,
  silently poisoning labels.
- Dictionary-citation sources may have licensing terms that prohibit use.
- Mining may yield far fewer natural sentences than predicted for rare senses.
- Generated sentences may be stylistically uniform, so the model learns the
  generator rather than the language.
- LLM labelling may collapse toward the frequent sense of each group.
- Generated contexts may leak the answer through metalinguistic phrasing.
- Azure spend may exceed budget across ~300 k annotation items.
- A mixed-licence corpus may be unpublishable.
- The additional service call may raise API latency or reduce availability.

## Mitigations

- Human review gates the inventory before any generation depends on it; blind
  verification agreement per group is reported as a sense-quality signal.
- Rule-based declension fallback for dictionary misses, flagged as generated
  rather than presented as dictionary data.
- Stress is transferred only for fixed-stress paradigms; mobile-stress cases
  require accented dictionary data or manual entry and are never inferred.
- Licence resolution is a hard gate before crawling; public-domain Wikisource is
  the declared fallback for the same register.
- Coverage is measured before the generation budget is set, replacing the
  estimate with a count.
- A mined/generated blend ratio with per-sense floors, real citations used as
  style anchors, and a natural-text evaluation set as the reported metric.
- Per-sense recall on the human-labelled gold set detects majority-class
  collapse.
- Leakage filters plus blind verification with the target masked.
- Batch endpoints, a pilot gate, per-batch token and cost counters, and a hard
  client-side spend cap.
- Per-row licence tags from the first write, so a publishable subset is
  derivable by filter.
- The model call is batched per request, bounded by timeout, and falls back to
  the existing ambiguous response.

## Rollback

- **Model rollback**: pin the previous model version in the inference service
  manifest and redeploy.
- **Feature rollback**: disable the model call in the Go API by configuration.
  The API returns the current ambiguous response, which is the pre-change
  behaviour.
- **Corpus rollback**: corpus versions are immutable; rebuild from a previous
  version key.
- **Data rollback**: none required. This change does not mutate the lexicon.
