# Plan — dataset creation for homograph stress disambiguation

Companion to [PLAN_MARIAN_HOMOGRAPHS.md](PLAN_MARIAN_HOMOGRAPHS.md). That document covers the model;
this one covers how the training corpus gets built — crawling first, LLM second.

Target output: ~260 k sentences, each carrying one ambiguous Ukrainian word form labelled with the
sense that determines its stress.

## 1. Blocking finding — the existing lexicon does not cover the homograph list

Measured against `output/9dacc408065a68ee-forms-v4/` (29 579 lexemes, 363 272 word forms from
ukwiktionary) and `output/homographs_precess_pilot_enriched2.csv` (4 829 senses / 2 250 groups):

| Check | Result |
| --- | --- |
| Homograph spellings present as a Wiktionary **lemma** | **264 / 2 250 (12 %)** |
| Homograph spellings present as **any** Wiktionary word form | 282 / 2 250 (13 %) |
| Sense-level stressed forms matching a parsed lexeme (stress position included) | **250 / 4 578 (5.5 %)** |
| Matched lexemes that do carry a paradigm | 324 / 324, median 14 forms |
| `reports/ambiguous_forms.json` | `[]` — empty, produces nothing usable |

The homograph inventory comes from `patriotyk/homographs-storage` (SUM-derived), and Ukrainian
Wiktionary simply does not contain 88 % of it. Consequences, both structural:

- **Paradigm expansion cannot use the PostgreSQL lexicon.** It needs an external Ukrainian
  morphological dictionary.
- **Crawling cannot search for lemmas only.** Without paradigms there is no way to find `за́мку`,
  `замка́`, `замка́ми` in running text, and lemma-only matching would throw away most of the corpus.

So Phase A below is a prerequisite for crawling, not a parallel track.

## 2. What already exists and gets reused

| Need | Existing code | Note |
| --- | --- | --- |
| Resumable download + sha256 + manifest | `etl/src/ukstress/downloader.py:35` | Works for any dump URL; point it at ukwiki/ukwikisource |
| Bounded-memory MediaWiki XML streaming | `etl/src/ukstress/dump_reader.py:39` | Namespace-filtered, clears each element; ukwiki is the same schema |
| Parallel page parsing with order preservation | `etl/src/ukstress/dump_reader.py:86` | Reuse for the crawl pass |
| NFD / apostrophe / hyphen normalization, `lookup_key` | `etl/src/ukstress/normalizer.py:33` | **Must** be the only normalization path, so corpus matching and dictionary lookup agree |
| Stress position encoding | `normalizer.stress_signature:81` | Label format for the model target |
| Stable content-hash keys | `deduplicator.stable_natural_key:12` | Sentence IDs, provenance keys |
| Run manifest with parser/schema/normalization versions | `output/.../manifest.json` via `pipeline.py` | Copy the shape for corpus runs |
| Wikitext → text stripping | `etl/src/ukstress/wikicode.py` | Written for entries; needs an article-prose variant |

Prior art to check before building, not to ignore: `lang-uk/ukrainian-word-stress` (dictionary +
rule/tag-based ambiguity handling) and `patriotyk/ukrainian-accentor` (the source of the homograph
list). Both give a free baseline and possibly reusable stress dictionaries.

## 3. Phase A — paradigms and the ambiguous-form surface

Nothing can be crawled until every sense knows what its inflected forms look like.

1. **Adopt VESUM / `dict_uk`** (brown-uk) as the morphology source: ~380 k lemmas with full
   paradigms and grammatical tags, and accented variants in recent releases. Vendor a pinned release
   tarball into `data/dict_uk/` with a sha256 manifest, using `download_dump()`. Fall back to
   `pymorphy3` + `pymorphy3-dicts-uk` only for lemmas VESUM misses.
2. **Match each of the 4 829 senses to a VESUM paradigm** by `(lemma_normalized, POS, gender/aspect
   from gram_style)`. Expect three buckets: clean single match, several candidate paradigms
   (needs the `gram_style` field to pick), and no match (proper nouns, toponymic adjectives,
   dialectal forms) — those get a rule-based paradigm generated from the declension class and
   flagged `paradigm_source: generated`.
3. **Transfer stress to every form.** For fixed-stress paradigms the acute position carries over by
   vowel ordinal (`stress_signature`). For mobile-stress paradigms it does not — those need the
   accented VESUM variant or manual entry, and must be flagged rather than guessed.
4. **Compute the ambiguous form set per group**: unstressed form strings that collide across two or
   more senses of the group. Only these need training data — forms that separate orthographically
   are already solved by `stress_lookup`.
5. Emit `output/ml/ambiguous_forms.jsonl`:

```json
{"group_id": 812, "form": "замок", "feats": "nom.sg",
 "candidates": [{"sense_id": "812.a", "stressed": "за́мок", "signature": "0"},
                {"sense_id": "812.b", "stressed": "замо́к", "signature": "1"}],
 "paradigm_source": "vesum", "stress_transfer": "fixed"}
```

Deliverable metric that gates Phase B: **≥ 90 % of senses with a paradigm**, and a written count of
how many groups are ambiguous in the nominative only versus across the paradigm — that number sets
the per-group crawl budget.

## 4. Phase B — crawling

Three tiers, descending label quality, ascending volume.

### Tier 1 — sense-bound citations (gold labels, no LLM needed)

Dictionary citations are already attached to a numbered sense. This is the only source where the
label comes for free, and it is the correct place to start.

| Source | Content | Access | Caution |
| --- | --- | --- | --- |
| Ukrainian Wiktionary usage examples | Quotes inside entries, per sense | **Already downloaded** — `data/ukwiktionary-latest-pages-articles.xml.bz2` | Covers only the 12 % overlap, but costs one extra parse pass |
| SUM-11 / SUM-20 (`sum.in.ua`, `ukrlit.org`) | Illustrative quotations per numbered sense | HTTP crawl | **Licensing gate — resolve before crawling.** Rate-limit ≤ 1 req/s, honour robots.txt, cache raw HTML, never redistribute raw citations without clearance |
| `r2u.org.ua`, `slovnyk.ua` | Additional glosses/examples | HTTP crawl | Same gate |

Expected yield: small in absolute terms (a few tens of thousands of sentences) but disproportionately
valuable — it is the only *human-authored* sense-labelled data, so it becomes the seed set and part
of the evaluation gold, not just training filler.

### Tier 2 — bulk natural text (LLM labels it)

| Source | Approx. size | Licence | Acquisition |
| --- | --- | --- | --- |
| Ukrainian Wikipedia dump | ~2.5 GB bz2, ~1.4 M articles | CC BY-SA | `download_dump()` + `stream_pages()` — near-zero new code |
| Ukrainian Wikisource dump | ~0.3 GB | public domain / CC BY-SA | Same path; **this is where obsolete, poetic and dialectal senses live** |
| UberText 2.0 (lang-uk) | ~6 B tokens (news, fiction, social, wiki) | check terms | Bulk download, streamed |
| HPLT v2 / CulturaX Ukrainian | tens of GB | ODC-By / CC0 per subset | HF streaming; use only if Tier 2 above still starves senses |

All sizes approximate — confirm at download and record actual bytes in the run manifest.

### Tier 3 — targeted long-tail crawl

Only for senses still starved after Tiers 1–2 (predicted: proper-noun adjectives such as
`су́хівський`/`сухі́вський`, dialectal and obsolete entries). Site-scoped crawl of `chtyvo.org.ua`,
`ukrlib.com.ua`, regional news archives, with: robots.txt honoured, ≤ 1 req/s per host, descriptive
User-Agent matching the existing `ukstress-etl/0.1 (…)` convention, resumable frontier, raw HTML
cached to disk before parsing, and a hard page cap per host. Do not crawl anything requiring a login
or paywall bypass.

### The matching pass (the actual engineering)

One streaming pass per corpus, one Aho–Corasick automaton over all ambiguous surface forms:

1. Build the automaton from `ambiguous_forms.jsonl` — roughly 4 829 senses × ~14 forms, deduplicated
   to the ambiguous subset, so on the order of 30–60 k patterns. Fits in memory easily.
2. Stream pages (`stream_pages`), strip wikitext to prose, split into sentences (a Ukrainian-aware
   splitter — abbreviations `тис.`, `см.`, `ім.`, initials).
3. Normalize each sentence with `lookup_key()` for matching, but **keep the original surface string**
   for the corpus. Match on word boundaries, not substrings.
4. On hit: emit the sentence with character offsets of the matched span, page id, revision id, and a
   `stable_natural_key` sentence id.
5. **Reservoir-sample per (group, form)** with a cap (e.g. 400 candidates) so `замок` does not
   contribute 90 000 sentences while `сухівський` contributes 3. Sampling happens during the pass —
   never write the full hit set to disk.

Filters applied inline: length 6–40 tokens, no more than one occurrence of any ambiguous form, no
existing stress marks, Ukrainian-script ratio ≥ 0.9, drop list/table/infobox fragments, drop
sentences that are themselves dictionary definitions.

Output: `output/ml/mined/{corpus}/candidates.jsonl.zst` + a coverage report
`mined_coverage.json` giving hits per sense. **That report is the input to the LLM budget** — it is
the moment the plan stops guessing which senses need generation.

Honest expectation: roughly half to two-thirds of senses should reach ≥ 25 natural hits from Tiers
1–2; the remainder go to generation. That split is a prediction, and Milestone M2 exists to replace
it with a measurement.

## 5. Phase C — LLM work (Azure AI Foundry)

Four distinct jobs. Different jobs, different model tiers, different failure modes — do not collapse
them into one prompt.

| Job | Input | Model tier | Volume | Gate |
| --- | --- | --- | --- | --- |
| **L0 — inventory completion** | 2 018 senses with no definition, 465 with no `gram_style` | frontier | ~1 200 calls | Human review before any generation depends on it |
| **L1 — sense labelling** | Mined sentences from Tier 2/3 | mini, Batch API | ~200 k items | Agreement with L3 |
| **L2 — generation** | Senses starved after mining | frontier, Batch API | ~100 k sentences | Human QA sample ≥ 95 % |
| **L3 — blind verification** | Every labelled/generated sentence, target masked, **different deployment from L1/L2** | mini, Batch API | ~300 k items | Disagreement ⇒ quarantine |

### L0 — completing the sense inventory

Per group, given all sibling senses at once, the model proposes the missing `definition`, `register`
and `contrast` (how this sense differs from its siblings). Output is written as
`review_status: llm_proposed` and **never used as ground truth until a human confirms it** — this is
a label definition, and an error here corrupts every sentence generated from it. Prioritize the
1 200 groups where at least one sense is missing a definition; within those, prioritize the 1 714
same-POS groups, where the definition is the only thing separating the senses.

### L1 — labelling mined sentences

```
Речення: {sentence}
Цільова словоформа: {form} (позиція {start}–{end})
Можливі значення:
  A ({stressed_A}) — {definition_A}
  B ({stressed_B}) — {definition_B}
Яке значення вжите? Якщо контекст не дозволяє визначити — відповідь "unclear".
```

Structured output: `{"sense_id": "...", "confidence": 0..1, "cue": "...", "unclear": bool}`.
`unclear` is a first-class answer and must not be penalized — genuinely ambiguous sentences are
noise in training and gold for a separate "abstain" eval slice. `cue` (the words in the sentence that
decide it) is a cheap self-check: if `cue` is empty or absent from the sentence, drop the row.

Cost control: Batch API (24 h, ~50 % discount), one call per sentence batch of 20, prompt prefix
shared per group so the sense definitions are written once per batch.

### L2 — generation for starved senses

Identical to the generation design in `PLAN_MARIAN_HOMOGRAPHS.md` §6.2: one call per **group** with
all sibling senses visible so the contexts actually discriminate, `temperature` 0.9, 10 sentences per
call, rotating topic hints (`побут`, `історія`, `техніка`, `новини`, `художня література`) to break
scene repetition, and a `cue` field per sentence.

Two additions specific to this plan:

- **Seed with mined examples.** Where Tier 1 supplied even 2–3 real citations for a sense, put them
  in the prompt as style anchors. Generated text drifts toward encyclopedic register otherwise.
- **Generate for the inflected forms, not the lemma.** The prompt names the exact target form and its
  features (`замку`, `dat.sg`), drawn from the ambiguous form set, so the corpus matches the
  distribution the model sees at inference.

### L3 — blind verification

The strongest single filter, and worth its cost: a different deployment sees the sentence with the
target span **masked** and picks a sense from the list. If it cannot recover the intended label, the
context did not disambiguate — regardless of whether a human or an LLM wrote it. Quarantine on
disagreement; measure per-group agreement rate and treat a low rate as evidence the *sense boundary*
is broken, not the sentences.

### Operational requirements

Resumable by `group_id` checkpoint; every raw API response persisted to `output/ml/raw/*.jsonl.zst`
before parsing; per-batch token and cost counters logged; deployment name, model version and API
version recorded in the run manifest next to the `homographs_v1` content hash. Credentials via
`AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` (or Entra ID) in `.env`, never committed. Hard spend
cap enforced client-side, checked before each batch submission.

## 6. Phase D — assembly

Validation filters, splits and the Marian source/target encoding are specified in
`PLAN_MARIAN_HOMOGRAPHS.md` §7–8 and are not repeated here. What this phase adds:

- **Provenance per row** — `source_tier` (citation / wiki / ubertext / crawl / generated),
  corpus name, page and revision id, licence tag, LLM job id and model version. Cheap now,
  impossible to reconstruct later, and it decides whether the dataset can ever be published.
- **Blend ratio** target ≈ 60 % mined / 40 % generated overall, with per-sense floors (25 sentences)
  and a per-group cap on the frequency ratio (4:1).
- `test_natural` is drawn **only** from Tier 1 + Tier 2 with human confirmation. It is the number
  that gets reported. Generated-only groups are flagged so distribution shift stays visible.

## 7. Artifacts and layout

```
data/dict_uk/                             # pinned VESUM release + manifest
data/ukwiki-latest-pages-articles.xml.bz2 # via download_dump()
data/ukwikisource-latest-...xml.bz2
output/ml/
  homographs_v1.jsonl                     # Phase 0/A: senses, definitions, contrasts, paradigms
  ambiguous_forms.jsonl                   # Phase A: the crawl target surface
  mined/{corpus}/candidates.jsonl.zst     # Phase B
  mined_coverage.json                     # hits per sense -> LLM budget
  raw/{job}/*.jsonl.zst                   # Phase C: unparsed API responses
  labelled.jsonl.zst                      # Phase C output
  quarantine.jsonl.zst                    # L3 disagreements, kept not deleted
  corpus_v1/{train,dev,test_*}.jsonl.zst  # Phase D
  manifest.json                           # versions, hashes, counts, cost
```

New package `ml/` beside `etl/`, same toolchain (uv, ruff, mypy, pytest, `line-length = 100`,
`requires-python >= 3.12`), same CLI style as `etl/src/ukstress/cli.py:35`:

```bash
ukstress-ml paradigms --dict data/dict_uk --senses output/ml/homographs_v1.jsonl
ukstress-ml mine --corpus ukwiki --dump data/ukwiki-latest-pages-articles.xml.bz2 --workers 4
ukstress-ml coverage
ukstress-ml llm label --batch --resume
ukstress-ml llm generate --batch --resume
ukstress-ml llm verify --batch
ukstress-ml assemble --version v1
```

## 8. Milestones

| # | Deliverable | Gate |
| --- | --- | --- |
| M0 | `homographs_v1.jsonl` — definitions and contrasts complete (L0 + human review) | 200-sense human review |
| M1 | `ambiguous_forms.jsonl` from VESUM | ≥ 90 % of senses have a paradigm; mobile-stress cases flagged, not guessed |
| M2 | Tier 1 + Tier 2 mined, `mined_coverage.json` | Coverage measured — **replaces the yield guess in §4** |
| M3 | Tier 3 targeted crawl for starved senses | robots/rate-limit compliance verified in logs |
| M4 | L1 labelling + L3 verification complete | Agreement ≥ 90 %; per-group agreement reported |
| M5 | L2 generation for the remainder | Human QA sample ≥ 95 % |
| M6 | `corpus_v1` assembled, split, provenance complete | Per-sense floor met; `test_natural` is human-confirmed |

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| VESUM misses proper nouns / toponymic adjectives (a large slice of this inventory) | Rule-based declension fallback, flagged `paradigm_source: generated`; these senses lean on L2 generation |
| Mobile stress transferred wrongly across a paradigm — silently poisons labels | Only transfer for fixed-stress paradigms; mobile cases require accented VESUM data or manual entry; never infer |
| SUM / dictionary-site crawling has unclear terms | Licensing gate **before** M2; Wikisource (public domain) is the fallback for the same register |
| Mining yields far less than predicted for rare senses | M2 measures it; generation budget is sized from the measurement, not the guess |
| Generated text is stylistically uniform ⇒ model learns the generator | 60/40 blend, Tier 1 citations as style anchors, `test_natural` is the reported metric |
| LLM labels inherit LLM bias toward the frequent sense | L3 uses a different deployment; per-sense recall on the Tier 1 gold set catches a majority-class collapse |
| Azure cost overrun | Batch API, 200-group pilot gate, per-batch counters, hard client-side spend cap |
| Corpus cannot be published due to mixed licences | Per-row licence tag from day one; publishable subset derivable by filter |

## 10. Open questions

1. **SUM citation licensing** — resolve before M2. It changes whether Tier 1 exists at all, and
   Tier 1 is the only free-label source.
2. **UberText 2.0 terms** for derived-dataset redistribution.
3. Is the corpus itself a deliverable (HF dataset card) or internal training input only? Affects how
   strict the provenance and licence tagging must be — but the tagging cost is low enough that the
   plan does it either way.
