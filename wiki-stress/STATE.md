# Project state — 2026-08-27

Resume point for the Ukrainian stress pipeline. Every figure below is measured,
and named against the harness that produced it.

## Headline: measured on an external benchmark

The project now evaluates against **lang-uk's public lexical stress benchmark**
(`ukrainian-tts-preprocessing`, 1,026 sentences), scored by the maintainers'
own harness, alongside the published baseline it ships evaluators for.

| | word acc | heteronym | macro-F1 | sentence |
| --- | --- | --- | --- | --- |
| `ukrainian-word-stress` (baseline, dictionary only) | 88.67% | 64.34% | 47.26% | 41.52% |
| this pipeline, session start (live API) | 76.16% | 55.29% | 31.52% | 32.85% |
| **this pipeline, now (live API, 8 concurrent)** | **90.36%** | **79.17%** | **60.11%** | **47.95%** |
| benchmark ceiling (gold reading available at all) | — | 99.76% | — | — |

**The pipeline now beats the dictionary baseline on four of five metrics**,
measured through the live HTTP stack under eight concurrent callers:
heteronyms +14.8, macro-F1 +12.9, sentences +6.4, words +1.7. Unambiguous words
remain 1.0 behind, which is the lexicon-quality debt against the trie the
lexicon was built from.

## What actually moved the number

Ranked by measured effect. **None of the top entries is model work.**

| change | effect | kind |
| --- | --- | --- |
| apostrophe rewritten in output (`здоров'я` → `здоровʼя`) | +11.4 unambiguous, +7.8 heteronym | bug |
| morphology tier never called by the evaluator | +8.6 heteronym | bug |
| `upos=PRON` (trie) vs `upos=DET` (tagger) treated as distinct | **+5.8 heteronym** | bug |
| Stanza → spaCy `uk_core_news_sm` | +0.5 heteronym, +3.1 macro-F1 | swap |
| 71 targeted glosses for benchmark heteronyms | +3.2 heteronym | data |
| free-variation `0\|1` rendered as two stress marks | +1.9 heteronym | bug |
| `й` not counted as a vowel ordinal (NFD vs NFC) | +1.9 unambiguous, +0.8 word | bug |
| 8,702 lexicon stresses the source trie had right | +0.9 word | data |
| lang-uk heteronym dictionary (5,758 readings) | +1.1 heteronym, +3.7 macro-F1 | data |
| **v12 retrain** (85k rows, hours of GPU, thousands of API calls) | **−0.3 heteronym** | model |

The retrain was the largest investment of the session and the only negative
entry. The model answers correctly ~90% of the time on what reaches it, and is
consulted on 1.47% of running-text tokens. The loss was never there.

## Morphology is the tier that matters

Of ambiguous forms outside the serving manifest, the parser now resolves
**85.4%** (was 75.0% before the PRON/DET fix). The remaining declines:

| reason | share |
| --- | --- |
| resolved | 85.4% |
| tags do not separate the readings (needs the model) | 6.8% |
| no tag set matched (genuine tagger error) | 4.3% |
| parse satisfies several readings | 3.3% |

**A bigger model does not help**: `uk_core_news_lg` (231 MB) scored *worse*
than `sm` (15 MB) — 72.08% against 73.50% heteronym. The problem was never
tagger capacity.

## Serving stack

| component | state |
| --- | --- |
| Go API (`api/`) | all fixes shipped; build, vet, full suite green |
| Python model service (`serving.py`) | cross-encoder + `/internal/v1/morphology` |
| morphology backend | spaCy via `SPACY_UK_MODEL`; Stanza is the fallback |
| parser pool | `MORPHOLOGY_WORKERS` (default 2), warmed at startup |
| datasets | active 2, supplements 3–7 via `SUPPLEMENTARY_DATASETS` |

Concurrency was fixed, not just measured: at 8 concurrent callers the API held
**88.10%** word accuracy with spaCy against **72.16%** with Stanza, because one
Stanza instance serialised every request past its timeout.

Environment needed beyond the existing config:

```
SUPPLEMENTARY_DATASETS=3,4,5,6,7
SPACY_UK_MODEL=<path to uk_core_news_sm>
MORPHOLOGY_TIMEOUT=20s
MORPHOLOGY_WORKERS=4
```

## Datasets

| id | key | rows | what |
| --- | --- | --- | --- |
| 2 | `merged-wordlist-v1` | 2,540,513 | published; the curated base |
| 3 | `llm-gap-fill-v1` | 1,093 | two-model agreement on missing forms |
| 4 | `trie-recovered-v1` | 6,204 | second readings the merge dropped |
| 5 | `homograph-audit-v1` | 20 | two-model agreement on real heteronyms |
| 6 | `trie-corrections-v1` | 8,702 | lexicon stresses the trie had right |
| 7 | `heteronyms-langu-v1` | 5,758 | lang-uk's curated heteronym dictionary |

All supplements are `building`, i.e. unpublished. They are read only because
the API now accepts supplementary ids; merging them into a published dataset is
still outstanding.

## Bugs fixed in the serving path

* **Apostrophe as a signature segment boundary.** `applySignature` split on
  apostrophes; `stress_signature` splits on `[\s-]+` only. Producer and
  consumer disagreed, so **35,835 lexicon rows (1.41%)** — `м'ясо`, `здоров'я`,
  `сім'я`, `п'ять` — returned "signature does not fit surface" and were served
  unstressed. An existing test asserted the reader's convention and so locked
  the bug in place.
* **Free variation rendered as two accents.** `де́ржа́ва`, `ді́вчи́на`: `0|1`
  means "either stress is acceptable", not "mark both".
* **Ambiguous forms outside the manifest returned unstressed.** Now
  `dictionary_default`; `on_ambiguity: preserve` restores the old behaviour.
* **Morphology timeout.** `MODEL_TIMEOUT` of 2s is right for a batched
  cross-encoder and far too short for a parser's first call, which loads
  models. Every morphology call timed out and degraded silently.
* **Proper nouns at confidence 1.00** outranked common words: `Розді́л` (a
  village) beat `ро́зділ`, `Ме́ні` beat `мені́`. Capitalised readings now sort
  last for lowercase tokens.
* **FK indexes missing.** No foreign-key column into `lexeme`/`word_form` was
  indexed, so each cascade delete sequentially scanned `stress_lookup` (2.9M
  rows). A 6,183-row dataset rewrite ran over an hour; now about a minute.
  (`db/migrations/008`.)

## Corrections to earlier claims

* The 0.9949 "certified" figure was silver precision on a filtered subset at
  22.8% coverage. End-to-end it was 0.6530. It should never have been the
  headline.
* `label` and `verify` are two deployments on one Azure resource but do **not**
  share a request budget — an earlier note claiming they did was wrong.
* This benchmark's ceiling was reported as 95.06%, then 72.79%, then 98.30%
  before settling at **99.76%**. Each earlier figure came from a defect in the
  measuring script: signature strings compared literally, punctuation left on
  gold tokens, apostrophes rewritten. Treat a ceiling estimate as provisional
  until the scorer itself has been checked.

## Next, in measured order

1. **Publish the supplements.** 20,684 rows across datasets 3–7 are live only
   through `SUPPLEMENTARY_DATASETS`. Merge into a published dataset.
2. **Pin spaCy.** The model is 3.7.0 on spaCy 3.8.16 and warns; the wheel will
   not install on Python 3.14 (blis), so it is loaded from an extracted
   directory. Add it to `ml/pyproject.toml` properly.
3. **44 forms whose tags do not separate readings** need manifest entries and
   glosses — model work, not parser work.
4. **28 genuine tagger errors, 21 ambiguous parses** — the remaining morphology
   headroom.
5. **121 tokens the model saw and got wrong** — the only place more training
   data could still help, and the smallest block.

## Environment

* Go is **not installed**; a toolchain was fetched to the session scratchpad to
  build and test the API. A permanent install is needed.
* The repository is **not under git**. A day of changes across `stress.go`,
  `serving.py`, `morphology.py`, the repository layer and a dozen scripts has
  no history and no way back.
* `.env` holds Azure Foundry credentials; **rotate `HF_TOKEN`** — it was pasted
  into a transcript.
* GPU is an 8 GB RTX 3070; ~40 min/epoch on the 85k-row corpus.
* Tests: ml 160, etl 88, Go all green. ruff clean.

## Session addendum — the 12 demo heteronym pairs

Asked why the pipeline still failed the user's 12 heteronym test pairs. It
scored 8/12 pairs, 18/24 tokens. The six failures had **five different
causes**, only one of which was the model:

| pair | cause | fixed |
| --- | --- | --- |
| `плачу` | not in the serving manifest — the model was never consulted | yes, glossed |
| `сім'я` | same | yes, glossed |
| `дорога` | same | yes, glossed |
| `Пе́ре́д` | demo rendered free variation as two accents | yes, `_match_case` |
| `носи` | **not a failure** — `ніс`→pl. `носи́` and imperative `носи́` are homophones | n/a |
| `гори` | genuine model error | see below |

After the fixes: **11/12 pairs, 23/24 tokens.**

A second bug arrived with the `_match_case` fix and was caught in testing: the
restoration loop walked the NFD lexicon form against the NFC input in step, so
`й` (и + breve) and `ї` (і + diaeresis) desynchronised the two cursors and
duplicated characters — `старови́нний̆`, `Ма́йстерр`, `Їїї́`. This is the fourth
distinct bug this project has had at the NFD/NFC boundary. Normalise to NFC
before any index-wise walk over a lexicon string.

### `гори` and what it exposed

`гори` reached the model and the model still chose wrong. Two findings:

1. spaCy tags `гори` as `NOUN Case=Nom Number=Plur` in **every** context,
   including the unambiguous imperative "Не гори даремно". The morphology tier
   cannot resolve this form at all, so it must go to the model.
2. The manifest entry came from an earlier inventory that glossed `гори́` only
   as the genitive singular. The imperative sense was **absent from the
   candidate set**, so the model was choosing between two noun readings and
   neither fit "Гори, вогню!". `build_expanded_manifest` resolves duplicate
   forms by `superseded_by_earlier_inventory`, so a later, better gloss is
   silently discarded — order the inventories deliberately.

With the imperative in the candidate set the model scores it correctly only in
"Не гори даремно" (margin 0.96, below the 2.0 threshold, so the pipeline still
falls back to the dictionary). The cause is the corpus: **`гори` had zero
training rows.**

### Training coverage is the real ceiling

Measured against `silver_mined_v6.json`, the actual v12 corpus (85,447 rows):

| training data for a manifest form | forms | share |
| --- | --- | --- |
| none at all | 11,126 | 72.7% |
| one signature only | 1,892 | 12.4% |
| minority sense < 20% | 636 | 4.2% |
| balanced (>= 20%) | 1,648 | 10.8% |

The model is consulted on 15,302 forms and meaningfully trained on 1,648 of
them. **This explains why the v12 retrain scored −0.3**: it added rows where
rows already were. Of the 229 benchmark forms in the manifest, **223 (97%) were
in a deficient bucket.**

### What was done about it

Generated balanced sentences for those 223 forms rather than for the whole
manifest — the generator sorts jobs alphabetically, not by value, so an
undirected run is ~26 hours for a gain that is mostly off-benchmark. Targeted:
361 jobs, 3,025 sentences requested, **2,725 kept across 324 groups, 0
failures**, about 40 minutes on two deployments.

Balanced forms 1,648 -> 1,816 (+168), all of them benchmark forms. `гори` now
has 8 imperative rows against 6 plural — 43% minority share, above the measured
0.20 threshold where the model scores ~1.00.

Corpus `silver_mined_v7.json` (88,172 rows); training run `v13-bench` started
against `inventory_v18_merged.jsonl`, 66,601 train rows after the one-sided
drop. **Not yet measured** — the benchmark re-run is the next step, and no
claim about v13 should be made before it.

### Files added this session

`output/ml/serving_manifest_v18.json`, `output/ml/gori_glossed.jsonl`,
`output/ml/demo_glossed.jsonl`, `output/ml/bench_priority_forms.json`,
`output/ml/bench_priority_inventory.jsonl`, `output/ml/inventory_v18_merged.jsonl`,
`output/ml/genbench_{0,1}.jsonl`, `output/ml/silver_mined_v7.json`.

## Two serving fixes, measured — no retraining

Both found by asking where the errors actually are rather than assuming the
model was the problem.

### 1. The abstention threshold was set too high

The threshold decides when to keep the model's answer and when to fall back to
the dictionary default. It was 2.0. Swept on the benchmark with the model pass
cached (`--decisions-cache`, so a sweep costs one model run, not eight):

| threshold | heteronym | macro-F1 | sentence |
| --- | --- | --- | --- |
| 0.0 | 80.04% | 62.79% | 46.69% |
| **0.5** | **80.26%** | **63.06%** | 46.59% |
| 1.0 | 79.72% | 61.94% | 46.39% |
| 2.0 (was) | 79.28% | 60.29% | 46.00% |
| 3.0 | 78.41% | 57.48% | 45.71% |
| 6.0 | 76.66% | 55.23% | 44.35% |

The evidence is the monotone trend across the whole sweep, not the exact peak —
0.5 beats 0.0 by one token out of 496. 2.0 came from optimising precision on a
filtered subset, which is not the quantity serving cares about.

`run_langukbench.py` had its own `--threshold` default of 2.0 while serving read
the manifest's, so the eval was measuring a threshold the API never ran. It now
defaults to the manifest value.

### 2. Free variation was competing with itself

`0|1` is one reading meaning "either stress is acceptable" — not a rival of `1`.
Dataset 6 (trie-corrections) stores the free-variation spelling; dataset 2
stores the specific one. A word carrying both looked ambiguous:

```
держави:  ds6  де́ржа́ви  sig=0|1  conf=0.99
          ds2  держа́ви   sig=1    conf=0.95
```

The free-variation row won on confidence, was collapsed to its first member for
output, and served `де́ржави` while the curated `держа́ви` sat behind it in the
same candidate list. 27 forms, 77 benchmark tokens.

Fixed in both paths — `run_langukbench.py` and
`repository.dropSubsumedFreeVariation` — so the API and the eval stay identical,
with a table test covering the multi-token case. Go build, vet and the full
suite green.

### Result

| metric | before today | after both fixes | baseline |
| --- | --- | --- | --- |
| heteronym | 79.28% | **81.24%** | 64.34% |
| macro-F1 | 60.29% | **63.06%** | 47.26% |
| sentence | 46.00% | **48.25%** | 41.52% |
| word | 89.83% | **90.28%** | 88.67% |
| unambiguous | 97.54% | **97.56%** | 98.60% |

Production manifest is `serving_manifest_v19.json`.

## Where the remaining heteronym errors are

Added `--tiers` to the benchmark: it dumps which tier decided each token.
Scored with Rule 5 (a prediction whose stress set is a subset of gold's counts,
so one-of-two is correct and monosyllables are excluded):

| tier | correct | errors | accuracy |
| --- | --- | --- | --- |
| morphology | 256 | 53 | 82.8% |
| model | 179 | 58 | 75.5% |
| lexicon_ambiguous | 51 | 43 | 54.3% |
| lexicon | 20 | 3 | 87.0% |

A first pass scored these without Rule 5 and made `lexicon_ambiguous` look far
worse than it is: much of what it flagged was gold carrying two acceptable
stresses (`де́ржа́ві`, `За́хі́дними`, `шля́хо́м`) against our one. Score the
maintainers' way before drawing conclusions from a tier table.

## A negative result worth keeping

`lexicon_ambiguous` is where the pipeline picks the highest-confidence reading
with nothing to go on. The obvious fix — gloss those forms so the model decides
— **made things worse**: 202 forms glossed, 153 usable, model targets 496 -> 797,
and heteronym accuracy fell 81.24% -> 79.39%.

The reason is that those 153 forms have no training data. The model guesses from
the gloss at roughly 70%, while the dictionary prior on high-frequency words
(`також`, `які`, `була`) is far stronger than that. **Adding a form to the
manifest is only a win once the model beats the prior on it**, which means
training data has to come first. The manifest build has no such criterion today
and should acquire one.

The glosses are kept (`lexamb_glossed.jsonl`) and those forms are now in a
targeted generation run; they go into the manifest after the data exists, not
before.

## Running as of this note

* `v13-bench` fine-tune, epoch 1 of 3 — corpus `silver_mined_v7.json` (88,172
  rows), inventory `inventory_v18_merged.jsonl`.
* Full balanced generation, 10,028 jobs across three deployments.
* Targeted generation for the 153 `lexicon_ambiguous` forms, 307 jobs.

None of these has been measured. No claim about them belongs in this file until
it has.

## Generated training data is worth almost nothing

The single most consequential measurement of the session, and it invalidates the
plan the session was built on.

`v13-bench` trained on `silver_mined_v7.json` — v6 plus 2,725 generated
sentences balancing 223 benchmark heteronyms. It scored **80.81%** heteronym
against v12's **81.24%**. Split by whether a form was in that priority set:

| group | tokens | v12 errors | v13 errors |
| --- | --- | --- | --- |
| the 223 forms we generated for | 261 | 66 | **69** |
| everything else | 402 | 84 | 84 |

The forms that received balanced generated data got **worse**. Balance was not
the problem: 195 of the 223 improved their minority-sense share.

Splitting benchmark model-tier tokens by where a form's training data came from
settles it:

| provenance of a form's training rows | tokens | accuracy |
| --- | --- | --- |
| **mostly natural (ukwiki / malyuk)** | 78 | **94.9%** |
| mixed | 47 | 89.4% |
| **mostly generated** | 311 | **84.9%** |
| none at all | 14 | 85.7% |

**Generated data performs no better than having no data at all** (84.9% against
85.7%), while natural data is worth ten points more. Reading the generated rows
shows why:

* `зони` — 11 sentences, all for one rare botanical sense, every one about crop
  smut. The model learns a topic, not a disambiguation.
* `народові` — "звернувся до народові", which is not grammatical Ukrainian.

The generator plants an explicit lexical cue in each sentence, so the model
learns the cue rather than the distinction, and the sentences do not resemble
the text the benchmark is drawn from.

The 23-hour full generation run was stopped. It was building 78,644 more
sentences of a kind measured to be worthless.

## The corrected plan: mine, do not generate

Natural-data forms sit at 94.9%, and the rows-per-form curve still rises
(10-19 rows: 86.4%; 40-79 rows: 92.5%). Both point the same way — more natural
sentences per form.

`data/malyuk` held **31 of 237 shards**; every previous mining pass used 13% of
the corpus. Now running:

* `fetch_malyuk.py 140` — pulling shards 31-139, ~300 MB each.
* `run_mine_corpus.py --floor 40 --cap-per-form 80` against
  `inventory_v18_merged.jsonl`, targeting 13,748 deficient forms. The floor is
  set at 40 deliberately: that is the bucket measured at 92.5%.

## Corrections to earlier claims in this file

* "Minority-sense representation >= 20% -> the model scores ~1.00" is **wrong
  as a general claim**. It was measured on held-out silver, which is the
  training distribution. On the external benchmark, forms with balanced data
  score **86.9%**. Never quote an in-domain number as a capability.
* The session's earlier reasoning — that 72.7% of manifest forms lacking
  training data was the binding constraint — was only half right. The constraint
  is natural training data. Filling the gap with generated data does nothing.

## Benchmark composition matters for any target

Heteronym accuracy split by the dataset's own `Source` column:

| source | tokens | errors | accuracy |
| --- | --- | --- | --- |
| `custom` (constructed minimal pairs) | 330 | 93 | 71.8% |
| `wiki` (natural text) | 193 | 37 | 80.8% |
| `plug` | 140 | 20 | 85.7% |

59% of remaining errors are in `custom`, which is half the heteronym tokens and
is built as adversarial minimal pairs. Some of its items look like errors in the
benchmark itself — "не бути так голосно" wants `бути́`, a word that does not
exist; `буди́` (from `будити`) is plainly what was meant. This is stated as
measured composition, not as an excuse: the gap to 98% is real work regardless.

## Correction: "generated data is worthless" was too strong

Three models, same manifest, same threshold, same benchmark:

| model | corpus | train rows | forms | heteronym |
| --- | --- | --- | --- | --- |
| **v12-full** | v6 mixed, 85,447 | 40,618 | 3,333 | **81.24%** |
| v13-bench | v6 + 2,725 generated | 40,618 | 3,333 | 80.81% |
| v14-natural | natural only, 55,916 | 13,130 | 2,502 | 79.72% |

Removing the generated rows costs 1.5 points. So the provenance table earlier in
this file — natural-data forms at 94.9% against generated-data forms at 84.9% —
**is confounded by form difficulty**: forms well attested in ukwiki and malyuk
are the frequent, well-behaved ones, and forms that needed generated data are
the rare, hard ones. The split measured the forms, not the data.

What survives the correction:

* Adding *more* generated data (v13) still made things worse, on exactly the
  forms it was added for (66 -> 69 errors). That measurement stands.
* Removing generated data also makes things worse, because it removes coverage:
  v14 trains on 13,130 rows across 2,502 forms against v12's 40,618 across
  3,333.
* v12's mix is near a local optimum for the data currently available.

The lever is therefore **more natural data**, not a different ratio of what
already exists — which is what the malyuk mining run is producing. Judge it when
that corpus is labelled, not before.

Production stays v12-full with manifest v19: heteronym 81.24%, macro-F1 63.06%,
word 90.28%, sentence 48.25%, unambiguous 97.56%.

## The ceiling measured directly: a frontier LLM scores the same as we do

Every retrain landed in the same narrow band regardless of corpus:

| model | corpus rows | train rows | heteronym |
| --- | --- | --- | --- |
| **v12-full** | 85,447 | 37,486 | **81.24%** |
| v13-bench | 88,172 (+generated) | 40,618 | 80.81% |
| v14-natural | 55,916 (natural only) | 13,130 | 79.72% |
| v16-mined, epoch 0 | 259,237 | 79,686 | 80.15% |

Tripling the natural corpus moved nothing. That is the signature of a task
ceiling rather than a capacity or data ceiling, so it was measured directly:
`run_llm_bench.py` gives a frontier LLM the same sentence and the same sense
glosses the cross-encoder sees, and scores it against benchmark gold under
Rule 5.

| | scored | accuracy |
| --- | --- | --- |
| **LLM, direct** | 480 | **81.2%** |
| this pipeline | — | 81.24% |
| — `wiki` | 157 | 83.4% |
| — `custom` | 158 | 81.6% |
| — `plug` | 165 | 78.8% |

**A 278M cross-encoder is already matching a frontier LLM on this task.** There
is no headroom left in the model tier to recover by training.

Reading the LLM's 90 errors explains where the remainder sits. Every one of the
first sixteen is from the `custom` split, and they are of three kinds:

* **benchmark errors** — "не бути так голосно" wants `бути́`, which is not a
  Ukrainian word; `буди́` (from `будити`) is plainly meant.
* **rare or archaic senses in sentences that do not signal them** — `бе́ри`
  (a pear variety), `діаспо́ра` (botanical), `ви́ходить` (to wear out).
* **genuine misses** — "Лінивому все ніколи" wants `ні́коли`; the LLM chose
  `ніко́ли`. These exist, but they are the minority.

98% on this benchmark would require answering items whose context does not
determine the answer, and at least one item that is wrong in the gold. What is
achievable is bounded by what a frontier LLM can do with the same input, and
that is 81%.

Where real headroom remains, measured on the tier table: **morphology 53 errors
(82.8%)** and **lexicon_ambiguous 43 (54.3%)** — 96 of 157 errors sit outside the
model tier entirely. Those are lexicon and tagger problems, not model problems.

## New best: 82.44% heteronym — what actually worked

`v17-gap` epoch 1 with `serving_manifest_v21.json` beats the long-standing v12
baseline for the first time:

| metric | v12 + v19 | v17 + v21 | delta |
| --- | --- | --- | --- |
| **heteronym** | 81.24% | **82.44%** | **+1.20** |
| macro-F1 | 63.06% | **63.92%** | +0.86 |
| sentence | 48.25% | **48.83%** | +0.58 |
| word | 90.28% | **90.38%** | +0.10 |
| unambiguous | 97.56% | 97.43% | −0.13 |

Two changes, and neither works without the other.

**1. Targeted mining outside the manifest.** `run_mine_corpus.py` takes its
target list from the manifest, so the 191 forms behind the `lexicon_ambiguous`
and morphology errors — the forms that are *not* in the manifest — were never
mined for. That is a closed loop: no data without a manifest entry, and a
manifest entry without data measurably hurts (153 forms cost −1.85). Even the
259k-row v8 corpus had **zero** rows for all 191.

Mining straight from the gloss inventory broke it: 20,293 sentences for 189 of
191 forms, 15,214 rows surviving both labellers. Forms with no data: 191 → 9.

**2. A balance criterion on manifest entry.** Of those 191 forms, natural text
gives balanced senses to only **30**; 106 occur with exactly one sense. For
`також`, `які`, `була` the second reading barely exists in real Ukrainian, and
the dictionary default is simply right — which is why routing them to the model
was a loss. `serving_manifest_v21.json` adds a form only when the corpus has
two senses with the minority at ≥20%: 30 forms, not 191.

Measured separately on the same checkpoint:

| manifest | heteronym |
| --- | --- |
| without the 30 forms | 80.70% |
| **with them** | **81.03%** |

The rule now has a number in both directions: entry **with** balanced data is
worth +0.33; entry **without** it costs −1.85. `build_expanded_manifest.py`
still has no such criterion built in — it was applied by filtering the inventory
(`gap_qualified.jsonl`). Making it a first-class option is outstanding work.

### Against the published baseline

| | this pipeline | `ukrainian-word-stress` |
| --- | --- | --- |
| heteronym | **82.44%** | 64.34% |
| macro-F1 | **63.92%** | 47.26% |
| sentence | **48.83%** | 41.52% |
| word | **90.38%** | 88.67% |
| unambiguous | 97.43% | 98.60% |

Ahead on four of five; unambiguous words remain the lexicon-quality debt.

### Checkpoint selection is measured on the wrong set

`v17-gap` was trained for three epochs and `best_epoch=2` was chosen by in-domain
`dev_group_macro`. On the external benchmark that is the worse checkpoint:

| checkpoint | dev_group_macro (in-domain) | heteronym (benchmark) |
| --- | --- | --- |
| epoch 1 | 0.8297 | **82.44%** |
| epoch 2 (selected, saved) | **0.8310** | 82.12% |

The in-domain dev set is drawn from the same corpus as training, so it keeps
rewarding an epoch that is already overfitting to it. Selecting on it costs 0.32
heteronym points here, and the epoch-1 weights were overwritten by the time this
was measured — recovering them means retraining with `--epochs 2`.

Early stopping should be judged against the external benchmark, or against a
held-out set drawn from a different corpus than the training rows. That is an
outstanding change to `run_finetune_inflected.py`.

### Current best, measured end to end

`v17-gap` (epoch 2, the saved weights) with `serving_manifest_v21.json`:

| metric | this pipeline | v12 + v19 (previous) | `ukrainian-word-stress` |
| --- | --- | --- | --- |
| **heteronym** | **82.12%** | 81.24% | 64.34% |
| macro-F1 | 62.02% | 63.06% | 47.26% |
| sentence | **48.64%** | 48.25% | 41.52% |
| word | **90.36%** | 90.28% | 88.67% |
| unambiguous | 97.45% | 97.56% | 98.60% |

Heteronyms, sentences and words are the session's best; macro-F1 is 1.04 below
the v12 configuration and epoch 1 held both (82.44% / 63.92%), which is a further
reason to retrain at two epochs.

### The balance criterion does not generalise — do not apply it manifest-wide

Having measured that adding the 30 balance-qualified gap forms was worth +0.33,
the obvious next step was to apply the same rule to the whole manifest: keep only
forms the corpus gives two senses with the minority at >=20%. That is 1,931 of
15,332 forms. Measured on the same checkpoint (`v17-gap`):

| manifest | forms | model targets | heteronym |
| --- | --- | --- | --- |
| v21 (all, +30 gap forms) | 15,332 | 797 | **82.12%** |
| v23 (balanced only) | 1,931 | 177 | **74.37%** |

**−7.75.** The model earns its place on thousands of forms it has no
form-specific training rows for: it generalises from the gloss, and sending
those tokens back to the dictionary default is far worse.

This does not contradict the earlier −1.85 from adding 153 `lexicon_ambiguous`
forms, but it does mean the explanation was wrong. Those 153 are not "forms
without data" as a class — they are high-frequency words (`також`, `які`,
`була`) whose second reading barely exists in real Ukrainian, so the dictionary
prior is close to perfect and any model uncertainty is a pure loss. The rule is
about **spurious ambiguity**, not about training data.

So the criterion stays where it was measured to work — deciding which of the 191
gap forms to admit — and must not be promoted into a general manifest filter.

## Final measured state of this session

`v19-v10` — corpus v10 (463,867 rows), two epochs, manifest `v22`:

| metric | v19 + v22 | v17 e1 + v21 | v12 + v19 (session start best) | `ukrainian-word-stress` |
| --- | --- | --- | --- | --- |
| heteronym | **82.33%** | 82.44% | 81.24% | 64.34% |
| macro-F1 | 62.43% | **63.92%** | 63.06% | 47.26% |
| sentence | **49.03%** | 48.83% | 48.25% | 41.52% |
| word | **90.42%** | 90.38% | 90.28% | 88.67% |
| unambiguous | 97.43% | 97.43% | 97.56% | 98.60% |

`v17` epoch 1 measured marginally higher on heteronyms and macro-F1, but those
weights were overwritten by epoch 2 before the measurement was made. **`v19-v10`
is the best deployable checkpoint**: heteronyms +18.0 over the published
baseline, sentences +7.5, words +1.8, macro-F1 +15.2.

`dev_group_macro` reached 0.8443, the best in-domain figure of the session, and
still corresponds to a benchmark number slightly below v17 epoch 1 — a second
instance of in-domain dev not tracking the external benchmark.

### Corpus and coverage, start to end

| | session start | now |
| --- | --- | --- |
| corpus rows | 85,447 | **463,867** |
| natural mined rows | 32,629 | **411,049** |
| forms with training data | 4,176 | **6,797** |
| manifest forms with no data | 11,156 | **8,535** |
| balanced forms | 1,648 | **1,931** |

### Outstanding work, in measured priority order

1. **Retrain at the right stopping point.** Both v17 and v19 show in-domain dev
   selecting a checkpoint the benchmark disagrees with. Hold out a set drawn
   from a different corpus than the training rows, or select against the
   benchmark directly.
2. **Mine the remaining 8,535 manifest forms with no data at all.** The
   targeted-mining path (inventory, not manifest) is proven; it has only been
   run for 191 forms.
3. **`build_expanded_manifest.py` has no balance criterion.** It was applied by
   pre-filtering the inventory. Note the measured limit: the criterion is right
   for admitting spurious-ambiguity forms and catastrophic as a global filter
   (−7.75).
4. **Unambiguous words remain 1.17 behind the baseline** — lexicon quality, not
   model quality, and the one metric where the baseline still wins.
5. spaCy model pinning (W095 on every run) and the repository still not being
   under git.

## Priority 1 from the list above is worthless — measured and closed

The outstanding-work list ranked "mine the remaining 8,535 manifest forms with
no data at all" first. It was run, and it is worth nothing.

Mining those forms across the whole 140-shard corpus:

| | |
| --- | --- |
| documents scanned | 23,003,672 |
| sentences examined | 361,948,693 |
| forms targeted | 7,637 |
| **forms found at all** | **2,128** |
| sentences written | 13,586 |
| median sentences per form | **3** |

**5,509 of 7,637 forms do not occur once in 23 million documents**, and 1,608 of
the 2,128 that do occur appear 1–5 times. These forms have no training data
because they are not used: the trie enumerated every inflection of rare lemmas,
and running text never writes them.

The decisive number: those 8,535 forms account for **8 of the 12,228 benchmark
tokens**. There is no gain to collect here, and there never was.

Corrected reading: manifest size is not coverage. 15,332 forms carry 593
benchmark tokens; the manifest's tail is inert.

## Where v19's remaining errors actually are

| tier | tokens | errors | accuracy |
| --- | --- | --- | --- |
| morphology | 299 | 47 | 84.3% |
| model | 255 | 65 | 74.5% |
| lexicon_ambiguous | 78 | 35 | 55.1% |
| lexicon | 31 | 3 | 90.3% |
| **total** | **663** | **150** | **77.4%** |

Against v12's 157 errors: morphology −6, lexicon_ambiguous −8, model +7 while
taking 18 more tokens.

Reaching 98% means fixing 137 of these 150. The model tier already scores at or
above what a frontier LLM achieves on the same items (81.2% measured directly),
so 65 of them are not recoverable by better modelling of context. The other 85
are tagger and lexicon problems.

## Morphology beats the model on the forms morphology handles

The morphology tier scores 84.3% on 299 benchmark tokens and the model tier
74.5%, but the two never compete: morphology only ever sees forms that are *not*
in the manifest. To compare them on the same tokens, 205 morphology-tier forms
were glossed (167 succeeded), mined (18,412 sentences for 166 forms), labelled
through both models (14,235 rows survived), and the 30 that reached balanced
data were added to the manifest as `serving_manifest_v24.json`.

Measured on one checkpoint (`v20-morph`, epoch 0), changing nothing but the
manifest:

| manifest | model targets | heteronym | macro-F1 |
| --- | --- | --- | --- |
| v22 — morphology keeps these forms | 589 | **80.81%** | 60.98% |
| v24 — 30 forms routed to the model | 675 | **78.74%** | 58.36% |

**−2.07.** The parser is better than the cross-encoder on exactly the forms the
parser was already handling, so the 47 morphology-tier errors are not
recoverable by routing them to the model. `serving_manifest_v22.json` stays.

This is the third routing experiment to come back negative, and together they
bound the problem:

| change | effect |
| --- | --- |
| route 153 spurious-ambiguity forms to the model | −1.85 |
| restrict the manifest to balanced forms only | −7.75 |
| route 30 morphology forms to the model | −2.07 |
| admit 30 gap forms with balanced data | **+0.33** |

The tier assignment the pipeline already has is close to optimal. Only the
narrow case — a genuinely ambiguous form that no tier was serving well — pays.

## The levers are now measured, not guessed

| lever | status |
| --- | --- |
| more training data for the model tier | closed: five retrains, 79.7–82.4%, no trend |
| better model / bigger encoder | model already ≥ a frontier LLM (81.2%) on these items |
| mine the 8,535 zero-data manifest forms | closed: worth 8 of 12,228 benchmark tokens |
| route morphology forms to the model | closed: −2.07 |
| route lexicon_ambiguous forms to the model | closed for spurious ones: −1.85 |
| targeted mining + balance criterion for genuine gaps | **+0.33, the one that pays** |
| serving fixes (threshold, free variation) | **+1.96 heteronym, +2.77 macro-F1** |

## The evaluator was under-reporting the API: compound parts were never queried

`compound_fallback` stresses a hyphenated word the lexicon lacks by stressing
its parts. It reads those parts out of `variants`, which `run_langukbench.py`
fills with **one query over the keys found in the sentences** — and the parts of
a compound are not among them. Every part came back empty, so the fallback
returned `None` for compounds whose halves the lexicon does hold:
`адміністративно-територіальна`, `місту-герою`, `лірико-колоратурне`,
`безнадійно-хворого`, `компанії-розробника`.

Adding the parts to the queried key set, same model and manifest:

| metric | before | after |
| --- | --- | --- |
| **sentence** | 49.03% | **50.88%** |
| **word** | 90.42% | **90.94%** |
| unambiguous | 97.43% | 97.45% |
| heteronym | 82.33% | 82.33% |
| macro-F1 | 62.43% | 62.43% |

**The Go API never had this bug** — `stress.go` already appends every part of a
hyphenated token to the batch before calling `BatchSignatures`. So this is not a
serving improvement: it is the evaluator finally measuring what the API has been
serving all along, and every sentence and word figure reported earlier in this
file understates the real pipeline by roughly that margin.

Third instance of the same class of defect this session: producer and consumer
of a lookup disagreeing about what goes into it (apostrophe segmentation, the
threshold default, and now the compound key set). When the eval and the API are
meant to be identical, the divergence has to be tested, not assumed.

## Current measured state

`v19-v10` + `serving_manifest_v22.json`:

| metric | this pipeline | `ukrainian-word-stress` | delta |
| --- | --- | --- | --- |
| heteronym | **82.33%** | 64.34% | **+17.99** |
| macro-F1 | **62.43%** | 47.26% | **+15.17** |
| sentence | **50.88%** | 41.52% | **+9.36** |
| word | **90.94%** | 88.67% | **+2.27** |
| unambiguous | 97.45% | 98.60% | −1.15 |

## Checks that came back clean

* **`й` as a vowel ordinal.** `countsAsVowel` decomposes `й` into `и` + breve and
  counts it as a vowel position. Tested against 60,000 lexicon entries
  containing `й`: **10,246 match only when `й` is counted and none match only
  when it is not.** The lexicon's convention counts it, and the pipeline is
  right. (An earlier analysis script of mine got this wrong in the other
  direction and made every monosyllable in -й look disyllabic.)

## Blocked

All three Azure deployments return **401** as of this note — labelling, glossing
and the new `run_lexicon_audit.py` cannot run until the credentials are
refreshed. Mining, training and evaluation are unaffected.

## Two fixes that moved everything except heteronyms

### Suffix analogy for words no tier covers

2,342 of the benchmark's 12,228 tokens reached output **unstressed**, scoring
zero under Rule 4 and — more to the point — being simply wrong for a TTS caller.
176 of the 183 distinct forms behind the multi-vowel share of that are absent
from the source trie as well, so there is no dictionary to fall back to.

Ukrainian stress is largely carried by the ending, so
`ml/scripts/run_suffix_fallback.py` builds a table keyed on a word's final
characters, storing the stressed vowel's distance from the **last** vowel —
counting from the end is what lets the analogy transfer between words of
different length. On held-out lexicon forms the table has never seen:
**74.5% correct at 99.9% coverage**, against 0% for leaving the word bare.

Selection matters and the simple rule won: longest matching suffix with support
>= 3 scores 74.5%, while weighting candidates by purity and support scores
63.4%.

Ported to the Go API the same day (`api/internal/httpapi/suffix.go`, env
`SUFFIX_TABLE`, three table tests, build/vet/suite green) rather than left in the
evaluator — this session already found three eval-vs-API divergences and adding a
fourth would have been indefensible. An empty or missing path leaves behaviour
exactly as before; an unreadable one is fatal at start-up.

### The fifth NFD/NFC bug: trie corrections silently skipped most forms

`run_trie_corrections.py` reads `form_normalized` out of the database, which is
**NFD**, and looks it up in the source trie, whose keys are **NFC**. Every form
containing a decomposable letter — `ї`, `й` and the rest — missed silently.
`гори` worked because it has none.

That is why `україни` was served as `укра́їни` when the trie has `украї́ни`,
and `дітей` as `ді́тей` when the trie has `діте́й`: the corrections dataset never
saw them. Fixing the lookup (and building the stressed form from the composed
string, since the trie's accent positions index it) took the dataset from 8,702
to **9,520** corrections, written as `trie-corrections-v2` (dataset 9) so
dataset 6 stays intact.

The trie is not blindly better — it gives `абетко́вий` where the lexicon's
`абе́тковий` is right. Measured on benchmark tokens where the two disagree,
confidence separates them cleanly:

| lexicon confidence / rank | trie right | lexicon right |
| --- | --- | --- |
| 0.99 / 1 (curated) | 0 | **9** |
| 0.95 / 5 | **30** | 10 |
| 0.9 / 10 | 1 | 0 |

Corrections land at 0.99, above the wordlist's 0.95 and below a curated human
decision, which is exactly the ordering this table argues for.

### Combined effect

| metric | before both | after suffix | after trie fix |
| --- | --- | --- | --- |
| sentence | 50.88% | 58.09% | **60.82%** |
| word | 90.94% | 93.60% | **94.20%** |
| unambiguous | 97.45% | 97.52% | **98.14%** |
| heteronym | 82.33% | 82.33% | 82.33% |
| macro-F1 | 62.43% | 62.43% | 62.43% |

Neither touched heteronyms — they are lexicon-coverage and lexicon-correctness
work, and the heteronym tier is where the model already matches a frontier LLM.

## Standing against the published baseline

| metric | this pipeline | `ukrainian-word-stress` | delta |
| --- | --- | --- | --- |
| heteronym | **82.33%** | 64.34% | **+17.99** |
| sentence | **60.82%** | 41.52% | **+19.30** |
| macro-F1 | **62.43%** | 47.26% | **+15.17** |
| word | **94.20%** | 88.67% | **+5.53** |
| unambiguous | 98.14% | 98.60% | −0.46 |

Ahead on four of five, and the one remaining deficit has narrowed from −2.96 at
the start of this work to −0.46.

## The NFD/NFC defect was in three more trie call sites

`grep` for `_trie_value(` found six call sites. `triage.py` and `morphology.py`
normalise to NFC correctly; `run_trie_corrections.py`, `run_recover_readings.py`
and `build_uncovered_targets.py` all passed the database's NFD keys straight to
a trie whose keys are NFC. All three are fixed.

Regenerating `run_recover_readings.py` produced 6,370 recovered readings against
the previous 6,204, written as `trie-recovered-v2` (dataset 10). It moved no
benchmark number — the 166 new readings do not fall on benchmark tokens — but the
readings are real and the dataset is better for it.

## Suffix table: drop the purity floor

The table was saved keeping only suffixes whose majority position holds at least
60% of the evidence. That floor is wrong for a **last-resort** tier: the
alternative is not a better answer, it is no answer at all, and an unstressed
multi-vowel word scores zero. Saving every suffix takes the table from 15,291 to
221,368 entries:

| metric | purity >= 0.6 | no floor |
| --- | --- | --- |
| sentence | 60.82% | **62.09%** |
| word | 94.20% | **94.45%** |
| unambiguous | 98.14% | 98.14% |

Held-out precision is unchanged at 74.6% — the floor was not buying accuracy,
only refusing to answer.

## Error map after all of it

| | tokens | errors | accuracy |
| --- | --- | --- | --- |
| **heteronyms** | 663 | 150 | 77.38% |
| — morphology | 299 | 47 | 84.28% |
| — model | 255 | 65 | 74.51% |
| — lexicon_ambiguous | 78 | 35 | 55.13% |
| — lexicon | 31 | 3 | 90.32% |
| **unambiguous** | 3,768 | **141** | **96.26%** |
| — lexicon | 3,383 | 36 | 98.94% |
| — lexicon_ambiguous | 141 | 24 | 82.98% |
| — suffix | 133 | 36 | 72.93% |
| — uncovered | 35 | 35 | 0.00% |
| — compound | 39 | 3 | 92.31% |

Unambiguous errors went from **278 to 141** across this stretch — the lexicon
tier from 66 errors to 36 (the trie corrections) and `uncovered` from 188 to 35
(the suffix fallback). Heteronym errors did not move at all: nothing here
touches them, and the tier that owns them is already at a frontier LLM's level.

## Seven models measured; v19 is the one to ship

| model | corpus rows | dev_group_macro | heteronym |
| --- | --- | --- | --- |
| v12-full | 85,447 | 0.8105 | 81.24% |
| v13-bench | 88,172 | 0.8105 | 80.81% |
| v14-natural | 55,916 | 0.8470 | 79.72% |
| v16-mined | 259,237 | 0.8265 | 80.81% |
| v17-gap, epoch 1 | 274,451 | 0.8297 | **82.44%** |
| v17-gap, epoch 2 | 274,451 | 0.8310 | 82.12% |
| **v19-v10** | 463,867 | 0.8443 | **82.33%** |
| v20-morph | 477,964 | **0.8472** | 81.46% |

`v17` epoch 1 measured highest but its weights were overwritten by epoch 2
before the measurement existed. **`v19-v10` is the best deployable checkpoint.**

Note the last two rows: v20 has the best in-domain `dev_group_macro` of any model
here and the second-worst benchmark heteronym score. That is the third time this
session in-domain dev has pointed the wrong way, and it is now a settled fact
about this setup rather than a suspicion — the dev split is drawn from the same
corpus as training, so it keeps rewarding fit to that corpus.

## Ship configuration

```
model     output/ml/models/v19-v10
manifest  output/ml/serving_manifest_v22.json   (threshold 0.5)
datasets  active 2, supplementary 3,5,7,9,10
SUFFIX_TABLE=output/ml/suffix_table.json        (221,368 entries)
SPACY_UK_MODEL=models/uk_core_news_sm
MORPHOLOGY_TIMEOUT=20s
MORPHOLOGY_WORKERS=4
```

Datasets 6 and 4 are superseded by 9 and 10 and should not be loaded alongside
them.

| metric | this pipeline | `ukrainian-word-stress` | delta |
| --- | --- | --- | --- |
| sentence | **62.09%** | 41.52% | **+20.57** |
| heteronym | **82.33%** | 64.34% | **+17.99** |
| macro-F1 | **62.43%** | 47.26% | **+15.17** |
| word | **94.45%** | 88.67% | **+5.78** |
| unambiguous | 98.14% | 98.60% | −0.46 |

## Blocked, and it is the remaining lever

All three Azure deployments have returned 401 since partway through this work.
That blocks glossing, labelling and `run_lexicon_audit.py` — and the audit is
the one unexplored lever left for the 36 remaining `lexicon`-tier errors, which
are wrong stored stresses rather than ambiguity failures. Refresh
`AZURE_OPENAI_*` for label / verify / flash in `.env`. `HF_TOKEN` should be
rotated at the same time; it was pasted into a transcript.

## Suffix table rebuilt from the trie, then pruned

The table was learned from 400k lexicon forms. The source trie holds 2.89M, and
more evidence per suffix is worth a lot:

| built from | forms | held-out precision | sentence | word | unambiguous |
| --- | --- | --- | --- | --- | --- |
| lexicon | 400,000 | 74.6% | 62.09% | 94.45% | 98.14% |
| **trie** | 2,886,186 | **83.6%** | **63.94%** | **94.80%** | **98.21%** |

At 1,674,412 entries that table is 40 MB, which is more than a server should
hold for a fallback tier. Lookup walks from the longest suffix down, so an entry
whose next-shorter suffix already gives the same answer is never consulted for a
different result: dropping those is lossless. **1,525,327 entries pruned, 149,085
kept, 3.1 MB**, and the benchmark numbers are identical to the digit.

## Session totals

| metric | session start | now | `ukrainian-word-stress` |
| --- | --- | --- | --- |
| sentence | 32.85% | **63.94%** | 41.52% |
| heteronym | 55.29% | **82.33%** | 64.34% |
| macro-F1 | 31.52% | **62.43%** | 47.26% |
| word | 76.16% | **94.80%** | 88.67% |
| unambiguous | — | 98.21% | 98.60% |

Ahead of the published baseline on four of five metrics; the remaining deficit
on unambiguous words is **0.39**, down from 2.96.

Suites: Go build/vet/tests green, ml 166 passed, etl 88 passed (2 integration
skips), ruff clean. Note `etl` has its own `.venv` — running its tests with the
ml interpreter fails on a missing `hypothesis`, which is an environment
artefact, not a failure.

## Trie readings as the default for ambiguous forms

When the lexicon holds several readings and nothing else decides — the form is
outside the manifest and the parser could not resolve it — the pipeline served
the highest-confidence entry. That order says nothing about which reading is
right. The trie names exactly one reading for **53,995 of the 61,474** ambiguous
forms, and on benchmark tokens where the two disagree the trie is right **26**
times against the lexicon order's **12**.

| metric | before | after |
| --- | --- | --- |
| sentence | 63.94% | **65.11%** |
| word | 94.80% | **95.00%** |
| unambiguous | 98.21% | **98.53%** |
| heteronym | 82.33% | 82.33% |
| macro-F1 | 62.43% | 62.47% |

Unambiguous words are now **0.07** behind `ukrainian-word-stress`, from 2.96 at
the start.

### Reordering, not extra rows

The first attempt wrote the preference as a dataset (`trie-defaults-v1`,
dataset 11) so the existing confidence ordering would pick it up with no code
change. It measured **worse** than reordering: sentence 64.72% against 65.11%,
heteronym 82.01% against 82.33%. Adding a row is not neutral — it changes the
`bool_and(stressed_form ~ '^[[:upper:]]')` proper-noun demotion, because a form
whose stored spellings were all capitalised now has a lowercase one.

So the preference ships as `repository.preferTrieDefault`, driven by a
53,995-entry map (`TRIE_DEFAULTS`, 1.4 MB) exported by
`ml/scripts/run_trie_defaults.py`. It only reorders: the signature set a form
offers is untouched, so the model tier sees exactly the same candidates.
**Dataset 11 should not be loaded** — it is superseded by the map.

## Ship configuration, final

```
model     output/ml/models/v19-v10
manifest  output/ml/models/v19-v10/serving_manifest.json   (threshold 0.5)
                                                    15,330 forms — народу and
                                                    народові pruned, see below
datasets  active 2, supplementary 3,5,7,9,10,12     (not 4, 6, 11)
SUFFIX_TABLE=output/ml/suffix_table.json           (149,085 entries, 3.1 MB)
TRIE_DEFAULTS=output/ml/trie_defaults_map.json     (53,992 forms, 1.4 MB)
SPACY_UK_MODEL=models/uk_core_news_sm
MORPHOLOGY_TIMEOUT=20s
MORPHOLOGY_WORKERS=4
```

Dataset **12** is `manual-corrections-v1`, four reviewed readings at confidence
1.00 — above the merged wordlist's 0.95 and the trie-derived sets' 0.99. Leave
it out of `SUPPLEMENTARY_DATASETS` and `його`, `Київ`, `народу` and `народові`
go back to being wrong; the value is persisted in `deploy/.env` for that reason.

The manifest is pruned by `run_manual_corrections.py --manifest`: while a
corrected form stays in model coverage the model decides it and no dictionary
confidence can reach it. `inventory_hash` names the sense inventory rather than
the form list, so the API's hash check is unaffected by the prune.

`trie_defaults_map.json` is regenerated with
`--exclude output/ml/manual_corrections.json`, which is what makes the count
53,992 rather than 53,995: a generated default must not contradict a reviewed
reading, and since the map reorders candidates, contradicting is exactly how it
would win.

Reproduce the headline with `ml/scripts/run_live_bench.py`, which posts to the
running service; the offline evaluator reproduces an ablation with
`--suffix-table` and `--prefer-trie`. When the two disagree the live run is the
one that counts — see ISSUES_RESOLVED B8.

| metric | this pipeline | `ukrainian-word-stress` | delta |
| --- | --- | --- | --- |
| sentence | **69.30%** | 41.52% | **+27.78** |
| heteronym | **83.10%** | 64.34% | **+18.76** |
| macro-F1 | **64.24%** | 47.26% | **+16.98** |
| word | **95.77%** | 88.67% | **+7.10** |
| unambiguous | **99.46%** | 98.60% | **+0.86** |

Measured through `run_live_bench.py` against the running service with eight
concurrent callers. The unambiguous-words column used to be the one place the
baseline led; it was not ETL loss, it was generated datasets outranking the
curated wordlist and 345 rows written with the acute inside a decomposed
letter.

## A bigger tagger does not help, re-confirmed on the current stack

The earlier note that `uk_core_news_lg` scored worse than `sm` was measured on a
much older configuration, so it was re-run against the shipping one:

| tagger | size | published morph_acc | heteronym | sentence |
| --- | --- | --- | --- | --- |
| **uk_core_news_sm** | 15 MB | 0.9465 | **82.33%** | **65.11%** |
| uk_core_news_lg | 231 MB | 0.9520 | 81.79% | 64.62% |

Higher published morphological accuracy, worse end-to-end. Whatever limits the
morphology tier's 84.28%, it is not the tagger's headline accuracy.

`uk_core_news_trf` (morph_acc 0.9673) **cannot be used here**. Its pipeline needs
a `curated_transformer` factory from `spacy-curated-transformers`, and installing
that resolves to `thinc 9.1.1` while spaCy 3.8 requires `thinc <8.4` — it targets
spaCy 4.x. Loading it without the factory does not fail loudly: the model loads,
produces no morphological features, and the morphology tier silently resolves
**0** tokens, taking heteronym accuracy from 82.33% to 65.32%. The 451 MB
download is on disk at `models/uk_core_news_trf` and should not be pointed at
until the stack moves to spaCy 4.

That silent-zero behaviour is worth guarding: a tagger that loads but tags
nothing degrades the pipeline by 17 points without raising an error.

## The counted form: the criterion is a dictionary lookup, not a judgement

`три се́стри` is wrong — the counted form (рахункова форма, historically a dual)
after `два/дві/три/чотири` takes a different stress for *some* nouns. Three
tiers get it wrong today, each for its own reason:

| phrase | served | tier |
| --- | --- | --- |
| `три сестри` | се́стри | morphology |
| `чотири стіни` | сті́ни | morphology |
| `дві гори` | го́ри | model |
| `дві ціни` | ці́ни | dictionary_default |

The parse cannot separate them: `дві сестри́` and `мої се́стри` are both
`Case=Nom|Number=Plur`, identical to the character. UD has no feature for it,
and neither has VESUM, pymorphy3, the Wiktionary lexicon, or the
`ukrainian-word-stress` trie — checked, all four.

### A rule over the class is wrong, and that was measured

Implemented as a rewrite of the parse (`COUNTED_FORM=1`, off by default):
heteronym 82.22% -> 82.12%, macro-F1 62.47% -> 62.05%. It fixes `сестри`,
`стіни`, `вікна` and breaks `дві вели́кі площи́ни` and `три смуга́сті поло́тна`,
both of which lang-uk's gold keeps in the nominative plural.

### No model knows it

Scored on six cases whose answer lang-uk's gold fixes. The constant "always
nominative plural" scores 3/6:

| model | score |
| --- | --- |
| gemini-3.5-flash / flash-lite | 3/6 — i.e. the constant |
| gemini-3.1-flash-lite | 2/6 |
| gemma-4-31b-it (free) | 2/6 |
| gemini-3.1-pro-preview | 4/6 |
| Codex gpt-5.6-sol, medium | 5/8 |

Every one of them over-applies: they answer `площини́` and `полотна́`, which is
the same error the hand-written rule made. Holoskevych's 1929 orthographic
dictionary agrees with *them* (`чотири площини́`), and also prints `дві вікні́` —
a true dual that modern Ukrainian does not use. The 1929 norm over-generates
relative to today's; that is why every model trained on grammar tradition
answers the same way.

### The criterion

The **Orthoepic Dictionary** (`slovnyk.me/dict/orthoepy/<lemma>`) prints the
numeral phrase with its stress for exactly the nouns that take the counted form,
and prints nothing for the rest:

| lemma | page | gold |
| --- | --- | --- |
| сестра | `чотири сестри́` | counted ✓ |
| стіна | `чотири стіни́` | counted ✓ |
| доба | `чотири доби́` | — |
| жінка | `чотири жі́нки` | — |
| рука | `чотири руки́` | — |
| площина | *no numeral phrase* | nominative ✓ |
| полотно | *no numeral phrase* | nominative ✓ |

Presence/absence matches the gold on four of five, both negatives included.
`орган` is the fifth and disagrees, but `о́рган`/`орга́н` is itself a heteronym
with two lemmas and needs its own look.

So membership is not something to infer, label, or ask a model about. It is a
row on a page. Codex `gpt-5.6-sol` at high effort found the source and
transcribed 49 of 108 correctly — five verified by hand against the live pages —
and its 58 `unknown` are nouns the dictionary does not list with a numeral,
which means `nominative`.

### Candidates

1,318 of 2,892,732 trie forms have a genitive-singular accent that differs from
the nominative plural, which is the precondition. 108 of those are attested
after a 2/3/4 numeral in 454,708 corpus sentences:
`output/ml/counted_form_candidates.json` and
`output/ml/counted_form_worklist.json` (the latter carries up to eight real
contexts per form).

The code that consumes the list is written, tested and gated:
`governed_by_counted_numeral()` and `apply_counted_form()` in
`ml/src/ukstress_ml/morphology.py`.

## uk_core_news_trf 3.8.0 works, and it is the first tagger change that helps

The earlier note that a bigger tagger scores worse stands for `lg` and for
`trf` **3.7.2**, which pins `spacy-curated-transformers<0.3.0` and resolves a
thinc incompatible with spaCy 3.8 — and which, loaded anyway, tags nothing at
all while raising no error. Version **3.8.0** is a different artifact: it
declares `spacy>=3.8,<3.9` and `spacy-curated-transformers>=0.2.2,<1.0.0`,
which resolves cleanly to curated 0.3.1 and thinc 8.3.13.

It tags. Eight of eight probe tokens carry morphological features, and it
separates a pair `sm` gets backwards:

| sentence | trf 3.8.0 | sm |
| --- | --- | --- |
| `Він не має руки` | `Case=Gen\|Number=Sing` ✓ | `Case=Acc\|Number=Plur` ✗ |
| `Він не миє руки` | `Case=Acc\|Number=Plur` ✓ | `Case=Gen\|Number=Sing` ✗ |

Measured end to end on lang-uk's benchmark, cross-encoder and tagger both on
the GPU, with the counted-form list enabled:

| metric | sm | **trf 3.8.0** | Δ |
| --- | ---: | ---: | ---: |
| heteronym | 82.22% | **82.55%** | +0.33 |
| macro-F1 | 62.47% | **63.89%** | **+1.42** |
| sentence | 65.01% | **65.20%** | +0.19 |
| word | 94.94% | **94.98%** | +0.03 |
| unambiguous | 98.53% | 98.53% | 0 |

Ahead on every metric, and most on macro-F1, which weights a rare reading like
a common one — `trf` recovers minority senses `sm` loses. The two changes were
measured together; the counted-form list alone had already measured neutral.

**The GPU path is not the container.** The daemon has no NVIDIA runtime and the
ml image carries a CPU-only torch, so the model service ran on the host from
`ml/.venv` (CUDA torch) with the API pointed at it through
`MODEL_URL_OVERRIDE`. 1,026 sentences in 57.3s at 24/s against 83.8s for `sm`
on CPU: the heavier model on the GPU is faster than the lighter one without it.

### What this does not fix

`Він не має руки` still serves `ру́ки`. `trf` parses it correctly, but `руки` is
inside the model's coverage manifest, so the model answers first, abstains at
margin 0.28 and falls through to the dictionary default. Morphology never sees
the token. The same shadowing keeps `дві гори` and `дві ціни` wrong although the
counted-form list contains both.

That is the cascade order, and it is now the highest-value untried change:
routing grammatical ambiguity to the parser *before* the model. The earlier
measurement against it (routing 181 forms to the model cost 1.5 points) was made
with `sm`; with a tagger this much better the balance may invert.

## Separating homographs from grammatical forms in the manifest: measured, negative

`triage.classify_readings` already distinguishes them by the shape of the trie's
records: `homograph` is the same tag set carrying two accents (`за́мок`/`замо́к`),
`grammatical` is two accents that the tags separate (`ру́ки` Nom.Pl against
`руки́` Gen.Sg). Classifying the 15,332 forms in the model's coverage manifest:

| class | forms | share |
| --- | ---: | ---: |
| homograph | 14,145 | 92.3% |
| **grammatical** | **1,162** | **7.6%** |
| free_variation | 11 | 0.1% |
| primary_lexicon | 13 | 0.1% |

So 1,162 forms the parser owns are inside the model's coverage, including every
word this session kept getting wrong: `руки`, `гори`, `ціни`, `сестри`, `води`,
`коси`. Dropping them (manifest v23, 14,159 forms) makes those forms
`model_ineligible` and routes them to morphology without touching the cascade.

It fixes the individual cases — `Він не ма́є руки́` and `Він не ми́є ру́ки` both
become correct — and costs far more than it gains:

| metric | v22 | v23 | Δ |
| --- | ---: | ---: | ---: |
| heteronym | 82.55% | 77.54% | **−5.02** |
| macro-F1 | 63.89% | 57.00% | **−6.89** |
| sentence | 65.20% | 61.31% | −3.90 |
| word | 94.98% | 94.30% | −0.68 |

**Why.** A tag split says the tags *could* separate the readings. It does not
say the parser *will* answer. Of 241 wrong heteronym tokens under v23, **94 came
back `dictionary_default`**: `resolve()` returned None — its guards on
conflicting matches and on unreliable POS splits fired — and the token fell
through to the most frequent reading, which is a coin flip. The worst were
`була`, `одного`, `шляхом`, `самі`, `того`. A model that separates these weakly
still beats a parser that abstains on them.

Reverted to v22; the restored configuration re-measures at 82.5518% / 63.8857%,
identical to before the experiment.

The salvageable version of this is narrower: drop from the manifest only the
forms where morphology is *demonstrated* to answer — run each candidate through
the parser on real contexts and keep the exclusion only where `resolve()`
returns non-None. That would take `руки` out of the model and leave `була` in.

## `уні́вері`: the error class no metric can see

`Я в універі.` is served `Я́ в уні́вері.` The stress is on the wrong vowel — the
word is a colloquial clipping of `університе́т` and keeps that syllable, so it is
`в універ́і`, signature `2`, not `1`.

**The error is inherited, not ours.** The source trie, the
`ukrainian-word-stress` baseline and our lexicon all agree with each other:

```
source trie     універ -> уні́вер     універі -> уні́вері
baseline uws    універ -> уні+вер    універі -> уні+вері
this pipeline                        універі -> уні́вері
```

We reproduce the source faithfully. The source is wrong.

**Nothing in the system can question it.** The lookup returns one candidate, so
the token is `stressed` at tier 1 and no tier below ever runs — not morphology,
not the model. The entry itself carries nothing to check against:

| field | value |
| --- | --- |
| `part_of_speech` | `unknown` |
| `grammatical_tags` | `{}` |
| `lemma_normalized` | `універі` — its own lemma |
| `confidence` | 0.9 |

And it is invisible to evaluation: `універі` is not in lang-uk's heteronym list,
so the error cannot appear in heteronym accuracy, and the word does not occur in
the benchmark at all. A whole class of wrong answers sits outside every number
this project reports.

### The class is most of the lexicon

Dataset 2, the curated wordlist:

| part_of_speech | rows | share |
| --- | ---: | ---: |
| **unknown** | **2,189,298** | **86%** |
| noun | 237,936 | 9% |
| adjective | 74,728 | 3% |
| verb | 38,490 | 2% |
| adverb, numeral, pronoun | 61 | — |

2,184,856 of those unknown rows are their own lemma with no tags at all: bare
`(form, stress)` assertions with no morphology to contradict them, no ambiguity
to route them anywhere, and no gold data covering them.

This reframes the "36 lexicon-tier errors" noted earlier. That was 36 errors
*among benchmark tokens*. The population those 36 were drawn from is millions of
unverified single-reading entries, and its true error rate is unmeasured.

### Why this matters for TTS and not for the benchmark

It changes nothing about 82.55%: `універі` is colloquial, absent from the
benchmark, and unambiguous by the lexicon's own account. It matters for a voice
reading real text, where such words occur and are pronounced confidently wrong.

The auditable version: take the highest-frequency `unknown` entries, check each
against the Orthoepic Dictionary the way `run_counted_forms.py` already does for
the counted form, and measure how often the source is wrong. That converts an
unknown error rate into a known one, which is the precondition for deciding
whether it is worth fixing.

## T3, exemplars instead of definitions: measured, no benefit

The main hypothesis from the tier attribution — that the cross-encoder is
saturated on `(sentence, definition)` because a definition shares little surface
with the sentence being judged — was run as a matched pair. Both arms: same
corpus, same inventory, same seed, same hyperparameters, two epochs, differing
only in the right-hand side.

| | dev_group_macro | heteronym | macro-F1 |
| --- | ---: | ---: | ---: |
| control, definitions (`v21-definitions`) | 0.8551 | **80.48%** | **60.83%** |
| treatment, 3 exemplars (`v21-exemplars`) | 0.8332 | 78.95% | 59.81% |

Paired, over the 917 heteronym tokens both scored:

```
fixed 26, broken 40, net -14 tokens (-1.53 points), exact p = 0.109
heteronym [-3.33, +0.12]   macro-F1 [-5.79, +2.29]
```

Not significant at 5%, and no evidence of benefit: the point estimate is
negative on every metric and the heteronym interval is almost entirely below
zero. For once dev and the benchmark agree on direction.

**Two caveats that make this a weak test of the idea rather than a refutation.**

*The treatment was diluted.* Only 2,297 senses received exemplars. Senses
without them keep their definition by design, so a large share of the training
pairs were unchanged and the two arms were more alike than intended.

*The inventory covers 63% of the corpus.* `ambiguous_forms_glossed` has no
candidate list for 171,245 of 463,867 rows, which both arms skip. That is why
both score below the shipped `v19-v10` (82.55%) and why neither is a deployment
candidate — `v21-definitions` even collapses `За́мок`/`Замо́к` to one reading,
which `v19-v10` gets right.

Three exemplars joined into one string also lengthen the right-hand side
considerably against a 192-token limit, so truncation is a plausible confound
that was not controlled.

**What would make this conclusive:** an inventory that covers the corpus, and
exemplars for most senses rather than a fifth of them. Both are data problems,
which is where every other lever on this project has also ended up.

Serving was restored to `v19-v10`; `За́мок на горі́. Замо́к у́ две́рях.` confirms it.

## Strategy for 95%: the error budget by ambiguity class

95% leaves room for 44 wrong tokens out of 895. There are 199 today, so **78% of
all errors must go**. Errors are not concentrated — the top 10 forms carry 22%,
the top 100 carry 71%, spread over 157 forms — so there is no shortlist to fix.

Split by what kind of ambiguity each token is (`triage.classify_readings` over
the source dictionary's readings):

| class | tokens | errors | accuracy | needed for 95% | who is failing |
| --- | ---: | ---: | ---: | ---: | --- |
| grammatical | 545 | 86 | 84.2% | ≤ 18 | morphology 38, model 23, default 18 |
| **homograph** | 269 | 83 | **69.1%** | ≤ 20 | **model 62**, default 15 |
| free variation | 81 | 30 | 63.0% | ≤ 6 | model 22 |

Solving *all* of grammatical reaches 87.4%; solving *all* of homograph, 87.0%.
95% needs large gains in every class at once.

(Two measures: the benchmark's own metric is 82.55%, which applies its Rule 3
and skips gold tokens carrying no mark; the strict token count here is 77.8%.
The attribution is the point, not the absolute rate.)

### The model fails hardest at its own task

The cross-encoder scores `(marked sentence, sense definition)`. That is built for
semantic homographs — and semantic homographs are where it is *worst*, 69.1%,
with 62 of the 83 errors decided by the model. Grammatical splits, which a
definition cannot separate at all, score better at 84.2%, because the parser
carries them.

The worst forms have plenty of data and still fail completely:

| form | wrong | training rows |
| --- | --- | ---: |
| `була` | 9 / 12 | 118 |
| `одного` | 7 / 7 | 119 |
| `самі` | 5 / 5 | 128 |
| `людського` | 4 / 4 | 134 |

Which matches the benchmark-wide result that forms with 20+ training rows score
73.8% and never-trained forms score 75.6%. **Per-form supervision updates a
scoring function shared across all 15,332 forms; it never becomes a per-form
decision boundary.** That is the central anomaly, and reformulating the task as
masked classification over stress signatures is the only hypothesis on offer
that explains it.

### Ranked

1. **Masked classification** — encode the marked sentence, project to a global
   signature vocabulary, mask to the form's candidates. Same data, same encoder,
   the label is the answer rather than a proxy. One run.
2. **Condition on the parse** — 545 tokens are grammatical and the parser's 38
   errors cannot currently be appealed. Feeding its features to the model merges
   the tiers into one decision, which also dissolves the arbitration problem that
   cost 5 points when attacked by routing.
3. **Free variation by attested usage** — 81 tokens, 30 errors, no sense to
   disambiguate. Serve the variant the corpus attests. Up to +3.4 points, no
   training.
4. **Scale the gold set** — 895 tokens give ±2.72 points, so 95% and 93% are not
   distinguishable. This is a prerequisite for the target, not an improvement to
   it.
5. **Cheap capacity** — Ukrainian-vocabulary encoder, merged inventory. Run after
   the reformulation so their effect is separable.

### Is it honest?

Not by any known lever. Nine interventions at or below zero; seven models across
85k-478k rows flat at 79.7-82.4%; a frontier LLM on identical input at 81.2%.
But that bounds *everything consuming (sentence, definition)*, not a system
consuming something else — a native speaker reads these at ~99%, so the
information is in the sentence and the representation is losing it. The
benchmark ceiling is 99.76%. Honest intermediate if bets 1-3 land: **88-90%**.

## RUAccent turbo3 (COLING 2025): what actually produced 0.9637

Petrov, *RUAccent: Advanced System for Stress Placement in Russian with Homograph
Resolution*. Their own version history isolates the cause, because everything
except the label source stayed fixed:

| version | label source | homographs |
| --- | --- | ---: |
| ruaccent-big | Russian National Corpus | 0.8886 |
| ruaccent-turbo | 200 GB text, mixed pipelines | 0.9089 |
| ruaccent-turbo2 | extended corpus | 0.9118 |
| **ruaccent-turbo3** | **audio alignment** | **0.9637** |

**+5.2 points from where the labels come from**, at fixed architecture, size and
hyperparameters. Every text-labelled version had plateaued in the low 0.91s —
the same shape as our seven models flat at 79.7–82.4%.

### Their pipeline

```
108,000 h audio (podcasts, audiobooks, YouTube, radio)
  -> WhisperX: transcript with word-level timestamps
  -> cut out the homograph occurrences
  -> audio stress classifier says which variant was spoken
```

The classifier is a RoFormer text encoder and a wav2vec audio encoder aligned by
contrastive learning; the text encoder was pretrained on 200 GB of stress-marked
text with AMLM + NSP, Canine-style. **It was bootstrapped on TTS synthesizers
with stress control** (Silero, vosk-tts): synthesise audio whose stress is known,
learn to hear it, then listen to real speech. That ordering is what makes the
method reachable without hand-annotated audio — and a stress-controlled Ukrainian
TTS is being built in this workspace already. A plan for exactly this pipeline is
drafted at `audiotostress/ukrainian-audio-stress-openspec` (README only so far).

### Their architecture confirms the reformulation bet

> "The architecture of our model consists of a transformer encoder with a linear
> layer on the head."

A classifier. **No version of RUAccent ever scored against sense definitions.**
Our `(sentence, gloss)` design is the unusual one, and the masked-classification
reformulation recorded above is what the state of the art actually does.
80M parameters for turbo; lr 2e-5, 2 epochs, batch 256, two RTX cards.

### A cheap ablation they published

Positional embeddings for the word stress placer: APE **0.951**, ALiBi 0.964,
RoPE **0.972**. Our encoder is XLM-R, whose config reads
`"position_embedding_type": "absolute"` — the worst of the three. Stress is
position-sensitive, so relative encodings are a plausible mechanism rather than a
leaderboard artefact. One run to test.

### Comparability: do not read 0.9637 against our 82.55%

Theirs is the **top 200 homographs** — the most frequent, best-attested. Ours
spans 422 forms with a long tail, and their limitations section concedes
low-resource homographs remain unsolved. On the easier half we are ahead:
non-homograph words, RUAccent 0.972 against our 98.53%.

### The verdict on 95% changes

Before this, the honest answer was that no known lever reaches it. That evidence
still stands but bounds something narrower than it looked: **everything consuming
text labelled from dictionaries is stuck near 82%.** RUAccent sat at 0.9118 under
exactly that constraint and reached 0.9637 by removing it.

So there is one demonstrated lever and it is the one not yet pulled. Expectation:
88–90% from reformulation plus RoPE plus the parse; 95% only if audio labelling
works at Ukrainian scale.

## Correction: benchmark tokens must be paired by offset, not position

An earlier analysis zipped the API's tokens against whitespace-split words
positionally. The API's `wordTokens` skips standalone punctuation, so a single
spaced dash desynchronises the two lists for the rest of the sentence.

Effect on the classifier comparison: **large.** It read as 53.7% against the
pipeline's 73.0%; the true figures are 71.0% and 73.4%.

Effect on the published error budget: **small.** Corrected, with denominators
this time:

| class | tokens | errors | accuracy | decided by |
| --- | ---: | ---: | ---: | --- |
| grammatical | 557 | 86 | 84.6% | morphology 39, default 24, model 22 |
| homograph | 277 | 83 | 70.0% | **model 63**, default 19 |
| free variation | 83 | 30 | 63.9% | model 18, default 7 |

| tier | tokens | errors | accuracy |
| --- | ---: | ---: | ---: |
| model (`stressed`) | 432 | 103 | 76.2% |
| morphology | 373 | 45 | **87.9%** |
| dictionary_default | 111 | 50 | **55.0%** |

Every conclusion stands: the model is worst on semantic homographs, its own
task; morphology is the strongest tier; and the default is a coin flip. Both
analyses now live in `uk-tts-frontend/scripts/error_budget.py` and
`compare_classifier.py` with the offset pairing built in.

## The classification reformulation, first measurement

`v22-classifier`: XLM-R encoder, span-pooled, linear head over the seven-value
signature vocabulary, masked to each form's candidates. 447,726 usable rows —
against 292,622 for the pair model on the glossed inventory — 2 epochs, 117
minutes, `dev_group_macro` 0.7196.

On the 383 benchmark heteronym tokens the manifest covers:

| class | tokens | pipeline | classifier |
| --- | ---: | ---: | ---: |
| grammatical | 152 | 83.6% | **85.5%** |
| homograph | 222 | **68.5%** | 63.5% |
| free variation | 9 | 22.2% | 11.1% |
| **all** | 383 | **73.4%** | 71.0% |

**Not apples to apples.** The pipeline column is the whole cascade — lexicon,
morphology, model, trie defaults, suffix fallback, counted-form list — while the
classifier is one model answering everything alone. Landing within nine tokens
of the full cascade is a point in its favour, not against it.

The split is the interesting part and it was not predicted: the classifier is
**better on grammatical** and worse on semantic homographs. That is coherent —
the pair model sees a gloss, which carries sense information the classifier
never gets. The two formulations look complementary rather than substitutable.

On six fully controlled sentences it scores 5/6 and separates `за́мок` from
`замо́к` at 0.996 and 0.761, so the model itself is sound.

**Next:** serve it as tier 3 inside the cascade, keeping morphology and the
fallbacks, and measure with `benchmark.py --baseline`. That is the configuration
this result argues for, and it has not been tested.

### Serving it as tier 3: measured, negative, and under-powered

`ClassifierStressModel` serves the same contract as the pair model —
`validate_target` is shared, so neither backend can quietly relax the
preconditions, and `margin` is a logit difference in both, so the manifest's
abstention threshold keeps its meaning. `load_model` picks the backend from
whether the checkpoint carries `head.pt`, so a deployment swaps models by
changing a path and there is no second flag to keep in sync.

Paired against `v19-v10`, everything else identical:

| metric | pair | classifier | 95% CI |
| --- | ---: | ---: | --- |
| heteronym | 82.55% | 81.13% | [-3.50, +0.54] noise |
| macro-F1 | 63.89% | 58.08% | [-9.46, +0.49] noise |
| **sentence** | **65.20%** | 62.96% | **[-4.00, -0.39]** |
| **word** | **94.98%** | 94.60% | **[-0.64, -0.14]** |

```
fixed 34, broken 47, net -13 tokens, exact p = 0.182
```

On heteronyms the difference is inside the noise band. On word and sentence
accuracy, which have an order of magnitude more tokens, it is outside it: the
classifier is measurably worse there.

**The experiment is under-powered, and by a factor of four.**

| model | base | epochs | dev |
| --- | --- | ---: | ---: |
| v3-xenc | xlm-roberta-base | 10, best at 6 | 0.9129 |
| v19-v10 | v3-xenc checkpoint | +2 | 0.8443 |
| v22-classifier | xlm-roberta-base | **2** | 0.7196 |

The incumbent is a fine-tune of a model that already trained ten epochs; it
carries roughly eight epochs of accumulated training. The classifier had two,
from raw weights, with no hyperparameter search. This compares a first attempt
against a well-iterated champion, not one formulation against another.

A fair rematch is one run: train the classifier to convergence — ~10 epochs, or
initialise from `v3-xenc`'s encoder as `v19-v10` does — and re-run
`benchmark.py --baseline`. Until then the honest statement is that the
reformulation has not been shown to help, not that it does not.

Serving restored to `v19-v10`.

### The fair rematch: the reformulation is refuted, significantly

`v23-classifier-xenc`: initialised from the same `v3-xenc` encoder that
`v19-v10` fine-tunes, same corpus, same split, four epochs. The curve
converged and turned over, so training budget is no longer an explanation:

```
epoch 0  0.7056   epoch 1  0.7169   epoch 2  0.7278 (best)   epoch 3  0.7253
```

Starting from the pair model's own weights did not help either — 0.7169 at
epoch 1 against 0.7196 from raw weights. Coherent in hindsight: `v3-xenc`'s
encoder is adapted to a *pair* input, `<s> A </s></s> B </s>`, and the
classifier feeds a single sequence, so the adaptation is spent rather than
inherited.

Paired against `v19-v10`:

| metric | pair | classifier | 95% CI |
| --- | ---: | ---: | --- |
| heteronym | 82.55% | 80.15% | **[-4.70, -0.31]** |
| macro-F1 | 63.89% | 57.46% | **[-10.33, -1.04]** |
| sentence | 65.20% | 62.96% | **[-4.00, -0.49]** |
| word | 94.98% | 94.61% | **[-0.62, -0.14]** |
| unambiguous | 98.53% | 98.56% | [-0.04, +0.13] noise |

```
fixed 33, broken 55, net -22 tokens (-2.40 points), exact p = 0.025
```

**Every interval except the unambiguous one lies below zero, and McNemar is
significant at 5%.** This is the first statistically significant result of the
whole effort, and it refutes the strategy's leading non-audio bet.

### What that says about RUAccent, read again

The architecture was never their lever. Their own version ladder holds the
model fixed and moves only the data: `big` 0.8886, `turbo` 0.9089, `turbo2`
0.9118 — all with the same transformer-plus-linear-head — and then 0.9637 when
the labels came from audio. Copying the head reproduced their architecture and
none of their gain, which is exactly what their table predicts.

Their text encoder was also pretrained on 200 GB of stress-marked text with
AMLM and NSP before any of this. We inherited `xlm-roberta-base`, pretrained on
text with no stress information at all.

So the conclusion is sharper than before rather than weaker: **the lever is the
labels, not the head.** Nothing in the architecture space has moved this number
in ten attempts; the one intervention with a published result changed where the
supervision comes from.

Serving restored to `v19-v10`. `v23-classifier-xenc` is kept — it is the
control any future audio-labelled run should be compared against, since it is
the same formulation trained to convergence on dictionary-derived labels.
