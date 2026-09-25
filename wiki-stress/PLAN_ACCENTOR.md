# Ukrainian accentor — plan for an implementing agent

Written 2026-09-07 after comparing this pipeline with RUAccent
(Petrov et al., COLING 2025) and after the audio work in
`/home/devops/audiotostress`. This document is the brief: what exists, what is
measured, what to build, in what order, with what gates. Read `PLAN_NEXT.md`
first for the lessons that still apply, especially "measure which tier owns the
loss before touching a model".

## The target, decided 2026-09-08

**TTS on modern text.** That settles which number this project optimises, and
it is not lang-uk's.

lang-uk stays as a regression guard — external, human-annotated, and it catches
breakage. It is no longer the goal, because its remaining loss is not the
target domain: of the 382 tokens it still loses, the suffix tier's 116 are
mostly nineteenth-century and Russianised orthography (`єго`, `обовязком`,
`которой`, `особенність`, `обобщеніє`, `тогди`) and foreign proper nouns, and
109 of the 128 `stressed`-tier losses are forms that appear exactly once. A
model trained on modern normative forms would miss them too, so chasing them
buys nothing for speech synthesis.

**The primary metric is per-tier agreement with the recordings on Common
Voice** — currently 94.2% over 266,241 words, reported by tier, with the
prepositional blind spot excluded. Where the loss sits there:

| tier | share of words | disagree | recoverable above audio noise |
| --- | ---: | ---: | ---: |
| `dictionary_default` | 5.3% | 19.3% | **2,286** |
| `suffix` | 2.0% | 29.7% | **1,387** |
| morphology | 5.5% | 10.4% | 1,060 |
| `stressed` | 85.8% | 3.5% | 685 |
| compound | 0.4% | 42.3% | 428 |

That reorders the work. `dictionary_default` — the lexicon offering two
readings with no tier resolving them — is the largest bucket, and it is exactly
what the token classifier (§3) exists for. It now comes before the placer (§2),
which was ranked first while lang-uk was the target.

## 0. Where things stand (all measured, all reproducible)

| what | figure | how to reproduce |
| --- | ---: | --- |
| lang-uk word accuracy (live API) | 95.93% | `ml/.venv/bin/python ml/scripts/run_live_bench.py --benchmark <clone> --callers 8` |
| lang-uk heteronym accuracy | 84.62% | same |
| lang-uk sentence accuracy | 69.98% | same |
| lexicon, active dataset 2 | 2,540,508 rows / 2,350,464 forms | `stress_lookup where dataset_id=2` |
| forms with 2+ readings in dataset 2 | 42,390 | `group by form_normalized having count(distinct stress_signature)>1` |
| audio model, held-out speakers | 97.04% | `audiotostress/RESULTS.md` |
| audio model on unseen forms | 93.72% | same |
| audio↔pipeline agreement, 44,850 words | 93.8% | `audiotostress/scripts/run_verify_stress.py --corpus … --clips 10000` |

**Where the benchmark's loss actually sits** (`ml/scripts/run_tier_attribution.py
output/ml/live_review12.json`, which reproduces the official 95.96% exactly):

| tier | words | accuracy | lost | share of loss |
| --- | ---: | ---: | ---: | ---: |
| `stressed` (lexicon certain, or a tier answered) | 7,866 | 98.4% | 129 | 34% |
| **`suffix` fallback** | 419 | **72.1%** | **117** | **31%** |
| morphology | 552 | 88.4% | 64 | 17% |
| `dictionary_default` | 409 | 87.0% | 53 | 14% |
| compound fallback | 93 | 93.5% | 6 | 2% |
| prepositional rule | 38 | 97.4% | 1 | 0% |

Words the pipeline left unaccented that needed one: **eight**, all hyphenated
proper nouns or OCR damage (`сент-кіттс`, `землї`). Coverage is not the
problem; the suffix fallback is.

Agreement by pipeline tier from the audio sweep — a second view, useful where
the benchmark has few tokens, and not interchangeable with the table above:

| tier | words | agrees with audio |
| --- | ---: | ---: |
| dictionary certain | 38,549 | 96.4% |
| morphology | 2,358 | 89.5% |
| dictionary guessing (`dictionary_default`) | 2,454 | 76.3% |
| **suffix fallback** | 946 | **68.6%** |
| **compound fallback** | 179 | **59.2%** |
| prepositional rule | 361 | 47.6% — **not evidence**, see §6 D2 |

The benchmark clone lives at
`/tmp/claude-1000/-home-devops-wiki-stress/2c2fdcbe-374f-42d7-8e8d-e11ab5d1a89c/scratchpad/ukrainian-tts-preprocessing`
for now; clone `lang-uk/ukrainian-tts-preprocessing` somewhere permanent and
put the path in `BENCHMARK.md`.

## 1. Target architecture: two runtime models, split by word type

```
text → tokenize → lexicon (active + SUPPLEMENTARY_DATASETS, REVIEWED_DATASETS leads)
   ├─ one reading            → serve it                         (~95% of tokens)
   ├─ several readings       → contextual tiers, in order:
   │                            morphology  (right 84% when it answers)
   │                            → token classifier                [model 1] ← REBUILT
   └─ no reading             → stress placer                      [model 2] ← NEW
                                (replaces suffix + compound fallbacks)

offline, not in the request path:
   Common Voice audio → audio ranker → gold spans → corpus for model 1
```

**Neither model exists in the form this plan calls for.**

Model 1 today is the XLM-R gloss cross-encoder
(`ml/src/ukstress_ml/crossencoder.py`, manifest `models/v3-xenc`, 570 forms).
§3 shows it is the wrong instrument for 59% of heteronyms and must be replaced
by a token classifier over the sentence; the cross-encoder stays only where it
still wins, which §5 decides by measurement.

Model 2 does not exist at all. Its job is done by `SUFFIX_TABLE` and
`applyCompoundFallback` in `api/internal/httpapi/stress.go`, at 68.6% and 59.2%
agreement with speech. That is the largest structural gap and the cheapest to
close, so it comes first.

The audio model is **not** a runtime component. It is a label factory and a
verifier — the same role RUAccent gave it — with one difference in direction:
RUAccent's text model (0.96 on homographs) labelled Common Voice for its audio
model; ours is the other way round, because the audio model (85.3% on
homographs, 93.7% on unseen forms) is stronger than the morphology tier (74%)
on exactly the words that need labels.

## 2. Work package A — the stress placer (model 2)

**Goal.** Given a word the lexicon does not contain, predict which vowel carries
the stress. Beat the suffix table and compound fallback on every measure below.

**Data.** Build from the lexicon; no annotation needed.

```sql
-- training pool: forms with exactly one reading in the active dataset
SELECT form_normalized, stress_signature, lemma_normalized
FROM stress_lookup WHERE dataset_id = 2
GROUP BY form_normalized, stress_signature, lemma_normalized
HAVING count(*) >= 1;
-- exclude forms that appear with 2+ distinct signatures (42,390 of them)
```

* Input: the bare form (NFC; strip the acute with `ukstress.normalizer`).
* Target: vowel ordinal of the stress. **`й` counts as a vowel ordinal in this
  project** (`countsAsVowel` in `stress.go`, `VOWELS` in the audio scripts) —
  keep that convention or every downstream consumer breaks.
* **Split by lemma, not by form.** Inflections of one lemma share a stress
  pattern; a form-level split leaks it and inflates the score. Hold out 10% of
  lemmas as dev and 10% as test, seeded.
* Add a second test set of **true OOV**: every token the live API returned
  with status `suffix` or `compound` on lang-uk plus the 10k audio sweep
  (`output/ml/live_review11.json` tokens; `audiotostress/artifacts/audio_runs/verify_10k.json`
  rows with those statuses). These are the words the placer will actually see.

**Model.** Character-level transformer encoder, small (4 layers, d=256 is
plenty for 2.3M forms), **rotary position embeddings**. RUAccent measured this
choice: 0.951 with absolute positions, 0.964 ALiBi, 0.972 RoPE — stress is a
relative position between syllables and RoPE encodes that directly. Head: one
logit per character, softmax over the vowel positions only (mask consonants),
cross-entropy on the gold vowel. This is the same listwise formulation as the
audio ranker (`audiotostress/src/ukstress/ranker/model.py`), for the same
reason: the question is *which* vowel, not whether each one is stressed.

Put it in `ml/src/ukstress_ml/placer.py`; training script
`ml/scripts/run_train_placer.py`; keep the Python env `ml/.venv` (3.14).

**Baselines to report alongside it, on the same test sets.**

* suffix table (`SUFFIX_TABLE` json) — what ships today
* compound fallback — what ships today
* "most frequent ordinal for this word length"
* "penultimate vowel"

**Acceptance.** Measured on modern text, not on lang-uk: the suffix tier holds
5,235 words at 70.3% agreement there, of which about 1,387 are recoverable
above the audio model's own error rate. Reaching ~90% recovers roughly a
thousand. On lang-uk the same tier loses 116 tokens, but most are archaic or
foreign and a model trained on modern forms will not read them either — do not
count those. Measure with `run_tier_attribution.py` as the regression guard and
with a fresh verifier sweep for the goal, not with an internal test split — and note that the
lexicon has no paradigm structure to hold out by (92% of its rows have
`lemma_normalized` equal to the form itself, and only 17,331 lemmas group two
or more forms), so a "split by lemma" would be a split by form wearing a
different name.

**Serving.** Add `POST /internal/v1/place` to `serving.py` taking
`{"forms": [...]}` and returning one ordinal + confidence per form. In Go, add a
tier in `handleStress` between the lexicon miss and the suffix table: status
`placer`, `candidates` = the single signature, and fall through to the old
fallbacks only if the service is unavailable. Keep `suffix` and `compound`
reachable behind a flag for one release so the benchmark can be run both ways.

## 3. Work package B — the token classifier (model 1, replacing the gloss encoder)

"Homograph" in §1 hides two populations, and the benchmark's loss sits almost
entirely in the one the current resolver cannot reach. Classifying lang-uk's
492 heteronym forms by the lexicon the API reads (datasets 2,3,5,7,9,10,12):

| bucket | forms | lang-uk tokens | accuracy now | errors |
| --- | ---: | ---: | ---: | ---: |
| **grammatical** — same lemma, readings separated by case/number (`ко́леса`/`коле́са`) | 289 | 658 | **79.3%** | **136** |
| sense — different lemmas (`за́мок`/`замо́к`) | 133 | 412 | 86.2% | 57 |
| absent from the lexicon | 57 | — | (fallbacks) | — |
| lexicon holds one reading only | 13 | 61 | 77.0% | 14 |

Seventy percent of heteronym errors are grammatical. Who answers those 658
tokens (attribution by surface match; about a quarter could not be joined to a
token and are omitted here):

| tier | right | wrong |
| --- | ---: | ---: |
| morphology | 197 | 37 |
| `stressed` (served as certain) | 165 | 51 |
| `dictionary_default` | 16 | 16 |

Three conclusions:

1. **The gloss cross-encoder is the wrong instrument for 59% of heteronyms.**
   It scores (sentence, definition) similarity; a grammatical pair has one
   definition. Its manifest covers 570 forms; the lexicon has 42,390 ambiguous
   ones. Model 1 must be a **token classifier over the sentence** — RUAccent's
   design, a transformer encoder with a linear head, NER-style — that takes
   (sentence, span) and picks among the lexicon's candidate signatures. No
   glosses, so every ambiguous form is coverable, grammatical or sense.
   Keep the cross-encoder only where it already wins; §5 trains both arms and decides by measurement.

2. **Audio supplies exactly the training rows this model needs**, because a
   token classifier needs (sentence, span, gold ordinal) and nothing else. From
   the 10,000-clip sweep at confidence ≥ 0.99: 1,154 grammatical occurrences
   over 139 forms and 873 sense occurrences over 97 forms — about **8,400 and
   6,400 occurrences on the full corpus**, on lang-uk's own heteronym list alone, and ~35,000
   over all ambiguous forms (§4). Add the lexicon's unambiguous forms in context
   as easy negatives so the model learns positions, not a form list.

3. **51 grammatical tokens were served as certain and wrong.** `stressed` means
   no tier was asked — the candidate list collapsed to one (trie default,
   `dropSubsumedFreeVariation`, or a supplementary dataset carrying only one
   reading). List them from `live_review11.json`, find which collapse did it,
   and route those forms to the contextual model instead. This is the same
   defect family as §6 D1, seen from the benchmark side.

Morphology stays as the first pass: when it answers it is right 84% of the
time (197/234) and it costs nothing. The token model takes what morphology
declines — today that falls to `dictionary_default` at 50% — and the
`stressed`-but-wrong forms once (3) reroutes them.

**Acceptance**: on lang-uk heteronym tokens, grammatical
bucket ≥ 88% (from 79.3) and sense bucket not below 86.2%; measured through the
live API with the per-bucket split above, and again on a fresh audio sweep's
`dictionary_default` rows, which are the tokens it will actually receive.

## 4. Work package C — build the dataset

Both models need rows and both get them here. **One recipe, one builder.** An
earlier draft of this plan carried two overlapping filter lists with different
thresholds; if you find that anywhere else, this section wins.

One row per (sentence, span). Schema, NFC throughout:

```json
{"sentence": "…", "start": 12, "end": 16, "form": "того",
 "candidates": ["0", "1"], "gold": "1",
 "source": "cv-audio | lexicon-unambiguous | reviewed | xenc-corpus",
 "weight": 1.0, "speakers": 3, "pipeline_agreed": false, "split": "train"}
```

`start`/`end` are the API's token offsets. `run_verify_stress.py` stores them
from commit `9fd26cd` onward; for rows swept before that, recover them with one
`/v1/stress` call per distinct sentence — 46,644 distinct sentences across all
73,166 clips, so at most that many calls.

**Sources, in order of trust.**

| source | rows | gold from | use |
| --- | ---: | --- | --- |
| reviewed decisions (19 forms, 463 counted-form verdicts) | hundreds | a person | train, and always in dev/test |
| audio, gated as below | ~35,000 occ / ~3,000 forms | speech | the only source of contextual homograph gold at scale |
| existing cross-encoder corpus (`output/ml/corpus/*`) | thousands | mining + generation | train, mapped sense_id → signature |
| lexicon-unambiguous tokens in the same sentences | sample 1:1 | lexicon | negatives |
| lang-uk benchmark | 1,026 sentences | — | **never train**; test only |

**How the yield scales — measured, not assumed.** Occurrences grow linearly
with clips; distinct forms do not.

| sweep | words | usable occurrences | distinct forms |
| --- | ---: | ---: | ---: |
| 10,000 clips | 44,882 | 4,829 | 1,202 |
| ~28,000 clips | 124,974 | 13,700 | 2,037 |

2.8× the data gave 2.84× the occurrences and 1.7× the forms. Extrapolating the
full 73,166 clips: **~35,000 occurrences over ~3,000 forms** — not the ~8,000
forms a linear extrapolation suggests. Plan coverage against 3,000, and note
the resolver's current manifest covers 570.

**Step 1 — finish the sweep.** `run_verify_stress.py --corpus … --clips 80000`
resumes from `verify_*.jsonl`. Keep every row, including low confidence: filter
at build time, not sweep time, so a threshold can change without re-running
hours of alignment.

**Step 2 — collapse speakers.** 11,349 of the 46,644 sentences are read by two
or more speakers, so one sentence yields several rows for the same span.
Collapse to one row per (sentence, span): `gold` = the majority ordinal,
`speakers` = how many agreed. **Drop the span if the speakers split** — that is
label noise, not a homograph. Agreement across voices is a stronger label than
one voice at 0.99; use `speakers` as the weight.

**Step 3 — gate the audio rows.** In this order, counting what each step
removes so the numbers reach the build log:

1. `len(candidates) >= 2` — the lexicon offers a choice
2. the audio ordinal is one of those candidates — never invent a reading
3. confidence ≥ 0.99 from one speaker, or ≥ 0.95 from two who agree
4. not a prepositional clitic after a governing preposition (§6 D2)
5. per-form cap of 60 occurrences — `того` alone has 105 in 10k clips and
   would otherwise be a tenth of the grammatical rows

Keep `pipeline_agreed` as a column. Do **not** drop disagreements: they carry
the new information. But before trusting a form whose disagreements are
unanimous, check it against `audiotostress/DICTIONARY_AUDIT.md` — unanimity
there has meant a dictionary error twice and an alignment fault once.

**Step 4 — balance readings per form.** `PLAN_NEXT.md` §5 measured the threshold
on the cross-encoder: a form whose minority reading reaches ~20% of its rows
scores near 1.00, and degrades in proportion below that. Audio gives the
*spoken* distribution, which is skewed (`того́` 81 : `то́го` 24). Where a
minority reading falls under 20%, top up from the existing corpus or with
`run_generate_balanced.py --min-share 0.2`, tagged `source: generated`, weight
0.5. Never top up with audio rows below the gate.

**Step 5 — add negatives.** From the same sentences, every token whose form has
exactly one reading: `candidates` = that signature, `gold` = it. Sample 1:1
against the ambiguous rows. Without them a token classifier over-predicts
ambiguity, having seen only ambiguous spans.

**Step 6 — split, three ways.** They answer different questions and all three
get reported:

* **seen form, new context** — the ordinary case; split by sentence, seeded
* **unseen form** — hold out 10% of ambiguous forms entirely; §3's acceptance
  on lang-uk's 289 grammatical forms rests on this one
* **lang-uk** — the external number, the only one quoted outside this repo

Sentences from CV `test` speakers go to test and `dev` to dev, so the audio
side stays speaker-disjoint as it was for the audio model.

**Step 7 — validate before training.** All three are cheap and all three have
caught something in this project before:

* the 19 reviewed forms: audio gold must agree with the confirmed reading ≥ 95%
* per-form label entropy on the collapsed rows: list forms splitting near 50/50
  with no speaker agreement, exclude them, and report the list
* signature sanity: every `gold` is a valid vowel ordinal for its form under
  this project's convention (`й` counts), and every `candidates` list matches
  what `/v1/stress` returns for that sentence today

**Deliverable.** `ml/scripts/build_token_corpus.py` (env `ml/.venv`) reading the
audio JSONL, the lexicon, the reviewed files and the old corpus; writing
`output/ml/corpus/tok-v1/{train,dev,test}.jsonl` and a `manifest.json` carrying
every count from steps 2–7. The manifest is as much the deliverable as the
rows: a dataset whose filtering cannot be audited is one this project has
already been burned by.

## 5. Work package D — train and ship, gated on who owns the loss

`PLAN_NEXT.md` §5 is still right: the model is consulted on ~1.5% of tokens
and adding data is the lowest-yield lever *unless the model tier owns the
loss*. So the first task is to find out whether it does now:

```bash
# per-token errors by status, from the last live run
ml/.venv/bin/python - <<'EOF'
import json, collections
rows = json.load(open("output/ml/live_review11.json"))
# rows carry plain/gold/got/tokens; count wrong tokens by tokens[i]["status"]
EOF
```

Proceed only if `stressed`-by-model or `ambiguous`/`dictionary_default` errors
are a material share of the 121-ish remaining heteronym misses. If morphology
owns them, fix morphology (PLAN_NEXT §4) and stop here.

If the model tier owns the loss:

* Train the token classifier of §3 on `output/ml/corpus/tok-v1` from §4.
* Also continue the existing cross-encoder on the same corpus, mapped to its
  gloss format: `run_crossencoder.py tok-v1 models/v3-xenc audio1` (`run_tag`
  mandatory when resuming — the script header explains why). Two arms, because
  §3's claim that the gloss encoder is the wrong instrument is an argument
  until it is a measurement.
* Ship whichever wins per bucket; they can coexist, the token model taking
  grammatical forms and the cross-encoder sense forms, if that is what the
  numbers say.
* Re-derive the manifest and threshold with `build_serving_manifest.py`; forms
  that gained rows and were outside the manifest can now enter it.
* **Measure only through the live API** (`run_live_bench.py`), never through the
  offline harness alone — PLAN_NEXT records a 17.7-point gap between them once.
  Must beat 84.62% heteronym on lang-uk; report word/sentence/macro-F1 too, and
  the per-tier table from §0 after a fresh audio sweep.

## 6. Work package E — the precedence defects

Documented in `docs/architecture.md` "Which reading wins". D1 is done
(commit `837535f`); D2 turned out not to be a defect at all.

**D1. Done.** A reviewed reading was overruled by the contextual model,
because two candidates make a form model-eligible. Corrections are now split
into leading (`REVIEWED_DATASETS`) and exclusive
(`REVIEWED_EXCLUSIVE_DATASETS`, candidate list collapsed to one), and
`--manifest` drops corrected forms from the model's coverage. lang-uk heteronym
84.62% → 84.73%, sentence 69.98% → 70.27%. What remains, per form:

* if the form has **one** correct reading (`маю`, `гроші`, `років` — the
  "entry wrong" group in `DICTIONARY_AUDIT.md`): the lexicon must present it as
  unambiguous so no tier fires. Today it looks ambiguous only because dataset 10
  carries `маю́` at 0.6. Either drop those rows in the supplementary dataset, or
  add a `reviewed-exclusive` dataset flag that makes `batchSignatures` collapse
  the candidate list to the reviewed signature. Prefer the flag: it is one
  branch in Go next to `decided[form]` and it is auditable.
* if the form has **two valid readings** with a dominant one (`того`, `була`,
  `кого`): keep both candidates, reviewed one first, and let the tiers decide —
  but measure whether the tiers are right on them. On the 10k sweep `того` was
  read `того́` by 51 distinct speakers 81 of 105 times; if the tier keeps
  choosing `то́го`, the tier is wrong, and that is a training-data problem for
  §5, not a lexicon problem.

**D2. The prepositional rule is correct; the audio witness is not.** An earlier
draft of this plan proposed restricting the rule to the pronouns audio agreed
with. That was wrong, and the measurement that settles it was already in the
code it proposed to change.

On lang-uk, which is human-annotated gold, the rule scores **36 of 37** (97.3%)
and `prepositional.go` records the same pattern from the other direction: after
a preposition 39 of 40 tokens take the first syllable, with no preposition 0 of
11 do. Against that, the audio sweep says 50.0% agreement over 1,102 words.

The audio is the unreliable witness here, and for a reason this project caused:
**1,936 prepositional clitics were dropped from the audio model's training** as
suspected-bad labels, so it never learned these forms. Its readings of `мене`,
`тебе`, `себе`, `нього` are the one place its 96.8%-at-high-confidence
calibration does not hold.

**Do not change the rule.** Instead:

* Exclude prepositional clitics from every audio-derived conclusion — the
  dataset builder already drops them (§4 step 3.4); the dictionary audit should
  too.
* If the audio model is retrained, include these rows with labels derived from
  the rule rather than dropped, and re-measure. Only then is audio a witness
  here at all.
* The remaining question for a Ukrainian speaker is narrower than it looked:
  not whether the shift exists, but whether Common Voice's read speech departs
  from the written norm that lang-uk annotates.

## 7. Order and gates

| # | package | what | gate |
| --- | --- | --- | --- |
| 1 | **E** (§6) | ~~D1~~ done; D2 was not a defect | — |
| 2 | **C** (§4) | finish the sweep, build `tok-v1` | the three checks in step 7 |
| 3 | **B** (§3) | token classifier — the largest bucket on modern text | acceptance in §3 |
| 4 | **A** (§2) | stress placer | acceptance in §2 |
| 5 | **D** (§5) | train, compare arms, ship | beats 84.62% heteronym through the live API |

E is done, so §5's measurement is now clean. C before A and B because both need its rows. A and B can run in
parallel on one 8 GB GPU — the verifier sweep uses ~4 GB and the placer is
small.

## 8. Things an agent will otherwise trip on

* **The Go API is built and deployed from `/home/devops/uk-tts-frontend/deploy`**
  (`docker compose -p uk-tts …`), with build context pointing at
  `wiki-stress/api`. There is no local Go toolchain. `up -d api` recreates
  PostgreSQL too via `depends_on` — this took the lexicon offline once
  (`docs/operations.md`, "The volume follows the project name"). Use
  `docker compose -p uk-tts up -d --no-deps --build api`.
* **Env vars that matter at runtime**: `SUPPLEMENTARY_DATASETS=3,5,7,9,10,12`,
  `REVIEWED_DATASETS=12`, `TRIE_DEFAULTS`, `SUFFIX_TABLE`, `MODEL_MANIFEST`.
  Read them from the running container with `docker inspect`, not from
  `wiki-stress/deploy/docker-compose.yml`, which is not the file in use.
* **Two Python environments**: `wiki-stress/ml/.venv` (3.14, the text stack) and
  `audiotostress/.venv` (3.11, WhisperX/torch). Scripts assume their own.
* **`+` goes after the vowel** in lang-uk's format, and the acute (U+0301) goes
  after the vowel in ours. Converting between them has been done wrong twice in
  this project, each time reading as "accuracy ≈ 0–2%".
* **Never diagnose from an aggregate.** Every wrong conclusion in this project's
  history came from a summary number; every right one came from a per-token or
  per-form list. `live_review*.json`, `verify_*.json` and the audit files hold
  the lists — use them.
* **Do not trust `pgrep -f <pattern>` inside a shell whose command line
  contains the pattern.** It matches itself. Check the GPU or the log instead.

## 9. What this plan does not promise

The shape of this pipeline is now the same as RUAccent's. The distance is in
data: they trained on 200 GB of stress-marked text and 108,000 hours of audio;
this project has 2.5 M lexicon rows and 118 hours. §4 is the only mechanism
here that grows labelled homograph data without hand annotation, which is why
it outranks any change to model architecture — and why its yield should be
reported as a number of forms and occurrences, not as an accuracy, until §5
turns it into one.
