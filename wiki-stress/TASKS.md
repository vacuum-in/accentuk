# Tasks — 2026-08-31

**Status:** T1, T3, T4, T6 done. T2 and T5 are open — T2 needs labelling
capacity, T5 needs a decision. T7 is recorded but not started.

Supersedes `PLAN_NEXT.md` (2026-08-27), which was written before the error
budget was attributed to tiers and before the benchmark's noise floor was
computed. Both of those change what is worth doing.

The diagnosis this comes from is in `STATE.md`. In one line: **every tier is at
the ceiling of the evidence it consumes, and seven consecutive attempts to
re-arrange which tier decides have measured at or below zero.** Adding evidence
and fixing defects has worked seven times out of seven.

Current (live stack, 2026-09-03): heteronym **83.10%**, macro-F1 **64.24%**,
sentence **69.30%**, word **95.77%**, unambiguous **99.46%**.

---

## T1 — Give the benchmark a confidence interval — **DONE**

**Blocks every other measurement task.** The benchmark scores 895 heteronym
tokens; at 77.8% the 95% interval is ±2.72 points. A 0.33-point delta is three
tokens. Six of this session's seven measured deltas were inside that band,
including one reported as a win.

- `scripts/benchmark.py` gains `--baseline report.json`: score the same tokens
  under both systems, report **b** (fixed), **c** (broken), the net change, a
  McNemar exact p-value, and a bootstrap 95% CI over sentences.
- Print the interval next to every delta. A bare delta is not a result.

**Done.** `--save` writes per-token correctness and per-sentence counts;
`--baseline` reports McNemar plus a bootstrap interval over sentences. Metrics
are unchanged, because they still come from the benchmark's own
`evaluate_stress_sentence_level` and `DatasetAccuracy`.

Its first result retracted a claim made the same day. `uk_core_news_sm` against
`uk_core_news_trf`:

```
fixed 11, broken 8, net +3 tokens (+0.33 points), exact p = 0.648
heteronym  [-0.59, +1.27]   macro-F1  [-0.91, +4.15]   all straddle zero
```

"trf is ahead on every metric" was reading noise. It may still be better; 917
tokens cannot tell. **Nothing on this project should be adopted or rejected on a
sub-1-point delta until T2 lands.**

---

## T0 — Label from audio — **the demonstrated lever**

RUAccent reached 0.9637 on Russian homographs by reading labels off 108,000 h of
recorded speech: WhisperX for word-level timestamps, homograph occurrences cut
out, an audio classifier saying which variant was spoken. That single change
moved them 0.9118 -> 0.9637 with the architecture held fixed. See STATE.md.

It also dissolves what no text source can: our free-variation class sits at 63%
because dictionaries disagree with each other and with modern usage. Recorded
speech has no conventions to disagree about.

The bootstrap makes it reachable without annotated audio: stress-controlled TTS
synthesises audio whose answer is known, the classifier learns to hear stress on
that, then labels real recordings. A Ukrainian TTS with stress control is being
built in this workspace, and a plan is drafted at
`audiotostress/ukrainian-audio-stress-openspec` — README only, no spec files yet.

**Effort** weeks. **Risk** high, and the only lever with a published result.

---

## T3b — RoPE instead of absolute positional embeddings

RUAccent's own ablation on the stress placer: APE 0.951, ALiBi 0.964, RoPE 0.972.
Our XLM-R config reads `"position_embedding_type": "absolute"`. Stress is a
position-sensitive task, so this is a mechanism rather than a leaderboard
artefact. One training run, measurable with T1.

**Effort** one run. **Risk** low.

---

## T2 — Enlarge the gold set until 1 point is detectable

895 heteronym decisions cannot resolve the differences being chased. Roughly
3,000 would bring the band under ±1.5 points.

- Sources already on disk: `output/ml/counted_form_worklist.json` (108 forms ×
  up to 8 real contexts), plus mined contexts for the highest-frequency
  ambiguous forms in `silver_mined_v10.json`.
- Label with two independent models and keep only agreement, the same protocol
  that produced the silver corpus — **but hold this set out of training
  permanently**, and record that in the file.
- Watch the trap this session hit twice: the internal 1,000-row gold set is
  partly one sentence in three wrappers, so its 1,000 rows are not 1,000
  independent contexts. Count independent contexts, not rows.

**Done when** the noise band is under ±1.5 points and the set is registered as
frozen.

**Effort** two to three days, most of it labelling. **Risk** low.

---

## T3 — Replace sense definitions with exemplar sentences — **DONE, negative**

The main model change, and the only one aimed at the 107-error pool.

The cross-encoder scores `(sentence, gloss)` where the gloss is a lexicographic
*definition*. Evidence that this signal is saturated: forms with 20+ training
rows score **73.8%**, forms never trained score **75.6%** — abundant supervision
buys nothing. Seven models across 85k–478k rows all landed 79.7–82.4%.

The corpus holds 463,867 labelled *usages*. Comparing a sentence to real usages
of a sense is a different signal, not more of the same.

- `crossencoder.candidate_glosses()` returns *k* exemplar sentences per sense,
  retrieved from `silver_mined_v10.json`, instead of (or alongside) the
  definition. Exemplars must come from the training split only — a benchmark or
  gold sentence leaking in makes the result meaningless.
- Retrain from `models/v3-xenc/checkpoint`, same corpus, same hyperparameters.
  One run is ~3.5 h on the 3070.
- Evaluate with T1. **Do not select on `dev_group_macro`** — it pointed the
  wrong way three times this session, most starkly for `v20-morph`, which had
  the best dev score and the second-worst benchmark score.

**Implemented**, and the ablation is running.

`crossencoder.build_exemplars()` and `candidate_exemplars()`;
`run_finetune_inflected.py --exemplars K`; `run_exemplar_ablation.py` runs both
arms on identical data.

It is a **matched pair**, not a comparison against `v19-v10`, because the corpus
and inventory that produced `v19-v10` were never recorded — the script takes
both as arguments and writes only `corpus_version` into its manifest. Comparing
against it would confound the change under test with an unknown data pairing.

Running it exposed something the previous crash hid: `ambiguous_forms_glossed`
covers **63%** of `silver_mined_v10`. 171,245 rows of 463,867 have no candidate
list and are skipped. Both arms share that, so the comparison holds, but the
absolute numbers will sit below `v19-v10` and are not comparable to it.

**Result: no benefit.** Both arms trained, both benchmarked, paired:

```
control (definitions)  80.48% heteronym   60.83% macro-F1   dev 0.8551
treatment (exemplars)  78.95%             59.81%            dev 0.8332

fixed 26, broken 40, net -14 tokens (-1.53 points), p = 0.109
heteronym [-3.33, +0.12]   macro-F1 [-5.79, +2.29]
```

Not significant, but no evidence of benefit either — every point estimate is
negative and dev agrees with the benchmark for once.

**It was a weak test, not a refutation.** Only 2,297 senses got exemplars, so
most training pairs were unchanged; and the inventory covers 63% of the corpus,
so both arms trained on 292,622 of 463,867 rows and both score below the shipped
`v19-v10`. Three exemplars also crowd a 192-token limit, an uncontrolled
confound.

Retrying it needs an inventory that covers the corpus and exemplars for most
senses — both data problems. Serving was restored to `v19-v10`.

---

## T4 — Make silent failure impossible — **DONE**

Every large loss this session was invisible while it was happening.

| what failed | cost | error raised |
| --- | --- | --- |
| `uk_core_news_trf` 3.7.2 loaded, tagged nothing | −17 points | none |
| `MORPHOLOGY_TIMEOUT` never read from the environment | no deadline at all | none |
| API never read its supplementary datasets | −17.67 points | none |
| Verbalizer rewrote plain sentences | 23.8% corrupted | none |
| Whole request passed as sentence context | confident wrong answers | none |

- Add a canary set of ~20 sentences with a known tier attribution, and a
  readiness check that fails when any tier resolves **zero** tokens on it. That
  single check catches rows one, three and four above.
- Log per-tier resolution counts on every request at debug level, so a tier
  going quiet in production is visible without a benchmark run.

**Done.** `/health/ready` runs four sentences of known morphological resolution
and reports how many the tier answered; zero returns 503 instead of serving
degraded. A probe that raises reports itself as not-ok rather than taking the
endpoint down.

```json
"morphology": {"ok": true, "resolved": 3, "probed": 4, "detail": "ok"}
```

One honest caveat: the original 3.7.2 failure can no longer be reproduced here.
Installing `spacy-curated-transformers` into `ml/.venv` for the 3.8.0 work also
repaired 3.7.2, which now resolves 3 of 4. The featureless-parse condition is
therefore covered by unit test, not by live reproduction.

---

## T5 — Decide the policy for the 73 undecided tokens

They are served at **49.3%** — chance, because nothing in the system knows the
answer. Three separate priors were tried and none beat it: lexicon order 49.3%,
corpus-frequency majority 45.2%, trie default already in place.

This is a product decision, not an engineering one. A TTS caller may prefer an
unstressed word to a confidently wrong one, and `on_ambiguity: preserve` already
offers exactly that — nothing downstream uses it.

- Decide whether the frontend requests `default` or `preserve`, and write down
  why.
- If `preserve`, `uk-tts-frontend` should surface the count so a caller can see
  how much of a passage was left undecided.

**Done when** the choice is recorded with its reasoning.

**Effort** an hour, once someone decides. **Risk** none.

---

## T6 — Establish what the unused coverage costs — **DONE**

55.7% of the 15,332 forms in the serving manifest (8,535) never appear in the
training corpus, and they score the same as trained forms. Either that coverage
is free — in which case the manifest is not a competence claim and should stop
being described as one — or it is dead weight.

**Done** — `uk-tts-frontend/scripts/coverage_cost.py`. It is free.

| | full manifest | trained forms only |
| --- | ---: | ---: |
| forms | 15,332 | 6,797 |
| file | 14.1 MB | 4.9 MB |
| candidate pairs / 150 sentences | 166 | 152 |

**0.09 extra candidate pairs per sentence.** The untrained 8,535 forms cost
nothing at inference and about 9 MB of manifest. So the manifest is an
*inventory of forms that have glosses*, not a claim about what the model can do,
and it should stop being described as coverage.

---

## T7 — Ablation: does morphology help the model as *input*?

Not attempted. Proposed 2026-09-03; recorded because it is cheap, genuinely
untested, and the reasoning behind it is sound even if the prediction below is
pessimistic.

The cross-encoder currently sees `(sentence, gloss)`. The proposal is to feed
the target's morphological analysis alongside it, and to measure three arms:

```
A  context only                          (control — what ships)
B  context + lemma
C  context + lemma + POS + case + number
```

Input shape, roughly:

```
Мої <target>сестри</target> приїхали.
[LEMMA] сестра  [POS] NOUN  [CASE] Nom  [NUMBER] Plur
[CANDIDATE] се́стри
```

Score with T1 — paired per token, McNemar plus a bootstrap interval — and read
**macro-F1 on ambiguous forms**, not dev accuracy, which has pointed the wrong
way three times.

**Cost**: three runs of ~3.5 h each on the 3070, plus the tagging pass to build
the fields. The tags are already produced by the morphology tier, so the
feature extraction is a re-use rather than new work.

**Prediction, to be recorded before the run: no effect.** The tokens where tags
decide the answer are exactly the ones the morphology *tier* already claims —
after the fix in ISSUES_RESOLVED B8 it resolves 594 of them and is right on
84.6% of the grammatical class. What reaches the model is mostly homographs,
where both readings carry the *same* tags by definition, so arms B and C would
be handing it a feature that is constant across the candidates it has to
separate. If that reasoning is right, C ≈ B ≈ A.

**Why run it anyway**: the prediction is falsifiable in one measurement, and if
it is wrong the payoff is on the largest remaining error class. Three previous
model-side changes (exemplars, classification, routing) all measured at or below
zero, so a fourth null result is also worth having on record.

**Related, and not the same thing**: a monolingual Ukrainian encoder
(`HPLT/hplt_bert_base_uk`, 12×768, 32,768 vocab) has not been tried either.
Three encoders have — `xlm-roberta-base` (ships), `ukr-models/xlm-roberta-base-uk`
(110M, net +24 tokens, p = 0.116), `mmBERT-base` (308M, abandoned at 21.6 h
without finishing epoch 0) — and none moved the metric. RUAccent's own history
says the same: 0.8886 → 0.9118 with the architecture held constant. Treat the
encoder as settled unless T7 shows the input representation matters.

**A ByT5 or character model for out-of-lexicon words** is the third untried
piece of that proposal. The cheap analogue is already in place — a suffix
fallback by analogy with lexicon endings, which took uncovered-word errors from
188 to 35 — so the remaining headroom there is small.

---

## Ruled out — do not re-attempt without new information

Each was measured on the benchmark this session.

| attempt | result |
| --- | --- |
| Counted-form rule over the whole accent class | −0.42 macro-F1; the class is lexical, not paradigmatic |
| Counted-form gated on the Orthoepic Dictionary list | 0.00 exactly; fires on no benchmark token |
| Abstention threshold 0.5 → 0.01 | −0.33; the dictionary default beats a weak model decision |
| Route grammatical forms out of the model (manifest v23) | **−5.02**, the only result outside the noise band |
| Route model abstentions down to morphology | 6 of 424 tokens; morphology declines exactly these |
| Corpus-frequency prior for the default | 49.3% → 45.2% |
| Larger tagger `uk_core_news_lg` | −0.54 |
| More training data | 7 models, 85k → 478k rows, flat 79.7–82.4% |

`output/ml/serving_manifest_v23.json` is kept for the one narrower version still
worth trying: exclude from the model only forms where morphology is
*demonstrated* to answer on real contexts, not merely permitted to by its tags.
That would take `руки` out and leave `була` in. Needs T1 to be measurable.

---

## Open, lower priority

- **`дві гори`, `дві ціни`** — in the counted-form list, still wrong: the rewrite
  covers `Case=Nom` only. Widening it to `Acc` would fire on `Обидві рідини`,
  where the dictionary says `рідини́` and the gold says `ріди́ни`. The list has at
  least one conflict with modern usage; its true error rate is unknown.
- **Lexicon-tier audit** — the only untouched lever for the 98.53% unambiguous
  figure, and larger than it first looked. `Я в універі.` is served `уні́вері`;
  the correct stress is `універ́і`, and the source trie and the
  `ukrainian-word-stress` baseline carry the same error, so we reproduce it
  faithfully. Nothing can question it: one candidate means tier 1 answers and no
  tier below runs, and the entry has `part_of_speech: unknown`, no tags, and
  itself as its lemma. **86% of the curated dataset (2,189,298 rows) looks like
  that** — bare (form, stress) assertions. The "36 errors" figure was 36 among
  benchmark tokens; the population is millions and its error rate is unmeasured.
  Auditable the way `run_counted_forms.py` audits the counted form: take the
  highest-frequency `unknown` entries and check them against the Orthoepic
  Dictionary. See STATE.md.
- **Licensing** — neither upstream carries a LICENSE, and 88.6% of the training
  corpus is malyuk under "mixed". Blocks publishing weights, not code.
- **Deployment** — the GPU path currently runs the model service on the host via
  `MODEL_URL_OVERRIDE`, because the daemon has no NVIDIA runtime and the image
  carries CPU torch. Either install the toolkit and build a CUDA image, or
  accept CPU and note that `trf` is slower there than `sm`.
- **Disk** — C: sits at 99%; Docker died from it once already. The WSL VHDX needs
  compacting, not just deletion inside the guest.
- **Rotate the two Gemini API keys** that were pasted into the session
  transcript: <https://aistudio.google.com/apikey>
