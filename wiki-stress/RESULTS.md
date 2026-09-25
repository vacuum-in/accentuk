# Homograph stress disambiguation — measured results

Every figure below comes from a recorded run on this machine (Apple Silicon,
10 cores, 32 GB, MPS). Nothing is estimated or pre-populated.

Companion documents: [PLAN_MARIAN_HOMOGRAPHS.md](PLAN_MARIAN_HOMOGRAPHS.md),
[PLAN_DATASET.md](PLAN_DATASET.md), [ANALYSIS_TRIAGE.md](ANALYSIS_TRIAGE.md),
[ml/CUDA.md](ml/CUDA.md), and the OpenSpec change
`build-homograph-stress-disambiguation`.

## 0. Status — 2026-08-16

**59 of 147 specified tasks complete.** A model meeting every declared release
gate exists and is measured. Task-level progress and the reasons for each
deviation are tracked in
`openspec/openspec/changes/build-homograph-stress-disambiguation/tasks.md`.

| Done | Not started |
| --- | --- |
| Inventory, ambiguous surface (lemma-only) | Paradigm expansion (§3) |
| Wikipedia mining, L1/L2/L3 annotation | L0 definition completion (§10.1) |
| Corpus assembly, splits, provenance | Inference service (§16), Go integration (§17) |
| Training, constrained scoring, evaluation | Observability (§18), API benchmarks (§19) |
| Serving latency benchmark | Seed-variance measurement |

The largest open gap is paradigm expansion: everything is lemma-only, so the
model sees `замок` but never `замка́ми`, and 36% of generated sentences were
rejected for using an inflected form.

## 0a. Update — 2026-08-25: paradigm expansion delivered

The gap §0 named as largest is closed. An accented morphological dictionary
(3.9M inflected forms) was merged into PostgreSQL, which is the "vendored
accented morphological dictionary" §7 said paradigm expansion required.

| | before | after |
| --- | --- | --- |
| `stress_lookup` rows | 364,130 | **2,540,514** (6.98x) |
| distinct forms | 215,258 | **2,350,464** |
| forms resolving deterministically | — | **98.2%** |
| servable homograph forms (manifest) | 570 | **10,986** |

`замками`, `замках`, `замку` now resolve — the exact forms §7 named as absent.

**Ambiguity triage.** The 42,395 apparently ambiguous forms are not all
homographs. Classified from the source dictionary's tag structure
(`ukstress_ml.triage`): 17,386 free variation, 5,833 morphologically
resolvable, 611 over-merging artifacts, leaving a **model tier of 16,057**.
A later correction removed 1,724 more: their candidates carried identical
glosses, so the cross-encoder saw identical inputs and picked arbitrarily at
high confidence. All 166 affected groups were adjudicated as free variation.

**Evaluated honestly, per §2.** 421 of 598 held-out forms occur with a single
sense, so aggregate accuracy mostly measures prior recall. On the 177
multi-sense forms, pooled over 4-fold cross-validation by form:

| | majority baseline | zero-shot | fine-tuned |
| --- | --- | --- | --- |
| multi-sense accuracy | 0.7893 | 0.7847 | **0.8527** |
| minority-sense recall | 0.0000 | 0.6376 | 0.6403 |

Zero-shot v3-xenc lands *below* the majority baseline on multi-sense forms;
the aggregate 0.8928 was single-sense prior recall. Fine-tuning on 22,004
silver rows (mined, labelled by DeepSeek-V4-Pro, blind-verified by
DeepSeek-V3.2) is a genuine +6.3pp over baseline.

**Precision, and the label ceiling.** Best pooled precision is **0.9731 at 26%
coverage**. Re-adjudicating high-margin errors blind, on both deployments,
overturned the silver label on **57% of them** — so a majority of measured
"errors" are annotation mistakes. Crediting only those, precision at margin 15
is **0.9855**. The 98% target is therefore not reached against silver labels
and is plausibly reached against correct ones; a human-adjudicated gold set is
what would settle it.

**Serving.** ONNX Runtime fp32 is decision-identical to eager torch on all 759
`test_natural` spans (argmax agreement 1.0000) and ~2.1x faster on CPU;
end-to-end through `resolve()` the gain is 1.1-1.3x, since tokenization is a
shared fixed cost. Dynamic int8 is *faster* than fp32 on x86/fbgemm, reversing
§6's qnnpack finding, but shifts logits by up to 13.0 and would invalidate the
calibrated abstention threshold.

## 1. Headline

The task: given a sentence containing a known ambiguous spelling, choose the
correct stressed form. The inventory is closed — the model is consulted only
for spans where PostgreSQL already returned more than one candidate.

Evaluated on `test_natural`: 759 held-out **natural Wikipedia sentences**, 202
known groups, generated text excluded, natural sense distribution preserved.

| Metric | Marian v3 (30M, from scratch) | **Cross-encoder (278M, pretrained)** | Majority baseline |
| --- | --- | --- | --- |
| Group-macro accuracy | 0.7161 | **0.9142** | 0.7678 |
| Multi-sense groups (306 rows) | 0.5899 | **0.8730** | 0.5930 |
| Minority-sense recall (150 rows) | 0.4200 | **0.8600** | 0.0 |

Best epoch 6 of a 14-epoch budget, early-stopped at epoch 9, 430 minutes on MPS.
All four release gates pass (`output/ml/models/v3-xenc/release_report.json`).

**Abstention**: a decision margin of **1.2482** yields **95.1% precision at 93.8%
coverage**. Below it the API keeps the dictionary's default sense rather than the
model's choice. 14 of 202 evaluated groups fall below a 0.6 accuracy floor and
are named in the release report as known limitations.

The cross-encoder is `xlm-roberta-base` scoring `(sentence with marked span,
candidate gloss)` pairs with a softmax across the row's candidates. Marian v3
lands **below** the majority baseline: it trades majority-sense accuracy for
minority recall and nets worse than always guessing the dominant sense.

## 2. Why the metrics are shaped this way

Two early metric choices were wrong and were corrected once measured.

**Aggregate accuracy hides everything.** 35% of evaluation groups appear with
only one sense, and those are answered by memorising a prior. They inflate both
the model and its baseline. `multi_sense_group_macro` restricts to groups that
genuinely occur with more than one sense.

**Minority-sense recall is the load-bearing number.** The majority baseline
scores exactly 0 on rows whose gold sense is not the group's dominant one, so
anything above 0 is context actually being read.

**Unseen-group accuracy was the wrong gate.** It measures generalisation to
homographs outside the inventory, which the product never encounters — the
dictionary only flags ambiguity for spellings already in the inventory. Worse,
reserving those groups removed 16% of the corpus, and specifically the
best-balanced groups, from training. Removed in v3; all 570 groups now train.

**Evaluation is mined-only.** Generated sentences were written under a prompt
demanding unambiguous context, so their cues are stronger than real text.
Scoring on them overstates served accuracy. They are training-only. Training is
balanced *after* splitting so evaluation keeps the natural skew, and a
cross-split 4-gram guard (0.5 overlap) removed 8 further rows.

## 3. Run history

| Run | Corpus | Encoding | Model | dev macro (baseline) | minority recall |
| --- | --- | --- | --- | --- | --- |
| v1 | 8 131, skewed | marked span | Marian 30M | 0.9908 (0.9908) | ~0 |
| v2 | 12 173, balanced | marked span | Marian 30M | 0.7191 (0.7135) | 0.1797 |
| v2g | 12 173, balanced | + glosses | Marian 30M | 0.7143 (0.7135) | 0.2136 |
| v3 | 12 351, honest split | marked span | Marian 30M | 0.7085 (0.7488) | 0.3878 |
| **v3-xenc** | 12 351, honest split | + glosses | **XLM-R 278M** | **0.9129 (0.7488)** | **0.7755** |

v1's 0.9908 is not a result: the corpus was so skewed that the majority baseline
also scored 0.9908. Balancing dropped the baseline to 0.71 and the model
followed it down, which is what exposed the problem.

Three levers were tried on Marian and none moved it meaningfully past baseline:
corpus balancing (v2), gloss conditioning (v2g, +3.4pp minority recall at 2×
training cost), and the corrected split with full training data (v3). The
constraint was not the input format or the data balance — a randomly initialised
model has no Ukrainian semantics with which to match a context against a
definition. Starting from pretrained weights moved minority recall from 0.39 to
0.79 on the same corpus.

## 4. Corpus

| Stage | Value |
| --- | --- |
| Wikipedia pages streamed | 2 530 185 |
| Sentences examined | 22 179 835 |
| Ambiguous-form hits | 1 705 779 |
| Kept after inline filters | 1 329 928 |
| Sampled to disk (reservoir, cap 80/form) | 58 813 |
| Forms with ≥1 natural hit | 1 511 / 2 035 |
| Forms starved (zero hits) | 524 |
| Mining wall clock | 20.3 min |

Annotation: 12 502 labels (`DeepSeek-V4-Pro`), 12 534 blind verifications
(`DeepSeek-V3.2`), 1 115 calls, 1.88M prompt + 880k completion tokens.
Generation: 843 requests → 7 724 proposed → 4 862 accepted (63%).

| Assembly filter | Rows dropped |
| --- | --- |
| Blind-verify disagreement | 2 095 |
| Cue absent from sentence | 1 088 |
| Label unclear | 532 |
| Verify unclear | 397 |
| Signature does not fit surface | 54 |
| Out-of-group sense | 70 |
| Duplicates / near-duplicates | 239 |
| Balance cap (train only) | 776 |
| **Accepted** | **12 351** |

Span mismatches, stress-validation failures, unstressed-projection failures and
label conflicts were all zero.

Final splits: train 11 165 · dev 427 · test_natural 759 · 570 groups · 1 064
senses.

## 5. Defects found and fixed

**Stressed `ї` rejected by the normalizer** (`etl/src/ukstress/normalizer.py:68`).
In NFD, `ї` is `і` plus a combining diaeresis, so the code point before the
acute is a mark, not a vowel. `validate_stress` therefore rejected every word
stressed on `ї` — including `Украї́на` — while `validate_characters` accepted
them, and a test already asserted `ї́хній` was valid. Fixed by recomposing the
base and its marks, which still correctly rejects stressed `й`. Recovered 35
senses; affects the main lexicon too.

**Acute inserted before the diaeresis** in `apply_signature`, producing invalid
`краі́̈на`. Caught by a round-trip test.

**Failed annotation batches counted as complete.** A transient 429 recorded a
null-content batch that resume then skipped permanently, silently dropping ~15%
of the corpus. Now only batches with content count as done, and rate-limit
errors back off 6s × attempt with jitter.

**Variant-notation senses are unlearnable labels.** 218 senses across 215 groups
record two acceptable stresses of one token (`ба́тьківщи́на`). Excluded and
reported, kept distinct from the 46 legitimate hyphenated compounds
(`а́льфа-ро́зпад`), which carry one stress per token.

**Rate limits are server-side, not concurrency-bound.** Raising workers 6 → 14
did not change throughput (3.7 batches/min); moving labelling off the saturated
deployment did.

## 6. Serving latency — measured, and the news is not good

CPU inference on this machine (Apple Silicon, 10 cores, torch 2.13, 4 threads),
40 timed runs after 5 warmup passes:

| Variant | 1 span | 2 spans | 4 spans | 8 spans |
| --- | --- | --- | --- | --- |
| fp32 (1 061 MB params) | **37.0 ms** | 58.6 ms | 117.6 ms | 284.6 ms |
| int8-dynamic (734 MB) | 80.6 ms | 171.9 ms | 411.5 ms | 2 394.2 ms |

Two results worth stating plainly:

**Dynamic int8 quantization is 2–8× *slower* than fp32 here.** The only
quantization backend available on this build is `qnnpack`, and per-operation
quantize/dequantize overhead exceeds any gain at these shapes. The usual
assumption that int8 is the deployment win does not hold on this hardware, and
it degrades catastrophically at batch 8. Do not ship it without re-measuring on
the actual serving host.

**Thread count does not help.** fp32 single-span latency measured 60.2 ms at 4
threads, 60.7 ms at 8, and 71.4 ms at 10 — more threads is slightly worse.
Run-to-run variance is significant (37–60 ms for the same configuration across
runs), so treat these as an order of magnitude, not a spec.

At roughly 30–60 ms per ambiguous span, a sentence with one or two homographs
adds tens of milliseconds. That is tolerable for offline TTS preprocessing and
heavy for a low-latency lookup API. Three levers exist before accepting it:

1. **Route by triage.** Only ~27% of ambiguous-span traffic genuinely needs
   context (see `ANALYSIS_TRIAGE.md`), though the routing is not yet certifiable.
2. **Optimised runtime.** ONNX Runtime or an equivalent, benchmarked on the real
   serving host rather than a laptop.
3. **Distillation** to a smaller encoder, trading measured accuracy for latency
   against the abstention threshold.

## 7. Known limitations

- **Coverage is 570 of 2 250 groups**, lemma forms only. The model sees `замок`
  but never `замка́ми`.
- **Paradigm expansion is not implemented.** It needs a vendored accented
  morphological dictionary; transferring stress across a mobile-stress paradigm
  without one would be guessing. This also caused 2 798 of 7 724 generated
  sentences (36%) to be rejected for using an inflected form.
- **1 900 senses still lack a definition.** The cross-encoder requires a gloss
  per sense, so this gates any coverage extension.
- **Every figure is a single run.** No seed variance was measured, so none of
  these numbers carries an error bar.
- **Serving latency is 30–60 ms per ambiguous span on CPU**, and int8 makes it
  worse on this hardware. Measured in §6; needs re-measuring on the real host.
- **Evaluation covers 202 groups**, those occurring with more than one sense in
  natural text. Groups seen with a single sense are not tested for
  disambiguation, only for recall of a prior.
- **The deliverable is not Marian.** The pipeline, corpus, constrained scoring
  and serving design are unchanged, but the scorer is XLM-RoBERTa.

## 8. Gold evaluation — 2026-08-27

The first evaluation against labels this pipeline had no part in producing:
`ukrainian_homographs_1000`, 1000 sentences over 31 target words, human-labelled,
balanced across senses.

### 8.1 The headline, and why the old one did not survive

| | figure |
| --- | --- |
| silver precision, mined split, margin 13 | 0.9949 |
| **gold accuracy, end to end** | **0.6530** |

The two differ by 34 points and they are not measuring the same thing. The
silver figure is precision on ambiguous forms the lexicon already knew, scored
against labels two models agreed on, at 22.8% coverage. The gold figure is
every token of every sentence, scored against a person, with no abstention.

Reported honestly: **0.6530 is the number, and the 0.9949 should never have
been the headline.**

### 8.2 Where the errors were

| tier | rows | accuracy | what it means |
| --- | --- | --- | --- |
| model | 480 | 0.9208 | the cross-encoder answered |
| dictionary | 224 | **0.5000** | lexicon reports one stress; the word has two |
| model_abstained | 136 | 0.4044 | below threshold, fell back |
| morphology | 64 | 0.6875 | resolved by the parse |
| outside_coverage | 64 | **0.0000** | ambiguous, absent from the manifest |
| not_in_lexicon | 32 | **0.0000** | no entry at all |

The contextual model is the strongest component. Every other tier is weaker,
and two of them return **nothing at all**.

### 8.3 Three findings that changed the diagnosis

**The `dictionary` tier scores exactly 0.5000 by construction.** 112 of 347
errors are 7 words — `мати`, `радій`, `насип`, `лупа`, `паша`, `ніколи`,
`зольник` — that the lexicon records with one stress and that have two. The
pipeline cannot route them to the model because it never sees an ambiguity, and
on a sense-balanced set one fixed reading is a coin flip.

**`outside_coverage` returning 0.0000 is faithful to the deployed API, not an
artefact of the harness.** `stress.go` gave an ambiguous form outside the
serving manifest the status `ambiguous` and left `OutputText` as the
*unstressed* word. Not "the wrong sense" — no answer.

**The abstention fallback was never frequency-tuned.** `PLAN_NEXT.md` described
`model_abstained` as falling back to a corpus-frequency default. It does not:
3,345 of 10,986 manifest forms carry no `default_signature` at all, and the code
fell through to `variants[min(variants)]` — the lexicographically smallest
signature. That arbitrary pick, not a frequency prior, is why the tier scored
below chance.

### 8.4 Policy sweep

Evaluation was split into evidence collection (`run_gold_eval.py`, one GPU pass,
writes every candidate, confidence, morphological parse and model score vector)
and scoring (`score_gold.py`, applies a threshold and a fallback). A threshold
sweep is now a file re-read.

Gold accuracy, with the recovered readings in the lexicon:

| threshold | fallback `none` | `dictionary` | `manifest` |
| --- | --- | --- | --- |
| 0.0 | 0.5450 | 0.7060 | 0.7060 |
| 0.5 | 0.5400 | 0.7100 | 0.7080 |
| **2.0** | 0.5130 | **0.7140** | 0.7030 |
| 3.0 *(shipped)* | 0.4740 | 0.7090 | 0.6900 |
| 13.0 *(the "certified" margin)* | 0.1170 | 0.5260 | 0.5260 |

always-serve-the-lexicon-default baseline: 0.4680

Three things follow. Serving *nothing* costs 20 points and is never right. The
shipped threshold of 3.0 is too high and 13.0 is catastrophic end-to-end — the
margin that maximised silver precision at low coverage is not the margin that
maximises accuracy. And the frequency default is *worse* than the lexicon's own
confidence order on balanced input, which is what the plan suspected.

### 8.5 What the model's own errors are made of

At threshold 2.0 with a dictionary fallback the model tier is 543 rows at
0.8858, and its errors are not spread evenly:

| form | training rows | minority sense | gold accuracy |
| --- | --- | --- | --- |
| поділ | 12 | *one sense only* | 0.500 |
| правило | 96 | 1% | 0.708 |
| копати | 46 | 6% | 0.774 |
| варення | 28 | *one sense only* | 0.808 |
| атлас | 85 | 3% | 0.871 |
| ... | | | |
| орган | 111 | 12% | 1.000 |
| мука | 31 | 26% | 1.000 |
| колос | 57 | 26% | 1.000 |
| **обід** | **0** | — | **1.000** |

**Volume predicts nothing; minority-sense representation predicts everything.**
`правило` has 96 rows and scores 0.708 because 95 of them are one sense.
`обід` has no training data at all and scores 1.000 because the cross-encoder
can read its gloss. A group whose second sense never appears is a group the
model can only guess at, and `поділ` scores exactly 0.500 — it answers the same
way every time.

Across the serving manifest, of 1,265 multi-sense groups:

| | groups | share |
| --- | --- | --- |
| no training data at all | 392 | 31.0% |
| at least one sense with zero rows | 353 | 27.9% |
| minority sense under 10% | 112 | 8.9% |
| minority sense 10–20% | 76 | 6.0% |
| **minority sense over 20%** | **332** | **26.2%** |

Only a quarter of servable groups are in the band that scores 0.96–1.00. That
is the ceiling on the model tier, and no threshold, fallback or coverage change
moves it.

## 9. Mining a second corpus — 2026-08-27

### 9.1 Generation was the wrong instrument

Balanced training data was the identified fix for §8.5, and LLM generation was
the obvious way to get it. Measured, it is a poor one.

The Azure resource admits roughly **one request in flight**: 19 of 20 concurrent
generation-sized calls returned 429. A generation call spends 900–3,200 output
tokens and yields about **seven** usable sentences after validation, so the
whole 1,583-job plan was a multi-day run. Concurrency does not help — raising
workers from 6 to 24 changed throughput not at all, and the extra threads spent
their time in exponential backoff.

Two corrections to earlier conclusions in this file's history, both wrong:

* `label` and `verify` are two deployments on the *same* Azure resource, but
  they do **not** share a request budget — fired simultaneously, both succeed.
  An earlier session note claiming a shared quota was mistaken.
* The real generation bottleneck was the escalation stage: 44% of jobs fell
  through to gpt-5.4, which failed 36% of the time after two minutes of backoff.

### 9.2 Mining is bounded by CPU, not quota

`lang-uk/malyuk` is UberText 2.0 + OSCAR + Ukrainian News: 38.9M documents
against Ukrainian Wikipedia's ~1.3M. `mine.mine_documents()` applies the dump
miner's rules unchanged to any `(id, text)` stream.

On the OpenSubtitles corpus already on disk, costing **no quota at all**:

| | |
| --- | --- |
| documents scanned | 14,996,176 |
| sentences kept | 30,462 |
| forms covered | 1,894 of 8,573 targeted |
| wall time | ~10 minutes of CPU |

Labelling is what costs quota, and it is far cheaper per row than generation: a
label call carries **40 sentences** and emits a few hundred tokens, and label
and blind-verify run concurrently on the two deployments.

| path | output tokens per call | rows per call |
| --- | --- | --- |
| generation | 900–3,200 | ~7 |
| label a mined batch | ~600 | ~20 |

### 9.3 It fixes the actual defect

The point is not volume — it is that an encyclopedia and a subtitle corpus have
*different sense distributions*. Senses Wikipedia starves (imperatives,
colloquial readings, the non-encyclopedic half of a pair) are ordinary in
speech.

Measured at 59% of the subtitle labelling, over the 180 groups it touched:

| band | before | after |
| --- | --- | --- |
| no training data at all | 11 | **0** |
| single sense only | 91 | **67** |
| minority sense under 20% | 60 | 88 |
| minority sense at or above 20% | 18 | **25** |

Every group with no data is now covered and 24 groups left the single-sense
band — the state that pins `поділ` at exactly 0.500 because the model has never
seen its second sense.

### 9.4 Quality gates held

Labelling agreement between DeepSeek-V4-Pro and the blind verifier
DeepSeek-V3.2 was **87.6%**, in line with the existing mined corpora.
`assemble_mined.py` reuses `corpus.build_rows` unchanged, so a sentence from
subtitles is admitted on exactly the same terms as one from Wikipedia. On the
subtitle mine it quarantined 912 verifier disagreements, 1,035 sentences whose
cue was not actually present, and 619 duplicates.

### 9.5 Defects fixed while building this

* The shard downloader reported success on truncated files — 15 of 32 had
  unreadable parquet footers. It now checks `Content-Length` *and* parses the
  footer before accepting.
* The miner aborted on the first corrupt shard; it now skips and says so, so one
  bad download cannot kill a multi-hour mine.
* `--limit 0` was falsy in both `run_generate_balanced.py` and
  `run_gloss_uncovered.py`, so a dry run started a *full* run. Both now test
  `is not None`.

## 10. External benchmark — lang-uk, 2026-08-27

Every figure before this section was measured on data this project produced or
curated. This one is scored by `lang-uk/ukrainian-tts-preprocessing`'s own
harness, on its own 1,026 sentences, against the four reference systems it
ships evaluators for.

### 10.1 Result

| | word acc | heteronym | macro-F1 | sentence |
| --- | --- | --- | --- | --- |
| `ukrainian-word-stress` (baseline) | 88.67% | 64.34% | 47.26% | 41.52% |
| session start (live API) | 76.16% | 55.29% | 31.52% | 32.85% |
| **final (live API)** | **90.36%** | **79.17%** | **60.11%** | **47.95%** |

The baseline is a dictionary library with no model, no manifest and no
inference service. It scores 88.67% word accuracy on its own — which is the
honest context for every word-accuracy figure this project has reported. The
value the pipeline adds is concentrated in heteronyms, **+14.9 points**, and
that is the right place for it to be.

### 10.2 The ceiling is 99.76%, and getting there took three wrong answers

A token is unreachable when no candidate the lexicon offers can satisfy the
gold under the benchmark's own comparison. Measured that way, 1,232 of 1,235
heteronym tokens are reachable.

The same question was answered 95.06%, then 72.79%, then 98.30% first. Each
was wrong for a different defect in the *measuring* script:

* signature strings compared literally, so `0` counted as absent when the
  lexicon held `0|1` — which serves as `0`;
* punctuation left on the gold token, so Rule 1 rejected every comparison
  before stress was considered;
* `apply_signature` normalising apostrophes to U+02BC while the gold used
  U+0027 — the same rewriting bug fixed in the serving path, still live in the
  oracle.

A ceiling is a measurement like any other and deserves the same scepticism as
the number it bounds.

### 10.3 What moved it, and what did not

| change | effect | kind |
| --- | --- | --- |
| apostrophe rewritten in output | +11.4 unambiguous, +7.8 heteronym | bug |
| morphology tier never called | +8.6 heteronym | bug |
| `upos=PRON` vs `upos=DET` | +5.8 heteronym | bug |
| Stanza → spaCy `uk_core_news_sm` | +0.5 heteronym, +3.1 macro-F1 | swap |
| 71 targeted glosses | +3.2 heteronym | data |
| free variation as two accents | +1.9 heteronym | bug |
| 8,702 trie corrections | +0.9 word | data |
| lang-uk heteronym dictionary | +1.1 heteronym, +3.7 macro-F1 | data |
| **v12 retrain** | **−0.3 heteronym** | model |

Five of the nine were defects. The single largest investment — mining 240k
sentences, generating 12.5k, labelling 42k, two retrains — scored negative.

Two results worth keeping because they contradict the obvious guess:

* **`uk_core_news_lg` (231 MB) is worse than `sm` (15 MB)**: 72.08% against
  73.50% heteronym accuracy. Tagger capacity was never the constraint.
* **A tag-vocabulary mismatch cost 5.8 points.** The trie writes `upos=PRON`
  for `цьому`, `всього`, `усі`; UD taggers write `DET`. Case, gender and number
  agreed exactly; the match was refused on the label alone. Morphology's
  resolution rate went 75.0% → 85.4% from eight lines.

### 10.4 A third normalisation bug: `й`

Stress signatures are generated over NFD, where `й` decomposes to `и` plus a
breve and `и` counts as a vowel — so `райо́н` carries signature "2" (а, и, о).
The Go reader applied that signature to the caller's NFC text, where `й` is a
single consonant, giving ordinals а=0, о=1. Signature "2" did not fit, the
token was rejected as an invalid candidate, and the word came back unstressed.

Every word with `й` before the stressed vowel was affected: `район` alone
accounts for 58 tokens on this benchmark, about 4.7% of the unambiguous metric.
Fixing it moved unambiguous accuracy 95.64% → 97.56% and word accuracy
88.71% → 90.36%.

This is the third instance of one root cause — a signature crossing the
NFD/NFC boundary. The other two were the apostrophe, in the Go reader and again
in the Python `apply_signature`. Any code that generates a signature in one
normalisation and applies it in another is suspect.

### 10.5 Concurrency was a silent failure, not a slow one

Stanza holds one parser and is not safe to share across threads. Under eight
concurrent callers the API scored **72.16% word / 78.46% unambiguous** against
**87.89% / 95.64%** served serially: requests queued past `MODEL_TIMEOUT`, the
tier reported itself unavailable, and every affected token fell to a dictionary
default while the API returned 200.

With spaCy in a warmed pool the same load holds **88.10%**. The fix was
possible because the model is 15 MB rather than 500 MB — a pool of four costs
~60 MB instead of ~2 GB.

---

## 11. Tier order, reviewed readings and a phrase rule — 2026-09-03

Four changes to the deployment, each measured on its own against the running
service with `run_live_bench.py`, eight concurrent callers, scored by the
benchmark maintainers' evaluator.

| # | change | sentence | word | heteronym | macro-F1 | unambiguous |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| — | starting point | 65.20% | 94.98% | 82.55% | 63.89% | 98.53% |
| 1 | morphology fills the model's abstentions | 65.30% | 94.98% | 82.66% | 64.24% | 98.53% |
| 2 | four reviewed readings at confidence 1.00 | 68.13% | 95.58% | 83.10% | 64.24% | 99.22% |
| 3 | prepositional stress shift | **69.30%** | **95.77%** | **83.10%** | **64.24%** | **99.46%** |

Per-token, paired: change 2 fixed 52 and broke 2; change 3 fixed 19 and broke
1. All three "breaks" are against malformed reference tokens — `йог+о`, a
lowercase `киї+в`, and one sentence that marks the prepositional shift on the
preposition (`Біжа́ть до́ мене́`) rather than on the pronoun.

Total disagreements outside single-vowel function words: **636 → 561**.

### The measurement that did not transfer

The offline harness said change 1 should be much larger. Letting morphology
decide every ambiguous token — overriding the model where both answer — scored
net +98 (+125/−27), p < 0.00001 there, and the grammatical class went from
68.62% to 84.57%.

Against the deployment the same change scored **−0.54 heteronym points**. Three
arms, live:

| | morphology for uncovered forms only | overriding the model | filling abstentions |
| --- | ---: | ---: | ---: |
| heteronym | 82.55% | 82.01% | **82.66%** |
| macro-F1 | 63.89% | 63.37% | **64.24%** |
| sentence | 65.20% | 64.72% | **65.30%** |

The harness ran a different model over a reduced manifest and agreed with
itself. What shipped is the narrow version: morphology now also sees the tokens
the model declined with a low margin, which previously fell to a dictionary
coin flip, and a confident model answer stands.

### Routing by ambiguity class: refuted

The structure of the error budget suggests a pipeline shaped by what *kind* of
ambiguity a form has — dictionary for one reading, a masked classifier for
homographs, morphology for grammatical alternations, dictionary default for
free variation. Implemented behind `--route` and measured:

| slice | fixed order | routed | net | p |
| --- | ---: | ---: | ---: | ---: |
| all ambiguous | 67.64% | 66.95% | −10 (+24/−34) | 0.237 |
| homograph | 64.88% | 63.90% | −2 (+23/−25) | 0.885 |
| grammatical | 68.62% | 68.62% | **0** (+0/−0) | 1.000 |
| free variation | 21.57% | 17.65% | −4 (+0/−4) | 0.125 |

The routing fires — 227 tokens reach the masked head, 109 the dictionary
default — and changes no outcomes. Grammatical is exactly zero because
morphology was already consulted first; the tier that looked mis-ordered was
switched off, not mis-ordered. Homographs are a wash on the one class the
masked classifier was meant to own. Free variation was a prediction and it was
wrong: the dictionary's first entry loses 4 and wins none against the model's
guess.

### Is the prepositional shift a rule or the annotator?

The test applied before implementing it: is the reference consistent *in both
directions*, and does the pattern hold across the word class rather than the
two words in question?

| context | first-syllable stress |
| --- | ---: |
| after a preposition | 39/40 |
| no preposition | 0/11 |

Across six pronouns — `мене`, `себе`, `тебе`, `нього`, `неї`, `них` — four of
which were not part of the original observation. A convention would not hold
that shape.

By the same test the free-variation class is *not* worth chasing: 172 of the
636 disagreements are tokens where the reference marks two acceptable stresses
and the pipeline emits one. There is no right answer there to converge on.

### A model, and a training run that was not worth its wall clock

`ukr-models/xlm-roberta-base-uk`, 110M parameters against the shipped model's
278M, trained on the name-free corpus: `dev_group_macro` 0.8556 after two
epochs, 7.1 hours. Head to head on lang-uk with everything else held equal it
is **net +24 tokens, p = 0.116** — not a measurable improvement, and not a
regression either. The gain is size, not accuracy.

An earlier attempt with `jhu-clsp/mmBERT-base` (308M) was killed after 21.6
hours without finishing epoch 0; the comparable XLM-R run does an epoch in 102
minutes. Vocabulary trimming reduces parameters and memory, not FLOPs — the
encoder is the same 12×768 either way.

## tok-v2: the token classifier trained on audiobooks (2026-09-14)

`tok-v1`, trained on Common Voice, scored 66.3% on forms it had not seen
against 66.4% for answering `candidates[0]` — it had learned which form, not
which context, because both readings of a form were present in 9% of forms.
`tok-v2` is the same architecture trained on 24 audiobooks (369 h, mined three
times over, labelled by the audio ranker at ≥0.95): 47,456 ambiguous rows over
4,759 forms, 884 with both readings present.

**Inside its own corpus** (held-out book positions and held-out forms):

| condition | rows | accuracy | `candidates[0]` |
| --- | ---: | ---: | ---: |
| seen form | 5,218 | 92.3% | 59.7% |
| unseen form | 4,758 | **76.6%** | 58.2% |

**On modern text** — 19,089 ambiguous Common Voice tokens with an audio gold
at ≥0.99, none of whose sentences the classifier saw:

| rule | accuracy on ambiguous tokens |
| --- | ---: |
| pipeline as deployed | 85.4% |
| classifier everywhere | 83.4% |
| classifier on forms the books contain, pipeline elsewhere | **90.4%** |
| … and the `stressed` tier always to the classifier | ~91.0% |

The classifier must not answer for forms the books never showed it: there it
is at 64% where `dictionary_default` is at 89%. Where the books did show the
form it beats every tier, by +15 to +20 points on `dictionary_default` and by
+14 on `stressed`.

**On lang-uk** (human gold, adversarial sentences built to force rare
readings), the classifier loses overall — 83.10% → 77.43% heteronym accuracy
with the weak tiers overridden — because 368 of its 1,553 tokens are forms the
books never contain, and because the books teach the common reading where the
benchmark asks for the rare one. The one slice it wins there is the same one it
wins on modern text: `dictionary_default` on forms the books saw, 60% → 78%.

**Caveat.** The modern-text gold and the training labels come from the same
audio ranker. A systematic ranker error on a form would count as agreement
here. The lang-uk result is the independent check, and it confirms the
direction only for `dictionary_default`. The `stressed` gain (+14 on modern
text, −2 on lang-uk over 265 tokens) needs human review before it is deployed.

**Not deployed.** Deployment needs an ONNX export and a tier in the Go API
that consults the classifier only for forms in its coverage list; the list is
`output/ml/corpus/tok-v2/train.jsonl`'s forms.

## tok-v3d: 118 audiobooks, the cap lifted, the cross-encoder's picks taken (2026-09-18)

A second and third batch of books (96 more, 1,770 h) were mined in one pass
each by `run_mine_book_v3.py` (audit lift +23.0 pp over 1128 h, ranker
96.20% against the lexicon). Trained on all of it, `tok-v3` came out *worse*
on modern text than `tok-v2` — 87.7% against 90.4% by the deployment rule.
Two causes, found by ablation:

* **archaic and verse books** (Чайковський, Лепкий, Кобилянська, Куліш,
  Мирний, Нечуй-Левицький, Пчілка, Франко, Хоткевич, Назарук, Осьмачка;
  Данте, Котляревський) teach readings modern speakers do not use — dropping
  the 21 recovers ~1 point;
* **the per-reading cap of 80** — with four times the rows both readings of
  a common form hit the cap and the classifier lost the prior that decides
  most modern-text cases. Cap 400 recovers most of it; cap 2000 all of it.

`tok-v3d` = modern books, cap 2000, 256,438 ambiguous rows over 7,696 forms.

**Deployed rule:** the classifier answers where the pipeline would take the
dictionary default *or the cross-encoder's pick*, for forms seen 100+ times
in its training rows (299 forms, 22% of ambiguous tokens on modern text).

| test | pipeline | tok-v2 live (dict. default, seen 10+) | **tok-v3d live** |
| --- | ---: | ---: | ---: |
| modern text, 6,000 ambiguous tokens | 85.9% | 88.0% | **90.8%** |
| … the slice the tier takes | | 78.7% → 96.0% (324) | 78.4% → **95.9%** (1,317) |
| lang-uk heteronyms | 85.06% | 84.84% | **84.95%** |
| lang-uk words | 96.04% | 96.03% | 95.98% |

Reviews: `output/ml/live_review16.json` (laptop), `live_review16_desktop.json`.

## Step 3 of PLAN_QUALITY: the stress placer (2026-09-19)

Words the lexicon lacks go to the suffix fallback: 461 lang-uk tokens at
71.6%, and 70–73% on modern text. The books produced 125,372 such words with
an audio-derived stress over 50,870 forms (at ranker confidence ≥0.95, cap 40
per form). Six placers were trained on them and scored on the same two sets —
the benchmark's 484 fallback tokens (human gold) and 2,698 Common Voice tokens
with no lexicon candidate (audio gold at ≥0.99):

| placer | held-out forms | lang-uk (484) | Common Voice (2,698) |
| --- | ---: | ---: | ---: |
| suffix table, as served | — | **72.5%** | 73.5% |
| character BiGRU, books only | 85.0% | 68.6% | 80.3% |
| character BiGRU, books + 600k lexicon forms | 82.5% | 70.7% | **80.7%** |
| character BiGRU, books + 2.3M lexicon forms, 3×512 | 75.2% | 67.6% | 74.2% |
| XLM-R-uk over the sentence | 83.6% | 67.6% | 76.1% |
| mmBERT-base over the sentence | 85.3% | 61.4% | 80.6% |
| character + table (table if ending ≥6 chars and placer <0.8) | — | 70.9% | 81.1% |

Two findings. **й must not be a candidate**: it carries an ordinal in this
project's signatures and the first placer put stress on it («й́ому», «ї́ї»);
excluding it was worth +2.7 on lang-uk. **The two sets disagree about what a
fallback word is**: lang-uk's are proper nouns (27% capitalised) and the
benchmark's own nineteenth-century sentences (одушевимо, єго, сербове), where
a 1.9M-form memory beats every learned model; modern text's are derivations
and dialect, where the character model wins by seven points. mmBERT's
sentence context does not help either kind, and its tokenizer warns about a
wrong regex under transformers 4.57.

Not deployed: the best variant is +7.6 on modern text and −1.6 on lang-uk
over a tier that holds ~1.7% of words, about +0.1 point overall. The models
are in `models/placer-*` on the laptop; `run_train_char_placer.py --eval-only`
reproduces the table.

## tok-v5: ambiguity by the served lexicon (2026-09-20) — PLAN_PARITY step 1

Measured RUAccent's way — the 200 most frequent ambiguous forms of Common
Voice, audio gold at ≥0.99, the live API — the pipeline stood at 94.61%
against the paper's 96.37%, and 1.7 of the 1.8 points sat in 558 tokens the
cross-encoder decided at 61%. The classifier reads those forms at 96% — when
it has seen them. It had not: the miner's trie calls 3,400 database-ambiguous
forms single-reading (мені, які, також, трохи, кого, мало), so their rows
never became training rows.

`tok-v5` is `tok-v3d`'s recipe (the same 100 books, cap 2000, gate 0.95) with
ambiguity decided by the served lexicon: 336,189 ambiguous rows over 11,054
forms, 386 covered at 100+ (was 299).

| test | tok-v3d | **tok-v5** |
| --- | ---: | ---: |
| top-200 forms, audio gold (13,201 tokens) | 94.61% | **95.88%** |
| … `stressed` slice | 558 at 61.1% | 334 at 94.0% |
| modern text, all ambiguous (6,000 live) | 92.0% | **92.9%** |
| tokens the classifier takes | 22% | 37% (95.0%) |
| lang-uk words | 95.93% | 95.98% |
| lang-uk heteronyms | 84.95% | 84.62% |

Live on both machines from 2026-09-20. The lang-uk cost is three tokens.

## tok-v6: 5.3M YouTube rows, and what the ranker gets wrong (2026-09-21) — PLAN_PARITY step 2

The book-less miner ran over 1,293 videos (1,527 h; six channels, four
accounts, five machines) and gave 5.28M rows. `tok-v6` is `tok-v5`'s recipe
over the 100 books plus those rows: 619,939 ambiguous rows over 13,691
forms, 747 covered at 100+ (was 386).

| model | YouTube rows | modern, classifier everywhere | modern, rule @100 | lang-uk words / heteronyms |
| --- | --- | ---: | ---: | ---: |
| tok-v5 (live) | none | 93.9% | 90.4% | 95.62% / 84.51% |
| tok-v6 | all, weight 1 | 93.7% | **90.7%** | 93.51% / 81.46% |
| tok-v6b | gate 0.99, weight 0.5 | 93.8% | 90.7% | 93.81% / 81.90% |
| tok-v6c | majority reading only | 92.1% | 89.3% | 94.03% / 82.22% |
| tok-v6, tok-v5's 386 forms | all | — | 89.8% | 95.62% / 84.51% |

None ships: the rule is lang-uk down ≤ 0.3. The last row locates the loss —
restricted to the forms `tok-v5` already covered, `tok-v6` scores lang-uk
exactly as `tok-v5`, so the two points sit entirely in the 361 forms the
YouTube rows newly pushed past 100, where the dictionary default beats the
classifier on adversarial sentences. `min_seen` 80 for `tok-v5` is +0.1 on
modern text at no lang-uk cost; 60 costs 0.5.

### The ranker measured on homographs, without synthetic speech

The plan's step 3 (synthesise both readings with the project's TTS) would
have measured the ranker on a synthesiser's prosody; dropped. Instead the
LLM sense labeller that built the cross-encoder corpus judged 4,221
classifier rows (399 heteronyms with glosses, 8 per form × reading ×
source), and the user listened to 100 of the disagreements:

| slice | ranker–judge agreement | by ear: ranker / judge / unclear | ranker precision (est.) |
| --- | ---: | ---: | ---: |
| books, majority reading | 81.0% | 24 / 1 / 0 | ≈ 99% |
| books, minority reading | 56.8% | 19 / 4 / 2 | ≈ 92% |
| YouTube, majority reading | 81.9% | 23 / 0 / 2 | ≈ 100% |
| YouTube, minority reading | 40.2% | 10 / 13 / 2 | **≈ 66%** |

The judge leans to the dictionary sense (за́раз for a spoken зара́з) and is
no filter for majority readings. The ranker's confidence does not separate
its errors (≥0.99 agrees 75%, <0.99 77%). A third of the minority-reading
labels on spontaneous speech are wrong — and the per-reading cap keeps every
one of them. Scripts: `run_judge_ranker.py`, `run_ear_check.py`; the 100
answers are `output/ml/ear_check/answers.json` on the laptop, the first
human gold on audio homographs.

**Next:** a per-form allowance instead of a global `min_seen` — the
classifier only on forms where it beats the live pipeline on held-out rows
— and the YouTube audio for the ranker (its weak spot is now known) rather
than for the classifier directly.

### A per-form allowance instead of a global gate (2026-09-22)

The clips of the modern test split in two by hash; on the dev half the
classifier is allowed on a form only where it beats the pipeline by two
tokens (`run_eval_tok_modern.py --allowance`). 88 forms for `tok-v5`, 93
for `tok-v6`. On the test half the rule scores 90.9% / 90.7% against the
global gate's 90.6% / 90.8% — the same — while taking a third of the
tokens; on lang-uk both fall to 94.92% / 83.75% (the live gate: 95.62% /
84.51%). What the modern clips say about a form does not carry to
adversarial sentences; not deployed. The gate stays `min_seen` 100 on
`tok-v5` (80 is +0.1 modern at no cost and can go live with the next
change).

### Zero-shot stress from a causal LLM, Jev-style (2026-09-22)

The fixed-choice scoring path (`ukstress_ml/llm_picker.py`: the sentence
and the word in a chat prompt, the stressed spellings as lettered options,
softmax over the letters' logits at the first answer position, both option
orders averaged to cancel the letter bias) on `Qwen/Qwen3-4B-Instruct-2507`
with nothing trained:

| | modern, every ambiguous token (2,000) | lang-uk heteronyms |
| --- | ---: | ---: |
| Qwen3-4B zero-shot | 42.8% | 50.05% |
| … with the sense lexicon's glosses | 48.9% | 53.00% |
| `tok-v5`, same tokens | 93.8% | 84.51% |

Coin-flip: the model has no Ukrainian stress; glosses add 3–6 points of
sense. The mechanism is the classifier's own (softmax over the served
readings), so the only thing a decoder backbone could add is pretraining,
and it would have to learn stress from the corpus as XLM-R did, at 15× the
size. Not pursued further without a reason to.

### LoRA on Qwen3-1.7B, the same decision (2026-09-22)

`run_train_llm_lora.py`: the zero-shot prompt, cross-entropy over the
option letters, order shuffled per row, one epoch over the 245k
multi-candidate rows of the `tok-v5` corpus (65 min on the 5090 at 62
rows/s). Dev 92.1% (`tok-v5` after four epochs: 93.7%).

| | modern, every ambiguous token (2,000) | modern rule @100 | lang-uk rule @100 |
| --- | ---: | ---: | ---: |
| `tok-v5` | 93.8% | 90.5% | 95.62% / 84.51% |
| LoRA, 60k rows | 88.0% | 90.2% | 95.61% / 84.51% |
| LoRA, one epoch | 88.5% | 90.5% | 95.61% / 84.51% |
| LoRA, three epochs | 91.8% | 90.8% | 95.62% / 84.51% |

Three epochs (3.5 h) land on the deployment rule's numbers: +0.3 modern
over `tok-v5`, lang-uk identical to the third decimal — the gate at
`min_seen` 100 hands both models the same tokens and they agree on nearly
all of them. Everywhere else the adapter is still two points behind, and
it costs a 1.7B forward pass per token against 278M. A decoder backbone
buys nothing here: the classifier's XLM-R already reads the sentence, and
what both lack is gold, not parameters. Not deployed; the scripts stay
for the day a stress-aware pretrained encoder is worth trying
(PLAN_PARITY step 4).

At 2e-4 with 50 warmup steps the three-epoch run collapsed to chance
(loss ln 2) for 13,000 steps before escaping; 1e-4 with 200 warmup steps
is the default now.

**Live from 2026-09-22:** the gate on `tok-v5` moved from `min_seen` 100 to
80 — 461 eligible forms instead of 386, +0.1 on modern text, lang-uk
95.97% / 84.62% on the live API (was 95.98% / 84.62%). Everything else
tried this week (the `tok-v6` series, the per-form allowance, zero-shot and
LoRA decoders) failed the rule and is not deployed.

## A learned combiner over the tiers (2026-09-23) — PLAN_QUALITY step 5

The pipeline decides an ambiguous token by a fixed tier order: the first
tier that answers wins and the others are never asked. The combiner asks
all of them for every candidate and learns how far to trust each —
`build_combiner_tokens.py` (labelled tokens: Common Voice audio gold at
≥0.99, lang-uk human gold; candidates and the pipeline's answer from the
live API), `build_combiner_features.py` (per candidate: lexicon rank and
capitalisation, the morphology tier's pick and the tagger's agreement with
the reading's tags, the cross-encoder's softmax, tok-v5's probability and
training profile, the share of narrators over 6,217 h who said this
reading, whether the readings are told apart by sense), `run_combiner.py`
(a one-hidden-layer scorer, softmax over the token's candidates).

It is residual: the pipeline's answer and its tier are features, so it
learns when to depart from the pipeline rather than rebuilding it, and a
gate keeps the pipeline's answer unless the combiner prefers another
candidate by at least τ in probability. Trained on cv:train plus
lang-uk:dev (×4), τ = 0.2 chosen on the dev halves, reported once on the
held-out halves:

| held-out test | tokens | pipeline | combiner, gated |
| --- | ---: | ---: | ---: |
| Common Voice | 3,001 | 92.74% | **94.77%** |
| … of which top-200 forms | 2,058 | 96.06% | **97.57%** |
| lang-uk | 748 | 83.96% | **83.96%** |

Without the gate the combiner is +2.7 on Common Voice and −3.9 on lang-uk
(trained on cv alone) — it learns "trust the prior and the classifier",
which is right on frequent readings and wrong on sentences built to force
the rare one. More lang-uk weight (×8, ×16) makes both worse. Tiers alone
on the same Common Voice test: classifier 90.9%, audio prior 92.6%,
morphology 86.7%, cross-encoder 72.3% (it answers 4,041 of 15,939 tokens),
lexicon order 68.4%.

**Live from 2026-09-23** (`COMBINER_ENABLED=1`, `models/combiner-v1`,
`/internal/v1/combine`). Served and offline decisions agree on 600 of 600
sampled tokens. Measured on the live API against the same pipeline with the
combiner off, on the halves it never trained on:

| live, held-out only | tokens | combiner off | combiner on |
| --- | ---: | ---: | ---: |
| top-200 forms, Common Voice test clips | 2,774 | 95.93% | **97.51%** |
| lang-uk test half, ambiguous tokens | 753 | 83.53% | 83.27% |

On everything (training halves included, so optimistic): top-200 95.88% →
97.27%, lang-uk words 95.93% → 96.02%, heteronyms 84.62% → 85.28%. The cost
is latency: the model service reruns the tagger, the cross-encoder and the
classifier for every ambiguous token on CPU, and a two-clause sentence went
from 341 ms to 733 ms. The tier answers the API already has (the
cross-encoder's scores, the morphology pick) could be passed in instead.

### The agreement repair, and a guard on the combiner (2026-09-23)

«Мої сестри.» came out сестри́: the tagger reads the whole phrase as a
genitive singular, «мої» included, though only «моєї» can be one. The
morphology tier now checks the word before a noun against the dictionary
(pymorphy3/VESUM): when that word is tagged DET/ADJ, every dictionary
reading of it is a modifier, and the tagger's case and number for it are
not among them, the noun takes the one reading that agrees with what the
modifier can be (status `agreement`). A first version that trusted any
possessive-adjective reading fired nine times on the top-200 set («ханів
була», «Артемові вони») and was wrong seven; the tightened rule fires on
none of it and leaves both tests where they were. Combiner-v1 then put
сестри́ back from the classifier and the prior, having never seen the new
tier in training, so the combiner now keeps any answer from a tier outside
its training data. A matching modifier feature in the combiner (v2) gave
no held-out gain and is not served.

Live after both, held out: top-200 95.93% → 97.51%, lang-uk 83.53% →
83.27% — unchanged from the combiner alone; on everything, top-200 97.27%,
lang-uk words 96.02%, heteronyms 85.28%.
