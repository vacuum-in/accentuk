# Ways to improve the pipeline

Ordered by measured value per unit of effort, not by how interesting they are.
Every estimate names the measurement it comes from; where there is no
measurement, it says so.

Current state: **90.36% word / 79.17% heteronym** on lang-uk's benchmark
against a **88.67% / 64.34%** baseline. Ceiling of the benchmark is **99.76%**,
so roughly 20 heteronym points are available without any new lexicon data.

---

## Where the remaining error actually is

| owner | tokens | current | if perfect |
| --- | ---: | ---: | ---: |
| morphology (grammatical alternations) | ~94 | 85.4% resolved, 74% correct | +7–9 pts |
| cross-encoder (semantic homographs) | 121 | ~90% | +5–6 pts |
| free variation — either answer valid | 13 | — | 0 |
| unreachable | 3 | — | 0 |

Plus a separate ~1.0 point on unambiguous words, where the baseline is ahead.

---

## 1. Morphology: the largest block

The parser owns ~94 failing tokens and declines 14.6% of what it sees.
Breakdown of the declines, measured:

| reason | share | what would fix it |
| --- | ---: | --- |
| tags do not separate the readings | 6.8% | route to the model (needs glosses) |
| no tag set matched — tagger error | 4.3% | better tagger, or an ensemble |
| parse satisfies several readings | 3.3% | a tie-break better than declining |

**1a. Ensemble the taggers.** spaCy and Stanza disagree on different words;
`resolve()` already refuses when readings conflict, so agreement between two
taggers is free precision and disagreement is an honest decline. Cost: one
extra parse per ambiguous token. No training.

**1b. Tie-break when several readings match.** Today `resolve()` returns None
and falls to a dictionary default — a coin flip. The candidates' relative
corpus frequency is already computable from the training corpus and would beat
a coin flip. 21 tokens on this benchmark; more in running text.

**1c. Fine-tune a tagger on the trie's own tags.** The trie carries 2.9M
(form, tags, accents) records — that is labelled data for exactly the four
features `resolve()` needs. A small tagger trained on it would be aligned with
the tag vocabulary by construction, which is what the PRON/DET fix showed
matters more than capacity.

**1d. Do not reach for a bigger model.** Measured: `uk_core_news_lg` (231 MB)
scored 1.4 points *worse* than `sm` (15 MB).

---

## 2. Lexicon quality: the one metric where the baseline wins

Unambiguous words: 97.56% against the baseline's 98.60%. The baseline reads the
source trie directly; this lexicon is derived from it through an ETL that loses
accuracy on the way.

**2a. Re-derive primary readings from the trie wherever it is unambiguous.**
8,702 corrections have been applied; the scan that found them covered only
forms where both sides hold exactly one reading. Widen it to forms where the
trie is unambiguous and the lexicon holds several.

**2b. Audit the remaining unambiguous failures directly.** The per-token error
list is the tool; it named `україни`, `мене`, `які` in one pass. Most remaining
entries are likely the same class.

**2c. Demote proper nouns in the data, not just in the ordering.**
`run_classify_toponyms.py` and `EXCLUDED_toponyms.csv` already exist from
earlier work and never reached the serving path. Confidence 1.00 on a toponym
is the underlying problem; the case-agreement rule is a workaround.

---

## 3. The cross-encoder: 121 tokens, and a known mechanism

The relationship is measured and mechanical: **a group whose weakest sense
reaches ~20% of its training rows scores ~1.00; below that it degrades in
proportion.** `правило` had 96 rows, 1% minority, and scored 0.640; `обід` had
*zero* rows and scored 1.000, because with no data the model reads the gloss
instead of learning a prior.

**3a. One-sided data is worse than no data.** `--drop-one-sided` with
`--min-sense-share` already implements this and took `правило` from 0.640 to
1.000 by *removing* its rows. Tune the threshold — 0.15 cut slightly too deep,
0.08–0.10 is probably better.

**3b. Generate for the senses that are rare everywhere.** Mining supplies
senses rare in encyclopedias but ordinary in speech; it gave `правило` 260 new
rows and zero of its minority sense. `run_generate_balanced.py --min-share`
targets exactly the deficit, proportionally.

**3c. Improve glosses rather than volume.** The cross-encoder scores
`(sentence, gloss)`. Two glosses that differ only in a case label are nearly
identical inputs — which is why grammatical heteronyms belong to the parser.
For the semantic ones, a gloss that names concrete collocates would give the
scorer more to work with than a dictionary definition.

**3d. Do not simply add training data.** Measured: a retrain on 85k rows,
after mining 240k sentences, generating 12.5k and labelling 42k, scored
**−0.3** on this benchmark. The model is consulted on 1.47% of tokens.

---

## 4. Coverage

**4a. Publish the supplementary datasets.** 20,684 rows reach the API only
through a deployment flag. `etl/merge_datasets.py` folds them into one
publishable dataset; then `ukstress publish`.

**4b. Import the rest of lang-uk's heteronym dictionary.** 37,030 curated
groups; 3,671 forms were added and 30,190 were already complete. Re-run
periodically — it is maintained.

**4c. Gloss the forms that are ambiguous but unservable.** ~10,000 remain.
Low yield per form (the last batch of 10 moved nothing), so drive it by corpus
frequency and stop when the curve flattens.

---

## 5. Architecture and operations

**5a. Batch morphology per request, not per sentence.** `parse_batch` exists
and amortises per-call overhead; the serving endpoint currently groups by
sentence within one request but could batch across the whole payload.

**5b. Cache parses.** The same sentence is often re-stressed; the same form
recurs constantly. `readings` is already cached; the parse is not.

**5c. ONNX for the cross-encoder.** `serving.py` already supports an ONNX
backend and the graph builder exists. Measured earlier at ~2.1x on CPU.

**5d. Put the benchmark in CI.** Every regression this session was found by
running it. A gate on heteronym accuracy would have caught the timeout, the
stale binary, and the truncated inventory hash immediately.

---

## 6. Evaluation discipline

Not an improvement to the pipeline, but the thing that made the improvements
findable.

* **Measure the product, not the harness.** The evaluation script reimplements
  the serving path; for most of a day it read 17.7 heteronym points higher than
  the live API, and the difference was a timeout that disabled a whole tier.
* **Read the per-token error list before proposing a fix.** Three consecutive
  diagnoses from aggregates — coverage, archaic orthography, proper nouns —
  were all wrong. The error list settled it in one pass.
* **Change one thing per measurement.** A manifest and a binary changed
  together produced a 6.7-point "regression" that was neither.
* **Check the scorer before trusting a bound.** The ceiling was reported as
  95.06%, 72.79% and 98.30% before settling at 99.76%.
* **A harness result is a hypothesis until the deployment confirms it.** Giving
  morphology every ambiguous token was worth +8.29 heteronym points offline and
  −0.54 live; the harness ran a different model over a reduced manifest and
  agreed with itself. `run_live_bench.py` posts to the running service and
  scores with the maintainers' own evaluator, which is what settled it.
* **Ask whether the pattern is a rule or the annotator.** Before implementing
  the prepositional stress shift, the test was whether the benchmark is
  consistent *in both directions and across the word class*: after a
  preposition 39/40 pronouns take the first syllable, with no preposition
  0/11 do, over six pronouns. A convention would not hold that shape. The
  free-variation class fails the same test by construction — 172 tokens where
  the reference marks two acceptable stresses — and chasing those is fitting
  the annotator, not the language.
* **A source's silence is not disagreement.** The dictionary records only
  `мене́`, which looked like evidence against the shift until the obvious
  point landed: a table keyed by one word cannot express a stress that depends
  on the previous word.

---

## 7. Tooling built to make the above answerable

* **`ml/scripts/run_live_bench.py`** — posts the benchmark's own sentences to
  the running API with eight concurrent callers and scores them with the
  maintainers' evaluator. `--save` writes reference and produced sentences side
  by side, which is what the per-token attribution is read from.
* **`ml/scripts/run_dataset_browser.py`** — a web view over every dataset in
  `output/ml`, with descriptions, byte-offset paging over files too large to
  load, review verdicts (`wrong` / `recheck` / `ok`), tags (`not-frequent`,
  `toponym`, `alt-needed`), click-a-reading-to-propose-it, and sorting or
  filtering by corpus frequency. Reviews are append-only JSONL under
  `output/ml/.annotations/`.
* **`ml/scripts/run_split_propernames.py`** — separates surnames, toponyms and
  given names out of the inventory and the training corpus: 1,142 of 15,460
  form groups, 4.5% of silver rows. A name's stress is a property of the name,
  not of the sentence, so the rows only teach the model to guess.
* **`ml/scripts/run_manual_corrections.py`** — reviewed readings at confidence
  1.00, with the manifest prune and the trie-default exclusion that make them
  stick.
* **Per-epoch training progress** — the loop printed nothing until an epoch
  ended, so an mmBERT run went 21 hours without anyone being able to tell it
  had not finished epoch 0. It now prints loss, step rate and time remaining
  every 500 steps.

---

## What is probably not worth doing

* **A larger tagger** — measured worse.
* **More training data without a balance target** — measured negative.
* **Replacing the cross-encoder architecture** — it is right ~90% of the time
  on 1.47% of tokens; even perfection is worth ~5 points, and the parser tier
  is worth more for less.
* **Chasing 98% end-to-end** — the ceiling is 99.76%, so it is not impossible,
  but it requires both the parser and the model near-perfect on genuinely
  ambiguous input. 95% is the defensible target; beyond that, each point costs
  disproportionately.
