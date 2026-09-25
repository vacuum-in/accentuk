# Plan: raising output quality after `tok-v3d`

Written 2026-09-18. Five steps, two cheap experiments first, one decision
gate at the end. Each step names its inputs, the exact command shape, what
"done" means in numbers, and how to undo it. Where a step needs the user, it
says so. Machines: **desktop** (this repo, the live API on `:8080`, the
lang-uk clone in the session scratchpad) and **laptop** (`gpu-host`,
RTX 5090, the mined books under `~/audiotostress/artifacts/books`, the same
live stack). Every deployment happens on both.

Where the loss is today (lang-uk, `live_review19`): 9,455 scored words, 385
wrong — suffix fallback 32%, lexicon and cross-encoder 34%, morphology 17%,
token classifier 12%, dictionary default 11%. Modern text (Common Voice,
audio gold): 90.8% on ambiguous tokens, suffix tier 70.3%.

Baselines to beat, stated once so every step reports against them:

| test | words | heteronyms | ambiguous (modern) |
| --- | ---: | ---: | ---: |
| live, 2026-09-18 | 95.93% | 84.95% | 90.8% |

---

## Step 0 — two experiments (half a day, no user time)

**0a. Confidence gate on lang-uk.** The classifier's confidence separates
right from wrong on modern text (`stressed`, seen 10+: 85.5% → 92.6% at
≥0.9) and has never been tried on lang-uk, where the losses are the
adversarial tokens — plausibly the low-confidence ones.

* add `--min-confidence` to `ml/scripts/run_eval_tok_langukbench.py` (the
  `/internal/v1/token` reply already carries it);
* run the grid `min_seen ∈ {30, 100}` × `min_conf ∈ {0, 0.9, 0.95, 0.99}` ×
  override `∈ {dictionary_default, dictionary_default+stressed}` on
  `live_review19.json`, and the same grid on modern text with
  `run_eval_tok_modern.py` (Step 0c);
* **done:** a 16-row table; if a cell beats the live rule on *both* tests,
  set it in `coverage.json` (`min_seen`) and the API (`min_conf` is a new
  field on the token decision the Go side must respect — 20 lines in
  `stress.go`), redeploy, re-measure.

**0b. `reading_not_offered`.** 10,029 book rows where the ranker chose a
vowel the lexicon does not list for the form. Group by form, keep forms with
≥10 such rows at ≥0.99 and the unlisted reading in ≥80% of them. Those are
either lexicon gaps (feed Step 1) or a ranker blind spot (feed Step 1's
exclusion list). One script, one table.

**0c. Move the modern-text evaluator into the repo.** It lives in two `/tmp`
copies. `ml/scripts/run_eval_tok_modern.py`: arguments `--model`,
`--coverage`, `--min-seen`, `--min-confidence`, `--override`; reads the three
verify sweeps; prints the per-tier table and the three rule totals. Commit,
sync. Everything after this reports through it and through
`run_eval_tok_langukbench.py`.

---

## Step 1 — audit the lexicon against the book audio (script: 1 day; review: user)

**Why first.** Round 2 of this, on 311k Common Voice words, produced 85
corrections and +2.0 points of heteronym accuracy. The books hold 4.5M words
the lexicon names, each with the ranker's reading and confidence, from ~130
narrators. воду́ and хо́ча were found by reading one paragraph; this finds
the rest of that class at once.

**Inputs.** On the laptop: `*.rows3.jsonl` (batches 2, 3), and the 24 first-
batch books via `*.rows2*.jsonl` + `*.labels*.jsonl` (labels carry the
ranker pick; rows carry the lexicon label for `readings == 1`).

**Script `audiotostress/scripts/run_audit_lexicon_books.py`.**

1. For every row with `readings == 1`, `label` set, `confidence ≥ 0.99`:
   count per form `(agree, disagree, disagreeing_reading, books, narrators)`.
2. Candidate = form with ≥ 8 disagreements, disagreement share ≥ 0.8, in ≥ 3
   books, and the disagreeing reading is one vowel ordinal (not scattered).
   Books alone can be wrong in unison (one narrator's habit) — hence ≥ 3.
3. For every row with `readings > 1` and the ranker's pick **not** in the
   lexicon's candidates (Step 0b's table): same thresholds; these are
   "the lexicon lists two readings and the narrators use a third".
4. For ambiguous forms where the lexicon's *first* candidate (the dictionary
   default) is the narrators' minority at ≥ 0.8 across ≥ 3 books: candidate
   for **reordering**, not correction — the default is served on every form
   the classifier does not cover.
5. Exclude: prepositional clitics (`is_prepositional_clitic`), single-vowel
   forms, forms in `EXCLUSIVE`/`CORRECTIONS` already, forms on the archaic
   books' exclusion list (Step 3 commits it).
6. Output `DICTIONARY_AUDIT_ROUND3.md` in the round-2 format: form, pipeline
   reading, narrators' reading, books, narrators, share, examples (3
   sentences from the book text with the span marked), and an empty
   **verdict** column.

**User.** Two verdicts per row, as before: "second is right" (→ `EXCLUSIVE`
+ `CORRECTIONS`) or "both valid, second more frequent" (→ reorder, dataset
12). Expect 100–200 rows; an hour.

**Apply.** Add to `run_manual_corrections.py`; for reorders, a new dataset
`manual-order-v1` written by the same script with the chosen reading leading
at confidence 0.99 (it joins `REVIEWED_DATASETS`, not `EXCLUSIVE`). Run with
`--manifest`, then `docker compose restart stress-model api` — **both** —
on both machines. Re-run both evaluations.

**Done:** heteronyms ≥ 85.5% and words ≥ 96.0% on lang-uk with no loss on
modern text; every correction has a reason string. **Undo:** the datasets
are separate; drop them from `REVIEWED_DATASETS` / `SUPPLEMENTARY_DATASETS`.

---

## Step 2 — `tok-v4`: every book, the minority reading weighted (automatic, ~4 h GPU)

**Why.** 299 covered forms is the whole reach of the classifier; the third
batch (36 books) is mined and unused. And the classifier learns each form's
*prior* more than its context, because the minority reading is rare in
training — cap 2000 still drops 92k rows, and the losses on the test
paragraph (підземні хо́ди, почервоніла по́ра) are minority readings in
plain contexts.

**2.1 Commit the exclusion list.** `ml/data/book_exclusions.json`: the 21
archaic/verse slugs from `/tmp/archaic.txt` on the laptop, each with a reason
(`"originals before ~1930"`, `"verse"`), plus the four wrong-audio folders
and the duplicate slugs from the inventory. `build_book_corpus.py` reads it
by default (`--exclude` stays as an override).

**2.2 Build.** On the laptop:
```
ml/.venv/bin/python ml/scripts/build_book_corpus.py \
  --books ~/audiotostress/artifacts/books --out output/ml/corpus/tok-v4 \
  --per-reading-cap 100000 --min-confidence 0.95
```
Report: ambiguous rows, forms, forms with both readings present (was 1,867),
and the frequency-bucket table (`docs/lessons.md` §4.3).

**2.3 Balance.** New trainer flag `--balance-readings`: each row's loss
weight = `n_form / (n_readings_present × n_form_reading)`, so within a form
every reading present carries equal total weight while forms keep their
relative size. Rows for negatives (single-reading) unchanged. Ten lines in
`run_train_token_resolver.py`, plus the weight passed to `cross_entropy`.

**2.4 Train four variants**, 4 epochs, batch 16, on the 5090 (~40 min each):
`cap 2000`, `uncapped`, `uncapped + balance`, `cap 2000 + balance`.

**2.5 Evaluate** each with `run_eval_tok_modern.py` and
`run_eval_tok_langukbench.py` at the live rule, and at the best Step 0a cell.
Also the trainer's own held-out-forms accuracy (v3d 76.0%, v3c 80.3%).

**Done:** pick the variant that is ≥ live on lang-uk heteronyms and best on
modern text; write `coverage.json` with the chosen `min_seen`; deploy on both
(`TOKEN_MODEL_DIR`, restart `stress-model` + `api`); re-measure live on both.
Record in `RESULTS.md`. **Undo:** point `TOKEN_MODEL_DIR` back at `tok-v3d`.

**Risk.** Balancing can hurt the *common* reading on modern text (that is
the prior it removes). The modern-text test will show it; if so, weight with
a square root instead of the full inverse.

---

## Step 3 — a stress placer for words the lexicon does not have (2 days)

**Why.** The suffix fallback is a third of the lang-uk loss (461 tokens,
72.0%) and scores 70.3% on modern text. The books produced 132k out-of-
vocabulary words with an audio-derived stress in batch 2 alone — proper
nouns, loanwords, dialect, derivations — exactly what the suffix tier meets.
`run_placer_baselines.py` measured the cheap alternatives against the 520
benchmark tokens routed to suffix/compound; none reached the target.

**3.1 Data.** From the same builder, a second corpus mode
`--placer`: rows with `readings == 0`, confidence ≥ 0.95, `candidates` = every
vowel ordinal of the form, `gold` = the ranker's pick, sentence and span as
for `tok`. Negatives: none (every word has a stress). Split **by form** only
— an OOV word at inference is by definition one the placer has not seen, so
"unseen form" is the only condition that matters. Expect ~350k rows over
~150k forms across all batches.

**3.2 Model.** The same architecture and trainer (`run_train_token_resolver.py
--corpus output/ml/corpus/placer-v1 --max-candidates 12`): XLM-R over the
sentence, softmax over the span's vowel ordinals. This gives context (a
foreign surname after «пан» vs before a verb) for free and reuses the serving
path. Alternative if it disappoints: a character-level model on the word
alone (BiLSTM over characters, softmax over vowels), which is what the
literature uses and trains in minutes.

**3.3 Evaluate** on three sets: held-out forms of its own corpus; the 461
lang-uk tokens the live pipeline sends to `suffix` (score offline from
`live_review19.json` tokens, gold from the benchmark); the 5,235 Common Voice
suffix-tier tokens from the verify sweeps (`candidates == []`, audio gold at
≥ 0.99).

**3.4 Serve.** A second resolver in `stress-model` sharing
`token_resolver.py` (`PLACER_MODEL_DIR`; a target with `candidates == []`
means "all vowels"); in the Go API a `placer` status applied to `not_found`
tokens with ≥ 2 vowels, **after** the compound fallback and **before** the
suffix fallback, so suffix and positional default remain the last resorts.

**Done:** ≥ 85% on the 461 lang-uk suffix tokens (from 72%) and ≥ 85% on
the Common Voice suffix set (from 70%); words on lang-uk ≥ 96.4%. **Undo:**
unset `PLACER_MODEL_DIR`.

---

## Step 4 — a human-gold modern-text set (user's time; unblocks 0a, 2, 5)

**Why.** lang-uk is human gold on sentences built to force rare readings;
the modern-text test's gold comes from the audio ranker that also labelled
the training books. Every deployment rule so far is a compromise between two
tests that are each wrong in a known way. One hundred and fifty passages in
the user's own register fix that, and become the acceptance test for
everything that follows.

**4.1 Format.** One passage per line, plain text with U+0301 after every
stressed vowel of every multi-vowel word — the paragraph of 2026-09-18 is
the model. File `ml/data/gold_modern.txt`, with `#` comment lines allowed.

**4.2 Drafting (me).** For each of the 493 lang-uk heteronym forms and the
top 300 book forms with both readings present: pull 2–3 real sentences from
the book texts per reading (the rows give the paragraph), strip and re-mark
them with the ranker's reading as a **draft**, and group into passages of
3–6 sentences mixing readings. Target 150 passages, ~3,000 words, ~500
heteronym tokens. The draft marks are the thing to check, not to trust.

**4.3 Review (user).** Fix the marks. An hour per 50 passages.

**4.4 Scorer.** `ml/scripts/run_gold_modern.py`: strip marks → call the live
API → compare per multi-vowel word with lang-uk's rule (unmarked output is
wrong) → per-tier table like `run_tier_attribution.py`. Split the file in
two halves by line parity: **dev** for choosing rules and thresholds, **test**
reported once per model.

**Done:** the file exists with ≥ 100 passages and the scorer prints a table
for the live API. From then on, Steps 0a/2/5 choose on dev and report test.

---

## Step 5 — a learned combiner over the tiers (after Step 4; go/no-go)

**Why.** The cross-encoder knows *senses* (gloss similarity: атла́с/а́тлас,
за́мок/замо́к, лу́па/лупа́) and the classifier knows the *distribution of
contexts*; morphology knows *grammar* (вина́/ви́на, обра́зи/о́брази). They
fail on different words, and today the winner is whoever sits higher in a
fixed precedence.

**5.1 Features per ambiguous token** (all already produced by the pipeline;
the API must return the cross-encoder's `scores`/`margin` and the classifier's
probabilities per candidate — the first exists, the second is a one-line
addition to `/internal/v1/token`): classifier probability per candidate and
entropy; cross-encoder margin and status; morphology answer and whether a
rule fired; which candidate the lexicon lists first; number of candidates;
form's count in the classifier's training rows; grammatical vs sense
heteronym (from lang-uk's list).

**5.2 Training data.** Book sentences from held-out books (not in `tok-v4`'s
training), run through the live API with `on_ambiguity=preserve` to collect
features, ranker pick as the target at ≥ 0.99. ~50k tokens. A logistic
regression or a 200-tree GBM over candidates (pairwise: "is this candidate
the right one").

**5.3 Evaluate** on Step 4's dev, report test and lang-uk. **Go** if it beats
the best fixed rule by ≥ 1 point on gold dev without losing on lang-uk;
otherwise stop here and record why.

**5.4 Serve** as one more call in the Go API replacing the precedence block
for tokens with ≥ 2 candidates; the precedence stays as the fallback when
the combiner is unavailable.

---

## Order and calendar

| day | doing | user |
| --- | --- | --- |
| 1 | Step 0 (a, b, c); Step 1 script → `DICTIONARY_AUDIT_ROUND3.md` | — |
| 2 | Step 2.1–2.4 (trains overnight) | reviews Round 3 |
| 3 | Step 1 apply + redeploy + measure; Step 2.5 pick + deploy | — |
| 3–5 | Step 3 (placer) | — |
| 2–6 | Step 4.2 drafts, in batches of 50 | Step 4.3 reviews |
| 7 | Step 4.4 scorer; re-choose the rule on gold dev | — |
| 8+ | Step 5 go/no-go | — |

Every deployment: both machines, `restart stress-model api`, both
evaluations, a line in `RESULTS.md`, `docs/state.md` updated. No rule is
changed on the strength of one test.

## What is deliberately not in this plan

* Retraining the audio ranker on books (ten narrators' voices per hundred
  hours; it is the labeller, and Common Voice's 1,099 speakers are the
  better teacher of acoustics).
* Lowering `min_seen` or letting the classifier override morphology before
  Step 4 exists — measured at −3 points on lang-uk, invisible on the
  modern-text test.
* An ONNX export of the classifier (latency, not quality).
