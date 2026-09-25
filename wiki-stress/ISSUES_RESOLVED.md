# Issues found and resolved

Everything here was found by measuring against an external benchmark and then
reading the per-token error list. Each entry names the defect, what it cost,
and how it was confirmed — a claim without a number attached is a guess.

The pattern worth noticing: **five of the six largest wins were defects, not
missing capability.** The pipeline was not under-built; it was mis-wired, and
the internal evaluation could not see it because it exercised the same wrong
assumptions.

---

## A. Normalisation: signatures crossing the NFD/NFC boundary

One root cause, three separate manifestations, found weeks apart in the same
session. A stress signature is *generated* over NFD text and *applied* over the
caller's original text. Wherever those two disagree about what a vowel is, the
ordinals shift and the token is rejected.

### A1. Apostrophe treated as a segment boundary — Go

`applySignature` split segments on `isJoiner`, which includes apostrophes.
`stress_signature` splits on `[\s-]+` only. So `м'ясо` was read as two tokens:
signature `"0"` asked for the first vowel of segment 0, which is just `м` and
has none.

* **Cost**: 35,835 lexicon rows (1.41%) — `м'ясо`, `здоров'я`, `сім'я`,
  `п'ять`, `ім'я`, `об'єкт` — returned `invalid_candidate` and were served
  **unstressed**.
* **Confirmed**: `stress_lookup` stores `пʼятьо́х` as `"1"`, never `"1:0"`.
* **Aggravating factor**: an existing test asserted `{"п'ятьох", "1:0"}` — it
  encoded the reader's convention rather than the producer's, so it locked the
  bug in place instead of catching it.

### A2. Apostrophe rewritten in output — Python

`canonical_stressed_form` folds every apostrophe variant to U+02BC. Correct for
building a lookup key, wrong for constructing output: input `здоров'я`
(U+0027) came back as `здоровʼя` (U+02BC). The stress was right every time;
the token failed because a character had been changed.

* **Cost**: +11.43 unambiguous, +7.75 heteronym when fixed — the single
  largest jump of the session.
* **Wider point**: the pipeline was silently rewriting the user's text.

### A3. `й` not counted as a vowel ordinal — Go

In NFD, `й` is `и` + combining breve, and `и` is a vowel, so the generator
gives `райо́н` signature `"2"` (а, и, о). The reader counted over NFC, where
`й` is one consonant: ordinals а=0, о=1. Signature `"2"` did not fit.

* **Cost**: every word with `й` before the stress — `район` alone is 58 tokens,
  ~4.7% of the unambiguous metric. Also `району`, `війна`, `майдан`.
* **Effect of fix**: unambiguous 95.64% → 97.56%, word 88.71% → 90.36%.

---

### A4. Acute written inside a decomposed letter — trie defaults

`stressed_at` placed the accent at the vowel's character index in the NFD
string. `ї` is `і` plus a combining diaeresis and `й` is `и` plus a breve, so
the acute landed *between* the two: `киі́̈в` where `ки́їв` was meant.

* **Cost**: 345 rows of `trie-defaults-v1`, all of them forms containing `ї`
  or `й`. The map orders candidates, so a mangled entry both served a broken
  spelling and outranked the correct one.
* **Confirmed**: `SELECT count(*) ... WHERE stressed_form ~ U&'\0301[\0308\0306]'`
  returns 345 for dataset 11 and 0 for every other dataset.
* **Fix**: the acute now goes after the letter *and* every combining mark that
  belongs to it.

---

## B. Serving logic

### B1. Ambiguous forms outside the manifest returned unstressed

An ambiguous form the model could not take was given status `ambiguous` with
`OutputText` left as the bare word. Not the wrong sense — **no answer**.

* **Cost**: 200 of 1,000 gold rows scored 0.0000 by construction.
* **Fix**: `on_ambiguity: default|preserve`; the default now serves the
  lexicon's highest-confidence reading as `dictionary_default`.

### B2. Free variation rendered as two stress marks

`0|1` is the wordlist's "either stress is acceptable" notation for one token.
Both the Go reader and the Python morphology path emitted *both* accents,
producing `де́ржа́ва`, `ді́вчи́на`, `цу+кру+` — not words.

* **Found twice**: fixed in the evaluator, then found still live in the Go API
  and again in the morphology output path, which builds its output separately.

### B3. The morphology tier was never reachable

`MODEL_TIMEOUT` defaults to 2s — right for a batched cross-encoder, far too
short for a parser's first call, which loads models (~30s for Stanza). Every
morphology call timed out, fell back to a dictionary default, and the warning
was returned in the response body rather than logged.

* **Cost**: the entire tier, worth +8.6 heteronym points, silently dead in
  production while the API returned 200.
* **Fix**: separate `MORPHOLOGY_TIMEOUT`, plus warm-at-startup so no request
  pays construction.

### B4. One parser serialised the whole API

Stanza holds a single pipeline and is not thread-safe. Under eight concurrent
callers requests queued past the timeout and degraded to dictionary defaults.

* **Cost**: word accuracy 87.89% → **72.16%**, unambiguous 95.64% → 78.46%
  under load, with no error surfaced.
* **Fix**: a warmed pool (`MORPHOLOGY_WORKERS`), viable only because spaCy is
  15 MB rather than ~500 MB.

### B5. Proper nouns outranked common words

The merged wordlist injects toponyms and given names at confidence 1.00, so
`Розді́л` (a village) beat `ро́зділ` (a chapter) and `Ме́ні` displaced `мені́`.

* **Fix**: readings whose every stored spelling is capitalised now sort last
  for a lowercase token, whatever their confidence.

### B6. The API read only the active dataset

Corrections, recovered readings, the audit and the heteronym dictionary all
live in their own datasets. The evaluation harness read them; the API did not.

* **Cost**: 17.7 heteronym points between the harness and the product — the
  gap that revealed every number reported before it described the evaluator,
  not the deployed system.
* **Fix**: `SUPPLEMENTARY_DATASETS`, plus `etl/merge_datasets.py` to fold them
  into one publishable dataset.

### B7. Monosyllables were never stressed

`vowelCount == 1` was treated as `not_required`. A single-vowel word is
unambiguous, not unstressable, and a TTS front-end still wants the mark.

---

### B8. Morphology was never offered a form the model covered

The tier was handed only tokens still marked `model_ineligible` — the forms
outside the serving manifest. Two populations therefore never reached a
parser: everything inside model coverage, and everything the model declined
with a low margin. Neither is a form a tagger has nothing to say about.

* **Cost**: 309 grammatical tokens on lang-uk went to the model (238) or to a
  dictionary coin flip (71) instead of to the tagger, which is right on 83.6%
  of the ones it reaches.
* **Confirmed**: in the offline harness, lifting the restriction moved
  morphology from 250 resolved tokens to 594 and the grammatical class from
  68.62% to 84.57%, net +98 (+125/−27), p < 0.00001.
* **Fix, and the correction to it**: the harness result did *not* transfer.
  Against the deployment, letting morphology override the model scored −0.54
  heteronym points. What shipped is narrower — morphology fills the model's
  abstentions and a confident model answer stands — worth +0.11 heteronym and
  +0.35 macro-F1. The harness had run a different model over a reduced
  manifest and agreed with itself; `run_live_bench.py` exists because of this.

---

## C. Tag vocabularies

### C1. `upos=PRON` vs `upos=DET`

The trie writes `PRON` for determiners like `цьому`, `всього`, `усі`; UD
taggers write `DET`. Case, gender and number agreed exactly; the match was
refused on the part-of-speech label alone.

* **Cost**: 96 of 644 ambiguous forms declined for this reason.
* **Effect of fix**: morphology resolution 75.0% → **85.4%**, heteronym
  accuracy **+5.78** — eight lines of code, the third-largest win of the
  session.

---

## D. Lexicon data

### D1. 8,702 stresses wrong where the source trie was right

`data/sterss-dict/run_stress.py` ran the stressifier over a bare word list with
Stanza disambiguation, so every word was parsed in isolation as a nominative
singular and Stanza chose among readings on features that do not exist without
a sentence.

* **Scale**: of 1,848,453 forms where both the lexicon and the trie hold one
  reading, 8,703 disagree (0.5% by type) — but they concentrate in common
  words: `украї́ни` read as `укра́їни`, `дія́льність` as `ді́яльність`, `воно́`
  as `во́но`, `о́бластях` as `областя́х`.
* **Caution recorded**: the first sample suggested "systematic corruption"
  because it was drawn from the failing words. The real rate is 99.5% faithful.

### D2. 6,028 forms missing a second reading

The merge kept whichever stress its highest-confidence source recorded and
discarded the other, so genuine homographs looked unambiguous and could never
reach the model. On a sense-balanced set they score exactly 0.5000.

### D3. A human override was silently undone

The trie recovery re-introduced `рече́ння`, which a person had explicitly
removed. Recovery now reads `stress_overrides.json` and skips those forms.

### D4. Provenance falsified in two places

* `merge_corpus.py` stamped every added row `corpus: "generated", licence:
  "synthetic"` — including 32,629 sentences mined from real text under
  different licences.
* `run_mine_corpus.py` hardcoded `corpus: "malyuk"`, so 30,462 subtitle-mined
  sentences were tagged as Malyuk.

Both matter for any corpus export that claims a licence.

---

### D5. Generated datasets outranking the curated wordlist

The lexicon is layered by confidence: the merged wordlist at 0.95, trie-derived
sets at 0.99. That ordering is backwards for any form the wordlist gets right
and a generated set gets wrong, and nothing in the pipeline could express "this
reading was checked".

* **Cost**: `Ки́їв` and `наро́ду` were both correct at 0.95 and both overridden
  — Київ by `trie-corrections-v2` and `trie-defaults-v1`, народу by
  `trie-defaults-v1` and then by the model, which decided it at margin 6.56
  because the form sat in the serving manifest.
* **Confirmed**: 59 tokens on lang-uk across `його`, `Київ`, `народу`,
  `народові`; correcting them fixed 52 and broke 2, both against malformed
  gold (`йог+о`, a lowercase `киї+в`).
* **Fix**: a `manual-corrections-v1` dataset at confidence 1.00; `--exclude` on
  the trie-default generator so a generated default never contradicts a
  reviewed reading; and `--manifest` to drop corrected forms from model
  coverage, because a reviewed reading is not a decision for the model to make.
  `inventory_hash` names the sense inventory rather than the form list, so the
  API's hash check survives the prune.

### D6. A stress the lexicon cannot hold

`мене́`, `себе́`, `тебе́` have one reading each in the source dictionary, and
always will: a stress that depends on the preceding word has no home in a table
keyed by one word. So the pipeline served the citation form everywhere,
including after a preposition, where Ukrainian moves the stress to the first
syllable.

* **Cost**: 26 of 40 governed pronouns on lang-uk.
* **Confirmed as a rule rather than an annotator's preference**: categorical in
  both directions and across six pronouns — after a preposition 39/40 take the
  first syllable, with no preposition 0/11 do.
* **Fix**: `applyPrepositionalShift`, a pass over the token list keyed on the
  preceding word. 19 tokens fixed, 1 "broken" against the one gold sentence
  that marks the same phenomenon on the preposition instead (`Біжа́ть до́
  мене́`).

---

## E. Database

### E1. No index on any foreign-key column

PostgreSQL indexes the referenced side of a foreign key automatically and the
referencing side never. Every child table was indexed on `(dataset_id, ...)`,
so a cascade lookup by `lexeme_id` alone could not use a prefix of any of them
and fell back to a sequential scan of `stress_lookup` (2.9M rows) **per deleted
row**.

* **Cost**: rewriting a 6,183-row dataset ran for over an hour before being
  killed. Dataset rewrites are the normal maintenance path, not an edge case.
* **Fix**: `db/migrations/008`, six indexes, ~8 seconds to build.

---

## F. Tooling and measurement

These cost time rather than accuracy, but each produced a wrong number that was
acted on.

* **`--limit 0` was falsy** in `run_generate_balanced.py` and
  `run_gloss_uncovered.py`, so a dry run started a *full* run. Hit twice.
* **A shared `limits` dict raced** in the retry path: threads whose sibling had
  already applied a parameter fix re-raised instead of retrying, failing a
  whole shard.
* **Weak retry in `generate.call_model`** — 3 attempts, linear sleep —
  abandoned 12% of calls. An abandoned call is not a slower run but a sense
  left at zero rows.
* **Truncated downloads accepted as successes**: 15 of 32 corpus shards had
  unreadable parquet footers because the fetcher never checked
  `Content-Length`.
* **The miner aborted on the first corrupt shard** instead of skipping it.
* **Three wrong ceiling estimates** — 95.06%, 72.79%, 98.30% before settling at
  **99.76%** — each from a different defect in the *measuring* script:
  signature strings compared literally, punctuation left on gold tokens, and
  apostrophes rewritten (A2 again, in the oracle).
* **`--add` appended to its argparse default**, so `generated_balanced.json`
  was processed twice.

---

### F3. A cached index silently lost the column that made sorting work

`load_index` rebuilt the dataset browser's index from cache field by field and
never read `row_forms` back, so a cached array index came up with no forms at
all: every row's corpus frequency was `None`, sorting was a no-op, and any
frequency band matched nothing. Object-shaped files skip the cache entirely,
which is why the manifests looked right and the corpora — the files worth
sorting — did not.

* **Fix**: read the field back, and treat a cache without it as stale rather
  than serving a crippled index.

---

## G. Wrong conclusions corrected

Recorded because each was stated confidently and acted on before being
disproved.

| claim | reality |
| --- | --- |
| "0.9949 certified" | silver precision on a filtered subset at 22.8% coverage; end-to-end 0.6530 |
| `label` and `verify` share a quota | separate deployments, separate budgets; the slowdown was a gpt-5.4 escalation bottleneck |
| the residual is coverage | only 19 of 4,694 unambiguous types were missing |
| the residual is archaic orthography | archaic forms are a small minority |
| the residual is proper nouns | real, but worth +0.06 |
| routing grammatical heteronyms to the model will help | it *lowered* heteronym accuracy by 1.5 points |
| a bigger tagger will help | `uk_core_news_lg` (231 MB) scored −1.4 against `sm` (15 MB) |
| the ETL systematically corrupted the lexicon | 99.5% faithful; the sample was drawn from the failures |
| the unambiguous-word deficit is ETL loss | generated datasets outranking the wordlist, plus 345 mangled rows; fixing both took unambiguous accuracy past the baseline |
| reordering the tiers by ambiguity class will help | net −10 (p=0.237); grammatical was exactly 0 because morphology was already consulted first — the tier that looked mis-ordered was simply switched off |
| morphology should override the model wherever both answer | +8.29 heteronym points in the harness, −0.54 against the deployment |
| the source dictionary's silence disproves the prepositional shift | a per-word table cannot express a stress that depends on the previous word; its silence is a structural limit, not disagreement |

The habit behind most of these: **diagnosing from an aggregate instead of
reading the per-token error list**, which was available the whole time and
settled each question in one pass.
