# Plan: parity with RUAccent on Ukrainian

Written 2026-09-20. RUAccent (Petrov, COLING 2025) is the reference: 96.4%
on the 200 most frequent Russian homographs, scored on two million sentences
labelled by its own audio classifier at confidence > 0.99. Measured the same
way — the 200 most frequent ambiguous forms of Common Voice, audio gold at
≥ 0.99, the live API — this pipeline is at **94.6%** (13,201 tokens). The
gap is 1.8 points, and 1.7 of them sit in one place: 558 tokens the
cross-encoder still decides, at 61.1%. Every other tier on that slice is
95–97%.

This plan is four steps, in the order they pay. Steps 1 and 2 are about the
rule and the data source, not about models; they are what RUAccent did and
what this pipeline has not. Steps 3 and 4 borrow the two ideas from the paper
that this project has the means for and has never tried.

Baselines (2026-09-20, `live_review20`, both machines):

| test | value |
| --- | ---: |
| top-200 forms, audio gold ≥ 0.99 | 94.61% |
| all 19,089 ambiguous tokens, same gold, live rule | 92.0% |
| lang-uk heteronyms (human gold, adversarial) | 84.95% |
| lang-uk words | 95.93% |

The rule for changing anything stays: the top-200 number and lang-uk are
both reported, and a change ships only if the first moves up and the second
does not move down by more than 0.3.

---

## Step 1 — take the cross-encoder's tokens on the frequent forms (a day)

**What.** On the top-200 forms, the cross-encoder decides 558 tokens at 61%;
the classifier reads those same forms at 96%. The live rule already lets the
classifier take `stressed` tokens — but only for forms with 100+ training
rows, and the cross-encoder's picks that survive are on forms *between* the
lexicon's confidence and the classifier's coverage.

**How.**
1. `run_eval_tok_modern.py --tiers` on `tok-v3d` at `min_seen` 10, 30, 50,
   100: which forms are in the 558, how many training rows each has, and
   what the classifier scores on exactly those tokens.
2. If the classifier is ≥ 90% on the 558 at some `min_seen` below 100 — it
   is 92.6% on `stressed seen 30+` today — lower `min_seen` **for the
   `stressed` tier only**: `coverage.json` gets a per-tier threshold
   (`min_seen: {"dictionary_default": 100, "stressed": 30}`), the resolver
   reads it, the Go side passes the tier in the target (one field).
3. lang-uk at the same setting. At `min_seen` 30 for both tiers it cost
   1.2 points; the stressed-only variant is untested.

**Done:** top-200 ≥ 96.0% with lang-uk ≥ 84.6%. **Undo:** the threshold.

---

## Step 2 — mine audio without books (two days of code, then unbounded)

**What.** RUAccent's 108,000 hours are podcasts, YouTube and radio: audio
with no matching text at all. Its pipeline needs none — WhisperX gives the
words and the times, the audio classifier gives the reading. This project's
v3 miner is one step from that: it uses the book only for the spelling and
the lexicon's reading. Drop the book and the supply of audio is every
Ukrainian podcast and audiobook the user is entitled to use, not the 372
books with a matching epub.

**How.** `run_mine_audio_v4.py`, from `run_mine_book_v3.py` with the book
removed:
1. Whisper large-v3-turbo on ten-minute windows, chunks ≤ 60 s on segment
   boundaries, forced alignment on the heard text, SSL frames, ranker — as
   today.
2. The **ASR transcript is the sentence**: for each kept word, the row keeps
   the chunk's text and the word's span in it. ASR errors in the *context*
   are tolerable for a classifier that reads context; ASR errors in the
   *target word* are not — a word is kept only if `readings(form) ≥ 1`
   (the lexicon knows it) and the aligner placed every vowel, which drops
   misheard words the way the book match used to.
3. **A spelling check that the book used to give**: the word's neighbours
   in the transcript must also be lexicon words at ≥ 80% over the chunk,
   or the chunk is dropped as a misrecognised stretch (music, another
   language, mumbling).
4. Per-window reporting unchanged: the longest-vowel lift and the ranker's
   agreement with the lexicon on single-reading words print every window,
   and a source whose lift is under +8 pp is stopped, as in the pilot rule.
5. The corpus builder reads these rows as it reads `rows3` — the sentence
   is already in the row, no locator needed.

**Pilot before scale**, as always: two windows from ten sources, the same
table as the book pilot, then a full run only on what passed. First sources:
the 33 books the pilot dropped for a mismatched edition (their audio is fine)
and the 20 hours of Common Voice not in the verify sweeps — both have a
truth to check against.

**Done:** the pilot table shows lift ≥ +15 and ranker ≥ 95% on ≥ 8 of 10
sources; a first 500 hours mined; `tok-v5` trained on books + audio-only
rows evaluated on both tests. **Risk:** ASR text as context shifts the
classifier's input distribution (no punctuation, no capitals). Mitigation:
Whisper's segments carry punctuation and case; keep them.

---

## Step 3 — synthetic audio for the ranker (three days)

**What.** RUAccent trained its audio classifier on speech synthesised with
controlled stress (Silero, vosk-tts), then aligned it to text contrastively.
This project's own TTS stack accepts stress marks. The ranker — the labeller
behind every book row and every modern-text gold — was trained on 118 hours
of Common Voice with lexicon labels; its precision on the homographs that
actually train the classifier has never been measurable. Synthesised speech
with a known stress makes it measurable and makes the training set as large
as wanted.

**How.**
1. Sentences: 20,000 from the book texts containing a heteronym form, half
   per reading, marked with each reading in turn (both readings of each
   sentence — the model must hear the difference, not the word).
2. Synthesis through the `uk-tts-frontend` stack with the mark forced
   (`already_stressed` is honoured), two or three voices.
3. Mine the synthesised audio with the v3 miner (text is known exactly, so
   the book path applies) → rows with a **known** gold.
4. Measure the ranker on them: precision on homographs by reading, by
   voice. This is the number `docs/open-issues.md` §11 says is missing.
5. Fine-tune the ranker on Common Voice + synthetic rows, hold out voices
   and sentences, re-measure on real book audio (agreement with the lexicon
   on single-reading words, the audit's number) — synthetic speech must not
   pull it away from real speech.

**Done:** ranker precision on synthetic homographs reported by reading; a
fine-tuned ranker only replaces the current one if its agreement with the
lexicon on real audio does not drop and its synthetic-homograph precision
rises. **Risk:** the TTS itself mispronounces a forced stress; check 50 by
ear before trusting the set.

---

## Step 4 — a stress-aware text encoder (a week, after Steps 1–2)

**What.** RUAccent pretrains its text encoder on 200 GB of stress-marked
text (AMLM + NSP, Canine-style). No such corpus exists for Ukrainian; the
paper's third version made one by running the previous model over raw text.
This project has 33 GB of Malyuk and a pipeline at 96% on words.

**How.**
1. Run the live pipeline over Malyuk in batches (the API does ~2,000
   sentences a minute per machine; two machines, a week for 20 GB), keeping
   only sentences where every multi-vowel word was decided by the lexicon,
   morphology or the classifier at ≥ 0.99 — no defaults, no suffix guesses.
2. Continue pretraining `xlm-roberta-base-uk` on the marked text with masked
   LM where the mask includes the accent mark, so the model learns where
   stress goes as part of the language.
3. Retrain the classifier and the placer from that encoder; compare on both
   tests.

**Done:** a classifier from the stress-aware encoder beats `tok-v3d` on the
top-200 test and on held-out forms. **Go/no-go:** decided after Step 2's
`tok-v5` — if audio-only mining alone closes the gap, this step is not
worth the week.

---

## What parity does not mean

RUAccent reports no human-gold adversarial number; lang-uk is stricter than
anything in the paper, and 84.95% there is not a gap to close by imitation.
The two numbers to carry side by side are the top-200 audio-gold accuracy
(their yardstick, ours at 94.6%) and lang-uk (ours alone). Step 4 of
`PLAN_QUALITY.md` — a human-gold modern-text set — remains the one thing
that lets both be trusted, and it still waits on the reviewer.

## Order

1 → 2 (pilot) → 3, with 2's full run and 3 overlapping; 4 only if 2 does not
close the gap. Each step: both tests, a line in `RESULTS.md`, `docs/state.md`
updated, both machines.
