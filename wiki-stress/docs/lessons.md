# Lessons: how the dataset and the model were built, and every way it went wrong

Written 2026-09-14, at the end of the audiobook iteration. This is the path
from "train a model that recognises stress from audio" to `tok-v2`, told by
its mistakes, so the next iteration does not pay for them again. Each entry
is what happened, what it cost, and the rule that came out of it. The rules
are collected at the end.

The path, in one paragraph: an audio stress ranker was trained on Common
Voice (97.0% on held-out speakers); it was used to verify the text pipeline
and to mine a token-classifier corpus from Common Voice (`tok-v1`), which
failed to generalise; audiobooks were brought in to fix that; the first
audiobook mining run was void; the second was piloted, mined three times over,
labelled, and trained into `tok-v2`, which generalises inside its corpus
(+18 pp over the trivial rule on unseen forms), helps on modern text (+15 pp on
the slice it would replace), and is neutral-to-slightly-negative on lang-uk.

---

## 1. Data: getting text and audio that match

### 1.1 The corpus was three times bigger than the plan said
`PLAN_BOOKS.md` said 152 hours. The plan was written for the first ten books;
seventeen more arrived and the number was never updated. Every time estimate
in the first day was made against 152 h and was wrong by ×2.4 before any
other error.
**Rule:** the driver prints the total it is about to process, in hours and
windows, before it starts. Never estimate from the plan; estimate from the
files.

### 1.2 Three books could never work, and mining them would have cost 44 hours
Макіавеллі had audio and no matching text; Гюго and Норвуд had epubs of a
different edition or language (Норвуд's was Russian — 5,163 out-of-vocabulary
words against 1,204 labelled before the language gate caught it). All three
produced **zero** rows in a two-window pilot.
**Rule:** a language gate on extraction (Russian-only letters per word > 0.05
drops the book), and a two-window pilot on every source before mining any
source. Zero rows in the pilot means drop, not investigate.

### 1.3 `index_split_002.html` matched a substring filter and a whole book vanished silently
Гарпер Лі's epub produced zero paragraphs because a furniture filter matched
`index` inside the file name. Nothing errored.
**Rule:** extraction reports words kept and dropped per book, and a book with
fewer than a few thousand words stops the run.

### 1.4 `soundfile` cannot open `.m4a`, and decodes `.mp3` from the start
Six books were m4a; the first mp3 windows near the end of a book cost 19
minutes each.
**Rule:** `ffmpeg -nostdin -ss <start> -t <len> -i <file>` for every window.
Seeking was verified sample-accurate (two decodes of the same second from
different seek points: correlation 1.0000, lag 0).

### 1.5 `ffmpeg` reads stdin and ate the driver's own input
A `while read` loop over books jumped from book 1 to book 6 while the counter
still said 3, because ffmpeg consumed the loop's stdin.
**Rule:** `-nostdin` on ffmpeg, `< /dev/null` on every step a shell loop
launches. Both, not either.

---

## 2. Alignment and mining: the run that was void

This is the expensive one. The first `run_mine_book.py` mined 2.2M rows over
16 hours, the labeller produced 348,835 labels over another day, a corpus was
assembled, and only then was the ranker asked about words whose reading the
lexicon already knew. It scored **39.1%**. Chance for that model is 37.3%.

### 2.1 What was wrong, in three layers
* **Anchors drifted by hundreds of words.** Ten-minute windows were placed in
  the book by rare-word voting on Whisper output. At 1200 s into Костенко the
  anchor said book word 3250; the narrator was at word 2838. The `overlap`
  figure the anchoring reported (0.68–0.87) did not reveal it.
* **Proportional slicing.** Each window's word range was cut into ten equal
  parts for ten 60-second slices, assuming uniform speech rate.
* **Forced alignment cannot refuse.** Handed a word list, WhisperX places every
  word somewhere in the audio. The 35% padding on each side, added so the true
  words were certainly inside, guaranteed that most of what it was asked to
  place was not there. Removing the padding did not help; measured, both score
  at chance, because the underlying range was wrong.

Vowel boundaries were noise, so the ranker pooled embeddings over the wrong
audio, so its labels were noise, so the corpus built from them was noise. And
that corpus looked *good*: 5,821 ambiguous forms, 4,582 (79%) with both
readings present. Random labels naturally show both readings. **A number that
is exactly what you hoped for is the first one to check.**

### 2.2 How it was caught, and why that was too late
Before setting a confidence gate, the ranker was scored on lexicon-named
words. It was flat: 0.999 was no better than 0.5. Four suspects were tested in
order — the labeller's slice clock (correct: first vowel +20…40 ms after the
word start), ffmpeg seeking (correct), empty prosody features (irrelevant: the
model runs in `position` mode, 97.8% either way), and finally the boundaries
themselves, with no model at all.

**The test that decided it needs no GPU, no model and no labels beyond the
lexicon's:** a stressed Ukrainian vowel is longer, so the longest vowel guesses
the stress 63.4% of the time on Common Voice against 37.6% chance. On every
book from the void run: 36–39% against 35–38%. It could have been run on the
first window of the first book. It was run last, after everything had been
built, as "calibration".

**Rule:** `run_audit_books.py` — the longest-vowel baseline — runs on every
mined file, sits *between* mining and labelling in the driver
(`run_books_end_to_end.sh`), and a book below **+8 pp lift over chance** does
not proceed. The void run managed +1 to +3; every good book makes +15 to +30.
The floor is a lift, not an absolute: chance moves with how many vowels the
words have, and an absolute 50% nearly dropped «Салимове лігво» (47%, lift
+10, ranker 95.9%).

### 2.3 What replaced it
`run_mine_book_asr.py` turns the dependency around: Whisper (large-v3-turbo)
transcribes the ten-minute window once; its own segment boundaries cut that
into chunks of ≤60 s, so a chunk's text is exactly its audio and no cut falls
inside a word; each chunk is force-aligned against what was heard in it; then
each aligned word is matched into the book by `difflib.SequenceMatcher` over a
window placed by per-chunk rare-word voting. The book supplies spelling,
paragraph, and the lexicon's reading. It never supplies the timing.

Three things had to be fixed inside that before it was fast and right:
* **Greedy first-match walked the cursor forward on false hits** of frequent
  words («не» occurs 2,130 times in one book); the next chunk then voted
  itself behind (median drift −49 words, range to −418). Sequence alignment
  fixed it (drift −5; word match 62% → 91%).
* **Chunk length was unbounded**, and one long Whisper segment stalled
  alignment for fifteen minutes with no error. Then the cap was applied after
  overshoot, so chunks reached 90 s, and 90 s costs five times what 60 s does
  on this card (5 min a window against 0.7). Close the chunk *before* adding
  the segment that would exceed the target.
* **Partial alignments were accepted at quality 0.8.** A word missing one
  vowel has every ordinal after the gap shifted; its label points at the wrong
  vowel. 1.7% of rows, each one a lie. `--min-quality 1.0`.

Verified after the rewrite, per book, on 27 books × 2 windows from the middle
of each (not the opening, which is publisher boilerplate): longest vowel 57.2%
over 22,787 words; ranker 96.34% [96.1, 96.6] over 22,647; 98.4–99.7% at the
0.99 gate on every book. Then, and only then, the full run.

### 2.4 The earlier miner bug that cost a re-mine
`.ravel()[:0]` stored an empty embedding on every row of the first-ever book
run — 224k homographs unlabellable, 16 hours lost. The fix stored vowel spans
instead of embeddings. Then the spans were relative to the slice while
`start_s` was absolute, which the labeller had to reconstruct. The ASR-first
miner stores absolute spans and the paragraph/word index, so nothing
downstream reconstructs anything.
**Rule:** open the first output file and read one row before letting a run
continue. Store absolute coordinates and the exact text position; never make
the consumer replay the producer's arithmetic.

---

## 3. Speed: the 8 GB card

Three models share the GPU during mining — Whisper, the wav2vec2 aligner, the
XLS-R encoder — and together they leave almost nothing free on 8 GB.

### 3.1 Every stage slowed by the same factor after the first window
Window 1: 32 s. Window 2: 6.7 min. Window 3: 12.7 min — with ASR, alignment
and pooling each growing by the same amount. No throttling (66 °C, 1725 MHz,
no flags). The allocator was thrashing. `torch.cuda.empty_cache()` after every
chunk plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and the ASR
batch from 16 to 8: windows went back to 0.5 min each, memory from 7.9 GB to
6.0 GB. This had almost certainly been slowing every earlier run too, which is
where the 25–35 hour estimates came from.
**Rule:** print per-stage seconds on every window, not at the end of a book.
The slowdown was invisible for an hour because the timing only printed at
exit.

### 3.2 Estimates from one window were wrong four times
6.5 h (from the first fast window) → 17–24 h → 25–35 h → 8 h per pass once the
allocator was fixed. Each was stated to the user as if it were a measurement.
**Rule:** estimate from at least one complete book, and say which book.

### 3.3 `pkill -f` kills the shell that runs it
Repeatedly: the pattern matched the invoking shell (exit 144). GPU at 0% is
the giveaway.
**Rule:** `ps -ef | awk '/pattern/ && !/awk/ {print $2}'`, then kill by PID.

---

## 4. Corpus building

### 4.1 `tok-v1` had nothing to learn context from
8,295 ambiguous rows over 2,047 forms, 53% of forms seen once, **both
readings present in 187 forms (9%)**. Trained anyway; it scored 66.3% on
unseen forms against 66.4% for `candidates[0]`. That number — the trivial rule
on the held-out-form condition — is the one that says whether a classifier
learned context or memorised forms, and it was not in the trainer's report
until after `tok-v2`.
**Rule:** `run_train_token_resolver.py` prints `candidates[0]` beside every
condition. Before training, count forms with both readings present; below a
few hundred, do not train.

### 4.2 The per-form cap threw away the minority reading
A cap of 60 rows per form fills on the majority reading of a common form. The
minority reading is the only part that teaches context.
**Rule:** cap per *(form, reading)*, not per form (`--per-reading-cap 80`).

### 4.3 The share of forms with both readings depends on depth, not breadth
After one stride-3 pass: 10% of forms had both readings — the same as
`tok-v1`, and it looked like the premise had failed. Broken down by how often a
form was seen: 0% at once, 11% at 2–3, 18% at 4–9, 32% at 10–24, 57% at
25–99, 67% at 100+. Half the forms had been seen once. Two more passes: 491 →
760 → 884 forms with both readings (at gate 0.95).
**Rule:** report the both-readings share *by frequency bucket*, and mine depth
(more of each book) before breadth (more books).

### 4.4 Half of "ambiguous" forms are not ambiguous in prose
Among forms seen 10+ times, the median minority-reading share is 0%; p75 is
8.5%. The lexicon lists readings prose does not use. These are fine as easy
rows; they are not evidence of anything.

### 4.5 The gate was set on the wrong material's calibration
0.99 was 96.8% right on Common Voice. On book audio (measured on 6,000
lexicon-named words after the rewrite) 0.95 is 98.2% and 0.99 is 98.8%, and
0.95 keeps almost twice the forms with both readings.
**Rule:** calibrate the gate on the material being gated, before building the
corpus, with `run_label_books2.py --calibrate`.

### 4.6 Sentence context must be the real text
The anchor text is lowercase and unpunctuated. The classifier will read
capitals and punctuation at inference. The miner records paragraph and word
index; the builder takes the sentence from the extracted text and checks that
the span it computes spells the form. 276,469 rows placed, zero lost.

---

## 5. Training and the overnight chain

### 5.1 `%` in an argparse help string crashed the build, and training ran on the stale corpus
"98.2% right" in a help string is a format specifier to argparse. The overnight
chain's build step died, the next step trained on the corpus from the day
before, and the log read as success until the row counts were compared.
**Rule:** the chain checks the corpus manifest's timestamp before training,
and the builder is run once by hand before it is put in a chain. `%%` in help
strings.

### 5.2 The trainer overwrote the encoder's `config.json`
`save_pretrained` wrote it; the trainer then wrote its own `{"base", …}` to
the same name. The model could not be loaded. Now `resolver.json`.

### 5.3 A stale `profile.py` in the scratchpad shadowed the standard library
Every probe script run from that directory failed inside `transformers` with
a circular-import error, for a day, for no reason anyone could see.
**Rule:** never name a scratch file after a stdlib module; when imports fail
mysteriously, `python -c "import profile; print(profile.__file__)"`.

### 5.4 The dev curve was still rising
87.8 → 91.3 → 92.2 → 92.5 over four epochs. Nobody tried six.

---

## 6. Evaluation

### 6.1 NFD leaked into the benchmark strings
`plus_form` replaced the acute with `+` in NFD and returned NFD; й and ї stayed
decomposed; the harness saw different words. Word accuracy "fell" 26 points.
**Rule:** every string handed to a scorer is NFC. Say so at the boundary.

### 6.2 The comparison was against a stale review
The first lang-uk comparison used `live_review8.json` (heteronyms 83.10%); the
deployed pipeline was `live_review13.json` (85.06%) after 85 corrections. The
slice where the classifier had won by 18 points shrank to −0.5 against the
current pipeline, because the corrections had already fixed much of it.
**Rule:** `docs/state.md` names the current review file; evaluate against
that one.

### 6.3 The evaluator does not count `й` as a vowel
This project's signatures do. Tier attribution disagreed with the official
harness by 15 points, twice, until the evaluator's own vowel class
(`[АаЕеЄєИиІіЇїОоУуЮюЯя]`) was copied. `not_found` looked like 15% of tokens;
it is eight.

### 6.4 The two tests disagree, and both are right about something
On modern text (Common Voice, gold from audio) the classifier beats every tier
on forms the books contain: +15 on `dictionary_default`, +14 on `stressed`.
On lang-uk (human gold, sentences built to force the rare reading) it is
neutral on the same slice and loses everywhere else. The books teach the
common reading; the benchmark asks for the rare one. And the modern-text gold
shares its labeller with the training set, so it can flatter.
**Rule:** the deployment rule is whatever is non-negative on the *independent*
test and positive on the target: the classifier replaces `dictionary_default`
only, only on forms seen 10+ times in its training rows. Everything else
stays with the pipeline until a human has read the disagreements.

### 6.5 On forms it has not seen, the classifier is a coin
64% where `dictionary_default` is 89%. Coverage gating is not an optimisation;
it is the difference between a gain and a regression.

---

## 7. Things asserted that were false

* "RUAccent is text-only." It is not; the COLING 2025 paper describes 108,000
  hours of audio, WhisperX, wav2vec, Common Voice. The user was right.
* "The prepositional-clitic rule is a dictionary defect." It is right (36/37 on
  gold); the audio model was blind to it because 1,936 clitic rows had been
  dropped from its training. A plan section and an audit entry were built on
  the false reading before it was retracted.
* "No change" — said once without checking, after a malformed tool call; 44k
  rows had been added.
* "79% of forms have both readings." An artefact of random labels.
* Three hypotheses that words-per-second predicts anchoring failure. All
  refuted by counter-examples (1.4 → 100%, 1.5 → 100%, 3.1 → 100%). Only
  text edition predicts it.
* Four time estimates.

**Rule:** a claim about data is preceded by the command that measured it.

---

## 8. Process: what the user had to say to get a pilot

After the void run: *"so much time was waste! … I need 100% to be sure that
this time it will provide necessary result! Test on small piece first!"*

The full run had already been launched a second time on the strength of a
single-book test. It was stopped, a 27-book × 2-window pilot ran in 105
minutes, three books were dropped before costing 44 hours, and the run that
followed was the one that worked.

**Rule:** for anything over ~30 minutes of compute, a pilot on 2–3 units per
source, a per-unit table with a model-free sanity metric and the real metric,
shown to the user, and only then the full job on the units that passed. The
check lives between stages in the driver so it cannot be skipped.

---

## The rules, collected

**Before any long run**
1. Pilot 2–3 units per source; per-unit table; show it; run only what passed.
2. Print the total to be processed from the files, not the plan.
3. Open the first output file and read one row.
4. Run the builder and trainer once by hand before chaining them.

**Mining**
5. Text for alignment comes from the audio (ASR); the book only confirms.
6. Chunks ≤60 s, closed before overshoot; never align a segment over the cap.
7. `--min-quality 1.0`: a word with a missing vowel is not a measurement.
8. Sequence alignment into the book, never greedy first-match.
9. `run_audit_books.py` between mining and labelling; lift ≥ +8 pp or stop.
10. Absolute coordinates and exact text positions in every row.

**GPU (8 GB)**
11. `empty_cache()` per chunk, `expandable_segments:True`, ASR batch 8.
12. Per-stage seconds on every window.
13. Kill by PID from `ps | awk`, never `pkill -f`.

**Corpus**
14. Calibrate the gate on the material being gated.
15. Cap per (form, reading).
16. Report both-readings share by frequency bucket; mine depth before breadth.
17. Sentence text from the source, NFC, span checked against the form.

**Training and evaluation**
18. `candidates[0]` beside every condition; unseen forms is the condition that matters.
19. Evaluate against the review named in `docs/state.md`.
20. NFC at every scorer boundary; the evaluator's own vowel class.
21. Gate the classifier by coverage; deploy the rule that is non-negative on the independent test.
22. `%%` in argparse help; `resolver.json`, never `config.json`; no stdlib names in the scratchpad.

**Serving**
23. After any manifest or coverage change, restart every container that read it — compose does not recreate a container for a changed bind-mounted file.
24. A classifier gets a window of context around its span, never a sentence truncated from the start.

**Speaking**
25. A claim about data follows the command that measured it.
26. Estimates come from a whole unit, and name it.
