# Audiobooks and their texts as a stress corpus

Written 2026-09-08, after `tok-v1` failed for a reason this material fixes.

## Why this corpus, specifically

The token classifier trained on Common Voice memorised forms instead of
reading context, and the corpus said why: 8,295 ambiguous rows over 2,047
forms, 53% of forms appearing once, and **both readings of a form present in
only 187 cases out of 2,047 — nine percent**. Nothing there to learn context
from.

Read-aloud sentences cannot fix that. Each Common Voice clip is one sentence
by one speaker, and a homograph appears in whatever single context that
sentence gives it. Continuous prose is the opposite: `за́мок` and `замо́к`,
`ко́леса` and `коле́са` recur across a novel in the constructions that
distinguish them, and ten novels give ten independent samples of each.

| | Common Voice | these books |
| --- | ---: | ---: |
| audio | 118 h | **152 h** |
| words of text | 46,644 sentences | **1,110,764** |
| speakers | 1,099 | 10 |
| context per form | one sentence | a whole book |

## What is here

| book | audio | epub words | words/second |
| --- | ---: | ---: | ---: |
| Шкляр, «Залишенець. Чорний ворон» | 18.8 h | 212,642 | 3.1 |
| Остен, «Емма» | 18.9 h | 144,148 | 2.1 |
| Гюго, «Собор Паризької Богоматері» | 22.1 h | 143,637 | 1.8 |
| Маркес, «Сто років самотності» | 17.2 h | 117,704 | 1.9 |
| Костенко, «Записки українського самашедшого» | 12.2 h | 97,339 | 2.2 |
| Дефо, «Робінзон Крузо» | 14.1 h | 95,475 | 1.9 |
| Лі, «Вбити пересмішника» | 13.0 h | 87,035 | 1.9 |
| Шкляр, «Троща» | 15.0 h | 84,307 | 1.6 |
| Муракамі, «Погоня за вівцею» | 10.5 h | 74,755 | 2.0 |
| Хемінгуей, «За річкою, в затінку дерев» | 10.1 h | 53,722 | 1.5 |

Ukrainian speech runs about 2.2 words per second, so **the ratio column is the
first quality signal**: 3.1 says the epub carries material the narrator does
not read (front matter, notes) or the reading is fast; 1.5–1.6 says the
audiobook is abridged, the epub has been split differently, or there is a lot
of non-speech. Neither is fatal — the alignment finds what matches — but a book
far from 2.2 will yield less and should not be trusted to yield proportionally.

**These are commercial recordings and ebooks.** The derived rows (form,
signature, timings) are facts about pronunciation and fine to keep; the audio
and full text are not redistributable. The dataset must never carry sentence
text from these books beyond the span needed, and nothing from here goes into
anything published.

## The shape of the job

Common Voice hands you a clip and its sentence. A book hands you eighteen hours
and three hundred pages, and nothing says which minute is which paragraph. That
is the whole difficulty, and it is solved once per book, not per row.

```
epub → chapters → normalised word stream          (cheap, minutes)
mp3  → 16 kHz wav → 10-minute windows             (cheap, one pass)
                ↓
        ASR each window (Whisper)                 (expensive, once)
                ↓
        locate the window in the word stream      (fuzzy match on the ASR text)
                ↓
        force-align that window against the BOOK's words, not the ASR's
                ↓
        vowel intervals → SSL features → stress ranker → gold spans
```

The last three steps are exactly what `run_mine_audio.py` already does; the
proof of concept established that aligning against the book's own words rather
than the ASR output is what makes the vowel boundaries trustworthy, because ASR
errors move them.

## Steps

**1. Text extraction.** `scripts/run_extract_books.py` (new). Read the epub,
drop front matter, notes and chapter headings, emit one JSONL per book:
`{"chapter": n, "paragraph": m, "text": "…"}` in reading order, NFC. Report
words kept and dropped per book. Ukrainian only — `Муракамі` and `Маркес` are
translations, which is fine, but check for stray English or Russian passages and
count them.

**2. Audio preparation.** Decode each book once to 16 kHz mono WAV
(`run_decode_audio.py` exists). **Do not skip this**: `soundfile` cannot seek
into mp3 and decodes from the start, which cost 19 minutes per window near the
end of a book the first time this was done. 152 hours at 16 kHz mono is about
17 GB — delete each WAV after its book is mined.

**3. Anchoring.** For each 10-minute window, ASR it, then find its position in
the book's word stream by fuzzy match. Anchors must be monotonic: a window's
match cannot precede the previous window's. Report, per book, the fraction of
windows anchored and the drift between expected and found position — a book
whose anchoring falls below ~80% is abridged or mismatched and should be set
aside rather than forced.

**4. Mining.** Per anchored window, feed the book's words for that span as the
forced-alignment transcript and run the existing chain. The filters carry over
unchanged: two vowels or more, an unambiguous trie entry for the label, at least
80% of expected vowels aligned, and not all vowel durations equal (that pattern
is the aligner falling back to the word span, not a measurement).

**5. Homograph spans.** This is the point of the exercise. For every token whose
form the lexicon reads two ways, keep `(book, chapter, paragraph, char span,
candidates, audio reading, confidence)` — the same schema as
`build_token_corpus.py` consumes, with the book position in place of the clip.
**Report, per form, how many distinct contexts and how many distinct books each
reading appears in.** That number is what `tok-v1` lacked, and it is the number
that decides whether this was worth doing.

## What to check before trusting any of it

* **Anchor quality per book** (step 3). Below 80% windows anchored, set aside.
* **Speaker count is ten.** Every earlier speaker-held-out result rests on 165
  test voices; here a held-out split means a held-out *book*. Report per-book
  accuracy, and expect it to vary more.
* **The audio model is the labeller and it has known blind spots.** Prepositional
  clitics are excluded (`is_prepositional_clitic`); its confidence calibration
  (96.8% at ≥0.99) was measured on Common Voice and must be re-measured here
  before the gate is trusted — different acoustics, different narrators, longer
  utterances.
* **Nineteenth-century and dialectal spelling.** Шкляр and Костенко write
  deliberately archaic Ukrainian; the lexicon will not know many forms, and the
  ones it does know it may know wrongly. Expect the `suffix` tier's share to be
  higher here than the 2% Common Voice showed, and treat disagreements on those
  forms as suspect from both directions.

## What this does not solve

Ten narrators. Every acoustic model trained here would learn ten voices, which
is why the books are a source of **context**, not of acoustic variety — the
stress ranker stays trained on Common Voice's 1,099 speakers and is used here
only as a labeller. If the two ever disagree systematically on a book, the book
is the outlier, not the model.

## The first mining run was void, and how that was found

Written 2026-09-12.

The book corpus was mined, labelled, and about to be trained on. Before setting
the confidence gate, the ranker was asked about 12,000 book words the lexicon
already names, so its confidence could be read against a known answer. It
scored **39.1%**. Chance for that model is 37.3%, and its confidence was flat:
rows at 0.999 were no better than rows at 0.5.

A model that is 97% on held-out speakers does not become a coin flip on new
material and simultaneously lose its calibration. Something was feeding it the
wrong audio. Four things were checked in turn:

| suspect | test | verdict |
| --- | --- | --- |
| the labeller's slice clock | first vowel starts +20…40 ms after the word | correct |
| ffmpeg seeking mp3 inexactly | two decodes of the same second from different seek points | correlation 1.0000, lag 0 |
| empty prosody features | the model scored on Common Voice with and without them | 97.8% either way — it runs in `position` mode |
| **the vowel boundaries themselves** | the longest vowel against the lexicon, no model at all | **noise** |

The last test is the one to reach for first next time, because it needs no GPU
and no model. A stressed Ukrainian vowel is longer than its neighbours, so
picking the longest vowel guesses stress **63.4% of the time on Common Voice
against 37.6% chance**. On all eight books checked it scored 36–39% against
35–38% chance. The boundaries were not measurements, so nothing built on them
meant anything: 2.2M mined rows, 348,835 ranker labels, and the `tok-v2` corpus
assembled from them.

### Three compounding causes

**The anchors drift by hundreds of words.** At 1200 s into Костенко the anchor
claims the narrator is at book word 3250; the narrator is reading word 2838.
Ten-minute windows matched by rare-word voting are simply not precise enough,
and the `overlap` figure the anchoring reported (0.68–0.87 here) did not
reveal it.

**The proportional cut adds its own error.** Splitting a window's word list
into ten equal parts assumes a uniform speech rate over ten minutes.

**Forced alignment cannot refuse.** Given a word list, it places every word
somewhere in the audio. Handed the wrong words it does not fail, it produces
confident nonsense — and the 35% padding on each side, added so the true words
would certainly be inside, guaranteed that roughly 70% of what it was asked to
place was not there. Removing the padding does not help; measured, it scores
the same as chance either way, because the underlying word range is wrong.

### What replaces it

`scripts/run_mine_book_asr.py`. The dependency runs the other way: Whisper
transcribes a whole window in one pass, its own segment boundaries cut that
into roughly sixty-second chunks (so a chunk's text is exactly its audio, and
no cut falls inside a word), each chunk is force-aligned against what was heard
in it, and only then is each word looked up in the book — placed by rare-word
voting per chunk, not per ten-minute window, with a running cursor. A word
Whisper heard that the book does not have at that point is dropped rather than
forced. The anchors survive only as a coarse prior for the first chunk and as a
fallback when a chunk cannot be placed.

Two filters were tightened at the same time:

* `--min-quality` is now 1.0, not 0.8. A word missing one of its vowels shifts
  every ordinal after the gap, so the row's label points at the wrong vowel.
  It was 1.7% of rows and every one of them was a lie.
* a chunk needs `--support` rare words agreeing on its position before it is
  used. Below that it is skipped, not guessed at.

### The check that must run before any book row is trusted again

`--audit` reports the longest-vowel baseline on words the lexicon names. It is
free, needs no labels, and is the only test that caught this. **Nothing built
on book audio should be trusted until that number is near 60%.**
