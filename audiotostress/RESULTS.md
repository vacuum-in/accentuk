# Measured results — Ukrainian stress from audio

Everything here was measured on this machine. Where a number is uncertain the
interval is given, because two of the headline figures rest on samples small
enough that the interval decides what may be claimed.

## Headline

Trained on an audiobook narrator, tested on a different voice reading a
different passage — 311 words, of which 34 are homographs the passage stresses
two ways.

| features | all 311 | 34 homographs |
| --- | ---: | ---: |
| chance | 38.3% | — |
| longest vowel | 64.0% | — |
| position only | 49.8% | 50.0% |
| prosody (25 hand-built features) | 73.0% | 73.5% |
| prosody + position | 74.6% | 73.5% |
| **wav2vec2 frame embeddings** | **94.9%** | **85.3%** |
| embeddings + prosody | 94.2% | 85.3% |

Stress is recoverable from a recording, and the encoder's own representations
carry it far better than duration, energy and pitch: +22 points overall on
half the training data (4,238 rows against 10,124). Adding the hand-built
features on top costs 0.7 — everything they measure, the encoder already has,
plus formant structure and vowel quality that three numbers cannot express.

## What these numbers do not support

**The homograph figure is 29 of 34, over 18 distinct forms.** The 95% interval
is [69.9, 93.6] — twenty-four points wide. The true accuracy could be 70% or
94% and this evidence cannot tell them apart. Repeats of one form are not
independent observations, so the effective sample is smaller still.

The overall figure is firmer: 295/311, interval [91.8, 96.8].

**The test audio is synthetic**, one voice, 200 seconds, from a passage written
to put homographs in maximally distinguishing contexts. Human speech will score
lower.

**The classifier confound is now measured, and it was small.** The table above
runs boosting for the narrow arms and regularised logistic regression for the
wide ones, because boosting cannot use a thousand dimensions from four thousand
rows. Running every arm through the same linear model (`--linear`) settles how
much of the gap was that:

| features, same linear model | all 311 | 34 homographs |
| --- | ---: | ---: |
| position only | 49.8% | 50.0% |
| prosody | 77.8% | 73.5% |
| prosody + position | 74.3% | 70.6% |
| embeddings | 94.9% | 85.3% |

Prosody gains 4.8 points from the change of model, so roughly a fifth of the
22-point gap was the classifier and the rest is the features. On homographs
nothing moves at all: 85.3% against 73.5% either way.

Position also *hurts* once acoustics are present — 74.3% against 77.8%, and
70.6% against 73.5% on homographs. The positional prior is learnt on ordinary
words, and a homograph is by definition a word that breaks the usual placement.

## What is solid

Training used one narrator; testing used another voice entirely, so
generalisation across speakers is demonstrated rather than assumed. The
position-only arm — a model told which vowel of how many, and nothing else —
scores 49.8%, so the acoustic arms are not simply reproducing the fact that
Ukrainian stress is unevenly distributed across syllables.

## Where a single feature gets to

Before the classifier, the proof of concept scored each vowel by one feature at
a time, on 131 words the lexicon names unambiguously:

| feature | accuracy |
| --- | ---: |
| F0 median | 56.5% |
| duration (four variants) | 51.1% |
| RMS / peak energy | 42.0% |
| F0 range | 40.5% |
| F0 slope | 39.7% |
| chance | 38.4% |

No single number finds the stressed vowel. That is what makes a learnt
combination necessary, and it matches RUAccent's approach rather than
contradicting it.

## Datasets

| file | rows | what |
| --- | ---: | --- |
| `artifacts/audio_runs/mined/rows.jsonl` | 10,188 | prosody only, 90 windows |
| `artifacts/audio_runs/ssl/rows.jsonl` + `.ssl.npy` | 4,238 | prosody and embeddings, 40 windows |
| `artifacts/audio_runs/stressed_eval/rows.jsonl` + `.ssl.npy` | 311 | the evaluation passage |
| `artifacts/audio_runs/commonvoice/rows.jsonl` + `.ssl.f16` | 116,340 | Common Voice 26.0, 1,099 speakers |

Labels come from the stress trie, and only from forms it names unambiguously:
for anything else the dictionary holds a guess, and training on a guess teaches
the guess. The evaluation set is the exception — its labels are the accents
written in the text, which is why it can score homographs at all.

## Common Voice: a thousand voices, and the test voices held back

Everything above was one narrator reading one novel. A model reaches 94.9% on
one voice partly by learning that voice, and the number says nothing about
whether it hears stress. Common Voice 26.0 Ukrainian is 118 hours over 1,107
speakers with the sentence shipped alongside each clip, which also removes ASR
from the mining loop: the transcript is exact, so a misheard word can no longer
become a wrongly labelled row.

Mining 32,598 clips took 91 minutes at 0.17 s/clip and produced **116,340
labelled words with 341,249 vowel embeddings, and no failed clips**. A cap of
150 clips per speaker keeps the busiest voice — which reads 7,346 of the 73,166
validated clips on its own — from supplying a tenth of the corpus.

Both splits keep every speaker on one side of the boundary, so the test voices
are ones the model has never heard.

| split | train | test | test speakers | accuracy | median voice | worst voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ours, 75/10/15 by speaker | 90,153 rows / 824 voices | 16,374 | 165 | **94.94%** | 95.7% | 86.3% |
| Common Voice official | 35,195 rows / 67 voices | 36,262 | **894** | **94.84%** | 95.5% | 71.4% |

Chance is 37.8% and picking the longest vowel is 63.5%.

The two rows landing on the same number is the finding. Training on 824 voices
scores what training on 67 does, once each has enough rows — at 32k rows the
gap was 94.95% against 92.44%, and more data closed it. Speaker diversity was
the bottleneck only while the corpus was small; it is not the bottleneck now.

### What the embeddings are worth once the voice is unfamiliar

| features | test | median voice | worst voice |
| --- | ---: | ---: | ---: |
| prosody and position | 71.08% | 72.5% | 47.7% |
| with SSL embeddings | **94.95%** | 95.4% | 84.3% |

Duration and F0 measure the speaker at least as much as the stress: prosody
alone collapses to 47.7% on its worst held-out voice, barely above chance. The
embeddings, taken against the word's own mean, are what survives a change of
voice — worth 23.9 points.

### Some of the error was the dictionary's

The misses on held-out speakers clustered on `мене`, `себе`, `тебе`, `кому`,
`поки` rather than scattering — the forms the prepositional rule moves. The
trie names the final syllable (мене́); after a governing preposition Ukrainian
says до ме́не, and the label calls a correct reading wrong.

The acoustics settle this without appealing to the rule:

| context | rows | agrees with trie label | model picks first syllable |
| --- | ---: | ---: | ---: |
| after a preposition | 86 | 67.4% | 62.8% |
| bare | 51 | 88.2% | 15.7% |

The model hears a shift the label does not know about. 904 such rows are
flagged by `scripts/run_tag_context.py` and dropped from training, and the true
accuracy is therefore somewhat above the figures quoted.

### Cross-corpus, on the words a dictionary cannot settle

The Common Voice checkpoint against the stressed passage — different speakers,
different recording chain, different material, never seen in training:

| subset | forms | chance | longest vowel | ranker |
| --- | ---: | ---: | ---: | ---: |
| all words | 264 | 38.3% | 64.0% | **93.6%** [90.3, 95.8] |
| homographs | 18 | 47.5% | 76.5% | **82.4%** [66.5, 91.7] |

The homograph figure is the only one worth quoting as a claim about hearing
stress, and its interval still overlaps the longest-vowel baseline. 34 rows
over 18 forms cannot separate them; that needs more gold passages, not more
training.

Misses: лупа×2, сходи, вина, виходити, ірис.

## Using it

```
.venv/bin/python scripts/run_stress_audio.py CLIP.mp3 --text "Він хотів стягти її з вагончика."
  → він хоті́в стягти ї́ї з ваго́нчика
```

The lexicon narrows the field before the model chooses: where it names one
reading the ranker is not consulted, where it names two the ranker picks
between them, and for a form the dictionary has never seen it picks freely.
`--model-only` drops the lexicon so the two can be compared; the run above
gives the same answer either way.

Words the aligner cannot measure — fewer than two vowels resolved, quality
below 0.8, or every vowel the same length — are left unaccented rather than
guessed. Three of the six words above were measured.

### Capacity is not the bottleneck

| ranker | test | median voice | worst voice |
| --- | ---: | ---: | ---: |
| hidden 256, 2 layers | 94.87% | 95.7% | 84.0% |
| hidden 384, 4 layers | 94.87% | 95.6% | 82.4% |

Three times the parameters and twice the depth score the same figure to two
decimals. The ceiling is in the representation — mean-pooled frames from a
frozen wav2vec2, centred on the word — not in the head that reads it.

The first attempt at the larger model kept the learning rate at 3e-4 and did
not train at all: loss stalled at 0.92 against the small model's 0.186, and
test came out at 51.5% against a 37.8% chance level. A deeper post-LN
transformer needs a smaller step (1e-4 here) or a warmup. Worth recording
because the failure looks like a capacity answer and is not one.

The remaining lever is therefore the encoder, not the head: unfreezing the top
wav2vec2 layers, which `configure_ssl_trainability` already supports. That
needs the raw audio inside the training loop rather than pooled vectors, so it
is a re-mining job as much as a training one.

### The shape of the vowel, not just its average

Mean-pooling a vowel's frames throws away what happens inside it, and the rise
and fall within a vowel is a stress cue. The miner's `--ssl-parts 3` stores the
whole-vowel mean followed by the mean of each of three equal slices, so the
first 1024 columns remain exactly the mean-pooled representation and both arms
train on identical rows from identical audio.

| features | test | discordant rows |
| --- | ---: | --- |
| mean only (1024) | 95.47% | 28 right where slices were wrong |
| mean + three slices (4096) | **95.92%** | 63 right where the mean was wrong |

McNemar exact, two-sided **p < 0.001** on 7,779 held-out rows. The gain is
0.45 points — small, and only visible as a paired test; comparing the two
means alone would have called it noise.

### A reported number that described no artifact

Finding this needed both models loaded from disk, which is how the trainer's
final figure turned out to describe the last epoch's state rather than the
checkpoint it had saved. Dev peaks a few epochs before the end, the saved
weights are that peak, and the model anyone would actually load scored better
than the line printed underneath it: 95.92% against a reported 95.62%, and
94.94% against 94.87% for the main run. The trainer now reloads its checkpoint
before the final evaluation, and the figures in this document are the
checkpoints'.

## The production model

Mining the full corpus with `--ssl-parts 3` — 116,340 rows, 341,249 vowels at
4096 columns, no failed clips, deterministic to the row against the
mean-pooled run — and training 14 epochs on the speaker re-split:

| | held-out speakers | median voice | worst voice | 10th pct |
| --- | ---: | ---: | ---: | ---: |
| mean-pooled | 94.94% | 95.7% | 86.3% | 90.9% |
| **production** | **95.54%** | 95.9% | 86.4% | 92.0% |

The same 16,374 rows, row for row. McNemar exact: 162 discordant in favour of
the production model against 64, **p = 5.5e-11**. The 10th-percentile voice
gains more than the median does, which is the shape you want — the floor comes
up rather than the middle.

Cross-corpus on the stressed passage:

| subset | forms | chance | longest vowel | mean-pooled | production |
| --- | ---: | ---: | ---: | ---: | ---: |
| all words | 264 | 38.3% | 64.0% | 93.6% | **94.9%** [91.8, 96.8] |
| homographs | 18 | 47.5% | 76.5% | 82.4% | **85.3%** [69.9, 93.6] |

Misses are down to лупа×2, сходи, вина, виходити — `виходити` and `ірис` from
the earlier run are now right.

The homograph interval still contains the longest-vowel baseline. It will keep
containing it at any accuracy until the sample grows: 34 rows cannot separate
85% from 76%, and that is a statement about the evaluation set, not the model.

**The artifact:** `artifacts/audio_runs/production/ranker.pt`, read by
`scripts/run_stress_audio.py`.

## Unfreezing the encoder is worth 1.9 points

The remaining lever after capacity and pooling was the encoder itself. Getting
a straight answer took three runs and a control, and the first two answers
were wrong.

Run against the production model, unfreezing looked harmful: four of twenty-
four layers trainable at 1e-5 for three epochs scored 95.32% against the
frozen model's 95.54%, paired on the same 16,374 rows, McNemar p = 0.034. A
gentler configuration — two layers, 3e-6, six epochs — was worse still at
94.76%, p = 5.9e-12.

That second result is what exposed the mistake. It moved the encoder *less*
and trained *longer*, and it lost by more. Encoder drift cannot explain that
shape, so the difference had to be somewhere other than the encoder — and it
was: the fine-tuning loop processes one clip at a time with accumulation over
four, an effective batch of roughly fifteen words, against the frozen
trainer's sixty-four, over three epochs against fourteen. The two runs
differed in more than the variable under test.

The control settles it. Same loop, same batching, same three epochs, encoder
frozen solid:

| same loop, three epochs | test | median voice | worst voice |
| --- | ---: | ---: | ---: |
| encoder frozen | 93.44% | 93.9% | 80.4% |
| two layers unfrozen, 3e-6 | 94.76% | 95.5% | 83.3% |
| four layers unfrozen, 1e-5 | **95.32%** | 95.8% | 82.4% |

Paired, frozen against four layers: 380 discordant in favour of unfreezing
against 73, McNemar exact **p = 4e-51**. Unfreezing is worth 1.88 points, and
more of it beats less of it.

The production model's 95.54% comes from a better training loop, not from
being frozen. With the loop's batching fixed — six layers, accumulation over
sixteen, eight epochs — unfreezing beats it outright:

| | test | median voice | worst voice | 10th pct |
| --- | ---: | ---: | ---: | ---: |
| frozen encoder, 14 epochs | 95.54% | 95.9% | 86.4% | 92.0% |
| **six layers unfrozen, 8 epochs** | **95.88%** | 96.4% | 86.4% | **93.3%** |

Paired on the same 16,374 rows: 200 discordant in favour of the fine-tuned
model against 144, McNemar exact **p = 0.003**. The tenth-percentile voice
gains 1.3 points against the median's 0.5, so again the floor rises faster
than the middle.

Dev was still climbing at epoch 8 (96.02%), so a fourteen-epoch run followed —
and came out slightly worse on test despite a better dev:

| run | dev | test | McNemar vs frozen |
| --- | ---: | ---: | --- |
| eight epochs | 96.02% | **95.88%** | 200 / 144, p = 0.003 |
| fourteen epochs | 96.30% | 95.76% | 217 / 181, p = 0.079 |

Both are positive and only one is significant, so the honest figure for
unfreezing against the frozen production model is **+0.2 to +0.35 points, at
the edge of significance** — not the 1.9 points it is worth inside a single
loop. Selecting the checkpoint on a dev set of 8,895 rows over fourteen epochs
starts fitting the selection rather than the model, which is the likeliest
reason the longer run picked a worse checkpoint from a better dev score.

### The gain does not leave Common Voice

Held-out speakers are still Common Voice speakers: the same recording chain,
the same room noise, the same microphones. The stressed passage is not, and
there the two models are indistinguishable:

| | all words (311) | homographs (34) |
| --- | ---: | ---: |
| frozen production | **94.9%** | **85.3%** (29/34) |
| fine-tuned | 94.2% | 82.4% (28/34) |

Two rows of 311 and one of 34 — nothing either way. What matters is the
absence: the 0.34 points unfreezing won on held-out Common Voice voices does
not appear here at all. The encoder was adapted to the corpus rather than to
Ukrainian stress, which is exactly why it helped only where the test audio
came from the same corpus as the training audio.

### Is it worth it

Against the frozen encoder the gain is real but small, and it costs a great
deal: eight to nine hours of training against twenty-five minutes, an epoch
that needs the audio rather than cached vectors, and a 1.27 GB checkpoint that
carries its own encoder — inference can no longer use a stock wav2vec2. For a
pipeline where the lexicon already settles most words, the frozen production
model remains the better trade. The unfrozen model is the one to reach for if
the homograph set ever grows enough to reward the last fraction of a point.

### What the wrong answer cost

Two runs and about five hours, because the first comparison changed the
encoder, the batch size and the epoch count at once and I read the result as
though only the encoder had moved. The gentler run was launched to test the
learning rate and accidentally caught the error instead: a configuration that
should have been safer came out worse, which is not a thing encoder damage can
do.

## A different encoder: WavLM-large loses by 1.4 points

Every result above uses one encoder, and the ceiling is the representation, so
the encoder is a lever in its own right. WavLM-large is the obvious candidate:
it was pretrained with utterance mixing and denoising and beats wav2vec2 on
SUPERB's speaker and emotion tasks, which are nearer to stress than phoneme
recognition is.

Same clips, same rows, same split, same fourteen epochs — only the encoder
differs:

| encoder | test | median voice | worst voice | 10th pct |
| --- | ---: | ---: | ---: | ---: |
| wav2vec2-xls-r-300m, Ukrainian ASR | **95.54%** | 95.9% | 86.4% | 92.0% |
| WavLM-large, as published | 94.18% | 94.3% | 80.4% | 90.3% |

Paired on the same 16,374 rows: 367 discordant in favour of wav2vec2 against
145, McNemar exact **p = 3e-23**.

The reason is the more useful part. The encoder in use is not stock
wav2vec2 — it is xls-r-300m fine-tuned on Ukrainian ASR, and WavLM-large is
stock. Adapting an encoder to the language beats the prosodic strength of its
pretraining objective, by a wide margin.

That also squares with the unfreezing result rather than contradicting it.
Language adaptation on a large corpus with a phonetic objective helps; nudging
six layers with 48 hours and a stress objective adapts to the corpus instead.
The scale and the objective are what separate them.

## An ensemble adds nothing, and the measurement floor is two rows

Averaging the frozen and fine-tuned models' probabilities is the obvious cheap
move: they disagree on 344 held-out rows, which is the condition under which
averaging pays.

| | held-out speakers (16,374) | passage (311) | homographs (34) |
| --- | ---: | ---: | ---: |
| frozen production | 95.54% | 94.9% | 85.3% |
| fine-tuned | 95.88% | 94.2% | 82.4% |
| ensemble | 95.91% | 94.9% | 85.3% |

On held-out speakers the ensemble beats the frozen model (McNemar p = 2e-6)
but beats the fine-tuned model by 0.03 points, so it is not an ensemble effect
— it is the stronger member showing through. Cross-corpus it is
indistinguishable from the frozen model alone. Two models to ship, no gain.

The run also measured the same production checkpoint twice, on the same audio,
through two evaluation paths, and got 295/311 and 293/311. Both are correct
implementations of the same pooling; they disagree because `part_pool` takes
its frame bounds from `time_to_frame_bounds` while `pool_spans` computes
`int(start / shift)` to `int(end / shift) + 1`, which differ by one frame.

Two rows out of 311. That is the same size as every cross-corpus difference
between models measured here, which puts a floor under what the 311-row
passage can resolve at all: differences of one or two rows are implementation
detail, not model quality. The held-out speaker set, at 16,374 rows, is where
the comparisons in this document actually carry weight.

## Alignment is not the bottleneck either

Features are pooled over vowel boundaries the aligner supplies, so their
quality bounds everything downstream, and a better aligner is a natural thing
to reach for. Measured on the held-out test set first:

| alignment quality | accuracy | rows |
| --- | ---: | ---: |
| 1.00 — every expected vowel found | 95.63% | 16,032 |
| 0.80 | 90.91% | 264 |

| shortest vowel in the word | accuracy | rows |
| --- | ---: | ---: |
| under 40 ms (two encoder frames) | 95.40% | 2,716 |
| 40–79 ms | 95.50% | 8,147 |
| 80–119 ms | 95.78% | 4,450 |
| 120 ms and over | 95.19% | 1,061 |

98.4% of rows already align perfectly. Imperfect ones do score worse, but
repairing every one of them would be worth at most 0.08 points. Accuracy is
flat across vowel duration, so neither boundary precision nor the encoder's
20 ms frame resolution is holding the result back — a two-frame vowel scores
what a six-frame one does.

## Does it hear stress, or recall the words it was trained on?

The question every number above depends on. A model trained on 90,153 labelled
words could reach 95% by learning where the stress falls in each of them, and
such a model would be worthless on the homographs it exists for.

Splitting the held-out rows by how often the exact form appeared in training:

| times the form was in training | accuracy | rows |
| --- | ---: | ---: |
| never | **93.72%** | 2,499 |
| 1–2 | 94.66% | 3,411 |
| 3–9 | 95.12% | 3,646 |
| 10–49 | 96.26% | 3,848 |
| 50 or more | 97.64% | 2,970 |

Familiarity is worth something — 3.9 points across the range — but the floor is
what settles the question. On 2,499 rows over 2,321 forms it has never
encountered, in voices it has never heard:

| | |
| --- | ---: |
| chance | 31.8% |
| longest vowel | 60.5% |
| always the first vowel | 23.3% |
| **the model** | **93.72%** [92.7, 94.6] |

Thirty-three points above the best heuristic, on words it cannot have
memorised. The model is reading the acoustics.

That also makes the homograph figure credible rather than lucky: 85.3% on
forms where the dictionary offers two readings is what a model that hears
stress should manage, and it is not what a lookup table could produce at all.

## It knows when it is wrong

Whether a stress reader can abstain matters more than its average for building
a dataset: a wrong accent in training data is worse than a missing one. The
model's confidence is close to honest, running a little high:

| it says | it is right | rows |
| ---: | ---: | ---: |
| 60.3% | 59.5% | 269 |
| 82.7% | 76.7% | 563 |
| 97.3% | 95.9% | 10,001 |
| 99.4% | 98.8% | 5,510 |

So a threshold buys predictable precision:

| threshold | coverage | precision | errors |
| --- | ---: | ---: | ---: |
| none | 100.0% | 95.54% | 731 |
| 0.95 | 88.6% | 97.42% | 374 |
| 0.99 | 33.7% | 98.80% | 66 |
| 0.995 | 12.8% | 99.24% | 16 |

There is no threshold that reaches certainty, and it would be wrong to imply
one — 99.24% at an eighth of the words is where this ends. But the measurement
is on words the lexicon already settles, which is not where the model is used.
In the pipeline the dictionary answers what it knows exactly and the model is
consulted only where it is silent or offers two readings, so pipeline
precision is higher than any row of this table.

`run_stress_audio.py --min-confidence 0.99` leaves a word unaccented rather
than answer below the threshold. Lexicon answers are never withheld: they are
not the model's guess.

## Even-handed across voices

A stress reader that works on men and not on women, or on careful speech and
not on fast, would be a poor foundation for a TTS dataset. Held-out rows carry
the speaker's stated gender and accent, so this is measurable rather than
assumed.

| speaker gender | accuracy | rows |
| --- | ---: | ---: |
| male | 95.70% | 7,298 |
| female | 95.54% | 4,194 |
| unstated | 95.29% | 4,882 |

| mean vowel length (speech rate proxy) | accuracy | rows |
| --- | ---: | ---: |
| under 50 ms (fast) | 95.95% | 148 |
| 50–79 ms | 94.70% | 4,584 |
| 80–119 ms | 96.11% | 8,879 |
| 120 ms and over (slow) | 95.04% | 2,763 |

The gender gap is 0.16 points, which is nothing. Speech rate shows no
monotonic trend: fast speech, whose vowels are half as long, reads no worse
than slow.

Stated accents range wider — 94.34% on 106 rows of Південь України against
97.41% on 540 of Галицький — but at those sample sizes most of that is noise,
and only the Галицький figure sits clearly above the 95.5% baseline. Worth
revisiting if accent-labelled data grows; not worth acting on now.

## What the model actually uses

Zeroing feature groups at inference, on the held-out set:

| zeroed | accuracy | cost |
| --- | ---: | ---: |
| nothing | 95.54% | |
| SSL embeddings | 54.10% | **41.44** |
| all prosody | 94.91% | 0.62 |
| position only | 95.30% | 0.23 |
| F0 only | 95.58% | **−0.04** |
| duration only | 95.55% | −0.01 |
| energy only | 95.54% | −0.01 |

The embeddings carry the task; without them the model falls to 54%, barely
above the longest-vowel heuristic. Everything else together is worth 0.62
points, and dropping acoustic prosody entirely — keeping only the free
positional features — costs 0.19 (McNemar p = 0.027).

These figures are an upper bound on prosody's contribution, not a measurement
of it: the model was trained *with* these features and then denied them, and
one trained without would have learned to compensate.

The striking row is F0. Pitch is the classic correlate of stress, it was the
best single feature in the proof of concept at 56.5%, and here it contributes
nothing at all — very slightly negative. The wav2vec2 frames already encode
it, and the explicit track is redundant.

That is worth knowing beyond this project: a pipeline pairing SSL embeddings
with hand-computed prosody can drop the F0 extraction, which was the original
performance bottleneck here, and lose nothing.

## The measured prosody was making it worse

Ablation put the measured features' contribution at 0.19 points — an upper
bound, since that model was trained with them and then denied them. Training
without them says something else entirely:

| features | test | median voice | worst voice | 10th pct |
| --- | ---: | ---: | ---: | ---: |
| SSL + full prosody | 95.54% | 95.9% | 86.4% | 92.0% |
| **SSL + position only** | **96.23%** | 96.4% | 87.0% | **93.5%** |
| SSL alone | 96.17% | 96.6% | 86.4% | 92.6% |

Paired against the full model on the same 16,374 rows: 219 discordant in
favour of dropping prosody against 106, McNemar exact **p = 3e-10**. Training
loss falls from 0.15 to 0.0965.

Duration, energy and F0 were not merely redundant beside the embeddings — they
were **costing 0.69 points**. Their absolute values carry the speaker, and on
a voice the model has not heard they mislead. Ablation could not see this
because the trained model had already learned to discount them; only training
without them shows what they cost.

The positional features are worth 0.06 over nothing at all, which is noise,
but they are free — five integers from the vowel index — so the shipped model
keeps them.

Cross-corpus the new model matches the old one at 94.9% and 85.3%, both within
the two-row floor the 311-row passage can resolve, and reads `сходи`
correctly where the old one did not. So the gain on held-out voices is not
bought at the cost of transfer.

**The artifact is now `artifacts/audio_runs/prosody_position/ranker.pt`.**
`run_stress_audio.py` reads the feature set from the checkpoint's own prosody
width and skips extracting what the model does not use — the F0 track
included, which was this project's original performance bottleneck.

## Two earlier conclusions were wrong, and one still holds

Prosody turned out to be harmful, which means every conclusion drawn while it
was present was drawn on a defective feature set. Two of them do not survive
re-testing without it:

| lever | measured with prosody | measured without |
| --- | ---: | ---: |
| head capacity (384/4 against 256/2) | 0.00 | **+0.50** in-domain |
| three-slice vowel pooling | +0.45 | +0.06 |

Capacity had looked settled at exactly zero; without the noisy features a
bigger head is worth half a point in-domain (McNemar p = 1e-5). The vowel
slices, worth 0.45 points before, are now noise — the 4096-column feature file
could be a 1024-column one.

Training longer is worth 0.10 (96.33% at 24 epochs against 96.23% at 14),
which is within noise.

### But the bigger head does not leave Common Voice either

| | held-out speakers | passage (311) | homographs (34) |
| --- | ---: | ---: | ---: |
| 256 / 2 layers | 96.23% | **94.9%** | **85.3%** |
| 384 / 4 layers | **96.73%** | 92.9% | 79.4% |

Six rows of 311, three times the two-row floor this passage can resolve, so
the drop is real. It is the same shape as the fine-tuned encoder: an in-domain
gain that does not transfer — except here it does not merely vanish, it
reverses. The larger head learns Common Voice rather than Ukrainian stress.

**The shipped model stays 256 / 2.** The cross-corpus figure is the one that
describes behaviour on audio this project did not collect.

### What this says about the order of experiments

Capacity and pooling were both closed as "measured" while the feature set was
defective, and both answers were wrong — one hid a real gain, the other
invented one. The cheapest experiment available, removing a feature group and
retraining, was also the most informative, and it was the last one run.

## Removing the per-speaker cap: the largest single gain

The mining cap of 150 clips per speaker existed so one prolific voice could
not supply a tenth of the corpus. Removing it mines all 73,166 validated clips
instead of 32,598.

| | rows | vowels | test on the same 16,374 rows |
| --- | ---: | ---: | ---: |
| capped at 150 clips | 116,340 | 341,249 | 96.23% |
| **every validated clip** | **260,757** | **766,863** | **97.04%** |

Paired: 252 discordant in favour of the larger corpus against 119, McNemar
exact **p = 4e-12**. On its own larger test set it reads 96.80% over 26,574
rows, with a median voice of 97.4% and a tenth percentile of 94.6%.

The estimate before running this was "+0.1 points or less", and it was wrong by
eight times. The reasoning behind it was that the speaker-scaling curve
flattens — +0.65, +0.23, +0.14 across doublings — and that 824 training voices
were already every voice available, so only clips could be added. What that
missed is that more clips from a known voice are not repetition: they are new
words in new contexts, which is the axis the curve above never varied.

Cross-corpus it reads 94.2% and 82.4% against the previous model's 94.9% and
85.3% — two rows and one row, at the floor this 311-row passage can resolve,
so transfer is unchanged within measurement precision. Unlike the larger head,
which lost six rows, nothing here is bought against it.

**The artifact is now `artifacts/audio_runs/all_model/ranker.pt`**: mean-pooled
vowels at 1024 columns, positional features only, 14 epochs.
