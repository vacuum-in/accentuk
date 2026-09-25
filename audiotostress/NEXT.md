# What to do next, and why

Ordered by what the evidence says is worth doing, not by what is interesting.

## 0. Done: Common Voice 26.0 is mined and benchmarked

116,340 rows over 1,099 speakers, 94.87% on held-out voices, 82.4% on
homographs cross-corpus. See RESULTS.md. Two things it settled:

* Speaker diversity is no longer the bottleneck — 824 training voices score
  what 67 do once each has enough rows.
* Prosody alone does not transfer between speakers (47.7% on its worst voice);
  the SSL embeddings are worth 23.9 points.

## Done: the encoder has been unfrozen and measured

Worth 1.9 points inside a single training loop (93.44% frozen against 95.32%
with four layers trainable, p = 4e-51), but only +0.2 to +0.35 against the
frozen production model, at the edge of significance, for eight hours of
training and a 1.27 GB checkpoint. See RESULTS.md. The frozen model remains
the better trade for now.

## Done: encoder choice has been tested

WavLM-large, despite being the stronger encoder on prosodic benchmarks, loses
by 1.36 points (p = 3e-23) because it is not adapted to Ukrainian and ours is.
If a better Ukrainian-adapted encoder appears, that is the lever to pull; a
generically better one is not.

## Closed: homograph labels cannot be bootstrapped from text

The obvious way around needing gold passages is to mine homographs from Common
Voice and label them with the text pipeline. The material is there — 32,439 of
79,711 validated clips contain a form the trie reads two ways, over 3,205
distinct forms, against the gold set's 18.

The labels are not. Only the morphology tier can settle these, and its measured
accuracy on exactly this class is 74% of the 79% it will answer at all. An
evaluation set labelled at 74% cannot measure a model at 85%: a perfect model
would score 74% and be indistinguishable from a poor one. The rows could still
train — a model that has never seen a homograph in training might learn from
noisy ones — but nothing available here could show whether it helped, and
unmeasurable work is not worth doing.

Gold passages are not the convenient path to the homograph claim. They are the
only one.

## 1. More gold homograph passages — the only thing blocking the claim

The homograph score is 82.4% with an interval of [66.5, 91.7], against a
longest-vowel baseline of 76.5%. Those overlap. 34 rows over 18 forms cannot
separate a model that hears stress from one that picks the long vowel, and no
amount of further training changes that — the fix is more passages in the
user's own format, stress marked, read aloud. Roughly 100-150 homograph forms
would bring the interval down to a few points.

Candidate forms to cover, from the misses: лупа, сходи, вина, виходити, ірис,
слова, замок, замки, доро́га/дорога́, за́мок/замо́к, ві́домість/відо́мість.

## 2. Common Voice 26.0 Ukrainian — the material this needed

`https://mozillafoundation.../datasets/cmqinqaqp00wunq07w9oyei38`

| | |
| --- | --- |
| audio | 118.46 h, 102.96 validated, 91,708 clips |
| speakers | **1,177** |
| corpus | 214,329 sentences |
| licence | **CC0-1.0** |
| size | 2.61 GB |

Three things make it decisive rather than merely larger.

**A thousand speakers instead of one.** Every model here is trained on a single
audiobook narrator. Transfer to another voice has been shown once, on synthetic
speech. With 1,177 speakers the model has to learn stress rather than a timbre.

**CC0.** The text side of this project cannot publish weights because 88.6% of
its corpus sits under mixed terms. Audio labels derived from public-domain
recordings carry no such constraint.

**Scripted speech.** The words are known exactly, so alignment needs no ASR and
no fuzzy match into a PDF — which removes the failure where a misheard word got
a dictionary label meant for a word nobody said.

Download: the collective's `/download` path returns 404 to an unauthenticated
request, so it likely needs an account and a licence acceptance.
`mozilla-foundation/common_voice_17_0` is on Hugging Face and is not gated —
older, but usable to start without waiting on access.

What it will *not* fix on its own: homograph density. Read sentences are as
sparse in homographs as any prose, and the evaluation problem below stays.

## 2. More evaluation material, in the format that already works

The homograph figure — the only one that matters — is 29 of 34 over **18
distinct forms**, with a 95% interval of [69.9, 93.6]. Twenty-four points wide.
No model change can be judged against an interval that size.

The passage supplied for evaluation has homograph density roughly thirty times
that of prose, because it was written for it. Two or three more such passages
would take the count to 100–150 forms and the interval to about ±6 points.
That is the cheapest thing on this list and it gates everything else.

## 3. Isolate the classifier from the features

The embeddings arm uses regularised logistic regression and the prosody arm
uses gradient boosting, because above 200 columns boosting is unusable on four
thousand rows. So the 22-point gap is measured across a change of both. Running
prosody through the same linear model would settle how much is the features.

## 4. Fine-tune the encoder

Only after 2 and 3. Frame embeddings from a frozen wav2vec2 already reach
94.9%; a stress head trained end to end is the obvious next lever, and equally
obviously not worth starting while the evaluation interval is twenty-four
points wide.

## Ruled out

**A general language model as a last tier for text.** Gemma 4 E4B answered the
blanks a confident text policy leaves at 50.3% and gpt-5.6-luna at 57.5%,
against 65.0% for the dictionary default they would replace. Both worse than
leaving the word alone. Measured, not assumed.

**Single acoustic features.** F0 median 56.5%, duration 51.1%, energy 42.0%,
chance 38.4%. The combination is the point.
