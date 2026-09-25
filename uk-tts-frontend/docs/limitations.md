# Limitations

What is known to be wrong, so you can judge whether it matters for your voice.

## The stress numbers are component numbers

The stress figures were measured through the running stack on lang-uk's
benchmark, which feeds **one sentence at a time** and scores stress only. The
verbalization figure comes from that project's own release set. Nothing scores
the two stages together, because no public set does.

So the end-to-end accuracy of this pipeline is unmeasured. It is bounded above
by the product of the two, and the stress tier sees verbalizer output — words
like `чотирнадцятій` — which is not the distribution its lexicon was measured on.

## Heteronyms are right about four times in five

82.22% on heteronyms means roughly one in six is wrong, and heteronym errors are
the audible ones. Seven models were trained on 85k to 478k rows and all landed
between 79.7% and 82.4%; a frontier LLM given the identical input scores 81.2%.
The model tier has no easy headroom left.

Context length matters more than it should:

```
Гори, вогню, не згасай!   →  Гори́  ✓
Гори, вогню!              →  Го́ри  ✗
```

Same word, same imperative, shorter sentence, wrong answer.

## A confident wrong answer looks like a right one

Stress decisions carry a status. `stressed` means a tier decided; `morphology`
means the parser did; `dictionary_default` means **nothing decided** and the
lexicon's most frequent reading was served. In the finished text they are
indistinguishable, and a default is right often enough to look deliberate.

Request `include_tokens` and read the statuses if you care which is which.
`on_ambiguity: preserve` emits undecided words unstressed instead.

## Monosyllables carry a mark

`У́ до́мі те́мно`, not `У до́мі те́мно`. Ukrainian orthography omits the accent on a
one-syllable word; the stress API marks it deliberately, on the reasoning that a
front-end still wants to know which vowel carries the stress and for a
single-vowel word the answer is free. If your downstream voice wants standard
orthography, strip it there — it is a rendering choice, not a stress decision.

## Verbalizer weak spots

Exact match is 82.90% on the release set, so roughly one line in six differs
from the reference somewhere. Known specifics:

- **Large numbers in oblique cases.** `на 1001 людину` is read as `на ста один
  людину`. A sibling checkpoint trades 0.47 points of exact match for better
  oblique counts; neither reads that example correctly.
- **Sentence-initial capitalization** is not always preserved (`двадцятий розділ`
  for `XX розділ`). Harmless for speech, visible in a diff.
- Numeric-value, identifier-group and digit fidelity are reported as 100% on
  every row of the release set — digits are not dropped or transposed, even when
  the wording around them is wrong.

## Chunking edge cases

The abbreviation guard is a fixed list. An abbreviation not on it, followed by a
capitalized word, splits the sentence — the model then sees less context than it
should. It never loses text: the round-trip invariant holds regardless.

A sentence longer than the source window is cut at clause seams, and each piece
is verbalized without the others' context. The result carries a warning saying
so.

## Scale

Both models run on CPU by default. Verbalization is roughly 300 ms for a short
paragraph on one core; the stress call adds one round trip per sentence. Set
`UKTTS_VERBALIZER_DEVICE=cuda` for the first stage if you have a GPU. There is
no queueing, no rate limiting, and no back-pressure in this layer.
