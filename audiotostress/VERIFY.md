# Two witnesses: the recording and the text pipeline

The audio model hears which vowel was lengthened and pitched. The text
pipeline reads a dictionary and a morphological parse. Neither sees what the
other saw, so agreement is evidence and disagreement is information.

`scripts/run_verify_stress.py` runs both over the same recordings and compares
them word by word. The audio side never consults the lexicon: letting it would
make the two witnesses the same dictionary twice.

## 400 recordings, 1,808 words compared

Overall agreement **93.6%**.

| pipeline tier | words | agreement |
| --- | ---: | ---: |
| dictionary certain | 1,521 | 96.3% |
| morphology | 111 | 91.9% |
| dictionary guessing | 104 | 75.0% |
| suffix fallback | 50 | 74.0% |
| prepositional rule | 15 | **46.7%** |
| compound fallback | 7 | 57.1% |

Agreement tracks the pipeline's own certainty, which is what it should do. The
two fallbacks and the guessing tier are where a recording adds what no text
has.

## Audio confidence is a usable gate

| audio confidence | words | agreement |
| --- | ---: | ---: |
| 0.99 and above | 1,456 | 96.4% |
| 0.90 – 0.99 | 277 | 86.3% |
| 0.70 – 0.90 | 53 | 73.6% |
| below 0.70 | 22 | 45.5% |

Eighty percent of words sit above 0.99 and agree with the pipeline 96.4% of
the time. That is a shape worth building on: accept where both agree and the
audio is confident, queue the rest.

## The prepositional rule is the tier the recordings dispute most

The rule shifts stress to the first syllable of a pronoun after a governing
preposition — до ме́не, у ньо́го. Word by word:

| form | rule says | recordings say | agreement |
| --- | --- | --- | ---: |
| неї, нею, ними | не́ї, не́ю, ни́ми | the same | 5/5 |
| мене, тебе, себе, нього | ме́не, те́бе, се́бе, ньо́го | мене́, тебе́, себе́, нього́ | 2/10 |

The rule applies categorically. Both measurements available here say the shift
is a tendency near 60%, not a law: mining found the first syllable chosen 62.8%
of the time after a preposition against 15.7% bare.

**Resolved: the rule is right and this measurement is not.** On lang-uk, which
is human-annotated gold, the prepositional rule scores 36 of 37 (97.3%), and
`api/internal/httpapi/prepositional.go` records the same pattern independently:
39 of 40 tokens take the first syllable after a preposition, 0 of 11 without.

The audio disagreement is a blind spot this project created. The 1,936
prepositional clitics were dropped from the audio model's training as
suspected-bad labels, so it never learned these forms; its 96.8% calibration at
high confidence does not hold on them. Exclude prepositional clitics from
audio-derived conclusions until the model is retrained with them included.

## Probable dictionary error found by this run

`того` — the pipeline reads `то́го` with its dictionary certain, the
recordings say `того́` in 7 of 9 occurrences at confidence 0.97. The standard
reading is `того́`.

Also disputed and needing a human: `самі` (pipeline `самі́`, recordings
`са́мі`, 3/3) and `нього` above.
