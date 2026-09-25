# Which homograph groups need context, and which do not

Question: some ambiguous spellings are effectively always used in one sense, so
the dictionary's default answer is already right and the model adds only cost
and risk. Which groups are those, and are adjectives (прикметники) among them?

Source: 469 groups with at least 8 mined Ukrainian Wikipedia sentences whose
LLM label survived blind verification. Generated sentences are excluded — they
are balanced by construction and would erase exactly the skew being measured.
Machine-readable output: `output/ml/triage.json`, produced by
`ml/scripts/run_triage.py`.

## 1. Headline: adjectives are not the safe category

The intuition that прикметники can be settled by prior probability does not hold.
Verbs are the prior-safe class; adjectives sit with nouns among the
context-hungry ones.

Point estimates of the dominant sense's share:

| Part of speech | Groups | Dominant sense ≥95% | Needs context (<80%) |
| --- | --- | --- | --- |
| verb | 123 | **83%** | 5% |
| unknown | 90 | 71% | 14% |
| **adjective** | 87 | **68%** | **18%** |
| noun | 145 | **54%** | **26%** |
| adverb | 4 | 0% | 25% |

Why verbs behave: the pairs are usually aspect or prefix distinctions
(`виносити`, `проводити`) where one member dominates written usage. Why nouns
and adjectives do not: they include toponymic adjectives (`богданівський`,
0.50 majority — genuinely which village), semantic pairs (`заняття` 0.54,
`захват` 0.58, `поверх` 0.52), and name/word collisions (`августин` 0.68,
`березина` 0.62).

## 2. The number of groups is the wrong unit — weight by frequency

Serving cost and error volume follow how often a form actually occurs, not how
many groups exist. `вона` occurs 263 426 times; a rare toponymic adjective
occurs 8 times. Both are one group.

| Routing (95% CI lower bound) | Groups | Share | Corpus occurrences | Share |
| --- | --- | --- | --- | --- |
| prior-dominant (≥0.80) | 222 | 47% | 693 521 | **73%** |
| context-needed (<0.80) | 247 | 53% | 254 557 | **27%** |

So roughly a quarter of real ambiguous-span traffic genuinely needs the model,
concentrated in a small number of frequent forms. The highest-traffic
context-needed groups are worth naming, because they are where accuracy is
actually decided:

| Form | Corpus occurrences | Dominant share | POS |
| --- | --- | --- | --- |
| використання | 55 121 | 0.91 | noun |
| уряд | 36 615 | 0.96 | noun |
| обʼєднання | 25 847 | **0.52** | noun |
| старший | 21 518 | **0.71** | adjective |
| скликання | 13 631 | 0.92 | noun |
| проводити | 7 368 | 0.96 | verb |
| поділ | 7 212 | 0.95 | noun |
| самий | 6 228 | 0.96 | pronoun |
| заняття | 5 810 | **0.54** | noun |
| поверх | 4 676 | **0.52** | noun+preposition |

## 3. The honest result: nothing can be routed to "prior-only" yet

Point estimates say 314 of 469 groups (67%) are prior-only. The confidence bound
says **zero** are.

That is not a contradiction, it is sample size. Labelling was capped at 24
sentences per form, and with 24 observations and a *perfect* record the best
achievable 95% lower bound on the dominant share is **0.86** — short of the 0.95
a prior-only routing decision would need.

| Certification target | Observed record | Observations needed |
| --- | --- | --- |
| share ≥0.95 | 100% | **73** |
| share ≥0.95 | 98% | 173 |
| share ≥0.95 | 95% | not reachable |
| share ≥0.80 | 100% | 35 |
| share ≥0.80 | 98% | 53 |

Routing a group to "never call the model" on 24 observations would be a guess
presented as a decision, and its failure mode is silent: the API would confidently
serve the wrong stress for the minority sense and never surface it.

## 4. What this changes

**Do now — nothing.** Route every ambiguous span to the model. The
prior-dominant bucket is a cost optimisation, not a correctness fix, and the
data does not yet support it.

**To unlock the optimisation**, label ~80 mined sentences per form instead of 24.
The mined pool already holds up to 80 per form on disk (reservoir cap), so this
needs no new mining — only another annotation pass, roughly 3× the current one,
over the 605 forms that have the material. That certifies the ≥0.80 bucket
outright and the ≥0.95 bucket for forms with a clean record.

**Independently useful**: `dominant_share` is a better default than the current
`priority` field, which is set on only 332 of 4 829 senses. Even uncertified, a
frequency-derived default improves the abstention fallback — when the model's
margin is below threshold, the API should fall back to the *observed* dominant
sense rather than an unset priority.

## 5. Caveats

- **Domain.** The distribution is Ukrainian Wikipedia. Encyclopedic text
  over-represents toponyms, surnames and institutional vocabulary. A TTS or news
  workload would shift these shares, and a group certified prior-only on
  Wikipedia is not certified for another domain.
- **Label noise.** Shares are LLM labels confirmed by a second, different
  deployment. Blind verification rejected 12.9% of labels outright, so the
  surviving labels are filtered but not human-verified.
- **Lemma forms only.** Inflected forms are absent, and inflection can change the
  balance — a sense may dominate the nominative and not the genitive.
- **469 of 2 250 groups** cleared the 8-observation floor. The rest are too rare
  in Wikipedia to estimate at all, which is itself the argument for generation.
