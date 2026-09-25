---
language: uk
license: apache-2.0
library_name: pytorch
tags: [ukrainian, stress, homograph, ensemble, tts]
---

# Ukrainian stress combiner (`combiner-v1`)

A small learned scorer over the pipeline's tiers. For each candidate reading of an ambiguous word, it sees:
- what every tier said about it;
- which reading the fixed tier order chose.

It overrules the tier order only when it prefers another reading by a margin.

## Input and output

32 features per candidate. `wiki-stress/ml/src/ukstress_ml/combiner.py` defines them; `model.pt` carries the list, and the list is part of the model:

| group | features |
| --- | --- |
| lexicon | rank, capitalisation |
| morphology tier | its pick; the tagger's agreement with the reading's case, number, gender and part of speech |
| cross-encoder | probability of the candidate |
| token classifier | probability of the candidate, the form's training count, how one-sided its training rows were |
| narrator prior | share of audiobook narrators who said this reading, and how many said anything |
| readings | whether the readings are told apart by sense, by grammar, or as proper nouns |
| tier order | its answer, crossed with the tier that gave it |

- **Model:** one hidden layer of 32 tanh units, softmax across a token's candidates.
- **Output:** a reading and its probability.
- **Gate τ = 0.2:** it departs from the tier order only when its preferred reading's probability beats the tier order's reading by τ.
- **Unseen tiers:** tiers that were added after training (the agreement repair) are always kept.

```python
from ukstress_ml.combiner import Combiner, Assets
combiner = Combiner("uk-stress-combiner/model.pt")
assets = Assets.load("uk-stress-combiner")   # readings.jsonl, audio_prior.json, classifier_profile.json
```

In the API it is off by default. A request enables it with `"combiner": true`.

## Training

- **Data:** labelled tokens from two sources.
  - Common Voice uk, with audio gold at ranker confidence ≥ 0.99, split 60/20/20 by sentence.
  - The dev half of the lang-uk benchmark, human gold, weighted ×4.
- **Features:** computed by running every tier independently over those tokens.
- **Model selection:** the hidden size and weight decay were chosen on dev (cv + lang-uk). τ was then chosen as the largest dev gain on Common Voice that costs lang-uk no more than 0.3 points.
- **Hardware:** CPU, about 1 min.
- **Reproducibility:** retrained from the same features, it is byte-identical to the released `model.pt`.

## Evaluation

Held-out test splits, the tier order against the combiner:

| test | tier order | combiner |
| --- | ---: | ---: |
| Common Voice test (3,001 tokens) | 92.74% | 94.77% |
| its top-200 forms (2,058) | 96.06% | 97.57% |
| lang-uk test half (748) | 83.96% | 83.96% |

## Limitations

- **Tied to the other models.** It was trained on the outputs of these exact tier models. A different cross-encoder or classifier shifts its features and needs a retrained combiner.
- **Modern-text gold.** It helps on audio-gold tests. On the project's small modern-text gold it scored 97.27% against the tier order's 98.07%, so it ships switched off.
- **Features at request time.** It needs every tier's opinion, which the service computes once per request and caches.
