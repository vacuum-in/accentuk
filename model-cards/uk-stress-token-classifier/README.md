---
language: uk
license: apache-2.0
library_name: transformers
base_model: ukr-models/xlm-roberta-base-uk
tags: [ukrainian, stress, homograph, token-classification, tts]
---

# Ukrainian stress token classifier (`tok-v5`)

Picks the stressed vowel of an ambiguous Ukrainian word from its sentence,
with no gloss. It is trained on stress *heard* in narrated audiobooks.

## Input and output

- **Input:**
  - a text window of ±300 characters around the target word, at most 160 subword tokens;
  - the character span of the word;
  - the candidate stress signatures: vowel ordinals, at most 8.
- **Model:** XLM-R-uk encodes the window. The hidden states over the word's subword tokens are mean-pooled. A two-layer head scores the 8 vowel positions, and positions that are not candidates are masked.
- **Output:** a softmax over the candidates.
- **Weights:** the encoder is `model.safetensors`, the head is `head.pt`.

```python
from ukstress_ml.token_resolver import TokenResolver   # from wiki-stress/ml
resolver = TokenResolver("uk-stress-token-classifier")
resolver.resolve([{"index": 0, "sentence": "Мені сподобалася саме ця книжка.",
                   "start": 17, "end": 21, "form": "саме", "candidates": ["0", "1"]}])
# -> [{"index": 0, "signature": <vowel ordinal>, "confidence": <probability>}]
```

**Serving gate.** `coverage.json` lists how often each form was seen in training. The resolver answers only for forms seen at least `min_seen` times (80): 461 forms. Measured against audio gold, covered forms improve sharply, while uncovered forms would fall from 89% to 64%.

## Training

- **Data:**
  - ambiguous-word occurrences in the text of 100 modern Ukrainian audiobooks;
  - the label is the stress an audio ranker heard in the narration, at confidence ≥ 0.95;
  - at most 2,000 rows per reading;
  - ambiguity decided by the served lexicon.
- **Size:** 515,161 training rows (336,189 of them ambiguous, over 11,054 forms), 50,790 dev and 106,427 test rows. Dev and test are held out by form.
- **The audio ranker:** 97.0% on held-out Common Voice speakers, 96.2–96.6% against the lexicon on book audio.
- **Settings:** 4 epochs, batch 16, lr 2e-5, max length 160, seed 17. Base encoder `ukr-models/xlm-roberta-base-uk`.
- **Hardware:** one RTX 5090, about 65 min.
- **Results:** dev 93.73%. Test 85.0% overall: 94.0% on forms seen in training, 74.5% on unseen forms.
- **Reproducibility:** retrained on 2026-09-25, dev 93.81%, test 94.1% on seen forms.

## Evaluation

As a single tier, on every form regardless of the gate:
- 90.9% on ambiguous Common Voice tokens;
- 91.9% on the top-200 forms;
- 75.7% on lang-uk.

lang-uk's sentences are built to force the rare reading, and there a classifier of frequencies does worst.

## Limitations

- **Frequency, not reading:** it learns a form's frequent reading and the contexts that go with it, so it does poorly on unseen forms and on deliberately rare readings. That is why the gate exists.
- **Label noise:** the audio labels carry the ranker's own errors, 1–4%.
- **Domain:** the books are literary prose, so conversational or technical text is less well covered.
- **Training data:** the book texts and audio are commercial works. They are not redistributed, and only the weights are released. Check that this fits your use before redistributing the weights.
