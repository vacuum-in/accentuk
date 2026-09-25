# accentuk

**Stress marks for Ukrainian text-to-speech.** Given plain Ukrainian text,
it returns the same text with U+0301 after every stressed vowel, choosing
between the readings of heteronyms from context:

```
Замок на горі. Замок у дверях. Мої сестри.
Замо́к на горі́. Замо́к у́ две́рях. Мої́ се́стри.
```

The first замок is still wrong, and it shows the limit of a tagger and a
classifier on a sentence this short. See *Results* for how often that happens.

## What is here

| directory | what |
| --- | --- |
| [`wiki-stress/`](wiki-stress/) | the stress service: a PostgreSQL lexicon built from the Ukrainian Wiktionary dump and the `ukrainian-word-stress` dictionary (`etl/`); a Go HTTP API (`api/`) that looks every word up and asks the model service about the ambiguous ones; the model service and every training script (`ml/`) |
| [`uk-tts-frontend/`](uk-tts-frontend/) | a text frontend for TTS: numbers, dates and abbreviations spelled out by the [marian-uk-verbalizer](https://github.com/vacuum-in/marian-uk-verbalizer) model ([E54 on Hugging Face](https://huggingface.co/aloudreader/marian-uk-verbalizer)), then stress marked by the service above; Docker Compose deployment and a Gradio UI |
| [`audiotostress/`](audiotostress/) | the audio side: finds which vowel a narrator stressed, from aligned speech. Its labels trained the token classifier |
| [`model-cards/`](model-cards/) | the cards of the published models |

## Models

The trained models are on Hugging Face:

| model | role |
| --- | --- |
| [`aloudreader/uk-stress-crossencoder`](https://huggingface.co/aloudreader/uk-stress-crossencoder) | XLM-R cross-encoder: picks the sense of a homograph from its gloss |
| [`aloudreader/uk-stress-token-classifier`](https://huggingface.co/aloudreader/uk-stress-token-classifier) | XLM-R-uk classifier trained on stress heard in audiobooks: picks the stress of frequent ambiguous forms |
| [`aloudreader/uk-stress-combiner`](https://huggingface.co/aloudreader/uk-stress-combiner) | a small learned scorer over all tiers; off by default, `"combiner": true` per request |

A word goes through these tiers:
1. reviewed corrections;
2. the lexicon;
3. morphology (spaCy tagger and agreement rules);
4. the cross-encoder;
5. the token classifier;
6. the dictionary's default reading;
7. compound, suffix and positional fallbacks.

`wiki-stress/docs/architecture.md` and `wiki-stress/docs/state.md` have the details.

## Results

Measured on 2026-09-25 against the running service. Numbers are word accuracy.

| test | combiner off (default) | combiner on |
| --- | ---: | ---: |
| 200 most frequent ambiguous forms of Common Voice uk, stress heard in the recordings (13,201 tokens) | 95.88% | 97.27% |
| [lang-uk stress benchmark](https://github.com/lang-uk/ukrainian-tts-preprocessing), words / heteronyms | 95.92% / 84.62% | 96.02% / 85.28% |
| modern-text gold, 3,887 words | 98.07% | 97.27% |

On CPU a ~115-character line takes about 139 ms without the combiner and about 128 ms with it.

## Running it

The datasets are not in this repository: the lexicon database, the training
corpora, the test sets, and the reviewed corrections.

To build the lexicon:

1. Follow the quick start in [`wiki-stress/README.md`](wiki-stress/README.md). It imports the Ukrainian Wiktionary dump and the `ukrainian-word-stress` dictionary into PostgreSQL.
2. Download the models from Hugging Face.
3. Deploy with [`uk-tts-frontend/deploy`](uk-tts-frontend/deploy) and start it with `deploy/restart.sh`.

`wiki-stress/db/lexicon_archive.sh` exports and imports a built lexicon as gzip CSV files. Use it to move a lexicon between hosts.

Without the reviewed corrections the service will differ slightly from the numbers above.

## Licence

Apache-2.0 for the code in this repository (see `LICENSE`). The data sources keep their own licences:
- the Ukrainian Wiktionary is CC BY-SA 4.0;
- `ukrainian-word-stress` is MIT;
- the lang-uk resources carry their own terms.

The model cards describe what each model was trained on.
