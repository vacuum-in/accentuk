# accentuk

> The offline Python package. The Docker service (stress API, lexicon database, TTS frontend, UI) is [vacuum-in/accentuk-api](https://github.com/vacuum-in/accentuk-api).

Offline Ukrainian stress marking and TTS text verbalization powered by a memory-mapped MARISA lexicon and deep neural disambiguation.

`accentuk` runs completely locally without network calls at inference time. It offers two first-class operational modes: **Light** (CTranslate2 INT8 quantized, ~150 MB RAM, ultra-low latency) and **Full** (PyTorch FP32 + spaCy TRF + learned Combiner, maximum ensemble coverage).

---

## Operating Modes: Light vs. Full

| Metric / Capability | Light Mode (`mode="light"`) | Full Mode (`mode="full"`) | Benefit of Light |
|---|---|---|---|
| **Lexicon Lookup** | MARISA BytesTrie (`lexicon.marisa`) | MARISA BytesTrie (`lexicon.marisa`) | Zero-copy mmap (<10 µs/word) |
| **Deterministic Rules** | Prepositions, compounds, suffix analogy table | Prepositions, compounds, suffix analogy table | Identical deterministic rules |
| **Cross-Encoder Model** | CTranslate2 INT8 (`267 MB`) | PyTorch FP32 XLM-RoBERTa (`1,061 MB`) | **3.1x faster**, **74.8% smaller** |
| **Token Classifier** | CTranslate2 INT8 (`106 MB`) | PyTorch FP32 XLM-RoBERTa (`420 MB`) | **6.0x faster**, **74.7% smaller** |
| **TTS Verbalizer** | Gemma-3-270M, CTranslate2 INT8 (`126 MB`, ~70 ms/sentence) | the same | one model for both modes |
| **Morphology Tier** | spaCy small (`uk_core_news_sm`, ~15 MB) | spaCy TRF (`uk_core_news_trf_380`, 380 MB) | small CPU tagger |
| **Total Model Disk** | **374 MB** (+ 126 MB with `--tts`) | **~1,680 MB** (+ 380 MB spaCy) | **74.7% disk reduction** |
| **Runtime Memory (RSS)** | **~120 – 180 MB** | **~1,500 – 2,200 MB** | **~10x lower RAM** |
| **Recommended For** | Production TTS, embedded systems, CLI, microservices | Research, batch indexing, maximum ensemble accuracy | |

---

## Installation

```bash
pip install accentuk
```

Optional TTS verbalization support:
```bash
pip install "accentuk[tts]"
```

---

## Quick Start (Python API)

### 1. Stress Marking (`Accentor`)

```python
from accentuk import Accentor

# Light mode: fast CTranslate2 INT8 models, low memory footprint
with Accentor(mode="light") as engine:
    result = engine.stress("Старовинний замок стоїть на горі.")
    print(result.text)  # "Старови́нний за́мок стої́ть на горі́."
    for token in result.tokens:
        print(f"{token.text} -> {token.output_text} ({token.status})")

# Full mode: PyTorch FP32 models + spaCy TRF + Combiner
with Accentor(mode="full") as engine:
    result = engine.stress("Старовинний замок стоїть на горі.")
    print(result.text)

# Auto mode (default): uses Light if CT2 models exist, otherwise Full/Dictionary
with Accentor(mode="auto") as engine:
    result = engine.stress("Мої сестри.")
    print(result.text)  # "Мої́ се́стри."
```

### 2. TTS Normalization & Stress Marking (`Frontend`)

Normalizes dates, numbers, currencies, abbreviations, and symbols, then applies stress marks:

```python
from accentuk import Frontend

# Light mode: Gemma INT8 verbalization + Light stress marking
with Frontend(mode="light") as frontend:
    result = frontend.prepare("Вона купила 25 яблук за 150 грн 3 квітня 2024 року.")
    print(result.text)

# Full mode: the same verbalizer + Full stress marking
with Frontend(mode="full") as frontend:
    result = frontend.prepare("Вона купила 25 яблук за 150 грн 3 квітня 2024 року.")
    print(result.text)
```

---

## CLI Usage

### Downloading Model Assets

Download the appropriate bundle for your target environment:

```bash
# Download Light assets (~425 MB CT2 INT8 models + lexicon)
accentuk download --light

# Download Full assets (~2.2 GB PyTorch weights + spaCy TRF + lexicon)
accentuk download --full

# Download with TTS verbalizer included
accentuk download --light --tts
accentuk download --full --tts
```

### Stress Marking from CLI

```bash
# Stress plain Ukrainian text using Light mode
accentuk stress "Він відчинив замок ключем." --mode light   # Він відчини́в замо́к ключе́м.

# Stress using Full mode
accentuk stress "Він відчинив замок ключем." --mode full

# Process text from standard input
echo "Мої сестри." | accentuk stress --mode light
```

### TTS Normalization from CLI

```bash
accentuk prepare "О 7:45 ранку 3 квітня потяг № 12 прибув до Києва." --mode light
```

---

## Accuracy

Measured on 2026-10-13 with the same benchmarks as the service (`vacuum-in/accentuk-api`), word accuracy:

| test | service | full mode | light mode |
|---|---:|---:|---:|
| 200 most frequent ambiguous forms of Common Voice uk, stress heard in the recordings | 95.88% | 95.88% | 95.54% |
| modern-text gold, 3,887 words (all / reviewed lines) | 98.07% / 96.83% | 98.07% / 96.83% | 97.63% / 96.25% |
| [lang-uk stress benchmark](https://github.com/lang-uk/ukrainian-tts-preprocessing), words / heteronyms | 96.10% / 84.62% | 96.10% / 84.62% | 96.10% / 84.62% |

The lexicon `accentuk download` installs (`lexicon-v2`, built from the public `data-v2` of accentuk-api) is the
service's, including the 51,298-word dictionary of stress heard in recordings; the light-mode column was measured on a
clean download of it.

Full mode gives the service's answers. Light mode differs on 34 lang-uk tokens: 18 where the small spaCy tagger
parses differently from the transformer one, the rest from INT8 quantization (the INT8 cross-encoder picks
another reading than FP32 on 2.1% of targets; its margins track FP32 with r = 0.984, so the served threshold 0.5 is kept).

---

## Benchmark Results

Evaluated on AMD EPYC / Intel Xeon CPUs with single-thread and batched execution:

| Model Tier | PyTorch FP32 Latency | CTranslate2 INT8 Latency | Speedup | Quality Parity |
|---|---|---|---|---|
| **Verbalizer (MarianMT, before 0.2)** | 2,891 ms / sentence | **4.38 ms / sentence** | **659.9x** | 100% exact match |
| **Token Classifier (XLM-R-uk)** | 31.0 ms / target | **5.20 ms / target** | **5.96x** | 100% top-1 match |
| **Cross-Encoder (XLM-R)** | 100.2 ms / target | **32.6 ms / target** | **3.07x** | 96.0% top-1 match |

### Verbalizer

The default verbalizer (from 0.2) is [`skypro1111/gemma-3-270m-uk-verbalizer`](https://huggingface.co/skypro1111/gemma-3-270m-uk-verbalizer)
converted to CTranslate2 INT8 ([`aloudreader/gemma-3-270m-uk-verbalizer-ct2-int8`](https://huggingface.co/aloudreader/gemma-3-270m-uk-verbalizer-ct2-int8)),
run as its card prescribes: one sentence at a time, greedy, digit tokens banned, bare digit strings and long numbers
prepared by its two deterministic helpers. Exact match on the held-out splits of
[`skypro1111/uk-text-normalization`](https://huggingface.co/datasets/skypro1111/uk-text-normalization), CPU:

| split | rows | Gemma CT2 INT8 | Marian CT2 INT8 (before 0.2) |
|---|---:|---:|---:|
| 139 domains | 2,564 | **89.59%** | 88.53% |
| mixed hard sentences | 557 | **90.13%** | 86.00% |
| transliteration | 380 | 72.63% | **72.89%** |
| numeral agreement | 63 | **100.00%** | 88.89% |
| reported bugs | 22 | **95.45%** | 90.91% |
| misc | 33 | **81.82%** | 75.76% |
| time per sentence | | 72 ms | 10 ms |

The INT8 conversion gives the same outputs as the original in transformers on these splits. The Marian model was
trained on an earlier version of the same dataset, so some of these rows may have been in its training data.
Gemma is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## Disambiguation Pipeline

When a word has multiple possible accents (heteronym), `accentuk` resolves stress through sequential tiers:

1. **Memory-Mapped Lexicon**: Exact form lookup via MARISA Trie (`lexicon.marisa`).
2. **Capitalization & Proper Noun Rules**: Differentiates names and toponyms (e.g. *Розді́л* vs. *ро́зділ*).
3. **Prepositional Pronoun Shift**: Shifts accent to the first syllable after governing prepositions (e.g. *без ньо́го*, *до ме́не*).
4. **Contextual Cross-Encoder**: Evaluates sentence context against candidate senses (*aloudreader/uk-stress-crossencoder*).
5. **Token Classifier**: Direct sequence labeling trained on audiobook audio alignment (*aloudreader/uk-stress-token-classifier*).
6. **Morphology & Agreement Rules**: Grammatical tags and case/number agreements (spaCy TRF in Full mode).
7. **Dictionary Defaults & Fallbacks**: Trie defaults, compound splitting (*пшенично-житній*), and analogy-based suffix tables.

---

## License

Apache-2.0. See `LICENSE` and `NOTICE.md` for third-party component licenses and dataset attributions.
