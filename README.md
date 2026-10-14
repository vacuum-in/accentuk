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
| **TTS Verbalizer** | span-to-span Marian, CTranslate2 INT8 (`54 MB`, 4-10 ms/sentence) | the same | one model for both modes |
| **Morphology Tier** | spaCy small (`uk_core_news_sm`, ~15 MB) | spaCy TRF (`uk_core_news_trf_380`, 380 MB) | small CPU tagger |
| **Total Model Disk** | **374 MB** (+ 54 MB with `--tts`) | **~1,680 MB** (+ 380 MB spaCy) | **74.7% disk reduction** |
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

# Light mode: span verbalization + Light stress marking
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

The default verbalizer (from 0.3) is [`aloudreader/marian-uk-verbalizer-spans-ct2-int8`](https://huggingface.co/aloudreader/marian-uk-verbalizer-spans-ct2-int8),
a 52.7M-parameter Marian model that reads **one span at a time**: the parts of a sentence that may need reading
(numbers, dates, money, units, Roman numerals, abbreviations, Latin words, symbols, footnote marks) are found from the
shape of the characters (`accentuk.spans`), the model sees the sentence with one span marked and writes only that
span, and everything outside the spans is copied from the input. Ordinary prose is never rewritten, and footnote
marks ("Жид[93]") are not read. Spoken match (case and punctuation are not heard):

| test | span Marian (0.3) | Gemma-3-270M (0.2) | whole-sentence Marian E54 |
|---|---:|---:|---:|
| held-out splits of [`skypro1111/uk-text-normalization`](https://huggingface.co/datasets/skypro1111/uk-text-normalization), 3,619 rows | 89.0% | ~90.2% | 89.5% |
| 1,102 audiobook sentences with digits, Latin, abbreviations or symbols (references by Qwen3.8-27B) | **70.1%** | 58.9% | 62.6% |
| audiobook sentences with a word outside the spans altered | **0.45%** | 8.1% | 5.0% |
| time per sentence, CPU | 4-10 ms | 72 ms | 10 ms |

Known weak spots: numbers written with irregular grouping ("1111 386,40" is read as two numbers), Roman numerals in
running text, and Latin words the model has not seen. The Gemma verbalizer of 0.2 still loads from an existing
bundle (`ct2_models/gemma-verbalizer-ct2-int8`); it is no longer downloaded.

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
