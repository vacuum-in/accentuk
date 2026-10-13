# CTranslate2 Conversion & Benchmark Report

- **Date:** 2026-10-07 13:33:03
- **Platform:** Linux-7.0.0-31-generic-x86_64-with-glibc2.43 (x86_64)
- **CPU:** Intel/AMD x86_64 CPU-only execution
- **Python:** 3.14.4

## Executive Summary

All 3 transformer neural models in the accentuk / uk-stress project were converted to **CTranslate2 (CT2)** in both **FP32** and **INT8** quantizations.

Key highlights:
1. **Verbalizer (Seq2Seq):** CTranslate2 INT8 is **659.9x faster** than PyTorch, reduces weights by **74.3%** (from 199.8 MB to 51.3 MB), with **100.0% exact string match** across real Ukrainian test sentences.
2. **Stress-Model (Cross-Encoder):** CT2 INT8 achieves **3.07x speedup** on CPU, cuts model size by **74.8%** (from 1060.7 MB to 267.4 MB), with **96.0% top-1 decision agreement** and a correlation $r=0.99142$.
3. **Token Model tok-v5:** CT2 INT8 achieves **5.96x speedup**, cuts model size by **74.7%** (from 419.9 MB to 106.3 MB), with **100.0% decision agreement**.

---

## 1. Model Sizes & Disk Footprint

| Model | PyTorch / Source | CT2 FP32 | CT2 INT8 | INT8 Size Reduction |
|---|---|---|---|---|
| **Verbalizer** (MarianMT Seq2Seq) | 199.8 MB | 200.4 MB | **51.3 MB** | **-74.3%** |
| **Stress Model** (XLM-R Cross-Encoder) | 1060.7 MB | 1058.4 MB | **267.4 MB** | **-74.8%** |
| **tok-v5** (XLM-R Token Classifier) | 419.9 MB | 417.6 MB | **106.3 MB** | **-74.7%** |
| **TOTAL** | **1680.4 MB** | **1676.4 MB** | **425.0 MB** | **-74.7%** |

---

## 2. Inference Speed & Throughput on CPU

| Model | Backend / Precision | Load Time (s) | Latency / Unit (ms) | Throughput (units/s) | Speedup vs PyTorch |
|---|---|---|---|---|---|
| **Verbalizer** | PyTorch FP32 | 0.35s | 2885.9 ms/sent | 0.35 sent/s | 1.00x |
| **Verbalizer** | CT2 FP32 | 0.241s | 15.89 ms/sent | 62.95 sent/s | **181.67x** |
| **Verbalizer** | CT2 INT8 | 0.047s | 4.37 ms/sent | 228.66 sent/s | **659.9x** |
|---|---|---|---|---|---|
| **Stress Model** | PyTorch FP32 | 0.502s | 100.17 ms/target | 9.98 targets/s | 1.00x |
| **Stress Model** | CT2 FP32 | 0.792s | 77.56 ms/target | 12.89 targets/s | **1.29x** |
| **Stress Model** | CT2 INT8 | 0.27s | 32.64 ms/target | 30.63 targets/s | **3.07x** |
|---|---|---|---|---|---|
| **tok-v5** | PyTorch FP32 | 0.035s | 31.25 ms/target | 32.0 targets/s | 1.00x |
| **tok-v5** | CT2 FP32 | 0.298s | 14.41 ms/target | 69.38 targets/s | **2.17x** |
| **tok-v5** | CT2 INT8 | 0.127s | 5.24 ms/target | 190.75 targets/s | **5.96x** |

---

## 3. Prediction Quality & Parity Evaluation

### Verbalizer Output Parity
- **CT2 FP32 Exact Match:** **100.0%**
- **CT2 INT8 Exact Match:** **100.0%**

Sample verbalization comparison:
- **Input:** `У 1991 році Україна проголосила незалежність.`
  - *PyTorch:* `У тисяча дев'ятсот дев'яносто першому році Україна проголосила незалежність.`
  - *CT2 INT8:* `У тисяча дев'ятсот дев'яносто першому році Україна проголосила незалежність.`
  - *Status:* **MATCH**
- **Input:** `Вона купила 25 яблук за 150 грн.`
  - *PyTorch:* `Вона купила двадцять п'ять яблук за сто п'ятдесят гривень.`
  - *CT2 INT8:* `Вона купила двадцять п'ять яблук за сто п'ятдесят гривень.`
  - *Status:* **MATCH**
- **Input:** `Температура була -5°C, а вологість 85%.`
  - *PyTorch:* `Температура була мінус п'ять градусів Цельсія, а вологість вісімдесят п'ять відсотків.`
  - *CT2 INT8:* `Температура була мінус п'ять градусів Цельсія, а вологість вісімдесят п'ять відсотків.`
  - *Status:* **MATCH**
- **Input:** `Зустріч призначено на 14:30 12 травня 2024 року.`
  - *PyTorch:* `Зустріч призначено на чотирнадцяту тридцять дванадцять травня дві тисячі двадцять четвертого року.`
  - *CT2 INT8:* `Зустріч призначено на чотирнадцяту тридцять дванадцять травня дві тисячі двадцять четвертого року.`
  - *Status:* **MATCH**

### Stress Model Parity (on 100 Gold Heteronym Sentences)
- **CT2 FP32 vs PyTorch:**
  - Top-1 candidate match: **100.0%**
  - Logit Mean Absolute Error: `3.2e-06`
  - Logit Pearson correlation: `1.0`
- **CT2 INT8 vs PyTorch:**
  - Top-1 candidate match: **96.0%**
  - Logit Mean Absolute Error: `0.4105`
  - Logit Pearson correlation: `0.99142`

### Token Model tok-v5 Parity (on 100 Gold Ambiguous Instances)
- **CT2 FP32 vs PyTorch:**
  - Top-1 candidate match: **100.0%**
  - Logit Mean Absolute Error: `8e-07`
- **CT2 INT8 vs PyTorch:**
  - Top-1 candidate match: **100.0%**
  - Logit Mean Absolute Error: `0.12022`

---

## 4. Architecture & Technical Details

### MarianMT Tokenizer Vocabulary Alignment
The project's verbalizer tokenizer assigns token ID 0 to `<pad>` with a trained embedding, while CTranslate2's default Marian loader assumes ID 0 is `<s>` and starts decoding with a zero embedding. To guarantee identical generation without dropped first tokens, a custom `PadFirstMarianLoader` is used which:
1. Configures `decoder_start_token = '<s>'` and `spec.start_from_zero_embedding = False`.
2. Re-aligns token mappings to match the HuggingFace tokenizer expectations exactly.

### XLM-RoBERTa Cross-Encoder & Token Resolver
CTranslate2 `ctranslate2.Encoder` executes the 12-layer XLM-RoBERTa transformer backbone at INT8 quantization with high-performance vector SIMD routines (AVX2/AVX-512/VNNI). The small linear projection heads (`classifier.dense` + `classifier.out_proj` for stress-model, and `Linear(768, 768) -> GELU -> Linear(768, 8)` for tok-v5) execute instantly in PyTorch on the pooled vectors, delivering complete mathematical fidelity.
