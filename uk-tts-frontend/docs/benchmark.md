# Benchmark

Scored on [`lang-uk/ukrainian-tts-preprocessing`](https://github.com/lang-uk/ukrainian-tts-preprocessing)
(Apache-2.0), its 1,026-sentence lexical stress dataset, **through the running
HTTP stack** — not through a harness-local reconstruction of it.

The metrics are not reimplemented. `scripts/benchmark.py` imports the
benchmark's own evaluator and hands it a stressify function, exactly as its
bundled examples do, so these numbers are comparable with any others produced by
that harness.

## Results

| metric | this pipeline | `ukrainian-word-stress` | Δ |
| --- | ---: | ---: | ---: |
| **heteronym accuracy** | **82.22%** | 64.34% | **+17.88** |
| **macro-F1 across heteronyms** | **62.47%** | 47.26% | **+15.21** |
| **sentence accuracy** | **65.01%** | 41.52% | **+23.49** |
| **word accuracy** | **94.94%** | 88.67% | **+6.27** |
| unambiguous words | 98.53% | **98.60%** | −0.07 |

1,026 of 1,026 sentences scored for both systems; none skipped for a structural
mismatch. 117 s and 132 s respectively, single client, CPU.

Ahead on four of five. The one deficit is unambiguous words, where the baseline
reads the source trie directly while this lexicon is derived from it through an
ETL that loses a little on the way — 0.07 points, roughly one word in 1,400.

**Heteronyms are the metric that matters.** Unambiguous words are a lookup that
both systems get right almost always; a heteronym is a decision, and getting one
wrong is immediately audible. The ceiling on this benchmark is 99.76% — 1,232 of
1,235 heteronym tokens have their gold reading somewhere in the lexicon — so the
remaining ~18 points are resolution quality, not missing data.

## Reproducing

```bash
git clone --depth 1 https://github.com/lang-uk/ukrainian-tts-preprocessing.git
make up                                    # the stack must be running

python scripts/benchmark.py --benchmark ../ukrainian-tts-preprocessing
python scripts/benchmark.py --benchmark ../ukrainian-tts-preprocessing --system baseline
```

`--system baseline` runs `ukrainian-word-stress` configured as the benchmark's
own example configures it (`stress_symbol="+"`, `OnAmbiguity.First`) and needs
no stack at all. `--out report.json` writes the metrics.

## What is and is not measured

This scores **stage 2 only**. The benchmark's inputs are already ordinary
Ukrainian prose, so the verbalizer has nothing to do on them, and its gold
labels are stress positions.

There is no end-to-end number, because no public dataset scores verbalization
and stress together. Verbalization is measured separately by its own project, at
82.90% exact match. Treat the two as independent stages with independent error
rates rather than multiplying them into a single figure: the stress tier is
being fed verbalizer output in production, which is not the distribution
measured here.

See [limitations.md](limitations.md) for what these numbers hide.
