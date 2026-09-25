# Architecture

```
source ──▶ NFC ──▶ chunk ──▶ verbalize (Marian) ──▶ stress (HTTP) ──▶ output
             │        │            │                      │
             │        │            │                      └─ one call per sentence
             │        │            └─ batched across every chunk of every input
             │        └─ lossless: concatenation is the identity
             └─ composed form, so `й` is one code point everywhere downstream
```

Five modules, each with one job:

| module | responsibility |
| --- | --- |
| `segment.py` | decide **where** to cut, never **what** the text says |
| `verbalize.py` | run the Marian checkpoint; no pre- or post-processing |
| `stress.py` | speak HTTP to the stress API; decide no stress locally |
| `pipeline.py` | order the stages, batch them, keep offsets coherent |
| `service.py` / `cli.py` / `gradio_app.py` | three front doors onto the same `Pipeline` |

## The boundary: no rules of our own

The verbalizer project's policy forbids hardcoded verbalization at runtime — no
dictionaries, no numeric tables, no output repair, no rejection. Quality is
changed by data and training. This pipeline honours it: a chunk goes to the
model unchanged and its output is returned raw.

The same discipline applies to stress. Placement is decided by the stress API,
which owns the lexicon and all four of its tiers. This repository never
re-derives a stress locally, because two implementations of the same decision
drift apart and then disagree in production.

What is left for this layer is genuinely its own: chunking, ordering, batching,
and offset bookkeeping.

## Chunking

The checkpoint was trained with a 384-token source window. A paragraph handed
over in one call is truncated, and a truncated sentence is missing words by the
time it reaches the voice — silently, with no error anywhere. So the input is
cut before inference.

Everything rests on one invariant:

```python
"".join(chunk.text for chunk in split_sentences(t)) == t   # for every t
```

Text becomes a list of `Chunk(text, speakable)`. Only `speakable` chunks reach
the model; whitespace and line breaks wait in place. Leading and trailing
whitespace is split off into its own non-speakable chunk so the model never
receives a padded string.

**Line breaks first.** Headings, list items and verse are boundaries and rarely
end in a period.

**Then sentence boundaries**, with three guards that suppress a false cut:

| guard | what it saves |
| --- | --- |
| no whitespace after the punctuation | `3.5`, `www.example.com` |
| the preceding token is a known abbreviation | `2024 р. о 14:30` stays one sentence |
| the preceding token is a single capital letter | an initial: `Т. Шевченко` |
| the next character is lowercase | a continuation, not a new sentence |

The abbreviation list is not a verbalization rule — it rewrites nothing. It only
prevents handing the model half a date, which it would then read back as a
different date.

**Then the window.** `fit_to_window` measures with the real tokenizer, not a
word-count heuristic. A sentence that still overflows is cut at clause seams
(`;` `:` `,` and spaced dashes), packed greedily under the budget, and only if a
single clause is still too long does it fall back to word boundaries. Either way
a warning names the actual length. The concatenation survives all of it.

## Pass-through

A chunk with nothing to spell out never reaches the model. This is routing, not
verbalization — it picks an owner and rewrites nothing — and it is there because
of what happens without it.

Sent every sentence unconditionally, the checkpoint alters **209 of 880** plain
Ukrainian sentences from lang-uk's benchmark: no digit, no Latin letter, nothing
to normalize. The failures are not subtle.

| input | output |
| --- | --- |
| `сидів спокійно` | `сидівох` |
| `на березі` | `від трьох до нуля` |
| `незрозумілих явищах` | `трьох цілих дев'яти десятих відсотка` |

A sequence-to-sequence model asked to copy its input will sometimes decline to.

`route.needs_verbalization()` sends a chunk when it holds a digit, a Latin
letter, a symbol a voice must read (`%`, `°`, `№`, `+`…), an acronym of two to
five capitals, or a word that is an abbreviation followed by a period. Two of
those bounds were themselves measured: an all-capitals line is a heading rather
than a run of acronyms (`ІСТОРИЧНИЙ НАРИС` came back `і ес те оРИЧНИЙ НАРИС`),
and a single capital before a period is an initial that nothing can expand
(`С. Смеречинський` came back without the initial at all).

| | altered | time |
| --- | ---: | ---: |
| no routing | 209 / 880 | 120 s |
| routing | 25 / 880 | 51 s |
| routing, both bounds fixed | **16 / 880** | **30 s** |

Most of the remaining sixteen are the router working: `напр.` → `наприклад`,
`Проф.` → `Професор`, `ст.` → `століття`, `СДА` → `ес де а`. What is left is the
model erring on text that did need it.

Set `UKTTS_ROUTE=0` to send everything and reproduce the first row.

## Batching

`prepare_many` flattens the chunks of *every* input into one list, remembers each
one's `(row, position)`, runs a single batched pass, and substitutes the outputs
back in place. Preparing a hundred short sentences costs one model pass, not a
hundred.

## One stress call per sentence

The stress API passes the whole request text to its contextual model as that
word's sentence (`api/internal/httpapi/stress.go:427`), and the cross-encoder was
trained on one sentence. Handed a paragraph it answers confidently and wrongly:

| input | reading | margin |
| --- | --- | --- |
| `Не вистачає руки.` alone | `руки́` ✓ | 0.68 |
| the same line inside four sentences | `ру́ки` ✗ | 1.56 |

Note that the margin *rises* when the answer is wrong, so the abstention
threshold does not catch it — it waves it through.

The chunker already knows where the sentences are, so the same boundaries drive
the stress calls. It is also faster: ten sentences take 1.17 s as ten calls
against 2.1 s as one blob, because each call is a small model input.

Token offsets from each call are rebased onto the whole verbalized text, so a
caller sees one coherent set of spans no matter how many calls it took.

## Failure behaviour

A stress outage is an error, not a silent degradation. `StressUnavailable`
propagates: the CLI exits 2 naming the endpoint, the service returns 503. It
never returns unstressed text as if it were finished, because a caller cannot
tell the difference by looking at it.

The same principle covers the verbalizer: a missing or unreadable checkpoint
fails at construction, so a bad path breaks the deployment rather than a user's
request. A backend that returns the wrong number of outputs raises rather than
misaligning chunks.

## Unicode

The pipeline normalizes to NFC once, at the entrance. The stress lexicon stores
NFD, where `й` is `и` plus a combining breve and `ї` is `і` plus a diaeresis; the
composed and decomposed forms have different lengths, and walking one while
indexing the other produces doubled characters. Normalizing at a single known
point keeps every offset in this layer meaning the same thing.

Note also that `й` counts as a vowel ordinal in the lexicon's signature
convention but is not a syllable — counting it makes `той` look disyllabic.
