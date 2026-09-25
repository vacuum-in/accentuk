# ukstress-ml

Contextual homograph stress disambiguation for the Ukrainian stress lexicon.

The PostgreSQL lexicon resolves every spelling that maps to a single stressed
form. This package resolves the ones that do not: it builds a sentence corpus
for ambiguous spellings and trains a Marian model to pick the stress the context
requires.

Canonicalization, stress validation, stress signatures, resumable download, and
bounded-memory dump streaming come from `ukstress` (the `etl/` package) and are
never re-implemented here — a second normalization path would let corpus
matching and dictionary lookup disagree silently.

## Pipeline

```bash
ukstress-ml inventory     # freeze senses, collapse duplicate-stress rows
ukstress-ml ambiguity     # compute the ambiguous surface to search for
ukstress-ml mine          # single streaming pass over a corpus dump
ukstress-ml label         # Azure AI Foundry sense labelling
ukstress-ml assemble      # validate, dedup, balance, split, encode
ukstress-ml train         # Marian, from scratch
ukstress-ml evaluate      # constrained candidate scoring, per-group metrics
```

## Status

See `openspec/openspec/changes/build-homograph-stress-disambiguation/` for the
specification and task list, and `PLAN_DATASET.md` for the corpus plan.
