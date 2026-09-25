# Contributing

## Setup

```bash
uv venv && uv pip install -e ".[service,ui,dev]"
make test          # 74 tests
make lint          # ruff
```

Tests that need the Marian checkpoint are marked `slow` and skip unless
`UKTTS_VERBALIZER_CHECKPOINT` points at one. `make test-fast` skips them
outright. Nothing in the suite needs the stress API: it is faked at the client
boundary.

## Two invariants a change must not break

**1. Chunking is lossless.** For every input,
`"".join(c.text for c in split_sentences(t)) == t`. If you touch `segment.py`,
the round-trip tests are the ones to watch — they are parameterized over the
awkward cases and a new case belongs in that list.

**2. This layer implements no verbalization or stress rules.** A chunk goes to
the model unchanged and its output comes back raw; stress is decided by the
stress API and never re-derived here. Both upstreams have this as an explicit
policy, and a second implementation of either decision will drift.

Rendering choices are a different thing and belong here — how to present what a
model decided is ours; deciding it is not.

## Style

- Comments explain *why*, not *what*. A comment that restates the line is noise;
  one that names the failure a line prevents is worth keeping.
- Prefer a test that fails for the real reason over one that asserts a shape.
- Errors name the endpoint, the path, or the setting that needs changing.
- Never return a degraded result that looks finished. Raise.

## Adding a stage

Both stages are protocols (`Verbalizer`, `Stresser`) with a passthrough
implementation. A new backend implements the protocol and is selected in the
corresponding `load()`; nothing in `pipeline.py` should need to know which one
is running.
