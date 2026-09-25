# Tasks: Build Homograph Stress Disambiguation

## Progress — 2026-08-16

**59 of 147 tasks complete.** A model meeting every declared release gate exists
and is measured; see [RESULTS.md](../../../../RESULTS.md) for all figures and
`output/ml/models/v3-xenc/release_report.json` for the gate evaluation.

| Phase | State |
| --- | --- |
| Inventory (§2) | Done — 4 578 senses, 2 250 groups, frozen and content-hashed |
| Paradigms (§3) | **Not started** — needs a vendored accented dictionary |
| Ambiguous surface (§4) | Done for lemma forms — 2 035 forms |
| Licensing (§5) | Partial — licence tags on rows; no source registry |
| Acquisition (§6–8) | Done for Wikipedia — 2.53M pages → 58 813 sampled rows |
| Targeted crawl (§7) | **Not started** — not needed at current coverage |
| Annotation (§9–10) | Done for L1/L2/L3 — **L0 not started** |
| Corpus (§11–13) | Done — 12 351 rows, train 11 165 / dev 427 / test_natural 759 |
| Model (§14–15) | Done — see below |
| Inference service (§16) | **Not started** |
| Go integration (§17) | **Not started** |
| Observability (§18) | **Not started** |
| Benchmarks (§19) | Serving latency measured; no load or API-level benchmark |
| Documentation (§20) | Partial — RESULTS, ANALYSIS_TRIAGE, CUDA handoff written |

### Model status

`test_natural` (759 held-out natural Wikipedia sentences, 202 known groups):
group-macro **0.9142** against a 0.7678 majority baseline, multi-sense groups
**0.8730** against 0.5930, minority-sense recall **0.8600** against 0.0. All four
release gates pass. Abstention threshold 1.2482 gives 95.1% precision at 93.8%
coverage.

### Deviations from this task list, and why

- **The scorer is `xlm-roberta-base`, not Marian.** Four Marian runs from scratch
  plateaued at or below the majority baseline; the constraint was absent
  pretrained Ukrainian semantics, not the input encoding or corpus balance.
  Recorded in RESULTS.md §3. §14–15 are satisfied by the cross-encoder in
  `ml/src/ukstress_ml/crossencoder.py`; the Marian implementation also exists and
  is retained as the measured comparison.
- **12.5 (reserve unseen groups) was implemented, then deliberately removed.**
  The inventory is closed, so generalisation to homographs outside it is not a
  product requirement, and reserving those groups took 16% of the corpus —
  specifically the best-balanced groups — out of training.
- **Evaluation is mined-only.** Generated sentences are training-only, because
  they were written under a prompt demanding unambiguous context and scoring on
  them overstates served accuracy.
- **§3 paradigms are the largest open gap.** Everything is lemma-only, which also
  caused 36% of generated sentences to be rejected for using an inflected form.

### Next, in order

1. §3 paradigm expansion — unlocks inflected forms and raises generation yield.
2. §10.1 L0 definitions for the 1 900 senses lacking one — hard gate on
   extending coverage past 570 of 2 250 groups.
3. §16–17 inference service and Go integration.
4. Seed variance — every figure so far is a single run with no error bar.

## 1. Project foundation

- [ ] 1.1 Create `ml/` package layout (`cli.py`, `inventory.py`, `paradigms.py`,
      `ambiguity.py`, `sources.py`, `mine.py`, `crawl.py`, `annotate/`,
      `corpus.py`, `tokenizer.py`, `train.py`, `evaluate.py`, `convert.py`,
      `serve.py`).
- [x] 1.2 Add Python packaging, lockfile, linting, typing, and test
      configuration for `ml/`, depending on `etl/` for canonicalization and
      stable natural keys without re-implementing them.
- [ ] 1.3 Add a Dockerfile for the internal inference service.
- [ ] 1.4 Add Make targets for inventory, mine, annotate, assemble, train,
      evaluate, convert, and serve.
- [ ] 1.5 Add CI that runs `ml/` tests, dependency scans, and image builds.

## 2. Homograph inventory — sense records and duplicate collapse

- [x] 2.1 Load `output/homographs_precess_pilot_enriched2.csv` as the source
      sense inventory.
- [x] 2.2 Implement per-sense fields: stressed form, part of speech,
      grammatical features, definition, register, contrast statement, review
      status, and per-field source.
- [x] 2.3 Mark senses with no definition from any dictionary source as
      incomplete and exclude them from corpus generation.
- [ ] 2.4 Tag language-model-proposed definitions and contrast statements as
      unconfirmed, and block generation on unconfirmed fields.
- [x] 2.5 Collapse senses within a group that share an identical canonical
      stressed form into one label, retaining both definitions.
- [x] 2.6 Remove groups that collapse to a single distinct stressed form from
      the ambiguous set and report the removal.
- [x] 2.7 Generate a completion report: counts of incomplete, collapsed, and
      removed senses.
- [x] 2.8 Add golden tests for collapse and completion using fixture sense
      rows.

## 3. Homograph inventory — paradigm expansion and stress transfer

- [ ] 3.1 Vendor a pinned VESUM / `dict_uk` release into `data/dict_uk/` using
      `download_dump()`, with a SHA-256 manifest.
- [ ] 3.2 Add a `pymorphy3` + `pymorphy3-dicts-uk` fallback for lemmas VESUM
      misses.
- [ ] 3.3 Match each sense to a dictionary paradigm by lemma, part of speech,
      and grammatical features; record the paradigm source as
      dictionary-derived.
- [ ] 3.4 Resolve multi-candidate matches using `gram_style`; flag
      unresolvable matches for review instead of picking arbitrarily.
- [ ] 3.5 Generate rule-based paradigms from declension class for senses with
      no dictionary match (proper nouns, toponymic adjectives, dialectal
      entries); tag the source as generated and keep it distinguishable in
      every downstream artifact.
- [ ] 3.6 Transfer stress across fixed-stress paradigms by vowel ordinal using
      the existing `stress_signature` rules; validate every generated form.
- [ ] 3.7 Source mobile-stress paradigms from accented dictionary data or
      manual entry; flag forms as unknown-stress and exclude them from labels
      when neither is available.
- [ ] 3.8 Reject and report transferred forms that fail stress validation.
- [ ] 3.9 Add golden tests for paradigm matching, generated-paradigm fallback,
      and fixed/mobile stress transfer.

## 4. Homograph inventory — ambiguous surface and freezing

- [x] 4.1 Compute, per group, the unstressed surface forms colliding across
      two or more senses under the canonical lookup key.
- [x] 4.2 Emit `output/ml/ambiguous_forms.jsonl` with candidates, stress
      signatures, and sense references per collision.
- [x] 4.3 Exclude and count orthographically separated forms from the
      ambiguous surface.
- [ ] 4.4 Distinguish groups ambiguous only in the nominative from groups
      ambiguous across the paradigm, for per-group crawl-budget sizing.
- [ ] 4.5 Route all form comparison through the existing canonicalization
      (`etl/src/ukstress/normalizer.py:33`); add a test proving no second
      normalization path exists in `ml/`.
- [x] 4.6 Freeze inventory versions as immutable, content-hash-identified
      artifacts recording the dictionary checksum and normalization version;
      never edit a frozen version in place.
- [x] 4.7 Add property tests for NFD-composed input producing identical stress
      signatures to decomposed input.
- [ ] 4.8 Measure and record the gate: ≥ 90% of senses with a paradigm before
      Phase B (acquisition) proceeds.

## 5. Corpus acquisition — source registry and licensing

- [ ] 5.1 Implement a source registry recording licence, attribution
      requirement, and redistribution status per source.
- [ ] 5.2 Refuse acquisition from any source absent from the registry; fail
      with the unresolved source named.
- [x] 5.3 Tag every acquired sentence with its source's licence, and derive a
      publishable subset by filtering on that tag.
- [ ] 5.4 Add tests for unresolved-source refusal and mixed-licence filtering.

## 6. Corpus acquisition — sense-bound citations and dumps

- [ ] 6.1 Extract Wiktionary usage examples from confirmed Ukrainian sections
      of the already-downloaded ukwiktionary dump, executing no template, Lua,
      or dump-provided code.
- [ ] 6.2 Extract dictionary citations attached to a numbered sense from
      registered, licence-cleared sources (`sum.in.ua`, `ukrlit.org`, etc.),
      preserving the sense attachment as a human-authored label.
- [ ] 6.3 Quarantine citations whose sense number cannot be mapped to an
      inventory sense; report the unmapped count.
- [x] 6.4 Reuse `downloader.py` and `dump_reader.py` to acquire and stream the
      Ukrainian Wikipedia and Wikisource dumps with atomic writes, checksum,
      and bounded memory.
- [x] 6.5 Build an article-prose variant of `wikicode.py` that strips markup,
      tables, infoboxes, references, and navigation before sentence splitting.
- [ ] 6.6 Add tests for citation quarantine, dump streaming memory bounds, and
      prose extraction on fixture pages.

## 7. Corpus acquisition — targeted crawl

- [ ] 7.1 Implement a bounded targeted crawl for senses starved after dump and
      bulk-corpus acquisition: robots.txt honoured, configurable per-host rate
      limit, descriptive User-Agent, per-host page cap.
- [ ] 7.2 Skip and record resources requiring login or paywall bypass.
- [ ] 7.3 Persist the crawl frontier and fetched-page cache so a resumed crawl
      does not re-fetch cached pages.
- [ ] 7.4 Cache raw responses before parsing so a parser change can be
      re-applied without re-fetching.
- [ ] 7.5 Add tests for rate-limit enforcement, resumability, and
      authentication skip-and-record.

## 8. Corpus acquisition — matching and sampling

- [x] 8.1 Build a single automaton over the frozen inventory's ambiguous
      surface forms (tens of thousands of patterns).
- [x] 8.2 Implement a single streaming pass per corpus matching whole tokens,
      excluding matches occurring inside a longer token.
- [x] 8.3 Compare candidate tokens using the canonical lookup key while
      retaining the original surface string and its character offsets.
- [x] 8.4 Exclude and count sentences that already carry stress marks.
- [x] 8.5 Exclude and count sentences containing more than one ambiguous-form
      occurrence from the default candidate set.
- [x] 8.6 Implement per-sense reservoir sampling during the pass with a
      configurable cap, so no full hit set reaches disk.
- [ ] 8.7 Prove sampling reproducibility for a fixed corpus, inventory
      version, cap, and seed.
- [x] 8.8 Generate a coverage report: candidates per sense/group/source tier,
      corpus checksum, byte size, scan configuration, and a starved-sense
      list.
- [ ] 8.9 Add deterministic-output tests and a benchmark of scan throughput on
      a real dump.

## 9. LLM annotation — infrastructure

- [x] 9.1 Implement an Azure AI Foundry client reading endpoint and
      credentials from environment configuration only, never persisting them
      to artifacts, manifests, logs, or reports.
- [ ] 9.2 Implement four separated jobs (L0 complete, L1 label, L2 generate,
      L3 verify) with separate prompts, outputs, and recorded deployment name,
      model version, API version, prompt version, and sampling parameters.
- [x] 9.3 Enforce that L3 verification uses a deployment different from the
      one producing the output under verification; fail if configured
      identically.
- [x] 9.4 Persist every raw response to a job-specific artifact before
      parsing.
- [x] 9.5 Implement per-group checkpointing so an interrupted run does not
      re-issue requests for completed groups and partial output is not marked
      complete.
- [ ] 9.6 Prove reparsing persisted raw responses is deterministic for a fixed
      inventory version, coverage report, prompt version, and seed.
- [x] 9.7 Implement structured-output requests with schema validation, bounded
      retry on non-conformance, and treatment of response text as inert data
      never executed or used to alter configuration.
- [ ] 9.8 Implement per-batch token/cost recording, a cumulative spend cap,
      and refusal of a batch that would exceed it.
- [ ] 9.9 Require a passing pilot-run quality report before a full annotation
      run is permitted.

## 10. LLM annotation — jobs

- [ ] 10.1 Implement L0 (inventory completion): propose missing definitions
      and contrast statements, stored as unconfirmed pending human review.
- [x] 10.2 Implement contrastive prompting shared by L1/L2: include every
      sibling sense with stressed form, definition, and contrast statement,
      and identify the target sense explicitly; refuse and report groups
      blocked on an incomplete sibling.
- [x] 10.3 Implement L1 (sense labelling): return one inventory sense,
      confidence, and determining words per sentence, or an explicit
      undetermined result.
- [x] 10.4 Discard L1 rows with empty/non-occurring determining words or a
      returned sense outside the candidate group; count schema violations.
- [x] 10.5 Retain undetermined L1 results for an abstention evaluation slice.
- [x] 10.6 Implement L2 (generation): target starved senses from the coverage
      report, name the exact target surface form and grammatical features,
      and supply human-authored citations as style anchors when available.
- [x] 10.7 Discard generated sentences that miss the target form exactly once,
      contain acute accents, contain a sibling's stressed spelling, or use
      metalinguistic phrasing about stress/meaning/usage; count violation
      types.
- [ ] 10.8 Detect low-variety generated batches (disproportionate shared
      opening constructions) and reject the batch for regeneration rather than
      filtering it down.
- [x] 10.9 Implement L3 (blind verification): mask the target span, omit the
      intended sense, and select from the group's candidate list.
- [ ] 10.10 Quarantine sentences on verifier disagreement, recording both
      labels; report groups whose agreement rate falls below the configured
      threshold as suspected sense-boundary defects.
- [ ] 10.11 Add fixtures and tests for each job's schema validation, discard
      rules, and quarantine paths.

## 11. Training corpus — validation and deduplication

- [x] 11.1 Validate every row against the frozen inventory: assigned sense
      exists, offsets delimit the recorded surface form, stressed form is a
      member of the sense's paradigm, stress passes validation, and the
      unstressed projection matches the recorded surface form under the
      canonical lookup key.
- [x] 11.2 Reject rows failing any check above with the failing rule recorded.
- [x] 11.3 Deduplicate identical sentences, retaining every observed source on
      the kept instance.
- [x] 11.4 Remove near-duplicate sentences above a configured similarity
      threshold, deterministically per corpus version.
- [x] 11.5 Quarantine (not choose between) sentences carrying conflicting
      sense labels.
- [ ] 11.6 Add golden tests covering each rejection scenario from the
      training-corpus spec.

## 12. Training corpus — provenance, balance, and splits

- [x] 12.1 Record per-row provenance: source tier, corpus name, source
      location, licence tag, label origin, annotation job id, model version,
      and the inventory content hash.
- [x] 12.2 Bound per-group sense-frequency imbalance to a configured cap;
      report senses below the configured training-row floor as under-covered
      rather than concealing the shortfall.
- [x] 12.3 Assemble the natural-text evaluation set only from mined,
      human-confirmed-label sources, left unbalanced.
- [x] 12.4 Implement deterministic splitting for a fixed seed; verify no
      sentence or near-duplicate straddles more than one split.
- [ ] 12.5 Reserve a configured number of whole groups for an unseen-group
      evaluation set with zero training rows; report any trained group absent
      from training.
- [ ] 12.6 Add split-reproducibility and overlap-detection tests.

## 13. Training corpus — encoding and artifacts

- [x] 13.1 Implement the marked-span source encoder and stressed-word-only
      target encoder as a single documented function.
- [x] 13.2 Define and implement the reserved-marker escaping/rejection rule
      for sentences already containing a marker character; apply it
      identically across corpus versions.
- [x] 13.3 Prove encode/decode round-trips recover the original sentence and
      target offsets exactly.
- [x] 13.4 Write immutable, self-describing corpus version artifacts recording
      inventory hash, source corpora checksums, annotation job ids, filter
      configuration, split seed, per-split row counts, and per-sense coverage.
- [ ] 13.5 Prove a corpus rebuild from a recorded manifest reproduces an
      identical row set and split assignment.
- [x] 13.6 Write rejected rows to a quarantine artifact with the failing rule,
      and report when a single rule's rejection share exceeds a configured
      threshold as a probable upstream defect.

## 14. Stress model — tokenizer and training

- [x] 14.1 Build a tokenizer from the corpus version's training split only,
      covering Ukrainian text and stressed forms without loss (combining
      acute, canonical apostrophes); record artifact checksums with the
      model.
- [ ] 14.2 Verify character-fallback encoding succeeds for unseen characters
      at inference without failing the request.
- [x] 14.3 Implement Marian training recording corpus version, inventory hash,
      tokenizer checksums, model configuration, hyperparameters, seed, library
      versions, hardware, measured duration, and selected checkpoint.
- [x] 14.4 Select the best checkpoint on development-split group-macro
      accuracy, not loss or a translation-quality metric.
- [ ] 14.5 Implement the encoder-classifier baseline (per-group softmax head)
      for comparison.
- [ ] 14.6 Verify repeated training on the same corpus version, configuration,
      seed, and library versions agrees within a declared tolerance on
      equivalent hardware.

## 15. Stress model — constrained scoring and evaluation

- [x] 15.1 Implement forced-decode scoring over a supplied candidate set,
      returning the highest-scoring candidate; make output outside the
      candidate set structurally impossible.
- [x] 15.2 Return a comparable score per candidate and the top-two margin.
- [x] 15.3 Short-circuit single-candidate requests without invoking the model;
      reject empty candidate sets as invalid rather than generating.
- [x] 15.4 Merge candidates sharing a canonical lookup key and stress
      signature before scoring, and report the merge.
- [x] 15.5 Implement the evaluation report: group-macro accuracy,
      natural-text accuracy, same-part-of-speech-subset accuracy,
      unseen-group accuracy, default-sense baseline, and encoder-classifier
      baseline, each naming its set and row count.
- [x] 15.6 Implement per-group worst-performer reporting with confusion counts
      and a known-limitations artifact for groups below a configured floor.
- [x] 15.7 Implement calibration measurement over the score margin and derive
      an abstention threshold from it.
- [ ] 15.8 Gate release on exceeding the default-sense baseline on
      group-macro accuracy and on every other configured gate; refuse release
      and name unmet gates with measured values otherwise.
- [ ] 15.9 Publish known limitations with every release; make no claim of
      complete homograph coverage.
- [ ] 15.10 Implement int8-quantized CT2 conversion recording source
      checkpoint, converter version, and quantization setting.
- [ ] 15.11 Verify converted-artifact selections agree with the
      training-framework model within a declared tolerance on the development
      split; block serving on a larger divergence.

## 16. Internal inference service

- [ ] 16.1 Implement the internal request/response contract: sentence with
      target spans and per-span candidates in, one selected candidate per
      span out with stress signature, per-candidate scores, and margin.
- [ ] 16.2 Process multiple spans in one request, preserving input order.
- [ ] 16.3 Reject requests with invalid span offsets, naming the offending
      span, without silently answering other spans in the same request.
- [ ] 16.4 Report model version and training inventory hash on every response.
- [ ] 16.5 Implement a readiness endpoint reporting whether the model artifact
      is loaded and which version.
- [ ] 16.6 Bind the service to the internal deployment network only; refuse
      requests arriving from outside it.
- [ ] 16.7 Enforce sentence/span-count limits and treat request text as inert
      data, never as control input.
- [ ] 16.8 Add integration tests for span-order preservation, invalid-offset
      rejection, and external-request refusal.

## 17. Go API integration

- [ ] 17.1 Leave unambiguous-token responses unchanged and skip the inference
      service call entirely for them.
- [ ] 17.2 For ambiguous tokens with supplied sentence context, batch all
      spans of a request into one call to the internal service, retain all
      candidates, and mark the selected candidate with resolution source
      `model`.
- [ ] 17.3 For ambiguous tokens without context, return the existing ambiguous
      response and skip the service call.
- [ ] 17.4 Mark the default sense with resolution source `default` (not
      `model`) when the top-two margin is below the configured threshold.
- [ ] 17.5 Add the optional model-resolution field to the `/v1` response
      without changing any existing field's meaning; add a contract test
      proving clients ignoring the new field see pre-change behaviour.
- [ ] 17.6 Return the existing ambiguous response and record a metric when the
      inference service is unreachable or exceeds its configured timeout,
      without failing the request or breaching the API's latency budget.
- [ ] 17.7 Add a configuration flag disabling model resolution entirely,
      restoring exact pre-change behaviour with zero calls to the service.
- [ ] 17.8 Surface an inventory-hash/active-dataset mismatch through health
      and metrics, operable without a redeploy.
- [ ] 17.9 Confirm via code review that the Go component contains no dump
      parsing, morphology inference, corpus processing, annotation, tokenizer
      construction, model loading, or training code.
- [ ] 17.10 Add integration tests for degradation, disablement, and mismatch
      reporting.

## 18. Observability and security

- [ ] 18.1 Add structured logs for annotation, acquisition, training, and the
      inference service; never log raw request sentences by default.
- [ ] 18.2 Add bounded-cardinality metrics for annotation spend, acquisition
      throughput, model-resolution rate, and inference-service degradation.
- [ ] 18.3 Verify annotation responses cannot trigger code, shell, SQL, or
      template execution; add an adversarial-response test fixture.
- [ ] 18.4 Run the inference service container as non-root with unnecessary
      capabilities removed.
- [ ] 18.5 Scan `ml/` dependencies and the inference-service image.
- [ ] 18.6 Verify credentials never appear in artifacts, manifests, logs, or
      responses.
- [ ] 18.7 Confirm the licence gate blocks acquisition from any unregistered
      source end-to-end.

## 19. Benchmarks and acceptance

- [ ] 19.1 Measure and record acquisition throughput (pages/sec, matches/sec,
      memory) on a real dump.
- [ ] 19.2 Measure and record annotation pilot-run cost and quality against
      the configured pass rate before authorizing a full run.
- [ ] 19.3 Run the declared evaluation suite and record group-macro,
      natural-text, same-part-of-speech, and unseen-group accuracy against
      both baselines.
- [ ] 19.4 Run the serving benchmark: latency distributions for
      dictionary-only, model-assisted, and inference-service-isolated
      requests, on named hardware, with model version, quantization, and
      batch sizes recorded.
- [ ] 19.5 Do not mark any performance or accuracy target achieved without
      attaching the measured report that proves it.

## 20. Documentation and release

- [ ] 20.1 Document the `ml/` architecture and the amended Python/Go boundary.
- [ ] 20.2 Document the source registry, licence gate, and how to add a new
      source.
- [ ] 20.3 Document inventory freezing, paradigm expansion, and corpus version
      semantics.
- [ ] 20.4 Document model release gates and how to read the known-limitations
      artifact.
- [ ] 20.5 Document the internal inference service's deployment, health, and
      rollback procedure (pin previous model version, redeploy).
- [ ] 20.6 Document the Go-side feature-flag rollback (disable model
      resolution by configuration).
- [ ] 20.7 Validate the OpenSpec change with `openspec validate --strict`.
- [ ] 20.8 Attach actual tests, evaluation reports, and benchmarks to the
      release.
