# OpenSpec package: Ukrainian Stress Lexicon

This package contains OpenSpec changes for a hybrid implementation:

- Python: all dump processing, linguistic logic, database creation, import,
  publication, reports, and exports.
- Go: only the read-only PostgreSQL HTTP lookup API.

## Change IDs

```text
build-ukrainian-stress-lexicon
build-homograph-stress-disambiguation
```

`build-ukrainian-stress-lexicon` builds the dictionary: every stressed form that
can be extracted from Ukrainian Wiktionary, served by exact lookup.

`build-homograph-stress-disambiguation` resolves what the dictionary cannot: for
2 250 spellings that carry several stresses, it builds a crawled and annotated
sentence corpus and trains a Marian model to select the stress that the context
requires. It takes into scope two items the first change listed as out of scope
(machine-learning stress prediction, and contextual homograph disambiguation),
and amends the architecture boundary to allow an **internal** Python inference
service. Go remains the only public HTTP interface.

Apply them in order. The second change depends on the lexicon's canonicalization
rules, dump streaming, and lookup projection.

## Validate

From a repository with OpenSpec installed:

```bash
openspec validate build-ukrainian-stress-lexicon --strict
```

```bash
openspec validate build-homograph-stress-disambiguation --strict
```

Depending on the installed OpenSpec command profile, the equivalent OPSX
validation or status command may also be available.

## Apply

```text
/opsx:apply build-ukrainian-stress-lexicon
```

```text
/opsx:apply build-homograph-stress-disambiguation
```

Review `proposal.md`, capability specs, `design.md`, and `tasks.md` before
implementation.
