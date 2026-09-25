# Coverage, limitations, and attribution

## What is extracted

Ukrainian-language sections are identified by heading markers in
`etl/src/ukstress/language_sections.py`
(`{"українська", "українська мова", "uk", "{{-uk-}}", "{{uk}}"}`); text
under any other language heading is never extracted, even from a page that
also has a Ukrainian section elsewhere.

Within a Ukrainian section, `etl/src/ukstress/wikicode.py` recognizes:

- **Explicit headword templates**: `uk-noun`/`uk-іменник` (noun),
  `uk-adj`/`uk-прикметник` (adjective), `uk-verb`/`uk-дієслово` (verb),
  `uk-pron` (pronoun), `uk-num` (numeral), `uk-adv` (adverb).
- **Declension/conjugation templates**: `uk-decl`, `uk-conj`, `uk-імен`,
  `uk-відм`, `uk-діє`, `uk-pron-decl`, `uk-num-decl`, `uk-part-decl`
  (participle), `uk-adv`, `uk-cmpr` (comparative), `uk-supr`
  (superlative).
- **Manually written wiki tables**, including `<br>`-, comma-, and
  slash-separated cells (`table_parser.py`).
- **Bold headwords** as a fallback when no recognized template is present
  (`extract_bold_headwords`).
- **A controlled, reduced-confidence fallback** for text that looks like a
  stressed Ukrainian word but matches none of the above
  (`extract_controlled_fallback`, confidence 0.5 — always below any
  template-derived confidence, so it never wins a tie in `stress_lookup`'s
  ranking).
- **Redirects**, recorded as aliases pointing at their target, never as a
  stressed form in their own right (`entry_parser.redirect_alias`).

Every other template — anything not in the lists above — is **not
executed and not silently ignored**: it's recorded in
`unhandled_template` (visible via `ukstress report-unhandled`, or
`reports/unhandled_templates.json` after a parse) with its normalized
name, an occurrence count, and a representative sample page/invocation.
This is the mechanism for prioritizing what to support next; it is not a
completeness claim.

## Measured coverage (last real build)

From `output/9dacc408065a68ee-forms-v4/manifest.json`, a real run against
the full `ukwiktionary-latest-pages-articles.xml.bz2` dump:

| | |
| --- | --- |
| Pages processed | 65,346 |
| Lexemes extracted | 29,579 |
| Word forms extracted | 363,272 |
| Parse errors recorded | 6,750 (not dropped — see `reports/parse_errors.json`) |

These are counts from one specific dump snapshot, not a claim about
Ukrainian vocabulary size or about future dump snapshots.

## What this project does not claim

- **Not complete coverage of the Ukrainian language.** Ukrainian
  Wiktionary itself is incomplete and inconsistent; this project extracts
  what's structurally present in it. `openspec/config.yaml`'s
  `context.rules` states this as a hard constraint: "Never claim complete
  coverage of the Ukrainian language."
- **Not contextual (homograph) disambiguation.** `stress_lookup` returns
  every valid candidate for an ambiguous spelling and never silently picks
  one — see the `мова` example in [api.md](api.md), a real duplicate the
  API reports rather than hides. Choosing *which* stress applies in a
  specific sentence is the explicit subject of the separate, unimplemented
  `build-homograph-stress-disambiguation` openspec change.
- **Not machine-learned or inferred stress.** Every stressed form in
  `stress_lookup` traces back to an explicit template, table, or bold
  headword in the source dump (or a flagged, lower-confidence fallback) —
  never a guess.
- **Not arbitrary optional-form expansion.** Parenthesized optional word
  parts in wikitext are not algorithmically expanded into every
  combination; this is a deliberate scope boundary, not an oversight.
- **Not a claim about which stress is "correct" when Wiktionary itself
  disagrees with itself.** The duplicate-provenance `мова` example in
  [api.md](api.md) is real: two source parses of the same word produced
  two `stress_lookup` rows with different metadata. The API surfaces both
  rather than resolving the disagreement.

## Source attribution and licensing

Every extracted record retains its page title, page ID, revision ID,
source section, and a bounded source-text fragment (`source_ref` table;
`reports.bounded_text` caps fragment length). Download source:
<https://dumps.wikimedia.org/ukwiktionary/latest/>.

Ukrainian Wiktionary content is distributed by the Wikimedia Foundation
under Creative Commons Attribution-ShareAlike (CC BY-SA) and, for some
content, GFDL — see Wiktionary's own terms at
<https://uk.wiktionary.org/wiki/Вікісловник:Авторські_права> and the
Wikimedia Foundation's terms of use for the current, authoritative
statement. This project is not a substitute for reading those terms: it
retains the provenance fields (title/page ID/revision ID/section) needed
to satisfy an attribution requirement, but whether a particular
downstream use (e.g., redistributing exports, training a model) satisfies
CC BY-SA's share-alike and attribution terms is a licensing question for
whoever performs that use to resolve, not a conclusion this document
draws on their behalf.
