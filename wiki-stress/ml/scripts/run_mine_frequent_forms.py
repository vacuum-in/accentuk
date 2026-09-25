"""Give the model the most frequent ambiguous forms, which the gloss inventory cannot.

The serving manifest is built from forms that have Wiktionary sense entries. The
most frequent ambiguous words in Ukrainian text do not: `вони`, `тому`, `яка`,
`вона`, `була` are ambiguous in the stress trie but carry no senses, so they are
outside the model's coverage entirely. Measured over 4.5M words of running text,
only **14 of the 100 most frequent ambiguous forms** are in the manifest, and
139 of the top 200 are missing — 106,788 occurrences.

Two facts make them recoverable without a dictionary:

* the trie holds their readings, so the *signatures* are known for all 139;
* 112 of the 139 are `grammatical`, meaning the tags separate the readings, so a
  morphological parse can say which one applies.

So this mines real contexts and labels them with the parser, which is the step
RUAccent describes for its own third version. Labels inherit the tagger's
errors; `resolve()` returning None is treated as no label rather than a guess,
and the yield per form is reported so a form the parser cannot handle is visible
rather than silently thin.

Output feeds the signature classifier, which needs only signatures — the pair
model cannot use these forms at all, because it scores against definitions that
do not exist.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)


def signature_of(form: str, accents: tuple[int, ...], vowels: str) -> str | None:
    """Convert the trie's character positions into the lexicon's vowel ordinal.

    The trie stores where the acute goes; the lexicon numbers vowels. `й` counts
    as a vowel ordinal even though it is not a syllable — verified against the
    manifest, where counting it reproduces 29,710 of 30,712 stressed spellings.
    """
    if len(accents) != 1:
        return None                      # several accents is free variation
    composed = unicodedata.normalize("NFC", form)
    position = accents[0]
    if not (0 < position <= len(composed)):
        return None
    seen = -1
    for index, ch in enumerate(composed):
        if ch.lower() in vowels:
            seen += 1
            if index == position - 1:
                return str(seen)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=Path,
                        default=Path("output/ml/ambiguous_frequency.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_v22.json"))
    parser.add_argument("--corpus", type=Path,
                        default=Path("output/ml/silver_mined_v10.json"))
    parser.add_argument("--spacy", default="models/uk_core_news_sm")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--per-form", type=int, default=200,
                        help="contexts to keep per form; more buys little and "
                             "costs parse time")
    parser.add_argument("--out-rows", type=Path,
                        default=Path("output/ml/top_forms_labelled.json"))
    parser.add_argument("--out-manifest", type=Path,
                        default=Path("output/ml/top_forms_manifest.json"))
    args = parser.parse_args(argv)

    sys.path.insert(0, "ml/src")
    from ukstress_ml.morphology import SpacyMorphologyTier, apply_accents
    from ukstress_ml.morphology import resolve as morph_resolve
    from ukstress_ml.triage import classify_readings

    vowels = "аеєиіїоуюяй"
    frequency = json.loads(args.frequency.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]
    ranked = sorted(frequency, key=lambda f: -frequency[f])[: args.top]
    wanted = [f for f in ranked if f not in manifest]
    print(f"top {args.top} ambiguous forms: {len(wanted)} missing from the manifest "
          f"({sum(frequency[f] for f in wanted):,} occurrences)", flush=True)

    tier = SpacyMorphologyTier(model_path=args.spacy)

    # Manifest entries straight from the trie: signature and stressed spelling,
    # no definition, which is exactly what the classifier consumes.
    entries: dict[str, Any] = {}
    readings_by_form: dict[str, list] = {}
    for form in wanted:
        readings = tier.readings(form)
        if not readings:
            continue
        readings_by_form[form] = readings
        candidates: dict[str, dict[str, str]] = {}
        for index, (_, accents) in enumerate(readings):
            signature = signature_of(form, tuple(accents), vowels)
            if signature is None or signature in candidates:
                continue
            candidates[signature] = {
                "sense_id": f"trie:{form}:{signature}",
                "signature": signature,
                "stressed": apply_accents(form, tuple(accents)),
                "definition": "",
            }
        if len(candidates) < 2:
            continue
        entries[form] = {
            "group_id": None,
            "candidates": sorted(candidates.values(), key=lambda c: c["signature"]),
            "signatures": sorted(candidates),
            "sense_ids": [c["sense_id"] for c in sorted(
                candidates.values(), key=lambda c: c["signature"])],
            "triage": classify_readings(readings)[0],
            "corpus_frequency": frequency[form],
        }
    print(f"built manifest entries for {len(entries)} forms", flush=True)

    # Collect contexts, capped per form.
    contexts: dict[str, list[tuple[str, int, int]]] = collections.defaultdict(list)
    seen: set[str] = set()
    for row in json.loads(args.corpus.read_text(encoding="utf-8")):
        sentence = row["sentence"]
        if sentence in seen:
            continue
        seen.add(sentence)
        lowered = unicodedata.normalize("NFC", sentence)
        for match in WORD.finditer(lowered):
            form = match.group().lower()
            if form in entries and len(contexts[form]) < args.per_form:
                contexts[form].append((sentence, match.start(), match.end()))
    found = sum(len(v) for v in contexts.values())
    print(f"collected {found:,} contexts across {len(contexts)} forms", flush=True)

    # Label with the parser. No decision means no label.
    rows: list[dict[str, Any]] = []
    yield_by_form: collections.Counter[str] = collections.Counter()
    batch: list[tuple[str, str, int, int]] = [
        (form, sentence, start, end)
        for form, items in contexts.items() for sentence, start, end in items
    ]
    for index in range(0, len(batch), 256):
        chunk = batch[index : index + 256]
        parsed = tier.parse_batch([sentence for _, sentence, _, _ in chunk])
        for (form, sentence, start, end), spans in zip(chunk, parsed, strict=True):
            parse = spans.get((start, end))
            if not parse or not parse[1]:
                continue
            resolution = morph_resolve(readings_by_form[form], parse[0], parse[1])
            if resolution is None:
                continue
            signature = signature_of(form, tuple(resolution.accents), vowels)
            if signature is None or signature not in entries[form]["signatures"]:
                continue
            rows.append({
                "sentence": sentence, "form": form, "start": start, "end": end,
                "gold_signature": signature, "group_id": f"trie:{form}",
                "source": "morphology", "corpus": "malyuk+ukwiki",
            })
            yield_by_form[form] += 1
        if index and index % 5120 == 0:
            print(f"  parsed {index:,}/{len(batch):,}, labelled {len(rows):,}", flush=True)

    args.out_rows.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    args.out_manifest.write_text(
        json.dumps({"forms": entries}, ensure_ascii=False, indent=1), encoding="utf-8")

    covered = sum(1 for f in entries if yield_by_form[f] > 0)
    print(f"\nlabelled {len(rows):,} rows for {covered} of {len(entries)} forms")
    print(f"  forms the parser never resolved: "
          f"{[f for f in entries if not yield_by_form[f]][:12]}")
    thin = [f for f, n in yield_by_form.items() if n < 20]
    print(f"  forms with fewer than 20 labels: {len(thin)}")
    print(f"-> {args.out_rows}  and  {args.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
