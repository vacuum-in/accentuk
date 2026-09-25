"""Split surnames, toponyms and other proper names out of the training data.

The inventory carries a sizeable population of forms whose two readings differ
only because one of them is a name: *барвіні́вський* from the surname
Барвінівський against *барві́нівський* from the plant барві́нок, or *бала́хна*
against *балахна́*, two Russian towns that no context in a Ukrainian sentence
distinguishes. These are a different problem from the heteronyms the model is
meant to learn. A name's stress is a property of the name, not of the sentence
it sits in, so no amount of context makes the choice inferable; the rows teach
the model to guess, and a guess on a name is a coin flip that costs accuracy
everywhere else through the shared encoder.

So they come out — into their own dataset rather than the bin, because they are
exactly the material a gazetteer tier would want later, and because keeping the
category label means a decision to re-admit, say, only the toponyms is a filter
away rather than a re-run.

The split is by *form group* almost everywhere, because when only one reading of
a form is a name, removing that reading leaves a single reading behind — no
longer ambiguous, so the form leaves the inventory regardless. The exception is
the handful of forms carrying three or more readings, where a name sits beside
a real ambiguity: *чопо́ві* (the pin) against *чопові́* (of a pin) is a heteronym
worth learning, and only Чо́пові, the town, has to go. Those keep their common
readings and lose just the name.

Classification reads the glosses. The cues are the defining phrases a
dictionary entry uses — "прізвище", "топонім", "місто в X", "жіноче ім'я" — not
the bare words, which appear all over ordinary definitions ("реєстрація
мешканців даного міста" is not a toponym). Anything left holding a capitalised
lemma or the words "власна назва" is a proper name of a kind the cues did not
name, and is split out under `proper`.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

# A place noun, in any of the cases a gloss puts it in.
PLACE = (
    r"міст[оаеу]|сел[оаище]\w*|річк\w+|озер\w+|гор[аи]|хребет|хребта|"
    r"населен\w+ пункт\w*|громад\w+|повіт\w*|округ\w*|штат\w*|провінці\w+|"
    r"комун\w+|муніципалітет\w*"
)

# Order matters: the first match wins, and a gloss reading "від прізвища або
# топоніма Барвінівський" is more usefully filed under the surname it names.
CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("surname", re.compile(r"прізвищ|спадкове родове найменування", re.IGNORECASE)),
    ("toponym", re.compile(
        rf"топонім|гідронім|ойконім|"
        rf"(?:{PLACE})\s+(?:в|у|на|біля|поблизу|коло|під)\s+(?-i:[А-ЯЇІЄҐA-Z])|"
        rf"назв[аи]\s+(?:{PLACE})|"
        rf"(?:{PLACE})\s+(?:в|у|на)\s+\w+ій\s+(?:області|окрузі|провінції|землі)",
        re.IGNORECASE)),
    # "ім'я" alone is not a cue: *назву́* is glossed "дати комусь певне ім'я",
    # which is the verb, not a name. The name has to be in a naming role.
    ("given", re.compile(
        r"(?:чоловіче|жіноче|особове|власне)\s+ім|на ім['ʼ]?я\b|по батькові|"
        r"імен[іяе]м?\s+(?-i:[А-ЯЇІЄҐ])", re.IGNORECASE)),
    # Last resort: the gloss says the lemma is capitalised, or says outright
    # that this is a proper name, without naming which kind. The capital-letter
    # test is scoped case-sensitive — under the pattern's own re.I it matches
    # lowercase too, and every "від: адре́са" in the inventory reads as a name.
    ("proper", re.compile(r"(?-i:від:\s*[А-ЯЇІЄҐ])|власна назва", re.IGNORECASE)),
)


def classify(definition: str) -> str | None:
    """Which kind of proper name this gloss describes, or None for a common word."""
    for name, pattern in CATEGORIES:
        if pattern.search(definition or ""):
            return name
    return None


def split(forms: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Partition the inventory into common forms and proper-name forms.

    A form leaves as soon as one of its readings is a name. The returned
    proper-name entries carry a `category` on every candidate and a `categories`
    roll-up on the group, so a later pass can re-admit a subset.
    """
    common: dict[str, Any] = {}
    proper: dict[str, Any] = {}
    stats: collections.Counter[str] = collections.Counter()
    dropped_senses: set[str] = set()

    for key, entry in forms.items():
        marked = []
        for candidate in entry.get("candidates", []):
            category = classify(candidate.get("definition", ""))
            marked.append((candidate, category))

        hits = [c for _, c in marked if c]
        if not hits:
            common[key] = entry
            continue

        for category in set(hits):
            stats[category] += 1

        out = dict(entry)
        out["candidates"] = [dict(c, category=cat) for c, cat in marked]
        out["categories"] = sorted(set(hits))
        proper[key] = out

        keep = [c for c, cat in marked if not cat]
        if len({c["signature"] for c in keep}) > 1:
            # A real ambiguity survives the name: trim rather than drop.
            stats["forms_trimmed"] += 1
            trimmed = dict(entry)
            trimmed["candidates"] = keep
            trimmed["signatures"] = sorted({c["signature"] for c in keep})
            trimmed["sense_ids"] = [c["sense_id"] for c in keep if "sense_id" in c]
            common[key] = trimmed
            dropped_senses.update(
                c["sense_id"] for c, cat in marked if cat and "sense_id" in c)
        else:
            stats["forms_removed"] += 1

    return common, proper, dict(stats), dropped_senses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_v25.json"))
    parser.add_argument("--out-manifest", type=Path,
                        help="cleaned manifest; omit to only report")
    parser.add_argument("--out-names", type=Path,
                        default=Path("output/ml/propernames.json"),
                        help="the split-out proper-name inventory")
    parser.add_argument("--dataset", type=Path, nargs="*", default=(),
                        help="silver/gold files to filter alongside the manifest")
    parser.add_argument("--suffix", default=".nonames",
                        help="inserted before .json for each filtered dataset")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    forms = manifest["forms"]
    common, proper, stats, dropped_senses = split(forms)

    print(f"{args.manifest.name}: {len(forms)} forms")
    print(f"  kept     {len(common)}")
    print(f"  removed  {stats.get('forms_removed', 0)}")
    print(f"  trimmed  {stats.get('forms_trimmed', 0)} "
          f"(kept, name reading dropped)")
    print("  name readings found, by kind:")
    for name, _ in CATEGORIES:
        if stats.get(name):
            print(f"    {name:<10} {stats[name]}")

    args.out_names.parent.mkdir(parents=True, exist_ok=True)
    args.out_names.write_text(
        json.dumps({k: manifest[k] for k in manifest if k != "forms"} | {"forms": proper},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out_names} ({len(proper)} forms)")

    if args.out_manifest:
        args.out_manifest.write_text(
            json.dumps({k: manifest[k] for k in manifest if k != "forms"} | {"forms": common},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out_manifest} ({len(common)} forms)")

    # Training rows carry the group and the sense they were labelled with, so
    # the same cut applies: a row goes if its whole group went, or if it was
    # labelled with a name reading that a trimmed group no longer offers.
    dropped_groups = {e["group_id"] for k, e in proper.items()
                      if "group_id" in e and k not in common}
    for path in args.dataset:
        rows = json.loads(path.read_text(encoding="utf-8"))
        kept = [r for r in rows
                if r.get("group_id") not in dropped_groups
                and r.get("gold_sense") not in dropped_senses]
        target = path.with_suffix("")
        target = target.with_name(target.name + args.suffix + ".json")
        target.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
        share = 100.0 * (len(rows) - len(kept)) / len(rows) if rows else 0.0
        print(f"{path.name}: {len(rows)} -> {len(kept)} rows "
              f"(-{len(rows) - len(kept)}, {share:.1f}%) -> {target.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
