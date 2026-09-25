"""One table of readings: what each stress of an ambiguous form *is*.

The lexicon the API serves is a flat list of stressed spellings. For
`пласту` it holds `пласту́` and `пла́сту` and nothing that says one is the
organisation Пласт and the other a layer; for `сестри` nothing that says
one is the nominative plural and the other the genitive singular. Every
tier therefore sees only the signal it was built around, and a fixed tier
order settles their disagreements. The combiner (step 2 of the plan) needs
each reading described in every dimension any signal speaks to:

  lemma      from Wiktionary (dataset 1) and pymorphy3/VESUM via the tags
  upos       from Wiktionary, the trie and the gloss inventory
  feats      Case/Number/Gender from Wiktionary and the trie
  proper     every stored spelling of the reading is capitalised
  senses     sense ids and definitions from the gloss inventory

Output: one JSON line per ambiguous form, and a coverage report saying how
many forms have their readings told apart by at least one dimension.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "etl" / "src"))
from ukstress.normalizer import NormalizationError, lookup_key, stress_signature  # noqa: E402

ACUTE = "́"
DATASETS = "2,5,7,9,10,12,13"
FEATS = ("Case", "Number", "Gender")
POS_NAMES = {"noun": "NOUN", "verb": "VERB", "adjective": "ADJ", "adverb": "ADV", "pronoun": "PRON",
             "numeral": "NUM", "participle": "VERB", "converb": "VERB", "preposition": "ADP",
             "conjunction": "CCONJ", "particle": "PART", "interjection": "INTJ",
             "proper_noun": "PROPN", "proper noun": "PROPN"}
PYMORPHY_POS = {"NOUN": "NOUN", "ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ", "VERB": "VERB",
                "INFN": "VERB", "PRTF": "VERB", "PRTS": "VERB", "GRND": "VERB", "NUMR": "NUM",
                "ADVB": "ADV", "NPRO": "PRON", "PREP": "ADP", "CONJ": "CCONJ", "PRCL": "PART",
                "INTJ": "INTJ"}
PYMORPHY_CASE = {"nomn": "Nom", "gent": "Gen", "datv": "Dat", "accs": "Acc", "ablt": "Ins",
                 "loct": "Loc", "voct": "Voc"}


def psql(query: str) -> list[list[str]]:
    """Rows from the serving database, tab-separated, through the container."""
    out = subprocess.run(
        ["docker", "exec", "-i", "uk-tts-postgres-1", "sh", "-lc",
         'psql -U "$POSTGRES_USER" -d ukstress -tA -F "$(printf \'\\037\')" -R "$(printf \'\\036\')"'],
        input=query, capture_output=True, text=True, check=True).stdout
    # unit and record separators: spellings and tags can hold tabs and newlines
    return [record.split("\x1f") for record in out.split("\x1e") if record.strip()]


def signature_of(stressed: str) -> str | None:
    try:
        return stress_signature(unicodedata.normalize("NFC", stressed))
    except NormalizationError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glosses", type=Path, default=Path("output/ml/ambiguous_forms_glossed.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/readings.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("output/ml/readings_coverage.json"))
    args = parser.parse_args()

    # 1. The candidates the API serves: form, signature, spelling, capitalisation.
    rows = psql(f"""
        with amb as (select form_normalized from stress_lookup where dataset_id in ({DATASETS})
                     group by 1 having count(distinct stress_signature) > 1)
        select form_normalized, stress_signature, stressed_form
        from stress_lookup where dataset_id in ({DATASETS})
          and form_normalized in (select form_normalized from amb)""")
    readings: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    spellings: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for form, signature, stressed in rows:
        if "|" in signature or ":" in signature:
            continue   # free variation and compounds are not rival readings
        form = unicodedata.normalize("NFC", form)
        readings[form].setdefault(signature, {"signature": signature, "stressed": stressed,
                                              "lemma": set(), "upos": set(), "feats": set(),
                                              "senses": [], "sources": set()})
        spellings[(form, signature)].append(stressed)
    readings = {f: r for f, r in readings.items() if len(r) > 1}
    for (form, signature), stored in spellings.items():
        if form in readings and signature in readings[form]:
            readings[form][signature]["proper"] = all(s[:1].isupper() for s in stored)
    print(f"{len(readings):,} ambiguous forms from the served lexicon", flush=True)

    # 2. Wiktionary (dataset 1): lemma, POS and tags per stressed spelling.
    wiki = psql("""
        select w.form_normalized, w.form, l.lemma_normalized, coalesce(p.code, ''),
               array_to_string(w.grammatical_tags, ',')
        from word_form w join lexeme l on l.id = w.lexeme_id
        left join part_of_speech p on p.id = l.part_of_speech_id
        where w.dataset_id = 1""")
    hits = 0
    for form, stressed, lemma, pos, tags in wiki:
        form = unicodedata.normalize("NFC", form)
        if form not in readings:
            continue
        signature = signature_of(stressed)
        reading = readings[form].get(signature or "")
        if reading is None:
            continue
        reading["lemma"].add(lemma)
        if pos in POS_NAMES:
            reading["upos"].add(POS_NAMES[pos])
        for tag in filter(None, tags.split(",")):
            reading["feats"].add(tag)
        reading["sources"].add("wiktionary")
        hits += 1
    print(f"wiktionary: {hits:,} reading rows attached", flush=True)

    # 3. The trie (VESUM-derived): UD tags per accent position.
    from ukrainian_word_stress.stressify_ import _load_dictionary, _parse_dictionary_value, _trie_value
    trie = _load_dictionary()
    trie_hits = 0
    for form, by_sig in readings.items():
        value = _trie_value(trie, form)
        if not value:
            continue
        for tags, accents in _parse_dictionary_value(value[0]):
            for position in accents:
                signature = signature_of(form[:position] + ACUTE + form[position:])
                reading = by_sig.get(signature or "")
                if reading is None:
                    continue
                for tag in tags:
                    key, _, val = tag.partition("=")
                    if key == "upos":
                        reading["upos"].add(val)
                    elif key in FEATS:
                        reading["feats"].add(f"{key}={val}")
                reading["sources"].add("trie")
                trie_hits += 1
    print(f"trie: {trie_hits:,} reading tags attached", flush=True)

    # 4. pymorphy3/VESUM: lemmas, attached to the reading whose Case/Number match.
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer(lang="uk")
    morph_hits = 0
    for form, by_sig in readings.items():
        parses = morph.parse(form)
        for reading in by_sig.values():
            own = {f.split("=", 1)[1] for f in reading["feats"] if f.startswith(("Case=", "Number="))}
            if not own:
                continue
            for p in parses:
                case = PYMORPHY_CASE.get(p.tag.case or "")
                number = {"sing": "Sing", "plur": "Plur"}.get(p.tag.number or "")
                if case and case in own and (number is None or number in own):
                    reading["lemma"].add(p.normal_form)
                    if p.tag.POS in PYMORPHY_POS:
                        reading["upos"].add(PYMORPHY_POS[p.tag.POS])
                    reading["sources"].add("pymorphy3")
                    morph_hits += 1
    print(f"pymorphy3: {morph_hits:,} lemma attachments", flush=True)

    # 5. The gloss inventory: senses and definitions per signature.
    gloss_hits = 0
    for line in args.glosses.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        form = unicodedata.normalize("NFC", entry["form"])
        if form not in readings:
            continue
        for candidate in entry.get("candidates", []):
            signature = signature_of(candidate.get("stressed", "")) or str(candidate.get("signature"))
            reading = readings[form].get(signature)
            if reading is None:
                continue
            reading["senses"].append({"sense_id": candidate.get("sense_id"),
                                      "definition": candidate.get("definition")})
            if candidate.get("pos") in POS_NAMES:
                reading["upos"].add(POS_NAMES[candidate["pos"]])
            reading["sources"].add("glosses")
            gloss_hits += 1
    print(f"glosses: {gloss_hits:,} senses attached", flush=True)

    # Which dimension tells a form's readings apart?
    def differ(values: list[frozenset]) -> bool:
        """Readings are told apart if every pair has distinct, non-empty values."""
        return all(values) and len(set(values)) == len(values)

    coverage = collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for form, by_sig in sorted(readings.items()):
            items = sorted(by_sig.values(), key=lambda r: r["signature"])
            told = {
                "lemma": differ([frozenset(r["lemma"]) for r in items]),
                "upos": differ([frozenset(r["upos"]) for r in items]),
                "feats": differ([frozenset(r["feats"]) for r in items]),
                "proper": len({r.get("proper", False) for r in items}) > 1,
                "sense": all(r["senses"] for r in items),
            }
            for key, value in told.items():
                coverage[key] += value
            coverage["any"] += any(told.values())
            coverage["forms"] += 1
            handle.write(json.dumps({
                "form": form, "told_apart_by": [k for k, v in told.items() if v],
                "readings": [{**r, "lemma": sorted(r["lemma"]), "upos": sorted(r["upos"]),
                              "feats": sorted(r["feats"]), "sources": sorted(r["sources"])}
                             for r in items]}, ensure_ascii=False) + "\n")
    report = {k: coverage[k] for k in ("forms", "any", "feats", "sense", "upos", "lemma", "proper")}
    report["none"] = coverage["forms"] - coverage["any"]
    args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
