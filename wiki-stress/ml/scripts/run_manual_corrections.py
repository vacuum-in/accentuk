"""Reviewed stress readings, recorded at full confidence.

Some errors are one decision repeated. On lang-uk's benchmark *його* is wrong
37 times, *Київ* 12, *народу* 10 — three forms, 59 tokens, and none of them is
context-dependent: the reading is simply wrong wherever it appears. A model
will not learn its way out of that, and there is no reason for it to try.

The rows here are written at confidence 1.00, above every generated dataset
(the merged wordlist sits at 0.95, the trie-derived sets at 0.99), because they
record a reviewed decision rather than an inference. Each carries the evidence
that justified it, so the file can be argued with rather than trusted.

What does *not* belong here: readings that depend on context. *мене* and *себе*
disagree with the benchmark 17 times between them, and every one of those is
preceded by a preposition — «до ме́не», «у ме́не», «з се́бе», «на се́бе» —
where the stress moves to the first syllable. Pin either form and the
prepositionless cases («він знає мене́») break instead. That is a rule about
the sentence and it belongs in the morphology tier.
"""

from __future__ import annotations

import argparse
import importlib.util as importlib_util
import json
import unicodedata
from pathlib import Path

_spec = importlib_util.spec_from_file_location(
    "run_trie_corrections", Path(__file__).with_name("run_trie_corrections.py"))
_corrections = importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_corrections)

VOWELS = frozenset("аеєиіїоуюяй")

#: form -> (stressed reading, why). The stressed form is authoritative; the
#: signature is derived from it, so a typo shows up as a mismatch rather than
#: as a silently wrong ordinal.
# Forms whose other reading is simply wrong. These are written to their own
# dataset and served as the only candidate, because leading the candidate list
# is not enough: two candidates reach the contextual model, and on `маю` the
# model chose the reading the review had rejected.
EXCLUSIVE: frozenset[str] = frozenset({
    "воду", "хоча",
    "його", "Київ", "народу", "народові", "маю", "дівчата", "дівчат", "беру",
    "народом", "сором", "розумів", "гроші", "користуватися", "богдане",
    "одержав", "болярин",
    # Second round: 72 forms confirmed by a Ukrainian speaker from 311,123
    # verified words across three Common Voice sweeps. See
    # audiotostress/DICTIONARY_AUDIT_ROUND2_review.md for the evidence per
    # form — distinct voices disputing, and whether they agree with each other.
    "боляри", "болярине", "борітеся", "боях",
    "веди", "веду", "великороса-державника", "видимира",
    "визнається", "використання", "вилазь", "витягати",
    "виходило", "входили", "відомості", "вісімнадесять",
    "галузі", "думать", "дівчатами", "жупаніце",
    "заводу", "заводі", "застав", "захисника",
    "захисників", "західнього", "західні", "зевес",
    "казань", "казані", "князівського", "корисним",
    "корисно", "корисні", "користувачів", "користується",
    "літами", "народи", "народі", "нестямі",
    "олеся", "повідаємо", "повідаєш", "подію",
    "послухав", "проекти", "просили", "проходили",
    "пуста", "пустині", "радости", "ростові",
    "русин", "русини", "руська", "святами",
    "складної", "сміливо", "смітнику", "сміття",
    "сторожі", "страхом", "судко", "січні",
    "характерний", "чималі", "чучупака", "чучупаки",
    "шукать", "ярослав", "ярославом", "іпатіївський",
})

# Everything else here has two valid readings and only the default was wrong;
# those keep both candidates with the reviewed reading first.
CORRECTIONS: dict[str, tuple[str, str]] = {
    "воду": ("во́ду",
             ("Accusative of вода́. The lexicon lists воду́ first, which is "
              "dialectal and poetic; standard is во́ду, and it is served as "
              "the dictionary default in «пила воду́». 2026-09-18.")),
    "хоча": ("хоча́",
             ("The lexicon holds only хо́ча. Dictionaries give хоча́ as the "
              "primary reading with хо́ча as the variant; for TTS the "
              "primary is served. 2026-09-18.")),
    "його": ("його́",
             ("The lexicon holds only йо́го. Standard is його́, and the "
              "benchmark's 37 occurrences agree unanimously.")),
    "Київ": ("Ки́їв",
             ("The merged wordlist already says Ки́їв at 0.95; "
              "trie-corrections-v2 and trie-defaults-v1 both assert Киї́в at "
              "0.99 and win. 12 occurrences, all Ки́їв.")),
    "народу": ("наро́ду",
               ("Genitive of наро́д. The wordlist says наро́ду at 0.95; "
                "trie-defaults-v1 asserts на́роду at 0.99. 8 occurrences.")),
    "народові": ("наро́дові",
                 "Dative of наро́д, same override as народу. 2 occurrences."),
    # Found by comparing 10,000 Common Voice recordings against this pipeline
    # (audiotostress/DICTIONARY_AUDIT.md). The acoustic model never saw these
    # forms in training — the miner keeps only forms the trie names
    # unambiguously, and every one of these is ambiguous or absent — so it read
    # them by generalisation, and a Ukrainian speaker confirmed all of them.
    # Speaker counts are distinct voices that agree with each other.
    "маю": ("ма́ю", "25 speakers, 31 of 32 recordings. The lexicon served маю́."),
    "дівчата": ("дівча́та", "9 speakers, 10 of 10. The lexicon served ді́вчата."),
    "дівчат": ("дівча́т", "5 speakers, 5 of 5. The lexicon served ді́вчат."),
    "беру": ("беру́", "6 speakers, 7 of 7. The lexicon served бе́ру."),
    "народом": ("наро́дом", "5 speakers, 6 of 6. Instrumental of наро́д, "
                "same override as народу and народові."),
    "сором": ("со́ром", "4 speakers, 5 of 6. The lexicon served соро́м."),
    "розумів": ("розумі́в", "5 speakers, 5 of 5. The lexicon served ро́зумів."),
    "гроші": ("гро́ші", "12 speakers, 13 of 18. The lexicon served гроші́."),
    "користуватися": ("користува́тися",
                      "5 speakers, 5 of 5. The lexicon served кори́стуватися."),
    "богдане": ("богда́не", "4 speakers, 5 of 5. Vocative of Богда́н."),
    "одержав": ("оде́ржав", "3 speakers, 3 of 4. The lexicon served одержа́в."),
    "болярин": ("боля́рин",
                "9 speakers, 9 of 9, unanimous. The form is absent from the "
                "lexicon entirely and reached the suffix fallback, which "
                "guessed боляри́н."),
    # Below: both readings are valid and the pinned one is far more frequent.
    # Writing it at 1.00 changes which is served without deleting the other,
    # so the rarer reading survives in its own dataset.
    "того": ("того́", "51 speakers, 81 of 105. то́го remains valid and rare."),
    "була": ("була́", "42 speakers, 67 of 82. бу́ла remains valid and rare."),
    "кого": ("кого́", "24 speakers, 29 of 40. Absent from the trie entirely."),
    "років": ("ро́ків", "17 speakers, 22 of 22. Genitive plural of рік."),
    "всього": ("всього́", "9 speakers, 10 of 10. всьо́го remains valid and rare."),
    "залишилися": ("залиши́лися", "6 speakers, 6 of 8."),
    "ніяк": ("нія́к", "10 speakers, 10 of 10. ні́як remains valid and rare."),
    # --- second round, exclusive: the other reading is wrong ---
    "народи": ("наро́ди",
                 "19 speakers, 29 of 29. Confirmed: the other reading is wrong."),
    "використання": ("використа́ння",
                 "17 speakers, 19 of 25. Confirmed: the other reading is wrong."),
    "застав": ("заста́в",
                 "15 speakers, 15 of 17. Confirmed: the other reading is wrong."),
    "сміливо": ("сміли́во",
                 "15 speakers, 18 of 20. Confirmed: the other reading is wrong."),
    "сміття": ("сміття́",
                 "15 speakers, 18 of 20. Confirmed: the other reading is wrong."),
    "просили": ("проси́ли",
                 "14 speakers, 14 of 15. Confirmed: the other reading is wrong."),
    "проекти": ("прое́кти",
                 "14 speakers, 15 of 15. Confirmed: the other reading is wrong."),
    "ярослав": ("яросла́в",
                 "13 speakers, 17 of 17. Confirmed: the other reading is wrong."),
    "русини": ("руси́ни",
                 "13 speakers, 17 of 19. Confirmed: the other reading is wrong."),
    "чучупака": ("чучупа́ка",
                 "13 speakers, 18 of 19. Confirmed: the other reading is wrong."),
    "болярине": ("боля́рине",
                 "12 speakers, 24 of 26. Confirmed: the other reading is wrong."),
    "заводу": ("заво́ду",
                 "11 speakers, 13 of 13. Confirmed: the other reading is wrong."),
    "казані": ("каза́ні",
                 "11 speakers, 12 of 16. Confirmed: the other reading is wrong."),
    "боях": ("боя́х",
                 "10 speakers, 10 of 10. Confirmed: the other reading is wrong."),
    "народі": ("наро́ді",
                 "9 speakers, 9 of 9. Confirmed: the other reading is wrong."),
    "користується": ("користу́ється",
                 "9 speakers, 10 of 11. Confirmed: the other reading is wrong."),
    "повідаєш": ("пові́даєш",
                 "9 speakers, 10 of 11. Confirmed: the other reading is wrong."),
    "веду": ("веду́",
                 "9 speakers, 10 of 13. Confirmed: the other reading is wrong."),
    "відомості": ("ві́домості",
                 "8 speakers, 15 of 18. Confirmed: the other reading is wrong."),
    "послухав": ("послу́хав",
                 "8 speakers, 8 of 10. Confirmed: the other reading is wrong."),
    "думать": ("ду́мать",
                 "8 speakers, 8 of 8. Confirmed: the other reading is wrong."),
    "проходили": ("прохо́дили",
                 "8 speakers, 10 of 11. Confirmed: the other reading is wrong."),
    "подію": ("поді́ю",
                 "8 speakers, 9 of 11. Confirmed: the other reading is wrong."),
    "заводі": ("заво́ді",
                 "7 speakers, 7 of 7. Confirmed: the other reading is wrong."),
    "літами": ("літа́ми",
                 "7 speakers, 7 of 7. Confirmed: the other reading is wrong."),
    "князівського": ("князі́вського",
                 "7 speakers, 7 of 8. Confirmed: the other reading is wrong."),
    "дівчатами": ("дівча́тами",
                 "7 speakers, 7 of 7. Confirmed: the other reading is wrong."),
    "великороса-державника": ("великороса-держа́вника",
                 "7 speakers, 7 of 7. Confirmed: the other reading is wrong."),
    "ярославом": ("яросла́вом",
                 "7 speakers, 9 of 9. Confirmed: the other reading is wrong."),
    "русин": ("руси́н",
                 "6 speakers, 7 of 8. Confirmed: the other reading is wrong."),
    "радости": ("ра́дости",
                 "6 speakers, 7 of 8. Confirmed: the other reading is wrong."),
    "складної": ("складно́ї",
                 "6 speakers, 6 of 6. Confirmed: the other reading is wrong."),
    "корисні": ("кори́сні",
                 "6 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "корисно": ("кори́сно",
                 "6 speakers, 6 of 8. Confirmed: the other reading is wrong."),
    "страхом": ("стра́хом",
                 "6 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "користувачів": ("користувачі́в",
                 "6 speakers, 8 of 8. Confirmed: the other reading is wrong."),
    "повідаємо": ("пові́даємо",
                 "6 speakers, 6 of 8. Confirmed: the other reading is wrong."),
    "смітнику": ("смітнику́",
                 "6 speakers, 6 of 8. Confirmed: the other reading is wrong."),
    "західні": ("за́хідні",
                 "6 speakers, 6 of 6. Confirmed: the other reading is wrong."),
    "олеся": ("оле́ся",
                 "6 speakers, 7 of 10. Confirmed: the other reading is wrong."),
    "захисників": ("захисникі́в",
                 "6 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "веди": ("веди́",
                 "6 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "судко": ("судко́",
                 "6 speakers, 7 of 7. Confirmed: the other reading is wrong."),
    "входили": ("вхо́дили",
                 "6 speakers, 6 of 6. Confirmed: the other reading is wrong."),
    "визнається": ("визнає́ться",
                 "6 speakers, 6 of 8. Confirmed: the other reading is wrong."),
    "січні": ("сі́чні",
                 "6 speakers, 7 of 8. Confirmed: the other reading is wrong."),
    "шукать": ("шука́ть",
                 "6 speakers, 6 of 6. Confirmed: the other reading is wrong."),
    "ростові": ("росто́ві",
                 "6 speakers, 7 of 8. Confirmed: the other reading is wrong."),
    "борітеся": ("борі́теся",
                 "5 speakers, 5 of 7. Confirmed: the other reading is wrong."),
    "жупаніце": ("жупані́це",
                 "5 speakers, 8 of 11. Confirmed: the other reading is wrong."),
    "казань": ("каза́нь",
                 "5 speakers, 5 of 6. Confirmed: the other reading is wrong."),
    "чималі": ("чималі́",
                 "5 speakers, 5 of 6. Confirmed: the other reading is wrong."),
    "зевес": ("зеве́с",
                 "5 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "витягати": ("витяга́ти",
                 "5 speakers, 5 of 6. Confirmed: the other reading is wrong."),
    "виходило": ("вихо́дило",
                 "5 speakers, 9 of 11. Confirmed: the other reading is wrong."),
    "руська": ("ру́ська",
                 "5 speakers, 5 of 7. Confirmed: the other reading is wrong."),
    "галузі": ("га́лузі",
                 "5 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "західнього": ("за́хіднього",
                 "5 speakers, 6 of 6. Confirmed: the other reading is wrong."),
    "пустині": ("пусти́ні",
                 "5 speakers, 5 of 6. Confirmed: the other reading is wrong."),
    "святами": ("свята́ми",
                 "5 speakers, 5 of 6. Confirmed: the other reading is wrong."),
    "сторожі": ("сторо́жі",
                 "5 speakers, 5 of 7. Confirmed: the other reading is wrong."),
    "корисним": ("кори́сним",
                 "5 speakers, 5 of 6. Confirmed: the other reading is wrong."),
    "нестямі": ("нестя́мі",
                 "5 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "характерний": ("характе́рний",
                 "5 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "боляри": ("боля́ри",
                 "5 speakers, 8 of 8. Confirmed: the other reading is wrong."),
    "чучупаки": ("чучупа́ки",
                 "4 speakers, 7 of 10. Confirmed: the other reading is wrong."),
    "вилазь": ("вила́зь",
                 "4 speakers, 5 of 7. Confirmed: the other reading is wrong."),
    "вісімнадесять": ("вісімнадеся́ть",
                 "4 speakers, 6 of 8. Confirmed: the other reading is wrong."),
    "іпатіївський": ("іпа́тіївський",
                 "4 speakers, 8 of 10. Confirmed: the other reading is wrong."),
    "пуста": ("пуста́",
                 "4 speakers, 5 of 7. Confirmed: the other reading is wrong."),
    "видимира": ("видими́ра",
                 "4 speakers, 6 of 7. Confirmed: the other reading is wrong."),
    "захисника": ("захисника́",
                 "3 speakers, 5 of 7. Confirmed: the other reading is wrong."),
    # --- second round, ordering: both valid, this one dominant ---
    "прошу": ("про́шу",
                 "58 speakers, 114 of 161. Confirmed: both readings are valid, this one is far more common."),
    "діяння": ("дія́ння",
                 "28 speakers, 48 of 51. Confirmed: both readings are valid, this one is far more common."),
    "городу": ("го́роду",
                 "14 speakers, 22 of 30. Confirmed: both readings are valid, this one is far more common."),
    "проводити": ("прово́дити",
                 "13 speakers, 14 of 14. Confirmed: both readings are valid, this one is far more common."),
    "небоже": ("небо́же",
                 "12 speakers, 12 of 13. Confirmed: both readings are valid, this one is far more common."),
    "виходив": ("вихо́див",
                 "12 speakers, 15 of 15. Confirmed: both readings are valid, this one is far more common."),
    "проводили": ("прово́дили",
                 "11 speakers, 12 of 13. Confirmed: both readings are valid, this one is far more common."),
    "сходив": ("сходи́в",
                 "6 speakers, 7 of 10. Confirmed: both readings are valid, this one is far more common."),
    "зносити": ("зно́сити",
                 "6 speakers, 7 of 7. Confirmed: both readings are valid, this one is far more common."),
    "береги": ("береги́",
                 "6 speakers, 6 of 7. Confirmed: both readings are valid, this one is far more common."),
    "зерна": ("зерна́",
                 "5 speakers, 6 of 8. Confirmed: both readings are valid, this one is far more common."),
    "виходили": ("вихо́дили",
                 "5 speakers, 7 of 9. Confirmed: both readings are valid, this one is far more common."),
    "мостом": ("мо́стом",
                 "2 speakers, 5 of 6. Confirmed: both readings are valid, this one is far more common."),
}


def signature_of(stressed: str) -> str:
    """The vowel ordinal the acute falls on, counting й as the pipeline does."""
    decomposed = unicodedata.normalize("NFD", stressed)
    ordinal = -1
    for index, char in enumerate(decomposed):
        if char.lower() in VOWELS:
            ordinal += 1
        if char == "́":
            return str(ordinal)
    raise ValueError(f"{stressed!r} carries no acute")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset-key", default="manual-corrections-v1")
    parser.add_argument("--exclusive-dataset-key",
                        default="manual-corrections-exclusive-v1")
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/manual_corrections.json"))
    parser.add_argument("--manifest", type=Path,
                        help="serving manifest to prune: a reviewed reading is "
                             "not a decision for the model to make, and while "
                             "the form stays in coverage the model overrides "
                             "the dictionary whatever its confidence")
    parser.add_argument("--decisions", type=Path, nargs="*", default=[],
                        help="JSON files of reviewed decisions, one object per "
                             "form: {form, stressed, verdict, why}. verdict "
                             "'правильно' makes the reading the only one, "
                             "'обидва' makes it lead; anything else is skipped. "
                             "Round 3 of the audit has hundreds of these, which "
                             "is too many for the dict above")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be written and stop")
    args = parser.parse_args()

    decided: dict[str, tuple[str, str, bool]] = {}
    for path in args.decisions:
        for entry in json.loads(path.read_text(encoding="utf-8")):
            verdict = (entry.get("verdict") or "").strip().lower()
            if verdict not in ("правильно", "обидва"):
                continue
            decided[entry["form"]] = (entry["stressed"], entry.get("why", path.name),
                                      verdict == "правильно")
    rows, exclusive_rows = [], []
    for form, (stressed, why) in list(CORRECTIONS.items()) + [
            (f, (st, why)) for f, (st, why, _) in decided.items()]:
        normalized = unicodedata.normalize("NFD", form.lower())
        row = {"form": normalized, "stressed": unicodedata.normalize("NFD", stressed),
               "signature": signature_of(stressed), "reason": why}
        only = form in EXCLUSIVE or (form in decided and decided[form][2])
        (exclusive_rows if only else rows).append(row)
        kind = "only" if only else "leads"
        print(f"{form:<14} -> {stressed:<14} signature {row['signature']}  {kind}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.dry_run:
        print("dry run: nothing written to the database")
        return 0

    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        forms = manifest["forms"]
        # The manifest is keyed NFC; the corrections are normalised NFD above.
        dropped = [unicodedata.normalize("NFC", r["form"])
                   for r in rows + exclusive_rows
                   if unicodedata.normalize("NFC", r["form"]) in forms]
        if dropped:
            backup = args.manifest.with_suffix(".json.before-corrections")
            if not backup.exists():
                backup.write_text(args.manifest.read_text(encoding="utf-8"),
                                  encoding="utf-8")
            for form in dropped:
                del forms[form]
            # inventory_hash names the sense inventory, not this form list, so
            # dropping coverage leaves the API's hash check intact.
            args.manifest.write_text(json.dumps(manifest, ensure_ascii=False),
                                     encoding="utf-8")
        print(f"manifest: dropped {len(dropped)} covered forms "
              f"({', '.join(dropped) or 'none'}), {len(forms):,} left")

    written = _corrections.write_dataset(
        args.database_url, args.dataset_key, rows,
        confidence=1.0, source_rank=0, source_kind="manual_correction")
    print(f"wrote {written} rows to {args.dataset_key!r} at confidence 1.00")
    exclusive_written = _corrections.write_dataset(
        args.database_url, args.exclusive_dataset_key, exclusive_rows,
        confidence=1.0, source_rank=0, source_kind="manual_correction")
    print(f"wrote {exclusive_written} rows to {args.exclusive_dataset_key!r}")
    print("\nThe API reads the first only once its id is in REVIEWED_DATASETS,")
    print("and the second only once its id is in REVIEWED_EXCLUSIVE_DATASETS,")
    print("which collapses those forms to one candidate so no tier is asked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
