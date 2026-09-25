"""Evaluate the pipeline on lang-uk's public lexical stress benchmark.

https://github.com/lang-uk/ukrainian-tts-preprocessing

Every number this project has reported was measured on its own data — its own
mined corpus, its own silver labels, and latterly a private gold set. This is an
external benchmark with published baselines for four existing systems, scored by
the maintainers' own harness rather than a reimplementation of it, which is the
only way to know where the pipeline actually stands.

Their metric differs from ours in ways that matter:

* Stress is marked `+` after the stressed vowel, and a multi-vowel word emitted
  with **no** stress counts as wrong (Rule 4). The old serving behaviour of
  returning an ambiguous word unstressed therefore scores zero here, exactly as
  it did on the gold set.
* Single-vowel words and words the labellers left unmarked are excluded from the
  denominator rather than counted as free correct answers.
* Heteronyms are scored separately against their own 493-entry list, so the
  homograph question is reported on its own terms and not diluted by the
  overwhelmingly unambiguous majority of tokens.

Sentences are resolved in one batched pass and cached, because the harness calls
the stressify function once per sentence and the cross-encoder is far cheaper
batched than called 1,026 times.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import psycopg
from ukstress.normalizer import lookup_key

from ukstress_ml.corpus import EncodingError, apply_signature
from ukstress_ml.morphology import MorphologyTier, SpacyMorphologyTier, apply_accents
from ukstress_ml.morphology import resolve as morph_resolve
from ukstress_ml.serving import (
    ClassifierStressModel,
    ContextualStressModel,
    Target,
)

WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
VOWELS = frozenset("аеєиіїоуюя")
ACUTE = "́"


def restore_apostrophes(original: str, marked: str) -> str:
    """Put back the apostrophe characters the caller actually wrote.

    `canonical_stressed_form` folds every apostrophe variant to U+02BC so that
    lookup keys match. That is right for a key and wrong for output: returning
    `здоро+вʼя` for an input spelled `здоров'я` rewrites the user's text. The
    benchmark rejects it before even comparing stress, and an API doing this to
    real input would be corrupting it.
    """
    out, source = [], iter(original)
    for char in marked:
        if char == "+":
            out.append(char)
            continue
        original_char = next(source, char)
        out.append(original_char)
    return "".join(out)


def to_plus(surface: str, signature: str) -> str | None:
    """Render `surface` with `+` after each stressed vowel, or None if it will not fit."""
    try:
        if ":" in signature:
            # A per-token signature for a hyphenated compound: apply each part
            # to its own token, which is what the signature is indexed by.
            parts = surface.split("-")
            wanted: dict[int, list[str]] = {}
            for item in signature.split("|"):
                token, _, ordinal = item.partition(":")
                wanted.setdefault(int(token), []).append(ordinal)
            out = []
            for index, part in enumerate(parts):
                if index in wanted:
                    out.append(apply_signature(part, "|".join(wanted[index])))
                else:
                    out.append(part)
            stressed = "-".join(out)
        else:
            # `0|1` is the wordlist's "either stress is acceptable" notation for
            # ONE token, not two accents to emit. Rendering both produced
            # `цу+кру+`, which is not a word and fails the benchmark outright.
            stressed = apply_signature(surface, signature.split("|")[0])
    except (EncodingError, Exception):  # noqa: BLE001
        return None
    marked = unicodedata.normalize("NFC", stressed).replace(ACUTE, "+")
    return restore_apostrophes(surface, marked)


def resolve_dataset(sentences: list[str], args: argparse.Namespace) -> dict[str, str]:
    """Stress every sentence, batching all model calls into one pass."""
    tokenised: list[list[tuple[int, int, str, str]]] = []
    keys: set[str] = set()
    for sentence in sentences:
        spans = []
        for match in WORD.finditer(sentence):
            surface = match.group()
            if sum(1 for c in surface.lower() if c in VOWELS) < 1:
                continue
            key = lookup_key(surface)
            spans.append((match.start(), match.end(), surface, key))
            keys.add(key)
            # `compound_fallback` stresses a hyphenated compound the lexicon
            # lacks from its parts, but it reads them out of `variants`, which
            # is filled by one query over exactly these keys. The parts were
            # never queried, so every part came back empty and the fallback
            # returned None for compounds whose halves the lexicon does have --
            # `адміністративно-територіальна`, `місту-герою`, `лірико-колоратурне`.
            if "-" in key:
                for part in surface.split("-"):
                    if part:
                        keys.add(lookup_key(part))
        tokenised.append(spans)

    with psycopg.connect(args.database_url) as conn:
        active = int(conn.execute(
            "SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()[0])
        datasets = [active, *args.extra_dataset]
        rows = conn.execute(
            "SELECT form_normalized, stress_signature, max(confidence), min(source_rank), "
            "       min(stressed_form) "
            "FROM stress_lookup WHERE dataset_id = ANY(%s) AND form_normalized = ANY(%s) "
            "GROUP BY form_normalized, stress_signature", (datasets, sorted(keys))).fetchall()
    variants: dict[str, list[tuple[str, float, int]]] = {}
    proper: dict[tuple[str, str], bool] = {}
    for form, signature, confidence, rank, stressed in rows:
        key, sig = str(form), str(signature)
        variants.setdefault(key, []).append((sig, float(confidence), int(rank)))
        # A reading whose stored spelling is capitalised is a proper noun. The
        # merged wordlist injects toponyms and given names at the highest
        # confidence, so `Розді́л` (a village) outranks the common noun
        # `ро́зділ` and `Ме́ні` displaces `мені́`. For a lowercase token in
        # running text that answer is simply wrong, whatever its confidence.
        proper[(key, sig)] = proper.get((key, sig), True) and str(stressed)[:1].isupper()
    # Where the lexicon offers several readings and nothing else decides, the
    # default is simply the highest-confidence one. The source trie often names
    # exactly one reading for the same form, and measured on this benchmark it
    # is the better default: right 26 times against the lexicon order's 12.
    # Only an unambiguous trie entry counts -- if the trie is undecided it has
    # no more to say than the lexicon does.
    trie_choice: dict[str, str] = {}
    if args.prefer_trie:
        import marisa_trie
        loaded = marisa_trie.BytesTrie()
        loaded.load(str(args.prefer_trie))
        for key in variants:
            composed = unicodedata.normalize("NFC", key)
            if composed not in loaded:
                continue
            ordinals = set()
            for record in loaded[composed]:
                if not record or not 1 <= record[0] <= len(composed):
                    continue
                prefix = unicodedata.normalize("NFD", composed[: record[0]]).lower()
                ordinals.add(sum(1 for c in prefix if c in VOWELS) - 1)
            if len(ordinals) == 1:
                trie_choice[key] = str(next(iter(ordinals)))
        print(f"trie defaults: {len(trie_choice):,}", flush=True)

    for key, candidates in variants.items():
        # `0|1` is one reading meaning "either stress is acceptable", not a
        # rival of `1`. Dataset 6 stores the free-variation form while dataset 2
        # stores the specific one, so a word like `держави` arrived here looking
        # ambiguous, and the free-variation entry -- higher confidence -- won
        # and then collapsed to its first member, giving `де+ржави` while the
        # curated `держа+ви` sat right beside it. Drop a free-variation
        # signature whenever a specific member of it is also present: it adds no
        # information and its collapse-to-first is arbitrary.
        specific = {c[0] for c in candidates if "|" not in c[0]}
        if specific:
            candidates[:] = [
                c for c in candidates
                if "|" not in c[0] or not (set(c[0].split("|")) & specific)
            ]
        preferred = trie_choice.get(key)
        candidates.sort(key=lambda c: (c[0] != preferred,
                                       proper.get((key, c[0]), False), -c[1], c[2], c[0]))

    model = ContextualStressModel(args.model, args.manifest, backend="torch", device=args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]

    pending: list[tuple[int, int, Target]] = []
    for index, (sentence, spans) in enumerate(zip(sentences, tokenised, strict=True)):
        for position, (start, end, _surface, key) in enumerate(spans):
            candidates = variants.get(key, [])
            if len(candidates) < 2:
                continue
            entry = manifest.get(key)
            signatures = sorted(c[0] for c in candidates)
            if entry is None or sorted(entry["signatures"]) != signatures:
                continue
            pending.append((index, position,
                            Target(sentence, start, end, key, tuple(signatures))))

    # Morphology decides the ambiguities whose readings differ by case or
    # number rather than by sense. On this benchmark's vocabulary that is 279
    # of 382 unglossed ambiguous forms — answering them with a dictionary
    # default instead is a coin flip on cases a parser settles outright.
    morphology = (SpacyMorphologyTier(args.spacy_model) if args.spacy_model
                  else MorphologyTier())
    reading_cache: dict[str, Any] = {}
    morph_choice: dict[tuple[int, int], str] = {}
    if not args.no_morphology:
        covered = {(i, p) for i, p, _ in pending}
        for index, (sentence, spans) in enumerate(zip(sentences, tokenised, strict=True)):
            wanted = [(p, s_, e_, sf, k) for p, (s_, e_, sf, k) in enumerate(spans)
                      if len(variants.get(k, [])) > 1
                      and (args.morphology_all or (index, p) not in covered)]
            if not wanted:
                continue
            try:
                parsed = morphology.parse(sentence)
            except Exception as error:  # noqa: BLE001
                print(f"  parse failed: {str(error)[:60]}", flush=True)
                continue
            for position, start, end, surface, key in wanted:
                if key not in reading_cache:
                    reading_cache[key] = morphology.readings(key)
                if not reading_cache[key]:
                    continue
                parse = parsed.get((start, end))
                if not parse:
                    continue
                resolution = morph_resolve(reading_cache[key], parse[0], parse[1])
                if resolution is None:
                    continue
                accents = list(resolution.accents)
                if "-" not in surface and len(accents) > 1:
                    # Several accents on a single token is the dictionary's
                    # "either stress is acceptable" notation, not two stresses
                    # to emit. The lookup path already guards this; the
                    # morphology path builds its output separately and did not,
                    # so it produced `гі+лля+`, which is not a word.
                    accents = accents[:1]
                stressed = apply_accents(surface, accents)
                marked = unicodedata.normalize("NFC", stressed).replace(ACUTE, "+")
                morph_choice[(index, position)] = restore_apostrophes(surface, marked)
        print(f"morphology resolved: {len(morph_choice):,}", flush=True)

    # Ambiguity class per form. The routing below turns on what *kind* of
    # ambiguity a form has, which is a property of the form, not of the run.
    triage: dict[str, str] = {}
    if args.triage:
        for line in args.triage.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            triage[row["form_normalized"]] = row["triage_class"]
        print(f"triage classes: {len(triage):,}", flush=True)

    chosen: dict[tuple[int, int], str] = {}
    # The model pass dominates the run on CPU, and it does not depend on the
    # threshold at all -- only the comparison below does. Caching the decisions
    # turns a threshold sweep from N full runs into one.
    cache: dict[str, dict[str, Any]] = {}
    if args.decisions_cache and args.decisions_cache.exists():
        cache = json.loads(args.decisions_cache.read_text(encoding="utf-8"))
        print(f"decisions from cache: {len(cache):,}", flush=True)
    if pending:
        print(f"model targets: {len(pending):,}", flush=True)
        missing = [(i, p, t) for i, p, t in pending if f"{i}:{p}" not in cache]
        for offset in range(0, len(missing), args.batch):
            chunk = missing[offset:offset + args.batch]
            for (index, position, _), decision in zip(
                    chunk, model.resolve([t for _, _, t in chunk]), strict=True):
                cache[f"{index}:{position}"] = {
                    "signature": decision["signature"],
                    "margin": decision["margin"],
                }
            print(f"  {min(offset + args.batch, len(missing)):,}/{len(missing):,}", flush=True)
        if args.decisions_cache and missing:
            args.decisions_cache.write_text(
                json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        for index, position, _ in pending:
            decision = cache[f"{index}:{position}"]
            if decision["margin"] >= args.threshold:
                chosen[(index, position)] = decision["signature"]

    # The masked model is a second opinion on one class only. It is loaded and
    # run separately rather than swapped in for the cross-encoder because the
    # two disagree about *which* forms they are good at: measured head to head
    # across every ambiguous token the classifier lost by 22 tokens (p=0.025),
    # which is a reason to give it the homographs alone, not the whole tier.
    masked: dict[tuple[int, int], str] = {}
    if args.mask_model and pending:
        wanted = [(i, p_, tgt) for i, p_, tgt in pending
                  if triage.get(tgt.form, "") == "homograph"]
        print(f"masked-model targets (homograph forms): {len(wanted):,}", flush=True)
        if wanted:
            second = ClassifierStressModel(
                args.mask_model, args.manifest, device=args.device)
            for offset in range(0, len(wanted), args.batch):
                chunk = wanted[offset:offset + args.batch]
                for (index, position, _), decision in zip(
                        chunk, second.resolve([t_ for _, _, t_ in chunk]), strict=True):
                    if decision["margin"] >= args.threshold:
                        masked[(index, position)] = decision["signature"]
                print(f"  masked {min(offset + args.batch, len(wanted)):,}/{len(wanted):,}",
                      flush=True)

    def compound_fallback(surface: str, key: str) -> str | None:
        """Stress a hyphenated compound the lexicon lacks as a whole.

        `адміністративно-територіальна` is absent from `stress_lookup`, but both
        halves are present. An ad-hoc compound is not a lexicon entry and never
        will be — there are unboundedly many — so it has to be resolved from its
        parts. 55 of the 89 hyphenated forms missing from this benchmark's
        vocabulary have every part covered.
        """
        if "-" not in key:
            return None
        pieces, offset = [], 0
        for part in surface.split("-"):
            part_key = lookup_key(part)
            candidates = variants.get(part_key, [])
            if not candidates:
                # A part with no entry keeps its own spelling unstressed; the
                # rest of the compound is still worth marking.
                pieces.append(part)
            else:
                marked = to_plus(part, candidates[0][0])
                pieces.append(marked if marked else part)
            offset += len(part) + 1
        joined = "-".join(pieces)
        return joined if "+" in joined else None

    suffix_table: dict[str, int] = {}
    if args.suffix_table:
        suffix_table = {k: int(v) for k, v in
                        json.loads(args.suffix_table.read_text(encoding="utf-8")).items()}
        print(f"suffix table: {len(suffix_table):,} entries", flush=True)

    def suffix_fallback(surface: str) -> str | None:
        """Stress a word no tier covers, by analogy with lexicon word endings.

        An unstressed multi-vowel word scores zero under the benchmark's Rule 4,
        so a 74.5%-precise guess strictly beats leaving it bare. The table is
        keyed on distance from the *last* vowel, which is what lets the analogy
        carry between words of different length.
        """
        if not suffix_table:
            return None
        decomposed = unicodedata.normalize("NFD", surface).lower()
        positions = [i for i, c in enumerate(decomposed) if c in VOWELS]
        if len(positions) < 2:
            return None
        for length in range(min(13, len(decomposed)), 0, -1):
            from_end = suffix_table.get(decomposed[-length:])
            if from_end is None or from_end >= len(positions):
                continue
            return to_plus(surface, str(len(positions) - 1 - from_end))
        return None

    def route_token(index: int, position: int, surface: str, key: str,
                    candidates: list) -> tuple[str | None, str]:
        """Pick a tier from the *kind* of ambiguity rather than a fixed order.

        A homograph's readings carry the same tags, so a tagger has nothing to
        separate them by and morphology is not merely second choice, it is
        inapplicable — measured, it claims 1 homograph token in 202. A
        grammatical alternation is the mirror image: the tags are exactly what
        differs, and morphology is right on 83.6% of the ones it reaches.
        Free variation is neither, both readings being acceptable; the model
        scores 10.0% there not by misreading the sentence but by guessing which
        of two right answers the annotator wrote down, so it goes to the
        dictionary, whose order at least is stable.
        """
        cls = triage.get(key, "unknown")
        if cls == "homograph":
            signature = masked.get((index, position)) or chosen.get((index, position))
            if signature is not None:
                return to_plus(surface, signature), (
                    "mask" if (index, position) in masked else "model")
        elif cls == "grammatical":
            marked = morph_choice.get((index, position))
            if marked is not None:
                return marked, "morphology"
            signature = chosen.get((index, position))
            if signature is not None:
                return to_plus(surface, signature), "model"
        elif cls == "free_variation":
            return to_plus(surface, candidates[0][0]), "lexicon_default"
        else:
            marked = morph_choice.get((index, position))
            if marked is not None:
                return marked, "morphology"
            signature = chosen.get((index, position))
            if signature is not None:
                return to_plus(surface, signature), "model"
        return to_plus(surface, candidates[0][0]), "lexicon_ambiguous"

    resolved: dict[str, str] = {}
    # Which tier actually decided each token. Accuracy alone cannot say whether
    # more training data is the right lever; this can.
    tiers: list[dict[str, Any]] = []
    for index, (sentence, spans) in enumerate(zip(sentences, tokenised, strict=True)):
        pieces, cursor = [], 0
        for position, (start, end, surface, key) in enumerate(spans):
            candidates = variants.get(key, [])
            if not candidates:
                marked = compound_fallback(surface, key)
                tier = "compound"
                if marked is None:
                    marked, tier = suffix_fallback(surface), "suffix"
                if marked is None:
                    tiers.append({"sentence": index, "surface": surface,
                                  "tier": "uncovered", "marked": None})
                    continue
                tiers.append({"sentence": index, "surface": surface,
                              "tier": tier, "marked": marked})
                pieces.append(sentence[cursor:start])
                pieces.append(marked)
                cursor = end
                continue
            if args.route and len(candidates) > 1:
                marked, tier = route_token(index, position, surface, key, candidates)
            else:
                marked = morph_choice.get((index, position))
                tier = "morphology"
                if marked is None:
                    signature = chosen.get((index, position))
                    tier = "model" if signature is not None else (
                        "lexicon_ambiguous" if len(candidates) > 1 else "lexicon")
                    marked = to_plus(surface, signature or candidates[0][0])
            tiers.append({"sentence": index, "surface": surface,
                          "tier": tier, "marked": marked,
                          "options": len(candidates)})
            if marked is None:
                continue
            pieces.append(sentence[cursor:start])
            pieces.append(marked)
            cursor = end
        pieces.append(sentence[cursor:])
        resolved[sentence] = "".join(pieces)
    if args.tiers:
        args.tiers.write_text(json.dumps(tiers, ensure_ascii=False), encoding="utf-8")
        print(f"tier attribution -> {args.tiers}", flush=True)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="clone of lang-uk/ukrainian-tts-preprocessing")
    parser.add_argument("--model", type=Path, default=Path("output/ml/models/v11-share"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_v11.json"))
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--extra-dataset", type=int, action="append", default=None)
    # Unset means "whatever the manifest says", which is what serving uses.
    # A separate default here silently measured a threshold the API never runs.
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--prefer-trie", type=Path,
                        help="path to stress.trie; an unambiguous trie reading "
                             "becomes the dictionary default for a form the "
                             "lexicon leaves ambiguous")
    parser.add_argument("--suffix-table", type=Path,
                        help="last-resort stress for words no tier covers; "
                             "74.5%% precise on held-out lexicon forms, against "
                             "the 0%% an unstressed word scores")
    parser.add_argument("--mask-model", type=Path,
                        help="second checkpoint consulted only for homograph "
                             "forms; a classifier head scores the masked "
                             "signature vocabulary instead of ranking glosses")
    parser.add_argument("--triage", type=Path,
                        help="ambiguous_surface jsonl; enables class routing")
    parser.add_argument("--route", action="store_true",
                        help="route by ambiguity class instead of a fixed tier "
                             "order: one reading to the dictionary, homographs "
                             "to the masked model, grammatical forms to "
                             "morphology, free variation to the dictionary")
    parser.add_argument("--tiers", type=Path,
                        help="dump which tier decided each token")
    parser.add_argument("--decisions-cache", type=Path,
                        help="reuse the model pass across threshold sweeps")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--no-morphology", action="store_true")
    parser.add_argument("--morphology-all", action="store_true",
                        help="ask morphology about every ambiguous token, not "
                             "only the ones outside model coverage; without it "
                             "a form the manifest covers never reaches a tagger")
    parser.add_argument("--spacy-model",
                        help="path to a spaCy Ukrainian model; replaces Stanza "
                             "as the morphological tagger")
    parser.add_argument("--out", type=Path, default=Path("output/ml/langukbench.json"))
    args = parser.parse_args()
    if args.extra_dataset is None:
        args.extra_dataset = [3, 4, 5]
    if args.threshold is None:
        args.threshold = float(json.loads(
            args.manifest.read_text(encoding="utf-8"))["threshold"])
        print(f"threshold from manifest: {args.threshold}", flush=True)

    import pandas as pd

    data = args.benchmark / "lexical_stress_benchmark" / "data"
    frame = pd.read_csv(data / "lexical_stress_dataset.csv")
    gold = [str(s) for s in frame["StressedSentence"]]
    plain = [s.replace("+", "") for s in gold]
    print(f"benchmark sentences: {len(plain):,}", flush=True)

    resolved = resolve_dataset(plain, args)

    # Import the maintainers' scorer and run it against their own dataset, from
    # their examples/ directory: the harness resolves data paths relative to cwd.
    import os
    import sys
    sys.path.insert(0, str(args.benchmark))
    cwd = os.getcwd()
    os.chdir(args.benchmark / "lexical_stress_benchmark" / "examples")
    try:
        from lexical_stress_benchmark import evaluate_stressification
        accuracies = evaluate_stressification(
            lambda text: resolved.get(text, text), show_progress=False,
            raise_on_mismatch=False)
    finally:
        os.chdir(cwd)

    names = ["sentence_accuracy", "word_accuracy", "heteronym_accuracy",
             "unambiguous_accuracy", "macro_f1_heteronyms"]
    values = list(accuracies.values())
    report: dict[str, Any] = dict(zip(names, values, strict=False))
    report["model"] = str(args.model)
    report["threshold"] = args.threshold
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'metric':<38}{'score':>8}")
    for name, value in zip(names, values, strict=False):
        print(f"{name:<38}{value * 100:>7.2f}%")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
