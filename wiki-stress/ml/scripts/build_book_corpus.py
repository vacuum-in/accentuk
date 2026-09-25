"""Build the token-classifier corpus from the audiobook sweeps.

`tok-v1` scored 66.3% on forms it had not seen, against 66.4% for answering
`candidates[0]` every time — it had learned which form, not which context. The
corpus said why: both readings of a form were present in 187 cases out of
2,047. Continuous prose fixes exactly that, because a homograph recurs through
a novel in the constructions that separate its readings.

Three things have to be recovered that the label rows do not carry. The row
knows its second on the clock, not its place in the book, so the anchor puts it
back: `anchor["text"]` is `stream[word_start:word_end]` verbatim, the miner cut
that into 60-second slices, and the rows of a slice are a subsequence of its
words in order. Matching them greedily gives each row a stream index, the
stream index gives a paragraph and a character span, and the paragraph gives
the sentence — with its punctuation and capitals, which the anchor text has
lost and the classifier will see at inference.

The per-form cap is per *reading*. A global cap fills up on the majority
reading of a common form and throws away the minority one, which is the only
part of the row that teaches context.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

# ukrainian_word_stress ships inside the uv archive the miner points at; the
# ml venv has marisa_trie but not the package itself.
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

VOWELS = "аеєиіїоуюяй"
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")
BREAK = re.compile(r"[.!?…]['\"»)\]]*\s")


def book_words(path: Path):
    """The book as one lowercase word stream, plus where each word came from.

    Must stay identical to run_anchor_book.book_words — the anchors were built
    with it, and a different tokenisation silently shifts every index.
    """
    stream, origin, texts = [], [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts[(row["doc"], row["paragraph"])] = row["text"]
        for index, word in enumerate(WORD.findall(row["text"])):
            stream.append(unicodedata.normalize("NFC", word.lower()))
            origin.append((row["doc"], row["paragraph"], index))
    return stream, origin, texts


def sentence_around(text: str, start: int, end: int, width: int = 320):
    """The sentence holding this span, trimmed to something a 160-token
    encoder can hold. Returns the text and the span's place inside it."""
    left = 0
    for match in BREAK.finditer(text, 0, start):
        left = match.end()
    right = len(text)
    found = BREAK.search(text, end)
    if found:
        right = found.end()
    if start - left > width:
        left = max(left, text.rfind(" ", start - width, start) + 1)
    if right - end > width:
        cut = text.find(" ", end + width)
        right = cut if cut != -1 else right
    return text[left:right].strip(), start - left, end - left


def readings_of(trie, word: str) -> list[str]:
    """Every distinct reading the lexicon offers, as vowel ordinals."""
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value

    normalised = unicodedata.normalize("NFC", word)
    value = _trie_value(trie, normalised)
    if value is None:
        return []
    out = set()
    for accents in {tuple(a) for _, a in _parse_dictionary_value(value[0])}:
        ordinals = []
        for position in accents:
            ordinal = -1
            for index, character in enumerate(normalised, 1):
                if character.lower() in VOWELS:
                    ordinal += 1
                if index == position:
                    ordinals.append(ordinal)
                    break
        if ordinals:
            out.add("|".join(str(o) for o in sorted(ordinals)))
    return sorted(out)


KEEP = ("book", "sentence", "start", "end", "token", "form", "readings", "label",
        "audio", "confidence", "pipeline_agreed")


def rows_of_book(slug: str, books: Path, labels_suffix: str,
                 negatives: int = 0, extra_ambiguous: frozenset = frozenset()):
    """Every labelled row of one book, placed in its sentence.

    The miner records the paragraph and the word's index within it, so this is
    a lookup rather than a reconstruction. The previous version had to replay
    the miner's slice arithmetic to guess where a row came from; that miner is
    gone and so is the guessing.
    """
    label_files = [books / f"{slug}{suffix}" for suffix in labels_suffix.split(",")]
    label_files = [f for f in label_files if f.exists()]
    # v3 rows carry the lexicon's reading and the ranker's answer in one row,
    # so the same file is both the labels and the source of negatives.
    v3 = books / f"{slug}.rows3.jsonl"
    text_path = books / f"{slug}.jsonl"
    if not ((label_files or v3.exists()) and text_path.exists()):
        return [], collections.Counter({"missing_input": 1})

    # Two sources. The labeller only answers what the lexicon cannot, so the
    # unambiguous rows — the negatives that stop the classifier from assuming
    # every word it sees is a homograph — come from the mined rows themselves,
    # where the lexicon's own reading is the gold.
    def read(path: Path):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)

    paragraphs = {}
    for line in text_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            paragraphs[(row["doc"], row["paragraph"])] = row["text"]

    rows = [r for f in label_files for r in read(f)]
    mined = [f for f in (books / f"{slug}.rows2.jsonl", books / f"{slug}.rows2b.jsonl",
                         books / f"{slug}.rows22.jsonl") if f.exists()]
    if v3.exists():
        every = list(read(v3))
        rows += [r for r in every if r.get("readings") != 1
                 or (r.get("audio") is not None and r["form"] in extra_ambiguous)]
        mined_v3 = [r for r in every if r.get("readings") == 1 and r.get("label") is not None]
    else:
        mined_v3 = []
    if negatives and (mined or mined_v3):
        plain = mined_v3 + [r for f in mined for r in read(f)
                            if r.get("readings") == 1 and r.get("label") is not None]
        random.Random(17).shuffle(plain)
        for row in plain[:negatives]:
            rows.append({**row, "audio": row["label"], "confidence": 1.0})

    out = []
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        text = paragraphs.get((row["doc"], row["paragraph"]))
        if not text:
            counts["no_paragraph"] += 1
            continue
        spans = list(WORD.finditer(text))
        if row["word_index"] >= len(spans):
            counts["span_missing"] += 1
            continue
        match = spans[row["word_index"]]
        sentence, start, end = sentence_around(text, match.start(), match.end())
        if not sentence or sentence[start:end].lower() != row["form"]:
            counts["sentence_mismatch"] += 1
            continue
        out.append({**row, "sentence": sentence, "start": start, "end": end})
        counts["located"] += 1
    return out, counts


VOWEL_SET = frozenset("аеєиіїоуюяй")


def build_placer(located: list[dict], args) -> int:
    """Rows for words the lexicon lacks. The ranker's pick is the gold; every
    vowel ordinal is a candidate; a tenth of forms goes to dev and a sixth to
    test, whole forms only."""
    rows, per_form = [], collections.Counter()
    drops: collections.Counter[str] = collections.Counter()
    random.Random(args.seed).shuffle(located)
    for row in located:
        if row.get("readings", 1) != 0:
            continue
        form = row["form"].lower()
        vowels = sum(1 for c in form if c in VOWEL_SET)
        # й has an ordinal but is never stressed: not a candidate, and a
        # ranker pick on it is an alignment slip, not a reading.
        candidates, seen = [], -1
        for c in form:
            if c in VOWEL_SET:
                seen += 1
                if c != "й":
                    candidates.append(str(seen))
        if len(candidates) < 2 or str(row["audio"]) not in candidates:
            drops["bad_span"] += 1
            continue
        bookless = row.pop("bookless", False)
        gate = (args.audio_min_confidence if bookless and args.audio_min_confidence is not None
                else args.min_confidence)
        if row["confidence"] < gate:
            drops["low_confidence"] += 1
            continue
        if per_form[form] >= args.placer_cap:
            drops["per_form_cap"] += 1
            continue
        per_form[form] += 1
        rows.append({**row, "candidates": candidates,
                     "gold": str(row["audio"]), "source": "book-oov", "weight": 1.0})
    forms = sorted(per_form)
    random.Random(args.seed).shuffle(forms)
    dev_forms = set(forms[: len(forms) // 10])
    test_forms = set(forms[len(forms) // 10: len(forms) // 10 + len(forms) // 6])
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        form = row["form"].lower()
        where = "dev" if form in dev_forms else "test" if form in test_forms else "train"
        row["unseen_form"] = where != "train"
        buckets[where].append(row)
    print(f"placer: {len(rows):,} rows over {len(forms):,} forms; drops {dict(drops)}")
    counts = {"rows": len(rows), "forms": len(forms), **{f"dropped_{k}": v for k, v in drops.items()}}
    for name, part in buckets.items():
        (args.out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n", encoding="utf-8")
        counts[f"{name}_rows"] = len(part)
        print(f"  {name:<6}{len(part):>9,} rows")
    (args.out / "manifest.json").write_text(json.dumps(
        {"counts": counts, "gates": {"min_confidence": args.min_confidence, "placer_cap": args.placer_cap},
         "seed": args.seed}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/books"))
    parser.add_argument("--trie", default=None,
                        help="defaults to the installed ukrainian_word_stress package's")
    parser.add_argument("--labels",
                        default=".labels2.jsonl,.labels_rows2b.jsonl,.labels_rows22.jsonl",
                        help="suffixes of the ranker's output per book, one per "
                             "mining pass, comma-separated")
    parser.add_argument("--min-confidence", type=float, default=0.95,
                        help="calibrated on the books themselves: 98.2%% right "
                             "at 0.95 against 98.8%% at 0.99, and the lower gate "
                             "keeps almost twice the forms that show both readings")
    parser.add_argument("--per-reading-cap", type=int, default=80)
    parser.add_argument("--negatives-per-book", type=int, default=4000,
                        help="unambiguous rows sampled from each book before "
                             "gating; the ratio below trims them afterwards")
    parser.add_argument("--negatives", type=float, default=1.0,
                        help="unambiguous rows per ambiguous row, as a ratio")
    parser.add_argument("--merge-train", type=Path, default=None,
                        help="a tok-v1 corpus whose train rows are folded in; "
                             "dev and test stay books-only so the held-out "
                             "form condition means one thing")
    parser.add_argument("--out", type=Path, default=Path("output/ml/corpus/tok-v2"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--db-ambiguous", type=Path,
                        help="TSV of form|sig,sig from the served lexicon: forms "
                             "the API offers two readings for. The trie the miner "
                             "used calls 3,400 of them single-reading (мені, які, "
                             "також, трохи), so their rows never reached the "
                             "classifier while the cross-encoder read them at 61 "
                             "percent. With this file a row is ambiguous when the "
                             "database says so, with the database candidates")
    parser.add_argument("--placer", action="store_true",
                        help="build the stress-placer corpus instead: words the "
                             "lexicon does not have at all, every vowel a "
                             "candidate, the ranker's reading the gold, split by "
                             "form only — an out-of-vocabulary word at inference "
                             "is by definition one the placer never saw")
    parser.add_argument("--placer-cap", type=int, default=40,
                        help="rows per form in the placer corpus, so a name that "
                             "recurs through a novel does not own the loss")
    parser.add_argument("--audio-weight", type=float, default=1.0,
                        help="training weight of the book-less (YouTube) rows; "
                             "tok-v6 at 1.0 cost 2 points on lang-uk")
    parser.add_argument("--audio-min-confidence", type=float, default=None,
                        help="ranker confidence gate for the book-less rows alone "
                             "(default: --min-confidence); spontaneous speech gives "
                             "the ranker 90-94 pct against audiobooks' 96-97 pct")
    parser.add_argument("--audio-majority-only", action="store_true",
                        help="keep a book-less row only when its reading is the "
                             "form's majority reading over the whole corpus: the ear "
                             "check put the ranker at ~66 pct on the minority readings "
                             "of spontaneous speech and ~100 pct on the majority ones")
    parser.add_argument("--audio-rows", type=Path, nargs="*", default=[],
                        help="rows from the book-less miner (run_mine_audio_v4.py): "
                             "they carry the sentence and span already, so they join "
                             "the located rows as they are")
    parser.add_argument("--only", type=Path,
                        help="JSON list of slugs: build from these books alone, so a "
                             "corpus can be rebuilt on exactly the book set of an "
                             "earlier one")
    parser.add_argument("--exclude", default="",
                        help="slugs to leave out, comma-separated, on top of the "
                             "exclusion file")
    parser.add_argument("--exclusions", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/books/book_exclusions.json"),
                        help="JSON list of {slug, why}: archaic originals, verse, "
                             "duplicates and wrong-audio folders. Measured: "
                             "training on them cost ~1 point on modern text")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import marisa_trie

    if args.trie is None:
        import ukrainian_word_stress
        args.trie = str(Path(ukrainian_word_stress.__file__).parent / "data" / "stress.trie")
    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)

    first = args.labels.split(",")[0]
    extra_ambiguous: frozenset = frozenset()
    if args.db_ambiguous and args.db_ambiguous.exists():
        extra_ambiguous = frozenset(
            line.split("|", 1)[0].strip()
            for line in args.db_ambiguous.read_text(encoding="utf-8").splitlines() if "|" in line)
    slugs = sorted({p.name[: -len(first)] for p in args.books.glob(f"*{first}")}
                   | {p.name[: -len(".rows3.jsonl")] for p in args.books.glob("*.rows3.jsonl")})
    excluded = set(args.exclude.split(",")) if args.exclude else set()
    if args.exclusions.exists():
        excluded |= {e["slug"] if isinstance(e, dict) else e
                     for e in json.loads(args.exclusions.read_text(encoding="utf-8"))}
    slugs = [s for s in slugs if s not in excluded]
    if args.only:
        keep = set(json.loads(args.only.read_text(encoding="utf-8")))
        slugs = [s for s in slugs if s in keep]
    print(f"{len(excluded)} books excluded", flush=True)
    print(f"{len(slugs)} books", flush=True)

    located: list[dict] = []
    totals: collections.Counter[str] = collections.Counter()
    for slug in slugs:
        rows, counts = rows_of_book(slug, args.books, args.labels,
                                    negatives=args.negatives_per_book,
                                    extra_ambiguous=extra_ambiguous)
        totals.update(counts)
        located.extend(rows)
        print(f"  {slug:<50} {counts['located']:>7,} located "
              f"{sum(counts.values()) - counts['located']:>6,} lost", flush=True)
    for path in args.audio_rows:
        added = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sentence") and row.get("start") is not None:
                # Only what the gate and the trainer read: five million
                # YouTube rows with their durations and vowel spans were 27 GB.
                located.append({k: row[k] for k in KEEP if k in row}
                               | {"doc": row.get("doc", 0), "bookless": True,
                                  "paragraph": row.get("paragraph", row.get("window", 0))})
                added += 1
        print(f"  {path.name:<50} {added:>7,} audio rows", flush=True)
    print(f"\n{len(located):,} rows placed in the text; "
          f"losses {dict(totals - collections.Counter({'located': totals['located']}))}",
          flush=True)

    if args.placer:
        return build_placer(located, args)

    # Gate and cap. Both readings of a form matter equally, so the cap is per
    # reading — this is the whole reason the book corpus exists.
    db_candidates: dict[str, list[str]] = {}
    if args.db_ambiguous and args.db_ambiguous.exists():
        for line in args.db_ambiguous.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                form, sigs = line.rstrip().split("|", 1)
                db_candidates[form.strip()] = [c for c in sigs.split(",") if c.strip().isdigit()]
        print(f"{len(db_candidates):,} database-ambiguous forms", flush=True)

    candidates_of: dict[str, list[str]] = {}
    ambiguous, negatives = [], []
    drops: collections.Counter[str] = collections.Counter()
    per_reading: collections.Counter[tuple[str, str]] = collections.Counter()
    random.Random(args.seed).shuffle(located)
    majority: dict[str, str] = {}
    if args.audio_majority_only:
        tally: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for row in located:
            tally[row["form"].lower()][str(row["audio"])] += 1
        majority = {form: c.most_common(1)[0][0] for form, c in tally.items()}
    rng = random.Random(args.seed)
    seen_plain = 0
    reservoir = int(len(located) * min(args.negatives, 1.0) * 0.5) or len(located)
    for row in located:
        form = row["form"].lower()
        if form not in candidates_of:
            trie_readings = readings_of(trie, form)
            db = db_candidates.get(form)
            # The served candidates decide: what the API offers is what the
            # classifier must choose between. The trie fills in where the
            # database has no opinion.
            candidates_of[form] = db if db and len(db) >= 2 else trie_readings
        candidates = candidates_of[form]
        gold = str(row["audio"])
        if len(candidates) < 2:
            if len(candidates) == 1 and candidates[0] == gold:
                # Reservoir of the size the corpus can use (negatives is a
                # ratio of the ambiguous rows, at most their count at 1.0),
                # rather than every unambiguous row of five million.
                seen_plain += 1
                sample = {**row, "candidates": candidates, "gold": gold,
                          "source": "youtube-unambiguous" if row.pop("bookless", False) else "book-unambiguous",
                          "weight": 1.0}
                if len(negatives) < reservoir:
                    negatives.append(sample)
                else:
                    slot = rng.randrange(seen_plain)
                    if slot < reservoir:
                        negatives[slot] = sample
            else:
                drops["not_ambiguous"] += 1
            continue
        if gold not in candidates:
            drops["reading_not_offered"] += 1
            continue
        bookless = row.pop("bookless", False)
        gate = (args.audio_min_confidence if bookless and args.audio_min_confidence is not None
                else args.min_confidence)
        if row["confidence"] < gate:
            drops["low_confidence"] += 1
            continue
        if bookless and args.audio_majority_only and gold != majority.get(form):
            drops["bookless_minority"] += 1
            continue
        if per_reading[(form, gold)] >= args.per_reading_cap:
            drops["per_reading_cap"] += 1
            continue
        per_reading[(form, gold)] += 1
        ambiguous.append({**row, "candidates": candidates, "gold": gold,
                          "source": "youtube-audio" if bookless else "book-audio",
                          "weight": args.audio_weight if bookless else 1.0})

    forms = sorted({f for f, _ in per_reading})
    both = sum(1 for f in forms if sum(1 for g, _ in per_reading if g == f) > 1)
    print(f"{len(ambiguous):,} ambiguous rows over {len(forms):,} forms "
          f"({both:,} with two readings present); drops {dict(drops)}", flush=True)

    random.Random(args.seed).shuffle(negatives)
    negatives = negatives[:int(len(ambiguous) * args.negatives)]

    # Split. A tenth of the forms is held out entirely — that condition is what
    # tok-v1 failed — and the rest splits by book position so a paragraph never
    # appears on both sides.
    random.Random(args.seed).shuffle(forms)
    unseen = set(forms[: max(1, len(forms) // 10)])
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in ambiguous + negatives:
        form = row["form"].lower()
        if form in unseen:
            row["unseen_form"] = True
            buckets["test"].append(row)
            continue
        draw = random.Random(f"{row['book']}:{row['doc']}:{row['paragraph']}").random()
        buckets["dev" if draw < 0.08 else
                "test" if draw < 0.20 else "train"].append(row)

    merged = 0
    if args.merge_train:
        extra = [json.loads(line) for line in
                 (args.merge_train / "train.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        extra = [r for r in extra if r["form"].lower() not in unseen]
        buckets["train"].extend(extra)
        merged = len(extra)
        print(f"folded {merged:,} Common Voice train rows in", flush=True)

    counts = {"rows_located": len(located), "ambiguous_rows": len(ambiguous),
              "ambiguous_forms": len(forms), "forms_with_both_readings": both,
              "negative_rows": len(negatives), "held_out_forms": len(unseen),
              "merged_train_rows": merged,
              **{f"dropped_{k}": v for k, v in drops.items()},
              **{f"lost_{k}": v for k, v in totals.items() if k != "located"}}
    for name, part in buckets.items():
        (args.out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n",
            encoding="utf-8")
        counts[f"{name}_rows"] = len(part)
        print(f"  {name:<6}{len(part):>9,} rows")

    (args.out / "manifest.json").write_text(json.dumps(
        {"counts": counts, "gates": {"min_confidence": args.min_confidence,
                                     "per_reading_cap": args.per_reading_cap},
         "seed": args.seed}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
