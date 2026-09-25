"""Mine book text for the contexts the wiki corpus does not contain.

The silver corpus is 88.6% encyclopedic prose, and for the aspect pairs that
shows: `виходити` occurs 184 times and is labelled `вихо́дити` all 184, so the
model has never seen the perfective at all and cannot answer «вдалося
ви́ходити пацієнта» whatever its confidence says. The gap is in the data, and
encyclopedias simply do not write the sentences that would fill it.

Fiction does. This walks a book collection, pulls the sentences that contain a
form from the target list, and keeps them with enough context to label — with
the trigger word recorded where one is present, so the rows that decide the
question can be labelled first.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import logging
import re
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
SENTENCE = re.compile(r"[^.!?…]+[.!?…]+|\S[^.!?…]*$")
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")

#: Words that force the infinitive after them to one aspect. Recorded rather
#: than acted on: a labelling queue wants the evidence, not a verdict.
IMPERFECTIVE = frozenset({
    "почав", "почала", "почало", "почали", "почати", "починає", "починають",
    "став", "стала", "стало", "стали", "стати",
    "продовжив", "продовжила", "продовжили", "продовжував", "продовжувала",
    "продовжували", "продовжувати", "перестав", "перестала", "перестали",
    "припинив", "припинила", "припинили", "закінчив", "закінчила",
})
PERFECTIVE = frozenset({
    "вдалося", "удалося", "вдалась", "зумів", "зуміла", "зуміли", "встиг", "встигла", "встигли", "встигнути", "спромігся", "спромоглася",
    "зміг", "змогла", "змогли", "мусив", "мусила", "муситиме",
})


def text_of(page: dict) -> str:
    raw = page.get("html") or page.get("text") or ""
    return SPACE.sub(" ", html.unescape(TAG.sub(" ", raw))).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--targets", type=Path,
                        default=Path("output/ml/aspect_pairs.json"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/book_contexts.json"))
    parser.add_argument("--per-form", type=int, default=40,
                        help="cap per form, so one common verb cannot fill the file")
    parser.add_argument("--min-words", type=int, default=5)
    args = parser.parse_args()

    pairs = json.loads(args.targets.read_text(encoding="utf-8"))
    targets = {row["form"]: row for row in pairs}
    print(f"target forms: {len(targets):,}", flush=True)

    found: list[dict] = []
    per_form: collections.Counter[str] = collections.Counter()
    triggers: collections.Counter[str] = collections.Counter()
    books = sorted(args.books.glob("*.json"))
    for index, path in enumerate(books, 1):
        if index % 50 == 0:
            print(f"  {index}/{len(books)} books, {len(found):,} contexts", flush=True)
        try:
            book = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 - one bad book cannot stop a scan
            log.debug("skipping %s: %s", path.name, error)
            continue
        # A few files are a bare page list rather than the usual book object.
        if isinstance(book, list):
            book = {"title": path.stem, "pages": book}
        if not isinstance(book, dict):
            continue
        title = book.get("title", path.stem)
        for page in book.get("pages", []):
            if not isinstance(page, dict):
                continue
            body = text_of(page)
            if not body:
                continue
            for sentence in SENTENCE.findall(body):
                sentence = sentence.strip()
                words = WORD.findall(sentence)
                if len(words) < args.min_words or len(sentence) > 400:
                    continue
                lowered = [unicodedata.normalize("NFC", w).lower() for w in words]
                for position, word in enumerate(lowered):
                    if word not in targets:
                        continue
                    before = lowered[max(0, position - 4):position]
                    trigger = next((w for w in reversed(before)
                                    if w in IMPERFECTIVE or w in PERFECTIVE), "")
                    kind = ("imperfective" if trigger in IMPERFECTIVE
                            else "perfective" if trigger in PERFECTIVE else "")
                    # The cap keeps one common verb from filling the file, but
                    # it must not apply to a sentence carrying a trigger: those
                    # are the rows the corpus lacks, and capping them threw
                    # away exactly what the scan is for.
                    if not kind and per_form[word] >= args.per_form:
                        continue
                    match = re.search(rf"\b{re.escape(words[position])}\b", sentence)
                    if match is None:
                        continue
                    per_form[word] += 1
                    triggers[kind or "none"] += 1
                    found.append({
                        "sentence": sentence,
                        "form": word,
                        "start": match.start(),
                        "end": match.end(),
                        "trigger": trigger,
                        "expects": kind,
                        "readings": targets[word]["readings"],
                        "source": title,
                    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nbooks read: {len(books)}")
    print(f"contexts kept: {len(found):,} over {len(per_form):,} distinct forms")
    print("by trigger:")
    for kind, count in triggers.most_common():
        print(f"  {kind:<14}{count:>7}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
