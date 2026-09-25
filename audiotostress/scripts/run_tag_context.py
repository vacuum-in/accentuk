"""Back-fill the word before each mined row, and flag prepositional clitics.

Error analysis on held-out speakers found the misses clustered on `мене`,
`себе`, `тебе`, `кому`, `поки` — not a random scatter. These are the forms the
prepositional rule moves: the trie names the final syllable (мене́), but after
a governing preposition Ukrainian says до ме́не, and the label then calls a
correct reading wrong.

The acoustics settle it independently of any dictionary. After a preposition
the ranker picks the first syllable 62.8% of the time; bare, 15.7% — and
agreement with the trie label drops from 88.2% to 67.4% in exactly that
context. The model hears the shift; the label does not know about it.

So these rows are contaminated, not hard. This records the preceding word on
every row so the trainer can drop them, and so any later question about
context has an answer already in the file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PREPOSITIONS = {
    "до", "у", "в", "на", "за", "під", "над", "про", "від", "од", "біля",
    "для", "без", "при", "через", "перед", "між", "із", "з", "зі", "повз",
    "поза", "попід", "коло", "після", "проти", "серед", "крім", "щодо",
}
CLITICS = {
    "мене", "тебе", "себе", "нього", "неї", "них", "нас", "вас",
    "мною", "тобою", "собою", "ньому", "нім",
}
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/audio_runs/commonvoice/rows.jsonl"))
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/cv-corpus-26.0-2026-06-12/uk"))
    parser.add_argument("--tsv", default="validated.tsv")
    args = parser.parse_args()

    sentences: dict[str, str] = {}
    with (args.corpus / args.tsv).open(encoding="utf-8", newline="") as handle:
        for entry in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            sentences[entry["path"]] = entry["sentence"]

    tagged = flagged = 0
    target = args.rows.with_suffix(".tagged.jsonl")
    with args.rows.open(encoding="utf-8") as source, \
            target.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            words = [w.lower() for w in WORD.findall(sentences.get(row.get("clip", ""), ""))]
            form = row["form"].lower()
            previous = ""
            for index, word in enumerate(words):
                if word == form and index:
                    previous = words[index - 1]
                    break
            row["prev_word"] = previous
            row["prepositional_clitic"] = bool(
                form in CLITICS and previous in PREPOSITIONS)
            tagged += bool(previous)
            flagged += row["prepositional_clitic"]
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    target.replace(args.rows)
    print(f"tagged {tagged:,} rows with a preceding word; "
          f"{flagged:,} flagged as prepositional clitics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
