"""Draft gold passages for modern text, for a person to correct.

For each heteronym form, real sentences from the audiobooks in which the
narrator read it each way, taken from the v3 rows at ranker confidence
≥0.99. Every sentence is marked in full by the live API, and then the target
word's mark is replaced by the narrator's reading — so the draft carries the
pipeline's answer everywhere except on the word under test, where it carries
what was said. The reviewer corrects any mark; nothing here is trusted.

Output: one sentence per line, `#` lines name the form and its readings.
`run_gold_modern.py` scores the live API against the corrected file.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import unicodedata
import urllib.request
from pathlib import Path

VOWELS = "аеєиіїоуюяй"
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")
ACUTE = "́"


def remark(word: str, ordinal: int) -> str:
    out, seen = [], -1
    for ch in unicodedata.normalize("NFD", word):
        if ch == ACUTE:
            continue
        out.append(ch)
        if ch.lower() in VOWELS:
            seen += 1
            if seen == ordinal:
                out.append(ACUTE)
    return unicodedata.normalize("NFC", "".join(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("/home/devops/audiotostress/artifacts/books"))
    parser.add_argument("--heteronyms", type=Path, required=True, help="lang-uk's heteronyms_list.csv")
    parser.add_argument("--exclusions", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/books/book_exclusions.json"))
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--forms", type=int, default=50, help="forms in this batch")
    parser.add_argument("--skip-forms", type=int, default=0, help="forms already drafted")
    parser.add_argument("--per-reading", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.99)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    wanted = []
    with args.heteronyms.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wanted.append(unicodedata.normalize("NFC", row["Heteronym"].strip().lower()))
    wanted_set = set(wanted)
    excluded = {e["slug"] for e in json.loads(args.exclusions.read_text(encoding="utf-8"))} \
        if args.exclusions.exists() else set()

    # form -> reading -> [(slug, doc, paragraph, word_index)]
    hits: dict[str, dict[int, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(args.books.glob("*.rows3.jsonl")):
        slug = path.name[: -len(".rows3.jsonl")]
        if slug in excluded:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"readings": 1' in line or not line.strip():
                continue
            r = json.loads(line)
            if r["readings"] > 1 and r["form"] in wanted_set and r["confidence"] >= args.min_confidence:
                hits[r["form"]][int(r["audio"])].append((slug, r["doc"], r["paragraph"], r["word_index"]))

    # Forms with both readings attested, most attested first, in list order
    # as the tiebreak so batches are reproducible.
    ranked = sorted((f for f in wanted if f in hits and len(hits[f]) > 1),
                    key=lambda f: -min(len(v) for v in hits[f].values()))
    batch = ranked[args.skip_forms:args.skip_forms + args.forms]
    print(f"{len(ranked)} lang-uk heteronyms with both readings in the books; "
          f"drafting {len(batch)} (skipping {args.skip_forms})", flush=True)

    texts: dict[str, dict] = {}

    def paragraph(slug, doc, para):
        if slug not in texts:
            texts[slug] = {}
            for line in (args.books / f"{slug}.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    texts[slug][(row["doc"], row["paragraph"])] = row["text"]
        return texts[slug].get((doc, para), "")

    def sentence_of(text, start, end):
        cut = re.compile(r"[.!?…]['\"»)\]]*\s")
        left = 0
        for m in cut.finditer(text, 0, start):
            left = m.end()
        m = cut.search(text, end)
        right = m.end() if m else len(text)
        return text[left:right].strip(), start - left, end - left

    def stress(text):
        req = urllib.request.Request(f"{args.api}/v1/stress", data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    rng = random.Random(args.seed)
    lines = [f"# Gold draft, batch of {len(batch)} forms. Every mark is the pipeline's except on the",
             "# form under test, where it is the narrator's. Correct any mark; delete a line that",
             "# is not a sentence. One sentence per line.", ""]
    drafted = 0
    for form in batch:
        readings = sorted(hits[form])
        lines.append(f"# {form}: " + " / ".join(remark(form, o) for o in readings))
        for ordinal in readings:
            picks = hits[form][ordinal]
            rng.shuffle(picks)
            written = 0
            for slug, doc, para, index in picks:
                text = paragraph(slug, doc, para)
                spans = list(WORD.finditer(text))
                if index >= len(spans):
                    continue
                m = spans[index]
                sent, s, e = sentence_of(text, m.start(), m.end())
                if not (40 <= len(sent) <= 260) or sent[s:e].lower() != form:
                    continue
                try:
                    marked = stress(sent)
                except Exception:  # noqa: BLE001
                    continue
                # Rebuild with the target word re-marked from the narrator.
                out, pos = [], 0
                for t in marked["tokens"]:
                    out.append(sent[pos:t["start"]])
                    out.append(remark(t["text"], ordinal) if t["start"] == s else t["output_text"])
                    pos = t["end"]
                out.append(sent[pos:])
                lines.append(unicodedata.normalize("NFC", "".join(out)))
                written += 1
                drafted += 1
                if written >= args.per_reading:
                    break
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{drafted} sentences over {len(batch)} forms -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
