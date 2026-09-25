"""Audit the lexicon against what a hundred narrators actually said.

Every v3 row carries the lexicon's reading where the lexicon has exactly one,
and the ranker's reading always, at a confidence calibrated on this very
material (98.8% right at ≥0.99 on lexicon-named words). Where narrators from
several books agree with each other and disagree with the lexicon, the
lexicon is the thing to check. Round 2 of this on 311k Common Voice words
gave 85 corrections and two points of heteronym accuracy; воду́ and хо́ча
were found afterwards by reading one paragraph, and this finds their class
at once.

Three classes, three tables, one verdict column for a person:

  A  the lexicon holds one reading, the narrators use another;
  B  the lexicon holds several, the narrators use one it does not list;
  C  the lexicon holds several and serves the first as its default, but the
     narrators' majority is another — a reordering, not a correction.

Books on the exclusion list (archaic originals, verse) do not vote: their
narrators disagree with the modern norm legitimately.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mine_commonvoice import VOWELS  # noqa: E402

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")
CLITICS = {"мене", "тебе", "себе", "нього", "неї", "них", "нас", "вас", "мною",
           "тобою", "собою", "ньому", "нім", "нею", "ними"}


def stressed(form: str, signature) -> str:
    """Mark the vowels a signature names; "0|1" marks two, as the wordlist's
    "either" notation does."""
    wanted = {int(p) for p in str(signature).split("|") if p.isdigit()}
    out, seen = [], -1
    for ch in form:
        out.append(ch)
        if ch.lower() in VOWELS:
            seen += 1
            if seen in wanted:
                out.append("́")
    return unicodedata.normalize("NFC", "".join(out))


def lookup(api: str, form: str) -> tuple[list[str], str]:
    """The candidates in the order /v1/stress serves them — reviewed datasets
    first, then the lexicon's own — and what the word alone comes out as.
    /v1/lookup shows the raw rows with their duplicates and does not apply
    the reviewed ordering, so it is not what a caller sees."""
    req = urllib.request.Request(f"{api}/v1/stress",
                                 data=json.dumps({"text": form}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.load(r)
    except Exception:  # noqa: BLE001
        return [], ""
    token = next((t for t in body.get("tokens", []) if t["text"].lower() == form), None)
    if not token:
        return [], ""
    seen, out = set(), []
    for c in token.get("candidates", []):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out, unicodedata.normalize("NFC", token.get("output_text", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("artifacts/books"))
    parser.add_argument("--exclude", type=Path, default=Path("artifacts/books/book_exclusions.json"),
                        help="slugs whose narrators do not vote")
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--min-confidence", type=float, default=0.99)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--min-share", type=float, default=0.8)
    parser.add_argument("--min-books", type=int, default=3)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--audio-rows", type=Path, nargs="*", default=[],
                        help="book-less rows from the v4 miner, as a glob per argument; "
                             "they carry the sentence instead of a book paragraph")
    parser.add_argument("--out", type=Path, default=Path("DICTIONARY_AUDIT_ROUND3"))
    args = parser.parse_args()

    already: set[str] = set()
    corrections = Path("/home/devops/wiki-stress/ml/scripts/run_manual_corrections.py")
    if corrections.exists():
        text = corrections.read_text(encoding="utf-8")
        block = text[text.index("CORRECTIONS: dict"):]
        already = set(re.findall(r'^    "([^"]+)": \(', block, flags=re.M))
        already = {unicodedata.normalize("NFC", f.lower()) for f in already}
    excluded = set()
    if args.exclude.exists():
        excluded = {e["slug"] if isinstance(e, dict) else e
                    for e in json.loads(args.exclude.read_text(encoding="utf-8"))}

    # form -> per-reading counts, split by whether the lexicon agrees
    single: dict[str, dict] = collections.defaultdict(
        lambda: {"agree": 0, "picks": collections.Counter(), "books": collections.defaultdict(set),
                 "label": None, "examples": collections.defaultdict(list)})
    multi: dict[str, dict] = collections.defaultdict(
        lambda: {"picks": collections.Counter(), "books": collections.defaultdict(set),
                 "examples": collections.defaultdict(list)})
    texts: dict[str, dict] = {}

    def sentence(slug: str, doc: int, paragraph: int, index: int, form: str) -> str:
        if slug not in texts:
            texts[slug] = {}
            path = args.books / f"{slug}.jsonl"
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        texts[slug][(row["doc"], row["paragraph"])] = row["text"]
        text = texts[slug].get((doc, paragraph), "")
        spans = list(WORD.finditer(text))
        if index >= len(spans):
            return ""
        m = spans[index]
        left = max(0, text.rfind(". ", 0, m.start()) + 2 if text.rfind(". ", 0, m.start()) >= 0 else 0)
        right = text.find(". ", m.end())
        right = len(text) if right < 0 else right + 1
        return (text[left:m.start()] + "«" + text[m.start():m.end()] + "»" + text[m.end():right]).strip()[:220]

    files = sorted(args.books.glob("*.rows3.jsonl"))
    if args.audio_rows:
        files += sorted(p for pattern in args.audio_rows for p in pattern.parent.glob(pattern.name))
    print(f"{len(files)} sources, {len(excluded)} excluded", flush=True)
    rows_read = 0
    for path in files:
        bookless = path.name.endswith(".rows4.jsonl")
        slug = path.name[: -len(".rows4.jsonl" if bookless else ".rows3.jsonl")]
        if slug in excluded:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows_read += 1
            if r["confidence"] < args.min_confidence:
                continue
            form = r["form"]
            if form in CLITICS or sum(1 for c in form if c in VOWELS) < 2:
                continue
            pick = int(r["audio"])
            if r["readings"] == 1 and r.get("label") is not None:
                entry = single[form]
                entry["label"] = r["label"]
                if pick == r["label"]:
                    entry["agree"] += 1
                else:
                    entry["picks"][pick] += 1
                    entry["books"][pick].add(slug)
                    if len(entry["examples"][pick]) < args.examples * 3:
                        entry["examples"][pick].append(
                            (slug, r.get("doc", 0), r.get("paragraph", r.get("window", 0)),
                             r.get("word_index", -1)) if not bookless else
                            ("youtube:" + slug, r.get("sentence", ""), r.get("start", 0), r.get("end", 0)))
            elif r["readings"] > 1:
                entry = multi[form]
                entry["picks"][pick] += 1
                entry["books"][pick].add(slug)
                if len(entry["examples"][pick]) < args.examples * 3:
                    entry["examples"][pick].append(
                        (slug, r.get("doc", 0), r.get("paragraph", r.get("window", 0)),
                         r.get("word_index", -1)) if not bookless else
                        ("youtube:" + slug, r.get("sentence", ""), r.get("start", 0), r.get("end", 0)))
    print(f"{rows_read:,} rows read; {len(single):,} single-reading forms, "
          f"{len(multi):,} ambiguous forms with a confident pick", flush=True)

    def render_examples(form, entry, pick):
        out = []
        for slug, doc, para, index in entry["examples"][pick]:
            if isinstance(slug, str) and slug.startswith("youtube:"):
                # a book-less row carries its own sentence and the word's span
                text, start, end = str(doc), int(para), int(index)
                s = (text[:start] + "«" + text[start:end] + "»" + text[end:]).strip()[:220] if text else ""
            else:
                s = sentence(slug, doc, para, index, form)
            if s:
                out.append(s)
            if len(out) >= args.examples:
                break
        return out

    # A: one reading in the lexicon, another in the narrators' mouths.
    table_a = []
    for form, e in single.items():
        if not e["picks"] or form in already:
            continue
        top, n = e["picks"].most_common(1)[0]
        total = e["agree"] + sum(e["picks"].values())
        if (n < args.min_rows or n / total < args.min_share
                or len(e["books"][top]) < args.min_books
                or n / sum(e["picks"].values()) < 0.9):
            continue
        table_a.append({"form": form, "lexicon": stressed(form, e["label"]),
                        "narrators": stressed(form, top), "rows": n, "total": total,
                        "share": round(n / total, 3), "books": len(e["books"][top]),
                        "examples": render_examples(form, e, top)})
    table_a.sort(key=lambda r: (-r["books"], -r["rows"]))

    # B and C need the served candidate order.
    table_b, table_c = [], []
    for form, e in multi.items():
        total = sum(e["picks"].values())
        if total < args.min_rows:
            continue
        if form in already:
            continue
        candidates, served = lookup(args.api, form)
        if not candidates:
            continue
        top, n = e["picks"].most_common(1)[0]
        listed = {int(p) for c in candidates for p in c.split("|") if p.isdigit()}
        if top not in listed:
            if n / total >= args.min_share and len(e["books"][top]) >= args.min_books:
                table_b.append({"form": form, "lexicon": " / ".join(stressed(form, c) for c in candidates),
                                "narrators": stressed(form, top), "rows": n, "total": total,
                                "share": round(n / total, 3), "books": len(e["books"][top]),
                                "examples": render_examples(form, e, top)})
            continue
        if len(candidates) < 2:
            continue
        if stressed(form, top) != served and n / total >= args.min_share and len(e["books"][top]) >= args.min_books:
            table_c.append({"form": form, "lexicon": served or stressed(form, candidates[0]),
                            "narrators": stressed(form, top), "rows": n, "total": total,
                            "share": round(n / total, 3), "books": len(e["books"][top]),
                            "examples": render_examples(form, e, top)})
    table_b.sort(key=lambda r: (-r["books"], -r["rows"]))
    table_c.sort(key=lambda r: (-r["books"], -r["rows"]))

    args.out.with_suffix(".json").write_text(json.dumps(
        {"A": table_a, "B": table_b, "C": table_c}, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# Форми на вердикт — третє коло (книжки)", "",
             f"З {rows_read:,} книжкових слів у {len(files) - len(excluded)} книжках; голосує лише "
             f"читання ранкера з довіреністю ≥{args.min_confidence}, і лише коли з ним згодні "
             f"≥{args.min_books} книжки при частці ≥{args.min_share:.0%}.", "",
             "**Як читати.** «книжок» — скільки різних книжок (дикторів) читають так; «частка» — серед усіх "
             "упевнених прочитань форми. Приклади — з тексту книжки, слово в «лапках».", "",
             "**Що потрібно.** У колонку «вердикт»: `правильно` (другий варіант правильний → виключне читання), "
             "`обидва` (обидва чинні, другий частіший → перевпорядкувати), або порожньо (лишити як є).", ""]

    def section(title, why, table, col):
        out = [f"## {title}", "", why, "", f"| форма | {col} | диктори | рядків | частка | книжок | приклади | вердикт |",
               "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
        for r in table:
            ex = "<br>".join(r["examples"]).replace("|", "¦")
            out.append(f"| {r['form']} | {r['lexicon']} | **{r['narrators']}** | {r['rows']}/{r['total']} "
                       f"| {r['share']:.0%} | {r['books']} | {ex} |  |")
        return out + [""]

    lines += section(f"A. Лексикон має одне читання, диктори — інше ({len(table_a)})",
                     "Кандидати на виключне читання, як воду́ → во́ду.", table_a, "лексикон")
    lines += section(f"B. Лексикон має кілька читань, диктори — те, якого в ньому немає ({len(table_b)})",
                     "Читання, яке треба додати, або сліпа пляма ранкера — приклади скажуть.", table_b, "лексикон")
    lines += section(f"C. Словниковий дефолт — не те, що читають диктори ({len(table_c)})",
                     "Не помилка словника, а порядок: дефолт подається на кожній формі, яку класифікатор не покриває.",
                     table_c, "дефолт")
    args.out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"A {len(table_a)}, B {len(table_b)}, C {len(table_c)} -> {args.out}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
