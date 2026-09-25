"""Pull a clean, ordered word stream out of each epub.

The anchoring step matches ten-minute ASR windows against this stream, so what
matters is not prettiness but that the text is **in reading order** and carries
only what a narrator actually says. Front matter, notes, page furniture and
tables of contents all break anchoring in the same way: they put words in the
stream that never appear in the audio, and the fuzzy match drifts.

Reports per book how much was kept and what was dropped, because a book whose
epub is far from the audio's word count will not anchor and is better set aside
than forced. Ukrainian runs about 2.2 words a second; see PLAN_BOOKS.md.
"""

from __future__ import annotations

import argparse
import html
import json
import collections
import re
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’’-]+")
LATIN = re.compile(r"[A-Za-z]")
# Cyrillic letters Ukrainian does not use; a run of them is Russian text.
RUSSIAN_ONLY = re.compile(r"[ёъыэЁЪЫЭ]")
BLOCK = re.compile(r"</(p|div|h[1-6]|li|br)\s*>", re.I)
TAG = re.compile(r"<[^>]+>")
# Matched against the file's stem with digits and separators stripped, not as a
# substring: `index_split_002.html` is chapter two of a book, not a back-of-book
# index, and a substring match threw away every document of one epub.
SKIP_STEMS = {
    "toc", "nav", "cover", "title", "titlepage", "copyright", "colophon",
    "annotation", "about", "contents", "note", "notes", "footnote",
    "footnotes", "index", "bibliography", "biblio",
}


def is_furniture(name: str) -> bool:
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    stem = re.sub(r"[\d_\-.]+", "", stem)
    return stem in SKIP_STEMS


def spine_order(archive: zipfile.ZipFile) -> list[str]:
    """Document order as the epub declares it, not as the zip happens to store it."""
    try:
        container = archive.read("META-INF/container.xml").decode("utf-8", "ignore")
        opf_path = re.search(r'full-path="([^"]+)"', container).group(1)
        opf = ElementTree.fromstring(archive.read(opf_path))
        ns = {"o": "http://www.idpf.org/2007/opf"}
        base = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""
        ids = {item.get("id"): item.get("href")
               for item in opf.iterfind(".//o:manifest/o:item", ns)}
        order = []
        for ref in opf.iterfind(".//o:spine/o:itemref", ns):
            href = ids.get(ref.get("idref"))
            if href:
                order.append(f"{base}/{href}" if base else href)
        return order
    except Exception:  # noqa: BLE001 - fall back to zip order, still usable
        return [n for n in archive.namelist()
                if n.endswith((".xhtml", ".html", ".htm"))]


def paragraphs(raw: bytes) -> list[str]:
    text = raw.decode("utf-8", "ignore")
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    # Mark block ends before stripping tags, or every paragraph runs together.
    text = BLOCK.sub("\n", text)
    text = html.unescape(TAG.sub(" ", text))
    out = []
    for line in text.split("\n"):
        line = unicodedata.normalize("NFC", " ".join(line.split()))
        if len(WORD.findall(line)) >= 3:
            out.append(line)
    return out


def from_fb2(path: Path) -> list[dict]:
    """FictionBook is XML with real paragraph tags, so this is the easy case."""
    raw = path.read_bytes()
    declared = re.search(rb'encoding="([^"]+)"', raw[:200])
    text = raw.decode(declared.group(1).decode() if declared else "utf-8", "ignore")
    body = re.search(r"<body[^>]*>(.*)</body>", text, re.S)
    text = body.group(1) if body else text
    out = []
    for order, chunk in enumerate(re.findall(r"<p[^>]*>(.*?)</p>", text, re.S)):
        line = unicodedata.normalize("NFC", " ".join(
            html.unescape(TAG.sub(" ", chunk)).split()))
        if len(WORD.findall(line)) >= 3:
            out.append({"doc": 0, "paragraph": order, "text": line,
                        "words": len(WORD.findall(line))})
    return out


def from_pdf(path: Path) -> list[dict]:
    """A PDF has pages, not paragraphs, and page furniture in every one.

    Lines that repeat across many pages are running heads and page numbers;
    they are dropped by frequency rather than by pattern, which travels better
    between books than any regex would.
    """
    import pypdf

    reader = pypdf.PdfReader(str(path))
    pages = [(page.extract_text() or "").split("\n") for page in reader.pages]
    seen: collections.Counter[str] = collections.Counter()
    for lines in pages:
        for line in {l.strip() for l in lines if l.strip()}:
            seen[line] += 1
    furniture = {line for line, n in seen.items() if n > len(pages) * 0.2}
    out = []
    for index, lines in enumerate(pages):
        kept = [l.strip() for l in lines
                if l.strip() and l.strip() not in furniture
                and not re.fullmatch(r"[\d\s.·—–-]+", l.strip())]
        # Re-join hyphenated line breaks before splitting into sentences.
        joined = re.sub(r"(\w)-\s+(\w)", r"\1\2", " ".join(kept))
        for order, part in enumerate(re.split(r"(?<=[.!?…])\s+(?=[А-ЯЇІЄҐ])", joined)):
            line = unicodedata.normalize("NFC", " ".join(part.split()))
            if len(WORD.findall(line)) >= 3:
                out.append({"doc": index, "paragraph": order, "text": line,
                            "words": len(WORD.findall(line))})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("/home/devops/books"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/books"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{'book':<34}{'paras':>8}{'words':>10}{'skipped':>9}{'latin':>7}{'rus':>6}")
    summary = []
    for folder in sorted(p for p in args.books.iterdir() if p.is_dir()):
        sources = {p.suffix.lower(): p for p in folder.iterdir()
                   if p.suffix.lower() in (".epub", ".fb2", ".pdf")}
        if not sources:
            continue
        kept, skipped_docs, latin, russian = [], 0, 0, 0
        if ".epub" not in sources:
            # Prefer FictionBook: it keeps paragraph structure a PDF has lost.
            path = sources.get(".fb2") or sources[".pdf"]
            kept = from_fb2(path) if path.suffix.lower() == ".fb2" else from_pdf(path)
            for row in kept:
                latin += len(LATIN.findall(row["text"]))
                russian += len(RUSSIAN_ONLY.findall(row["text"]))
        else:
          with zipfile.ZipFile(sources[".epub"]) as archive:
            names = set(archive.namelist())
            for index, name in enumerate(spine_order(archive)):
                if name not in names:
                    continue
                if is_furniture(name):
                    skipped_docs += 1
                    continue
                for order, text in enumerate(paragraphs(archive.read(name))):
                    words = WORD.findall(text)
                    latin += len(LATIN.findall(text))
                    russian += len(RUSSIAN_ONLY.findall(text))
                    kept.append({"doc": index, "paragraph": order,
                                 "text": text, "words": len(words)})
        total = sum(p["words"] for p in kept)
        # A Ukrainian novel quotes Russian speech; a Russian book is Russian
        # throughout. Шкляр's «Залишенець» sits at 0.006 Russian-only letters
        # per word, «Жінки, які кохають до нестями» at 0.125 — twenty times
        # higher, and its epub turned out to be the Russian edition against a
        # Ukrainian reading. Aligning those produced 5,163 OOV rows against
        # 1,204 labelled ones before anyone noticed.
        slug = re.sub(r"[^a-z0-9]+", "-", folder.name.lower().translate(
            str.maketrans("абвгдежзийклмнопрстуфхцчшщьюяєіїґ",
                          "abvgdezzijklmnoprstufhccssjuaeiig"))).strip("-")[:40] or f"book{len(summary)}"
        density = russian / max(total, 1)
        if density > 0.05:
            print(f"  {folder.name[:32]:<32}{'SKIPPED':>8} — {density:.3f} "
                  f"Russian letters per word, this text is not Ukrainian")
            summary.append({"book": folder.name, "slug": slug, "paragraphs": 0,
                            "words": 0, "skipped_documents": skipped_docs,
                            "latin_chars": latin, "russian_chars": russian,
                            "excluded": "not Ukrainian"})
            continue
        (args.out / f"{slug}.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in kept) + "\n",
            encoding="utf-8")
        print(f"  {folder.name[:32]:<32}{len(kept):>8,}{total:>10,}"
              f"{skipped_docs:>9}{latin:>7,}{russian:>6,}")
        summary.append({"book": folder.name, "slug": slug, "paragraphs": len(kept),
                        "words": total, "skipped_documents": skipped_docs,
                        "latin_chars": latin, "russian_chars": russian})
    (args.out / "index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{sum(s['words'] for s in summary):,} words over "
          f"{len(summary)} books -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
