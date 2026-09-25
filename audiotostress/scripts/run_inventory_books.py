"""Inventory every book on disk: what it is, exactly how big, how long.

One record per book folder across the given roots, with every file's byte
size and SHA-256, the audio's duration to the millisecond, the extracted word
count, and where the book stands in the pipeline (planned, piloted, mined,
dropped). The inventory is merged with the previous one, so a run reports
what is new since last time; duplicates are found by content hash, by audio
duration, and by title, so the same book downloaded twice under two folder
names is seen before it is mined twice.

    inventory.json   the record, machine-readable
    inventory.md     the table, for reading
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

AUDIO = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus"}
TEXT = {".epub", ".fb2", ".pdf", ".txt"}
TITLE = re.compile(r"«([^»]+)»")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True)
        return round(float(out.stdout.strip()), 3)
    except Exception:  # noqa: BLE001
        return None


def title_key(name: str) -> str:
    found = TITLE.search(name)
    title = found.group(1) if found else name
    return re.sub(r"[^а-яіїєґa-z0-9]+", " ", title.lower()).strip()


def hms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--books", type=Path, default=Path("artifacts/books"),
                        help="where plan.json, index.json, pilot and rows files live")
    parser.add_argument("--out", type=Path, default=Path("artifacts/books/inventory.json"))
    args = parser.parse_args()

    previous = {}
    if args.out.exists():
        previous = {r["folder"]: r for r in
                    json.loads(args.out.read_text(encoding="utf-8"))["books"]}
    today = dt.date.today().isoformat()

    plan = {}
    if (args.books / "plan.json").exists():
        plan = {b["name"]: b for b in json.loads((args.books / "plan.json").read_text(encoding="utf-8"))}
    index = {}
    for name in ("index.json", "index_batch1.json"):
        if (args.books / name).exists():
            for r in json.loads((args.books / name).read_text(encoding="utf-8")):
                index.setdefault(r["book"], r)
    keep = set()
    for name in ("pilot_keep_v3.json", "pilot_keep.json"):
        if (args.books / name).exists():
            keep |= set(json.loads((args.books / name).read_text(encoding="utf-8")))
    mined = {p.name.split(".")[0] for p in args.books.glob("*.rows3.jsonl")} | \
            {p.name.split(".")[0] for p in args.books.glob("*.rows2.jsonl")}
    piloted = {p.name.split(".")[0] for p in args.books.glob("*.pilot3.jsonl")} | \
              {p.name.split(".")[0] for p in args.books.glob("*.pilot.jsonl")}

    books = []
    for root in args.roots:
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            old = previous.get(str(folder))
            files = []
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                kind = ("audio" if path.suffix.lower() in AUDIO else
                        "text" if path.suffix.lower() in TEXT else "other")
                stat = path.stat()
                known = next((f for f in (old or {}).get("files", [])
                              if f["name"] == path.name and f["bytes"] == stat.st_size), None)
                files.append({
                    "name": path.name, "kind": kind, "bytes": stat.st_size,
                    "sha256": known["sha256"] if known else sha256(path),
                    "duration_s": (known.get("duration_s") if known else
                                   duration(path) if kind == "audio" else None),
                })
            audio = [f for f in files if f["kind"] == "audio"]
            planned = plan.get(folder.name)
            slug = planned["slug"] if planned else index.get(folder.name, {}).get("slug")
            status = ("mined" if slug in mined else "kept" if slug in keep else
                      "dropped" if slug in piloted else "planned" if planned else
                      "extracted" if slug else "new")
            books.append({
                "folder": str(folder), "root": str(root), "name": folder.name,
                "title": title_key(folder.name), "slug": slug,
                "audio_seconds": sum(f["duration_s"] or 0 for f in audio) or None,
                "bytes": sum(f["bytes"] for f in files),
                "words": index.get(folder.name, {}).get("words"),
                "status": status,
                "first_seen": old["first_seen"] if old else today,
                "files": files,
            })

    # Duplicates: the same bytes anywhere, the same audio length, the same title.
    by_hash, by_length, by_title = defaultdict(list), defaultdict(list), defaultdict(list)
    # Keyed by folder path, not name: the same book in two roots has the same
    # name, and a set of names would collapse exactly the duplicate we want.
    label = {b["folder"]: f"{Path(b['root']).name}/{b['name']}" for b in books}
    for book in books:
        for f in book["files"]:
            by_hash[f["sha256"]].append(book["folder"])
        if book["audio_seconds"]:
            by_length[round(book["audio_seconds"])].append(book["folder"])
        by_title[book["title"]].append(book["folder"])
    duplicates = []
    for kind, groups in (("same file", by_hash), ("same audio length", by_length),
                         ("same title", by_title)):
        for key, folders in groups.items():
            folders = sorted(set(folders))
            if len(folders) > 1:
                duplicates.append({"kind": kind, "key": str(key),
                                   "books": [label[f] for f in folders], "folders": folders})
    for book in books:
        book["duplicate_of"] = sorted({label[f] for d in duplicates for f in d["folders"]
                                       if book["folder"] in d["folders"] and f != book["folder"]})

    new = [b for b in books if b["first_seen"] == today and b["folder"] not in previous]
    total_h = sum(b["audio_seconds"] or 0 for b in books) / 3600
    payload = {"generated": today, "roots": [str(r) for r in args.roots],
               "books": books, "duplicates": duplicates,
               "totals": {"books": len(books), "audio_hours": round(total_h, 2),
                          "bytes": sum(b["bytes"] for b in books),
                          "words": sum(b["words"] or 0 for b in books)}}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# Book inventory — {today}", "",
             f"{len(books)} books, {total_h:,.1f} h of audio, "
             f"{payload['totals']['bytes'] / 1e9:.1f} GB, {payload['totals']['words']:,} words. "
             f"{len(new)} new since the previous inventory.", "",
             "| book | root | audio | duration | size | words | status | first seen | duplicate of |",
             "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for b in sorted(books, key=lambda b: b["name"]):
        audio = ", ".join(f["name"] for f in b["files"] if f["kind"] == "audio") or "—"
        lines.append(f"| {b['name']} | {Path(b['root']).name} | {audio[:40]} | {hms(b['audio_seconds'])} "
                     f"| {b['bytes'] / 1e6:,.1f} MB | {b['words'] or '—'} | {b['status']} "
                     f"| {b['first_seen']} | {'; '.join(b['duplicate_of']) or ''} |")
    if duplicates:
        lines += ["", "## Duplicates", ""]
        for d in duplicates:
            lines.append(f"- **{d['kind']}** ({d['key'][:16]}): " + " / ".join(d["books"]))
    if new:
        lines += ["", "## New since the previous inventory", ""]
        lines += [f"- {b['name']} — {hms(b['audio_seconds'])}, {b['bytes'] / 1e6:,.1f} MB" for b in new]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(books)} books, {total_h:,.1f} h, {payload['totals']['bytes'] / 1e9:.1f} GB; "
          f"{len(new)} new; {len(duplicates)} duplicate groups -> {args.out} and .md")
    for d in duplicates:
        print(f"  {d['kind']}: " + " / ".join(n[:45] for n in d["books"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
