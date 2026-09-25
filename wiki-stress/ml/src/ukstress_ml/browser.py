"""Index and serve the ML datasets for inspection.

These files reach 275 MB, so nothing here loads one to show a page. Each file is
scanned once into a list of byte offsets — one per row — and afterwards any row
is a seek and a read. The index is cached beside the data, so the scan happens
once per file per change.

The value of the tool is not the list of rows. It is seeing the *label in its
context*: the sentence with the target word marked and the stress the row claims
actually rendered, so a wrong label looks wrong rather than reading as a number.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ACUTE = "́"
#: `й` takes a vowel ordinal in the lexicon's signature convention even though
#: it is not a syllable. Counting it reproduces 29,710 of 30,712 manifest
#: spellings; not counting it renders `байко́ва` as `байкова́`.
VOWELS = "аеєиіїоуюяй"
INDEX_VERSION = 3


def apply_signature(word: str, signature: str) -> str:
    """Render a signature as an accented spelling, or return the word unchanged."""
    if not signature:
        return word
    try:
        index = int(str(signature).split("|")[0])
    except ValueError:
        return word
    seen = -1
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", word):
        out.append(ch)
        if ch.lower() in VOWELS:
            seen += 1
            if seen == index:
                out.append(ACUTE)
    return unicodedata.normalize("NFC", "".join(out))


@dataclass
class DatasetIndex:
    path: Path
    kind: str                      # "array", "lines" or "object"
    offsets: list[int]
    lengths: list[int]
    keys: list[str]
    forms: list[tuple[str, int]]
    rows: int
    size: int
    mtime: float
    #: Member keys, for object-shaped files where the key is the word itself.
    names: list[str] = field(default_factory=list)
    #: Object files are read whole and kept here. They are small — the largest
    #: is 16 MB — and their members are often scalars, which have no nested span
    #: for a byte scan to find. Arrays stay on the seek path.
    materialised: list[Any] | None = None
    labels: list[tuple[str, int]] = field(default_factory=list)
    #: The form each row is about, one entry per row. Object files get it from
    #: the member key for free; array files are scraped for it at index time,
    #: because sorting or filtering by corpus frequency needs a form per row and
    #: reading 478k rows per request to find one is not an option.
    row_forms: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"version": INDEX_VERSION, "kind": self.kind, "offsets": self.offsets,
                "lengths": self.lengths, "keys": self.keys, "forms": self.forms,
                "rows": self.rows, "size": self.size, "mtime": self.mtime,
                "labels": self.labels, "names": self.names,
                "row_forms": self.row_forms}


def _scan_container(data: bytes, member_depth: int = 2
                    ) -> tuple[str, list[int], list[int], list[str]]:
    """Byte spans of the members of a top-level JSON array or object.

    Walks bytes tracking depth and string state rather than parsing, so the cost
    is one pass and no object is built. An object's member *keys* are captured
    too: for a manifest or a lookup table the key is the word, and losing it
    would make the file unreadable.
    """
    kind = "array"
    for byte in data[:4096]:
        if byte in (0x5B, 0x7B):
            kind = "array" if byte == 0x5B else "object"
            break

    offsets: list[int] = []
    lengths: list[int] = []
    names: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    string_start = -1
    pending_key: str | None = None
    for i, ch in enumerate(data):
        if in_string:
            if escaped:
                escaped = False
            elif ch == 0x5C:
                escaped = True
            elif ch == 0x22:
                in_string = False
                if depth == member_depth - 1 and start < 0:
                    # A string closing just outside a member, before it opens,
                    # is that member's key.
                    pending_key = data[string_start + 1: i].decode("utf-8", "replace")
            continue
        if ch == 0x22:
            in_string = True
            string_start = i
            continue
        if ch in (0x7B, 0x5B):
            depth += 1
            if depth == 2 and start < 0:
                start = i
        elif ch in (0x7D, 0x5D):
            if depth == 2 and start >= 0:
                offsets.append(start)
                lengths.append(i + 1 - start)
                if kind == "object":
                    names.append(pending_key or "")
                    pending_key = None
                start = -1
            depth -= 1
    return kind, offsets, lengths, names


def build_index(path: Path, sample: int = 40000) -> DatasetIndex:
    data = path.read_bytes()
    stat = path.stat()
    if path.suffix == ".jsonl":
        kind = "lines"
        offsets, lengths, position = [], [], 0
        for line in data.split(b"\n"):
            if line.strip():
                offsets.append(position)
                lengths.append(len(line))
            position += len(line) + 1
        names: list[str] = []
        materialised = None
    else:
        kind, offsets, lengths, names = _scan_container(data)
        materialised = None
        if kind == "object":
            # Read it whole: a manifest nests its rows under one key, and a
            # lookup table's values are scalars. Neither shape survives a scan
            # that only knows about brackets.
            try:
                loaded = json.loads(data)
            except Exception as error:  # noqa: BLE001
                log.warning("%s is not readable as JSON: %s", path.name, error)
                loaded = {}
            if isinstance(loaded, dict):
                # `{"forms": {...}}` — descend to where the rows actually are.
                inner = [v for v in loaded.values() if isinstance(v, dict) and len(v) > 3]
                target = inner[0] if len(loaded) <= 8 and inner else loaded
                names = [str(k) for k in target]
                materialised = [
                    v if isinstance(v, dict) else {"value": v} for v in target.values()
                ]
                offsets, lengths = [], []

    # `"form": "…"` inside one row's bytes. Cheaper than parsing the row, and
    # the fallback below covers the rows where it is not literal.
    form_pattern = re.compile(rb'"form"\s*:\s*"([^"\\]*)"')
    row_forms: list[str] = []
    if materialised is not None:
        row_forms = list(names)
    else:
        for offset, length in zip(offsets, lengths, strict=True):
            found = form_pattern.search(data, offset, offset + length)
            row_forms.append(found.group(1).decode("utf-8", "replace") if found else "")

    keys: list[str] = []
    forms: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    count = len(materialised) if materialised is not None else len(offsets)
    step = max(1, count // sample)
    for i in range(0, count, step):
        if materialised is not None:
            row = materialised[i]
        else:
            try:
                row = json.loads(data[offsets[i]: offsets[i] + lengths[i]])
            except Exception as error:  # noqa: BLE001 - one bad row cannot stop a scan
                log.debug("row %d of %s is unparsable: %s", i, path.name, error)
                continue
        if isinstance(row, dict):
            if not keys:
                keys = sorted(row)
            form = (row.get("form") or row.get("form_normalized")
                    or (names[i] if i < len(names) else None))
            if isinstance(form, str):
                forms[form] += 1
            label = row.get("gold_signature") or row.get("gold") or row.get("gold_sense")
            if label is not None:
                labels[str(label)] += 1
    return DatasetIndex(path=path, kind=kind, offsets=offsets, lengths=lengths,
                        names=names, keys=keys, forms=forms.most_common(300),
                        rows=count, size=stat.st_size, mtime=stat.st_mtime,
                        labels=labels.most_common(12), materialised=materialised,
                        row_forms=row_forms)


def load_index(path: Path, cache_dir: Path) -> DatasetIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (path.name + ".idx.json")
    stat = path.stat()
    if cached.is_file():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            if (payload.get("version") == INDEX_VERSION
                    and payload.get("row_forms") is not None
                    and payload.get("kind") != "object"
                    and payload.get("mtime") == stat.st_mtime
                    and payload.get("size") == stat.st_size):
                return DatasetIndex(path=path, kind=payload["kind"],
                                    offsets=payload["offsets"], lengths=payload["lengths"],
                                    names=payload.get("names", []),
                                    keys=payload["keys"], forms=payload["forms"],
                                    rows=payload["rows"], size=payload["size"],
                                    mtime=payload["mtime"], labels=payload.get("labels", []),
                                    row_forms=payload.get("row_forms", []))
        except Exception as error:  # noqa: BLE001 - a stale cache is rebuilt
            log.debug("index cache for %s unusable, rebuilding: %s", path.name, error)
    index = build_index(path)
    cached.write_text(json.dumps(index.to_json()), encoding="utf-8")
    return index


def read_rows(index: DatasetIndex, start: int, count: int,
              order: list[int] | None = None) -> list[dict[str, Any]]:
    """Read a window of rows by seeking, without touching the rest of the file.

    `order` re-points the window at an arbitrary sequence of row numbers, so a
    sort or a filter is a list of indices rather than a rewritten file. Rows
    keep their original `_row`, which is what an annotation is keyed by.
    """
    out: list[dict[str, Any]] = []
    if order is not None:
        wanted = order[start: start + count]
    else:
        wanted = list(range(start, min(start + count, index.rows)))
    if index.materialised is not None:
        for i in wanted:
            row = dict(index.materialised[i])
            row["_row"] = i
            if i < len(index.names):
                row["_key"] = index.names[i]
            out.append(row)
        return out
    with index.path.open("rb") as handle:
        for i in wanted:
            handle.seek(index.offsets[i])
            raw = handle.read(index.lengths[i])
            try:
                row = json.loads(raw)
            except Exception:  # noqa: BLE001
                row = {"_unparsable": raw[:200].decode("utf-8", "replace")}
            if isinstance(row, dict):
                row["_row"] = i
                if i < len(index.names) and index.names[i]:
                    row["_key"] = index.names[i]
            out.append(row)
    return out


def find_rows(index: DatasetIndex, form: str, limit: int,
              scan_cap: int = 400000) -> tuple[list[dict[str, Any]], int]:
    """Rows whose form matches, found by scanning. Reports how far it got."""
    wanted = form.lower()
    out: list[dict[str, Any]] = []
    scanned = 0
    if index.materialised is not None:
        for i, row in enumerate(index.materialised):
            scanned = i + 1
            key = index.names[i] if i < len(index.names) else ""
            value = (row.get("form") if isinstance(row, dict) else None) or key or ""
            if isinstance(value, str) and value.lower() == wanted:
                item = dict(row) if isinstance(row, dict) else {"value": row}
                item["_row"] = i
                if key:
                    item["_key"] = key
                out.append(item)
                if len(out) >= limit:
                    break
        return out, scanned
    with index.path.open("rb") as handle:
        for i in range(min(index.rows, scan_cap)):
            handle.seek(index.offsets[i])
            try:
                row = json.loads(handle.read(index.lengths[i]))
            except Exception as error:  # noqa: BLE001
                log.debug("skipping unparsable row %d: %s", i, error)
                continue
            scanned = i + 1
            if not isinstance(row, dict):
                continue
            key = index.names[i] if i < len(index.names) else ""
            value = row.get("form") or row.get("form_normalized") or key or ""
            if isinstance(value, str) and value.lower() == wanted:
                row["_row"] = i
                if key:
                    row["_key"] = key
                out.append(row)
                if len(out) >= limit:
                    break
    return out, scanned


def decorate(row: dict[str, Any],
             manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add what makes a row readable: the sentence marked, and the stress applied.

    A label is easy to mis-read as a number. Rendered onto the word, in the
    sentence it came from, a wrong one is visible at a glance.

    `manifest` supplies the alternatives for rows that carry only a label — the
    corpus stores `gold_sense` and nothing about what it was chosen against, and
    a lone accent says nothing about whether the decision was right.
    """
    if not isinstance(row, dict):
        return {"value": row}
    sentence = row.get("sentence")
    start, end = row.get("start"), row.get("end")
    view = dict(row)
    if isinstance(sentence, str) and isinstance(start, int) and isinstance(end, int) \
            and 0 <= start < end <= len(sentence):
        target = sentence[start:end]
        view["_before"] = sentence[:start]
        view["_target"] = target
        view["_after"] = sentence[end:]
        signature = row.get("gold_signature")
        if signature is None and isinstance(row.get("gold"), (int, str)):
            signature = str(row["gold"])
        if signature is not None:
            view["_stressed"] = apply_signature(target, str(signature))
            view["_signature"] = str(signature)
    word = row.get("_key") or row.get("form") or row.get("form_normalized") or ""
    candidates = row.get("candidates")

    # A counted-form candidate carries its two readings as named fields rather
    # than as a candidate list. Rendering them as options makes the review a
    # click: the reviewer picks the reading `три …` takes, and the signature
    # recorded is the verdict itself rather than a vowel ordinal.
    if candidates is None and row.get("nominative_plural") and row.get("genitive_singular"):
        # The question here is not "which of these two readings is right" —
        # both are, in their own contexts. It is whether this noun has a
        # counted form at all, and the only way to ask that unambiguously is to
        # show the phrase rather than the bare word.
        plural, genitive = row["nominative_plural"], row["genitive_singular"]
        candidates = [
            {"signature": "nominative", "stressed": f"три {plural}",
             "definition": f"no counted form — the numeral takes the plural, "
                           f"same as «мої {plural}»"},
            {"signature": "counted", "stressed": f"три {genitive}",
             "definition": f"has a counted form — after 2, 3, 4 it is stressed "
                           f"like the genitive singular, «без {genitive}»"},
        ]
        verdict = row.get("verdict")
        for option in candidates:
            option["chosen"] = option["signature"] == verdict
    if not candidates and manifest and word in manifest:
        entry = manifest[word]
        if isinstance(entry, dict):
            candidates = entry.get("candidates")
            gold_sense = row.get("gold_sense")
            if gold_sense and view.get("_signature") is None:
                match = next((c for c in candidates or ()
                              if isinstance(c, dict) and c.get("sense_id") == gold_sense), None)
                if match:
                    view["_signature"] = str(match.get("signature", ""))
                    if isinstance(row.get("_target"), str) or view.get("_target"):
                        view["_stressed"] = apply_signature(
                            view.get("_target", word), view["_signature"])
    if isinstance(candidates, list) and candidates:
        # Show the alternatives the label was chosen against; a lone rendered
        # accent says nothing about what the decision actually was.
        options = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            sig = str(c.get("signature", ""))
            options.append({
                "signature": sig,
                "stressed": c.get("stressed") or apply_signature(word, sig),
                "definition": (c.get("definition") or "")[:160],
                "chosen": sig == str(view.get("_signature", "\u0000")),
            })
        if options:
            view["_options"] = options
    return view


DATASET_GLOBS = ("*.json", "*.jsonl")
SKIP = re.compile(r"\.idx\.json$|\.bak\.|^bench_|\.report\.json$")


def list_datasets(root: Path) -> list[dict[str, Any]]:
    found = []
    for pattern in DATASET_GLOBS:
        for path in sorted(root.glob(pattern)):
            if SKIP.search(path.name):
                continue
            stat = path.stat()
            found.append({"name": path.name, "size": stat.st_size,
                          "mtime": stat.st_mtime})
    return sorted(found, key=lambda d: -d["size"])


@dataclass(frozen=True)
class Annotation:
    dataset: str
    row: int
    verdict: str          # "wrong" | "recheck" | "ok", or "" for tags only
    tags: list[str] = field(default_factory=list)
    proposed: str = ""    # the signature the reviewer believes is right
    note: str = ""
    form: str = ""
    sentence: str = ""
    was: str = ""         # the label as the dataset has it
    at: float = 0.0


VERDICTS = ("wrong", "recheck", "ok")

# Tags are orthogonal to the verdict: they describe the *row*, not the label on
# it. A toponym whose stress is correct is still worth flagging, because the
# three of them each route to a different fix — a rare form to the "leave it,
# it costs nothing" pile, a toponym to the gazetteer, and alt-needed to the
# cases where neither reading in the inventory is the right one and no amount
# of reweighting the two will help.
TAGS = ("not-frequent", "toponym", "alt-needed")


class Annotations:
    """Append-only review notes, one file per dataset.

    Append-only because a review is expensive and a rewrite is the operation
    that loses it: a crash mid-write can truncate a whole session's work. Later
    entries supersede earlier ones for the same row, so changing your mind is
    just another append and the history stays.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, dataset: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", dataset)
        return self.directory / f"{safe}.jsonl"

    def add(self, note: Annotation) -> None:
        with self._path(note.dataset).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(note.__dict__, ensure_ascii=False) + "\n")

    def latest(self, dataset: str) -> dict[int, dict[str, Any]]:
        """The current verdict for each row: the last one written wins."""
        path = self._path(dataset)
        if not path.is_file():
            return {}
        out: dict[int, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception as error:  # noqa: BLE001
                    log.debug("skipping unreadable annotation: %s", error)
                    continue
                out[int(entry["row"])] = entry
        return out

    def counts(self, dataset: str) -> dict[str, int]:
        tally: Counter[str] = Counter()
        for entry in self.latest(dataset).values():
            verdict = entry.get("verdict")
            if verdict:
                tally[verdict] += 1
            for tag in entry.get("tags") or ():
                tally[tag] += 1
        return dict(tally)

    def export(self, dataset: str) -> str:
        return "".join(
            json.dumps(e, ensure_ascii=False) + "\n"
            for e in sorted(self.latest(dataset).values(), key=lambda e: e["row"])
        )


def load_frequency(path: Path) -> dict[str, int]:
    """Corpus counts per ambiguous form, or an empty table if the file is gone."""
    if not path.is_file():
        log.info("no frequency table at %s; sorting by frequency is off", path)
        return {}
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        log.warning("frequency table %s is unreadable: %s", path, error)
        return {}
    return {str(k): int(v) for k, v in table.items() if isinstance(v, int | float)}


def frequency_of(index: DatasetIndex, row: int, table: dict[str, int]) -> int | None:
    """How often this row's form occurs, or None when it is off the table.

    None is not zero: the table covers the 5,000 commonest ambiguous forms, so
    an absent form is unmeasured rather than never seen. Keeping them apart is
    what lets "everything below 50" mean the rare tail and not the whole file.
    """
    if row >= len(index.row_forms):
        return None
    form = index.row_forms[row]
    return table.get(form) if form else None


def frequency_order(index: DatasetIndex, table: dict[str, int], *,
                    descending: bool = True, low: int | None = None,
                    high: int | None = None) -> list[int]:
    """Row numbers ordered by corpus frequency, optionally cut to a band.

    Rows whose form is off the table sort last in either direction — they carry
    no count, so neither end of the order is where they belong — and any
    explicit bound drops them, since an unmeasured form cannot be shown to fall
    inside one.
    """
    bounded = low is not None or high is not None
    counted: list[tuple[int, int]] = []
    unknown: list[int] = []
    for row in range(index.rows):
        count = frequency_of(index, row, table)
        if count is None:
            if not bounded:
                unknown.append(row)
            continue
        if low is not None and count < low:
            continue
        if high is not None and count > high:
            continue
        counted.append((count, row))
    counted.sort(key=lambda pair: (-pair[0], pair[1]) if descending else pair)
    return [row for _, row in counted] + unknown
