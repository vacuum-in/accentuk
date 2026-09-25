"""Write the gold draft with the reviewer's corrections applied.

The review page (`run_review_server.py`) keeps one answer per line of the
draft, keyed "group:line" exactly as the page numbers them: an edited text,
a dropped line, or both. This replays them over the draft and writes the
corrected file, comments and group headers kept, dropped lines left out.
Lines nobody touched are copied as drafted; the header records how many
were corrected, so the file does not pass for a fully reviewed set.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import unicodedata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=Path("ml/data/gold_modern_draft_1.txt"))
    parser.add_argument("--answers", type=Path, default=Path("output/ml/review/gold.json"))
    parser.add_argument("--out", type=Path, default=Path("ml/data/gold_modern_reviewed_1.txt"))
    args = parser.parse_args()

    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    out, group, line_no = [], -1, 0
    edited = dropped = kept = 0
    for raw in args.draft.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        if line.startswith("#"):
            # the page opens a group at every "# form: …" header
            if re.match(r"#\s*\S+:\s", line):
                group, line_no = group + 1, 0
            out.append(raw)
            continue
        if group < 0:
            group, line_no = 0, 0
        answer = answers.get(f"{group}:{line_no}", {})
        line_no += 1
        if answer.get("deleted"):
            dropped += 1
            continue
        text = unicodedata.normalize("NFC", answer.get("text") or line)
        if answer.get("text") and text != unicodedata.normalize("NFC", line):
            edited += 1
        kept += 1
        out.append(text)
    header = (f"# Reviewed gold, {datetime.date.today():%Y-%m-%d}: {edited} lines corrected by the reviewer, "
              f"{dropped} dropped, {kept} kept.\n# Lines not corrected are the draft's marks, "
              f"which may or may not have been read.\n")
    args.out.write_text(header + "\n".join(out) + "\n", encoding="utf-8")
    print(f"{edited} corrected, {dropped} dropped, {kept} lines -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
