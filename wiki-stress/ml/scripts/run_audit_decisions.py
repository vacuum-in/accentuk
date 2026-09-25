"""Turn an audit table into decisions.

Reads the round-3 audit JSON (tables A, B, C from
audiotostress/scripts/run_audit_lexicon_books.py) and a reviewed markdown
copy, and writes the decisions file run_manual_corrections.py consumes.

Two sources of verdict:
  * the markdown's «вердикт» column, filled in by a person;
  * for table C only, an automatic 'обидва' where the narrators' majority is
    overwhelming (--auto-books, --auto-share): the default is the reading
    served when no tier decides, and a default that is the minority reading
    across that many books is simply the wrong default. Table A is never
    automatic — a lexicon reading nobody says may still be the correct one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="the audit .json")
    parser.add_argument("--reviewed", type=Path, help="the audit .md with verdicts filled in")
    parser.add_argument("--auto-books", type=int, default=0,
                        help="table C: books that must agree for an automatic 'обидва'; 0 = never")
    parser.add_argument("--auto-share", type=float, default=0.9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    verdicts: dict[str, str] = {}
    if args.reviewed:
        for line in args.reviewed.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| форма"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 8 and cells[-1]:
                verdicts[cells[0]] = cells[-1].lower()

    decisions, auto = [], 0
    for table, rows in audit.items():
        for r in rows:
            verdict = verdicts.get(r["form"], "")
            if not verdict and table == "C" and args.auto_books and \
                    r["books"] >= args.auto_books and r["share"] >= args.auto_share:
                verdict, auto = "обидва", auto + 1
            if verdict not in ("правильно", "обидва"):
                continue
            decisions.append({"form": r["form"], "stressed": r["narrators"], "verdict": verdict,
                              "why": f"Audit round 3, table {table}: {r['rows']}/{r['total']} "
                                     f"confident readings in {r['books']} books say "
                                     f"{r['narrators']}; the lexicon served {r['lexicon']}."})
    args.out.write_text(json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
    only = sum(1 for d in decisions if d["verdict"] == "правильно")
    print(f"{len(decisions)} decisions ({only} only, {len(decisions) - only} lead; "
          f"{auto} automatic) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
