"""Build the book plan: which audio goes with which extracted text.

One entry per book folder that has both an audio file and an extracted text,
with the audio's duration from ffprobe. Entries already in the plan are kept
as they are, so a second batch of books extends the plan rather than
rewriting it; a folder whose slug is already planned is reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

AUDIO = (".mp3", ".m4a", ".wav", ".flac", ".ogg")


def audio_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/books"))
    args = parser.parse_args()

    plan_path = args.out / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else []
    known = {b["slug"] for b in plan}
    index = {row["book"]: row for row in
             json.loads((args.out / "index.json").read_text(encoding="utf-8"))}

    added, skipped = 0, []
    for folder in sorted(p for p in args.books.iterdir() if p.is_dir()):
        row = index.get(folder.name)
        if row is None or not row.get("words"):
            skipped.append((folder.name, "no extracted text"))
            continue
        audio = [p for p in folder.iterdir() if p.suffix.lower() in AUDIO]
        if len(audio) != 1:
            skipped.append((folder.name, f"{len(audio)} audio files"))
            continue
        if row["slug"] in known:
            skipped.append((folder.name, f"already planned as {row['slug']}"))
            continue
        try:
            hours = round(audio_seconds(audio[0]) / 3600, 2)
        except (ValueError, subprocess.CalledProcessError):
            # ffprobe reports N/A for a truncated or still-uploading file
            skipped.append((folder.name, f"no duration: {audio[0].name}"))
            continue
        plan.append({"slug": row["slug"], "audio": str(audio[0]),
                     "text": str(args.out / f"{row['slug']}.jsonl"),
                     "hours": hours, "words": row["words"], "name": folder.name})
        known.add(row["slug"])
        added += 1
        print(f"  {row['slug']:<42} {hours:>6.2f} h {row['words']:>9,} words "
              f"{row['words'] / (hours * 3600):.1f} w/s")

    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{added} added, {len(plan)} in the plan, "
          f"{sum(b['hours'] for b in plan):.0f} h in total -> {plan_path}")
    for name, why in skipped:
        print(f"  skipped {name[:50]}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
