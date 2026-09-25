"""Generate the aspect contexts neither the wiki corpus nor the books contain.

`виходити` is labelled `вихо́дити` in all 184 of its corpus rows, and the book
scan found 10 perfective contexts for it and none at all for `викидати` or
`скликати`. The class the model has to learn is simply absent from the text
that exists, so it has to be written.

Two things make the output usable rather than plausible-looking. The prompt
fixes the *syntactic frame* — the verb must stay an infinitive and sit directly
after a named trigger — and every returned sentence is checked for the form and
the trigger before it is kept. Asked without the frame, the model silently
returned finite forms («Він ви́кидав старі речі») or forced the infinitive into
word salad («Сусіди часто викида́ти сміття залишають»), and 15 of 18 sentences
were unusable.

The trigger gives a strong prior on aspect, not a guarantee: `зміг` takes an
imperfective happily in «зміг викидати сніг усю ніч». These rows are candidates
for review, exactly like the counted-form queue, not labels.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

PERFECTIVE_TRIGGERS = ("вдалося", "встиг", "зміг")
IMPERFECTIVE_TRIGGERS = ("почав", "став", "продовжував")

PROMPT = """Ти будуєш навчальні дані для української системи наголошування.

ЗАВДАННЯ. Для кожного дієслова напиши речення, у яких воно стоїть в
ІНФІНІТИВІ — точно в тій формі, що подана, не змінюй її — і кероване
зазначеним словом-тригером, який стоїть безпосередньо перед ним.

Дієслова:
{verbs}

Тригери доконаного виду: {perfective}
Тригери недоконаного виду: {imperfective}

Для кожного дієслова: по одному реченню на кожен тригер, тобто {count} речень.
Речення природні, 7-15 слів. Наголосів у тексті НЕ став.

Виведи РІВНО валідний JSON, без пояснень і без markdown:
[{{"form":"...","trigger":"...","aspect":"perfective","sentence":"..."}}]
"""


def run_codex(prompt: str, timeout: float) -> str:
    """One non-interactive Codex turn, returning its last message."""
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as handle:
        out = Path(handle.name)
    try:
        subprocess.run(
            ["codex", "exec", "--skip-git-repo-check",
             "-c", "model_reasoning_effort=low",
             "--output-last-message", str(out), prompt],
            check=False, capture_output=True, timeout=timeout,
        )
        return out.read_text(encoding="utf-8")
    finally:
        out.unlink(missing_ok=True)


def parse_rows(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [row for row in rows if isinstance(row, dict)]


def keep(row: dict) -> bool:
    """The sentence must actually contain the infinitive and its trigger.

    Without this the batch looks fine and is not: the form comes back
    conjugated, which teaches the model nothing about the infinitive it was
    asked for.
    """
    sentence = unicodedata.normalize("NFC", row.get("sentence", ""))
    form = unicodedata.normalize("NFC", row.get("form", ""))
    trigger = unicodedata.normalize("NFC", row.get("trigger", ""))
    if not (sentence and form and trigger):
        return False
    return bool(re.search(rf"\b{re.escape(trigger)}\s+{re.escape(form)}\b", sentence, re.IGNORECASE))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path,
                        default=Path("output/ml/aspect_needs_generation.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/aspect_generated.json"))
    parser.add_argument("--limit", type=int, default=60,
                        help="how many forms to cover, most frequent first")
    parser.add_argument("--batch", type=int, default=3,
                        help="verbs per Codex turn; three fits comfortably")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    targets = json.loads(args.targets.read_text(encoding="utf-8"))[: args.limit]
    kept: list[dict] = []
    if args.out.is_file():
        kept = json.loads(args.out.read_text(encoding="utf-8"))
        done = {row["form"] for row in kept}
        targets = [t for t in targets if t["form"] not in done]
        print(f"resuming: {len(kept):,} rows on file, {len(targets)} forms left", flush=True)

    rejected = 0
    for index in range(0, len(targets), args.batch):
        batch = targets[index:index + args.batch]
        verbs = "\n".join(f"{n}. {row['form']}" for n, row in enumerate(batch, 1))
        prompt = PROMPT.format(
            verbs=verbs,
            perfective=", ".join(PERFECTIVE_TRIGGERS),
            imperfective=", ".join(IMPERFECTIVE_TRIGGERS),
            count=len(PERFECTIVE_TRIGGERS) + len(IMPERFECTIVE_TRIGGERS),
        )
        rows = parse_rows(run_codex(prompt, args.timeout))
        good = [row for row in rows if keep(row)]
        rejected += len(rows) - len(good)
        kept.extend(good)
        args.out.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {index + len(batch):>4}/{len(targets)} forms  "
              f"+{len(good)} kept, {len(rows) - len(good)} rejected  "
              f"({', '.join(row['form'] for row in batch)})", flush=True)

    print(f"\nkept {len(kept):,} sentences, rejected {rejected}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
