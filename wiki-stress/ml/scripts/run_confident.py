"""Stress a text, marking only what can be trusted, and ask a model for the rest.

Built for preparing an audio dataset, where a wrong stress is worse than none.
Measured per tier on lang-uk, the pipeline's certainty is very unevenly spread:

    lexicon, one reading   99.0%      dictionary default   77.3%
    prepositional rule     97.4%      suffix fallback      71.5%
    morphology             87.6%      compound             66.0%
    model, margin >= 5     86.3%      model, margin < 2     49.0%

So the confident policy marks only the first kind — 78.0% of multi-vowel tokens
at 99.00% — and leaves the rest bare. `--llm` then offers those blanks to a
model through the Codex CLI, which is the tier this exists to test rather than
to trust: measured on the same benchmark, Gemma 4 E4B answered them at 50.3%
and gpt-5.6-luna at 57.5%, against 65.0% for the dictionary default they would
replace. Both were worse than leaving the word alone.

Free-variation forms are never offered. Both of their readings are correct and
the reference recorded one; a model asked about those is being scored on
someone else's coin flip.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import httpx

TRUSTED = frozenset({"prepositional", "counted_form", "aspect"})
VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")

PROMPT = """Ти визначаєш наголос в українському тексті.

Для кожного пронумерованого випадку обери, який варіант правильний САМЕ в
цьому реченні. Зважай на значення слова та граматику, а не на те, який наголос
трапляється частіше. Якщо обидва варіанти однаково прийнятні тут — постав 0.

{cases}

Виведи РІВНО валідний JSON, без пояснень і без markdown:
[{{"n": 1, "choice": 2}}, ...]
"""


def confident(token: dict) -> bool:
    status = token.get("status")
    if status in TRUSTED:
        return True
    return status == "stressed" and len(token.get("candidates") or []) < 2


def free_variation(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("triage_class") == "free_variation":
                out.add(row["form_normalized"])
    return out


def ask_codex(prompt: str, model: str, effort: str, timeout: float) -> str:
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as handle:
        out = Path(handle.name)
    try:
        subprocess.run(
            ["codex", "exec", "--skip-git-repo-check",
             "-c", f"model={model}", "-c", f"model_reasoning_effort={effort}",
             "--output-last-message", str(out), prompt],
            check=False, capture_output=True, timeout=timeout,
        )
        return out.read_text(encoding="utf-8")
    finally:
        out.unlink(missing_ok=True)



def excluded_forms(path: Path) -> set[str]:
    """Forms held out of the model queue by hand.

    The triage covers free variation, which has no right answer to find. This
    covers the other case: a form whose alternative reading belongs to a lexeme
    nobody writes, so the question looks open and is not. `світи́` is the plural
    of `світ` and the imperative of `світити` at the same accent; only the
    genitive singular of a rare feminine noun differs, and offering that as an
    option invites a model to invent a distinction the sentence never had.
    """
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(unicodedata.normalize("NFC", json.loads(line)["form"]).lower())
    return out

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="text to stress; omit to read stdin")
    parser.add_argument("--api", default="http://127.0.0.1:8080/v1/stress")
    parser.add_argument("--llm", action="store_true",
                        help="offer the blanks to a model as a last tier")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--effort", default="high", choices=("low", "medium", "high"))
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--triage", type=Path,
                        default=Path("output/ml/ambiguous_surface_v4.jsonl"))
    parser.add_argument("--exclude", type=Path,
                        default=Path("ml/data/excluded_forms.jsonl"))
    parser.add_argument("--report", action="store_true",
                        help="print what each tier decided instead of the text")
    args = parser.parse_args()

    source = args.text if args.text else sys.stdin.read()
    if not source.strip():
        print("nothing to stress", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ukstress_ml.browser import apply_signature

    client = httpx.Client(timeout=180.0)
    free = free_variation(args.triage) | excluded_forms(args.exclude)
    lines = source.splitlines() or [source]
    rendered: list[str] = []
    blanks: list[dict] = []

    for line_number, line in enumerate(lines):
        if not line.strip():
            rendered.append(line)
            continue
        body = client.post(args.api, json={"text": line})
        body.raise_for_status()
        payload = body.json()
        pieces, cursor = [], 0
        for token in payload["tokens"]:
            start, end = token["start"], token["end"]
            surface = line[start:end]
            if confident(token):
                marked = token.get("output_text") or surface
            else:
                marked = surface
                candidates = token.get("candidates") or []
                if (len(candidates) > 1
                        and unicodedata.normalize("NFC", surface).lower() not in free
                        and sum(1 for c in surface if c in VOWELS) > 1):
                    blanks.append({"line": line_number, "start": start, "end": end,
                                   "surface": surface, "sentence": line,
                                   "signatures": candidates,
                                   "served": token.get("output_text"),
                                   "status": token["status"]})
            pieces.append(line[cursor:start])
            pieces.append(marked)
            cursor = end
        pieces.append(line[cursor:])
        rendered.append("".join(pieces))

    filled = 0
    if args.llm and blanks:
        for start in range(0, len(blanks), args.batch):
            batch = blanks[start:start + args.batch]
            cases = []
            for number, item in enumerate(batch, 1):
                options = []
                for signature in item["signatures"]:
                    try:
                        options.append(apply_signature(item["surface"], signature))
                    except Exception:  # noqa: BLE001 - unusable signature, no option
                        options.append(item["surface"])
                item["options"] = options
                listed = "  ".join(f"{i}) {o}" for i, o in enumerate(options, 1))
                cases.append(f'{number}. Речення: {item["sentence"]}\n'
                             f'   Слово: {item["surface"]}\n   Варіанти: {listed}')
            reply = ask_codex(PROMPT.format(cases="\n".join(cases)),
                              args.model, args.effort, args.timeout)
            try:
                parsed = json.loads(reply[reply.index("["): reply.rindex("]") + 1])
                picks = {int(r["n"]): int(r["choice"]) for r in parsed if isinstance(r, dict)}
            except Exception as error:  # noqa: BLE001 - a bad batch fills nothing
                print(f"batch at {start} unparsable: {error}", file=sys.stderr)
                picks = {}
            for number, item in enumerate(batch, 1):
                index = picks.get(number, 0) - 1
                if 0 <= index < len(item["options"]):
                    item["picked"] = item["options"][index]
                    filled += 1

        # Splice the filled words back, right to left so offsets stay valid.
        for item in sorted(blanks, key=lambda b: (-b["line"], -b["start"])):
            if item.get("picked"):
                line = rendered[item["line"]]
                pattern = re.escape(item["surface"])
                rendered[item["line"]] = re.sub(
                    rf"(?<!\w){pattern}(?!\w)", item["picked"].replace("\\", "\\\\"),
                    line, count=1)

    if args.report:
        print(f"blanks left by the confident policy: {len(blanks)}")
        if args.llm:
            print(f"filled by {args.model} ({args.effort}): {filled}")
        for item in blanks:
            mark = item.get("picked") or "—"
            print(f"  {item['surface']:<16} [{item['status']:<18}] "
                  f"pipeline {item['served'] or '—':<16} llm {mark}")
    else:
        print("\n".join(rendered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
