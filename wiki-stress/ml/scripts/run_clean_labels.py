"""Find and correct silver-label errors using out-of-fold disagreement.

Re-adjudicating the held-out errors showed that **57% of high-margin mistakes
were annotation errors, not model errors** — both annotators, blind with the
target masked, overturned the silver label on 16 of the 28 they agreed on.
Correcting only those moved precision at margin 15 from 0.9637 to 0.9855.

If that rate holds on the held-out rows, the *training* rows carry the same
noise and the model is being taught from it. This finds the likely errors and
fixes them.

The signal is an out-of-fold confident disagreement: a model that never trained
on this form nonetheless scores a different sense than the label, by a wide
margin. That is the standard confident-learning cue, and it is available for
every row because cross-validation predicts each one from a fold that excluded
its form.

Two safeguards keep this from becoming the model rewriting its own training set:

* Adjudication is **blind** — the target is masked and the model sees only the
  sentence and the candidate glosses, never the silver label or the prediction.
* A label is changed only when **both deployments independently agree** on the
  same replacement. Where they disagree or hedge, the original label stands.
  The prediction opens the question; it never answers it.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity
from ukstress_ml.annotate import (
    LABEL_DEPLOYMENT,
    MASK,
    VERIFY_DEPLOYMENT,
    Usage,
    load_config,
)

SYSTEM = (
    "Ти — лінгвіст-анотатор. Визначаєш, у якому значенні вжите слово в реченні. "
    "Відповідаєш лише валідним JSON."
)


def build_prompt(row: dict[str, Any], candidates: list[Any]) -> str:
    sentence = row["sentence"]
    masked = sentence[: row["start"]] + MASK + sentence[row["end"] :]
    options = "\n".join(f"  {c.sense_id} — {c.definition}" for c in candidates)
    return (
        f"Приховане слово має написання: {row['form']}\n"
        f"Можливі значення:\n{options}\n\n"
        f"Речення (слово позначене {MASK}):\n{masked}\n\n"
        'Яке значення вжите? Якщо контекст не дозволяє визначити — "unclear": true.\n'
        'Формат: {"sense_id": "...", "unclear": false}'
    )


def ask(deployment: Any, config: Any, prompt: str, usage: Usage) -> dict[str, Any]:
    client = deployment.client(config)
    model = deployment.resolved_model(config)
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            if response.usage:
                usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception:  # noqa: BLE001
            if attempt == 3:
                usage.fail()
                return {}
            time.sleep(4 * attempt)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path, default=Path("output/ml/cv_inflected_v3_predictions.jsonl")
    )
    parser.add_argument("--silver", type=Path, default=Path("output/ml/silver_inflected.json"))
    parser.add_argument(
        "--inventory", type=Path, default=Path("output/ml/ambiguous_forms_inflected.jsonl")
    )
    parser.add_argument("--margin", type=float, default=9.0,
                        help="only re-adjudicate disagreements at least this confident")
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw/label_clean.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/ml/silver_inflected_clean.json")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    silver = {r["sentence_id"]: r for r in json.loads(args.silver.read_text(encoding="utf-8"))}
    forms = {f.form: f for f in ambiguity.load(args.inventory)}
    suspects = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suspects = [
        p for p in suspects if not p["correct"] and p["margin"] >= args.margin
        and p["sentence_id"] in silver
    ]
    done: set[str] = set()
    if args.raw.exists():
        for line in args.raw.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("a") is not None or record.get("b") is not None:
                done.add(record["sentence_id"])
    pending = [p for p in suspects if p["sentence_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"confident out-of-fold disagreements (margin >= {args.margin}): {len(suspects):,}")
    print(f"  already adjudicated: {len(done):,}   running: {len(pending):,}", flush=True)

    config = load_config()
    usage = Usage()
    tally: Counter[str] = Counter()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("a", encoding="utf-8") as handle:
        for position, suspect in enumerate(pending, 1):
            row = silver[suspect["sentence_id"]]
            candidates = forms[row["form"]].candidates
            prompt = build_prompt(row, list(candidates))
            first = ask(LABEL_DEPLOYMENT, config, prompt, usage)
            second = ask(VERIFY_DEPLOYMENT, config, prompt, usage)
            handle.write(
                json.dumps(
                    {
                        "sentence_id": row["sentence_id"],
                        "form": row["form"],
                        "silver": row["gold_sense"],
                        "a": first.get("sense_id"),
                        "b": second.get("sense_id"),
                        "unclear": bool(first.get("unclear") or second.get("unclear")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if first.get("sense_id") and first.get("sense_id") == second.get("sense_id"):
                tally["agreed"] += 1
                if first["sense_id"] != row["gold_sense"]:
                    tally["overturned"] += 1
            else:
                tally["no_consensus"] += 1
            if position % 25 == 0 or position == len(pending):
                print(f"  {position}/{len(pending)}  {dict(tally)}  {usage.snapshot()}", flush=True)

    corrections: dict[str, str] = {}
    for line in args.raw.open(encoding="utf-8"):
        record = json.loads(line)
        if (
            record.get("a")
            and record["a"] == record.get("b")
            and not record["unclear"]
            and record["a"] != record["silver"]
        ):
            corrections[record["sentence_id"]] = record["a"]

    cleaned = []
    for row in json.loads(args.silver.read_text(encoding="utf-8")):
        new = corrections.get(row["sentence_id"])
        if new:
            ids = [c.sense_id for c in forms[row["form"]].candidates]
            if new in ids:
                row = {**row, "gold_sense": new, "gold": ids.index(new), "label_revised": True}
        cleaned.append(row)
    args.output.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
    print(f"\ncorrections applied: {len(corrections):,} of {len(cleaned):,} rows")
    print(f"written to {args.output}")
    print(json.dumps({**dict(tally), **usage.snapshot()}, indent=2))


if __name__ == "__main__":
    main()
