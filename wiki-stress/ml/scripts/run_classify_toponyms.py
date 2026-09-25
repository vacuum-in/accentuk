"""Classify starved homograph groups by whether every sense is a toponym.

The 294 groups with no mined sentences are not one population. Some are pairs
of place-name adjectives whose two stresses refer to the same place — free
variation, where either stress is acceptable and no context can decide. Others
are genuine semantic homographs that happen to be rare.

The distinction decides the handling:

* **all senses toponymic** -> equal probability, excluded from the model
  dataset. A model asked to choose here would be guessing at high confidence.
* **mixed, or none toponymic** -> kept for manual review, because a real
  semantic contrast may be present.

Both deployments are asked independently and must agree; disagreement is
reported as `uncertain` rather than resolved, so a coin flip never silently
becomes an exclusion.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ukstress_ml.annotate import LABEL_DEPLOYMENT, VERIFY_DEPLOYMENT, Usage, load_config

SYSTEM = (
    "Ти — український лексикограф. Визначаєш, чи є значення слова топонімічним "
    "(стосується назви місця: села, міста, річки, регіону). "
    "Відповідаєш лише валідним JSON."
)


def build_prompt(stressed: list[str], definitions: list[str]) -> str:
    senses = "\n".join(
        f"  {i}. {s} — {d}" for i, (s, d) in enumerate(zip(stressed, definitions, strict=True))
    )
    return f"""Значення одного написання з різними наголосами:

{senses}

Для КОЖНОГО значення визнач, чи воно топонімічне — тобто стосується назви
місця (села, міста, річки, області, регіону) або похідне від такої назви.

Загальні слова (напр. "яр", "борода", "тканина") — НЕ топоніми,
навіть якщо звучать схоже на назву.

Формат (лише JSON):
{{"senses": [{{"index": 0, "toponym": true}}, {{"index": 1, "toponym": false}}]}}"""


def ask(deployment: Any, config: Any, prompt: str, usage: Usage) -> list[bool] | None:
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
                max_tokens=300,
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            if response.usage:
                usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            payload = json.loads(match.group())
            items = payload.get("senses", [])
            out: dict[int, bool] = {}
            for item in items:
                if isinstance(item.get("index"), int):
                    out[item["index"]] = bool(item.get("toponym"))
            return [out[i] for i in sorted(out)] if out else None
        except Exception:  # noqa: BLE001
            if attempt == 3:
                usage.fail()
                return None
            time.sleep(4 * attempt)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, default=Path("output/ml/starved_groups.json"))
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw/toponym_class.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/ml/toponym_review.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text(encoding="utf-8"))
    done: set[int] = set()
    if args.raw.exists():
        for line in args.raw.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("a") is not None:
                done.add(int(record["group_id"]))
    pending = [g for g in groups if g["group_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    config = load_config()
    usage = Usage()
    tally: Counter[str] = Counter()
    print(f"groups={len(groups):,}  done={len(done):,}  running={len(pending):,}", flush=True)

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("a", encoding="utf-8") as handle:
        for position, group in enumerate(pending, 1):
            prompt = build_prompt(group["stressed"], group["definitions"])
            first = ask(LABEL_DEPLOYMENT, config, prompt, usage)
            second = ask(VERIFY_DEPLOYMENT, config, prompt, usage)
            handle.write(
                json.dumps(
                    {"group_id": group["group_id"], "a": first, "b": second},
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if first is not None and first == second:
                tally["all_toponym" if all(first) else
                      ("none_toponym" if not any(first) else "mixed")] += 1
            else:
                tally["uncertain"] += 1
            if position % 25 == 0 or position == len(pending):
                print(f"  {position}/{len(pending)}  {dict(tally)}  {usage.snapshot()}", flush=True)

    verdicts: dict[int, list[bool] | None] = {}
    for line in args.raw.open(encoding="utf-8"):
        record = json.loads(line)
        a, b = record.get("a"), record.get("b")
        verdicts[int(record["group_id"])] = a if (a is not None and a == b) else None

    rows = []
    for group in groups:
        flags = verdicts.get(group["group_id"])
        if flags is None or len(flags) != len(group["stressed"]):
            verdict = "uncertain"
        elif all(flags):
            verdict = "all_toponym"
        elif any(flags):
            verdict = "mixed"
        else:
            verdict = "none_toponym"
        rows.append({**group, "toponym_flags": flags, "verdict": verdict,
                     "action": "exclude_equal_probability" if verdict == "all_toponym"
                     else "manual_review"})
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(r["verdict"] for r in rows)
    print("\nverdicts:", dict(counts))
    print(f"  excluded (all toponym): {counts['all_toponym']:,} groups, "
          f"{sum(r['n_forms'] for r in rows if r['verdict']=='all_toponym'):,} forms")
    print(f"  manual review         : {len(rows)-counts['all_toponym']:,} groups")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
