"""Adjudicate groups whose candidates ended up sharing one gloss.

166 groups (1,724 forms) carry the same definition on two candidates, which
makes them unresolvable: the cross-encoder sees identical `(context, gloss)`
inputs and picks arbitrarily, at high confidence. They were 32% of high-margin
errors.

There are two different situations underneath, and they need opposite handling:

* **Distinct referents, lazy gloss.** Two villages both named Богданівка, in
  different oblasts. The senses are genuinely separate; the gloss simply failed
  to say how. Naming the region makes them learnable again.
* **One referent, two stresses.** `ба́йрацька`/`байра́цька` both mean "relating
  to a байрак". No gloss can separate these because there is nothing to
  separate — this is free variation, and it belongs in the dictionary tier, not
  the model tier.

Only the annotator can tell these apart, so the prompt asks directly and treats
"same referent" as a first-class answer rather than pressing for a distinction
that may not exist. Inventing a difference to satisfy the schema would produce
exactly the confident-but-arbitrary behaviour this is meant to remove.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity
from ukstress_ml.annotate import LABEL_DEPLOYMENT, Usage, load_config
from ukstress_ml.glosses import GlossError, validate_gloss

SYSTEM = (
    "Ти — український лексикограф. Розрізняєш значення слів-омографів. "
    "Відповідаєш лише валідним JSON."
)


def build_prompt(stressed: list[str], gloss: str) -> str:
    variants = "\n".join(f"  {i}. {s}" for i, s in enumerate(stressed))
    return f"""Дві наголошені форми одного написання мають ОДНАКОВЕ тлумачення:

{variants}

Спільне тлумачення: {gloss}

Питання: це справді ДВА різні значення (наприклад, два різні села в різних
областях, різні особи, різні поняття) — чи це ОДНЕ значення з двома
допустимими наголосами?

- Якщо значення РІЗНІ: напиши для кожної форми окреме тлумачення, яке чітко
  називає відмінність (область, регіон, кого саме стосується, що саме означає).
- Якщо значення ОДНЕ: постав "same_referent": true і не вигадуй відмінності.

Правила для тлумачень: 5–20 слів, описують ЗНАЧЕННЯ, не згадують наголос,
складів чи вимови.

Формат (лише JSON):
{{"same_referent": false, "senses": [{{"index": 0, "definition": "..."}},
 {{"index": 1, "definition": "..."}}]}}"""


def parse(content: str, stressed: list[str]) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"status": "unparsed"}
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return {"status": "unparsed"}
    if payload.get("same_referent"):
        return {"status": "free_variation"}
    out: dict[int, str] = {}
    for item in payload.get("senses", []):
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(stressed):
            continue
        try:
            out[index] = validate_gloss(str(item.get("definition", "")))
        except GlossError:
            continue
    if len(out) < len(stressed) or len(set(out.values())) < len(out):
        # Still not distinguishing: treat as unresolved rather than shipping a
        # pair that is only cosmetically different.
        return {"status": "still_identical", "definitions": out}
    return {"status": "distinguished", "definitions": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path, default=Path("output/ml/ambiguous_forms_glossed.jsonl")
    )
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw/regloss.jsonl"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    inventory = ambiguity.load(args.inventory)
    groups: dict[int, Any] = {}
    for form in inventory:
        if form.complete and len({c.definition.strip() for c in form.candidates}) < len(
            form.candidates
        ):
            groups.setdefault(form.group_id, form)

    done = set()
    if args.raw.exists():
        for line in args.raw.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("content"):
                done.add(int(record["group_id"]))
    pending = [g for g in sorted(groups) if g not in done]
    if args.limit:
        pending = pending[: args.limit]

    config = load_config()
    client = LABEL_DEPLOYMENT.client(config)
    model = LABEL_DEPLOYMENT.resolved_model(config)
    print(f"model={model}  duplicate-gloss groups={len(groups):,}  running={len(pending):,}",
          flush=True)

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    usage = Usage()
    tally: dict[str, int] = {}
    with args.raw.open("a", encoding="utf-8") as handle:
        for position, group_id in enumerate(pending, 1):
            form = groups[group_id]
            stressed = [c.stressed for c in form.candidates]
            content = ""
            for attempt in range(1, 4):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": build_prompt(
                                    stressed, form.candidates[0].definition
                                ),
                            },
                        ],
                        max_tokens=600,
                        temperature=0.2,
                    )
                    content = response.choices[0].message.content or ""
                    if response.usage:
                        usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
                    break
                except Exception as error:  # noqa: BLE001
                    if attempt == 3:
                        usage.fail()
                        print(f"  group {group_id}: {error}", flush=True)
                    else:
                        time.sleep(5 * attempt)
            result = parse(content, stressed) if content else {"status": "no_content"}
            tally[result["status"]] = tally.get(result["status"], 0) + 1
            handle.write(
                json.dumps(
                    {
                        "group_id": group_id,
                        "sense_ids": [c.sense_id for c in form.candidates],
                        "stressed": stressed,
                        "content": content,
                        "result": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if position % 25 == 0 or position == len(pending):
                print(f"  {position}/{len(pending)}  {tally}  {usage.snapshot()}", flush=True)

    print(json.dumps({"tally": tally, **usage.snapshot()}, indent=2))


if __name__ == "__main__":
    main()
