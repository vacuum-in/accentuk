"""Author missing sense definitions on Azure AI Foundry.

Resumable by group: every raw response is written before parsing, so a parser
change costs a re-parse rather than a re-spend. Run with --limit first; the
pilot gate is a human read of the sample, because a gloss is the label the
model is scored against, not decoration.

    uv run python scripts/run_glosses.py --limit 20      # pilot
    uv run python scripts/run_glosses.py                 # full run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ukstress_ml.annotate import LABEL_DEPLOYMENT, Usage, load_config
from ukstress_ml.glosses import SYSTEM, build_prompt, load_tasks, parse_response

TASKS = Path("output/ml/gloss_tasks.jsonl")
RAW = Path("output/ml/raw/glosses.jsonl")


def completed(raw_path: Path) -> set[int]:
    done: set[int] = set()
    if not raw_path.exists():
        return done
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only records that carry content count as done; a transient failure
            # must be retried on resume, not skipped forever.
            if record.get("content"):
                done.add(int(record["group_id"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    config = load_config()
    client = LABEL_DEPLOYMENT.client(config)
    model = LABEL_DEPLOYMENT.resolved_model(config)
    tasks = list(load_tasks(args.tasks))
    done = completed(args.raw)
    pending = [t for t in tasks if t.group_id not in done]
    if args.limit:
        # Sample rather than take the head: the file is group-id ordered, so the
        # first N would all be spellings starting with the same letters.
        random.Random(args.seed).shuffle(pending)
        pending = pending[: args.limit]

    print(f"model={model}  tasks={len(tasks):,}  done={len(done):,}  running={len(pending):,}",
          flush=True)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    usage = Usage()
    accepted_total = rejected_total = 0
    with args.raw.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending, 1):
            content = ""
            for attempt in range(1, 5):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": build_prompt(task)},
                        ],
                        max_tokens=800,
                        temperature=0.3,
                    )
                    content = response.choices[0].message.content or ""
                    if response.usage:
                        usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
                    break
                except Exception as error:  # noqa: BLE001 - retry, then record the failure
                    if attempt == 4:
                        usage.fail()
                        print(f"  group {task.group_id}: failed — {error}", flush=True)
                    else:
                        time.sleep(6 * attempt)
            accepted, rejected = parse_response(task, content) if content else ([], ["no content"])
            accepted_total += len(accepted)
            rejected_total += len(rejected)
            handle.write(
                json.dumps(
                    {
                        "group_id": task.group_id,
                        "model": model,
                        "content": content,
                        "accepted": [a.__dict__ for a in accepted],
                        "rejected": rejected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(
                    f"  {index}/{len(pending)}  accepted={accepted_total} "
                    f"rejected={rejected_total} {usage.snapshot()}",
                    flush=True,
                )

    print(json.dumps({"accepted": accepted_total, "rejected": rejected_total,
                      **usage.snapshot()}, indent=2))


if __name__ == "__main__":
    main()
