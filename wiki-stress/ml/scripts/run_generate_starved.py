"""Generate training sentences for starved homograph groups.

264 groups have no natural occurrence anywhere in Ukrainian Wikipedia, so no
amount of mining reaches them. They are genuine rare homographs — the toponym
pairs that turned out to be free variation have already been excluded — and
generation is the only route into the training set.

One request per (form, sense): the model writes sentences using the target in
one specific sense, and every sentence is put through the same validator the
mined corpus uses (`generate.validate_sentence`) before it is kept. That
rejects sentences carrying stress marks, containing a sibling's stressed
spelling, using the target more than once, or explaining the word instead of
using it — each of which would let the model learn the annotation rather than
the language.

Generation runs on DeepSeek-V4-Pro; groups whose yield falls short escalate to
gpt-5.4, which handles the harder senses better.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from ukstress_ml import ambiguity, generate
from ukstress_ml.annotate import LABEL_DEPLOYMENT, Usage, load_config

#: Prompt, request and validation now live in `ukstress_ml.generate`, so the
#: balanced-data generator and this one cannot drift apart in what they ask for
#: or what they accept.
TOPICS = generate.TOPICS
SYSTEM = generate.SYSTEM
build_prompt = generate.build_prompt
harvest = generate.harvest


def call(client: OpenAI, model: str, prompt: str, usage: Usage, max_tokens: int) -> str:
    return generate.call_model(client, model, prompt, usage, max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path("output/ml/toponym_review.json"))
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/ambiguous_forms_inflected.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw/generated_starved.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/ml/generated_starved.json"))
    parser.add_argument("--per-sense", type=int, default=10)
    parser.add_argument("--forms-per-group", type=int, default=2)
    parser.add_argument("--escalate-below", type=int, default=4,
                        help="retry on gpt-5.4 when a sense yields fewer than this")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    # Excluded toponym groups are free variation; they must not be generated for.
    targets = {r["group_id"] for r in review if r["verdict"] == "none_toponym"}
    inventory = {f.form: f for f in ambiguity.load(args.inventory)}
    by_group: dict[int, list[Any]] = defaultdict(list)
    for form in inventory.values():
        if form.group_id in targets:
            by_group[form.group_id].append(form)

    jobs: list[tuple[Any, Any, str]] = []
    for index, group_id in enumerate(sorted(by_group)):
        forms = sorted(by_group[group_id], key=lambda f: f.form)[: args.forms_per_group]
        for form in forms:
            for candidate in form.candidates:
                jobs.append((form, candidate, TOPICS[index % len(TOPICS)]))
    if args.limit:
        jobs = jobs[: args.limit]

    done: set[str] = set()
    if args.raw.exists():
        for line in args.raw.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("content"):
                done.add(f"{record['form']}|{record['sense_id']}")
    pending = [j for j in jobs if f"{j[0].form}|{j[1].sense_id}" not in done]

    config = load_config()
    primary = LABEL_DEPLOYMENT.client(config)
    primary_model = LABEL_DEPLOYMENT.resolved_model(config)
    escalate = OpenAI(
        base_url=str(config["AZURE_OPENAI_ENDPOINT_INITAITESTING"]).rstrip("/"),
        api_key=str(config["AZURE_OPENAI_API_KEY_INITAITESTING"]), timeout=240.0, max_retries=2)
    escalate_model = str(config.get("AZURE_OPENAI_DEPLOYMENT_INITAITESTING") or "gpt-5.4")

    print(f"starved groups={len(by_group):,}  (form,sense) jobs={len(jobs):,}  "
          f"pending={len(pending):,}", flush=True)

    usage = Usage()
    rejected_total: Counter = Counter()
    kept_total = 0
    escalated = 0
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("a", encoding="utf-8") as handle:
        for position, (form, sense, topic) in enumerate(pending, 1):
            prompt = build_prompt(form, sense, args.per_sense, topic)
            content = call(primary, primary_model, prompt, usage, 2000)
            kept, rejected = harvest(content, form, sense.sense_id)
            model_used = primary_model
            if len(kept) < args.escalate_below:
                content2 = call(escalate, escalate_model, prompt, usage, 6000)
                kept2, rejected2 = harvest(content2, form, sense.sense_id)
                escalated += 1
                if len(kept2) > len(kept):
                    kept, rejected, content, model_used = kept2, rejected2, content2, escalate_model
            rejected_total += rejected
            kept_total += len(kept)
            handle.write(json.dumps({"form": form.form, "sense_id": sense.sense_id,
                                     "model": model_used, "content": content,
                                     "kept": kept}, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 25 == 0 or position == len(pending):
                print(f"  {position}/{len(pending)}  kept={kept_total} escalated={escalated} "
                      f"{usage.snapshot()}", flush=True)

    rows: list[dict[str, Any]] = []
    for line in args.raw.open(encoding="utf-8"):
        rows.extend(json.loads(line).get("kept", []))
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"\nkept {len(rows):,} sentences across "
          f"{len({r['form'] for r in rows}):,} forms / {len({r['group_id'] for r in rows}):,} groups")
    print("rejections:", dict(rejected_total.most_common(8)))
    print(f"escalated to {escalate_model}: {escalated}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
