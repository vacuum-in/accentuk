"""Generate the minority-sense sentences that cap the model's accuracy.

The gold evaluation put the model tier at 0.8858, and its errors are not spread
evenly — they sit almost entirely on groups whose training data covers one
sense far better than the other:

    поділ      12 rows, one sense only          0.500   (the model guesses)
    варення    28 rows, one sense only          0.808
    правило    96 rows, minority sense at 1%    0.708
    копати     46 rows, minority sense at 6%    0.774
    ...
    мука       31 rows, minority sense at 26%   1.000
    орган     111 rows, minority sense at 12%   1.000
    колос      57 rows, minority sense at 26%   1.000

Total volume predicts nothing: `обід` has zero training rows and scores 1.000
because the cross-encoder can read its gloss, while `правило` has 96 rows and
scores 0.708 because 95 of them are the same sense. What predicts accuracy is
whether the *minority* sense is represented at all.

Across the serving manifest, only 332 of 1,265 multi-sense groups have a
minority sense above 20%. 392 have no training data whatsoever and 353 have at
least one sense with zero rows. That is the ceiling: no threshold, fallback or
coverage change moves a group the model has never seen both halves of.

The floor has to be **proportional**, not absolute. Mining `правило` produced
260 sentences and *zero* of its minority sense, because that sense
("пристрій, на якому розпрямляють") is rare in every corpus, not just in
Wikipedia. A flat floor of 12 rows would leave it at 4.6% of the group and the
model still guessing; the measured threshold for the model to score ~1.00 is a
minority share around 20%. So the target for the weakest sense is
`max(floor, min_share x group_total)`.

This is the division of labour between the two data sources, measured on the
gold set: mining supplies senses that are rare in an encyclopedia but ordinary
in speech — it took `варення` from 0.808 to 1.000 and fixed `город`, `замок`,
`колос`, `терен`, `мука` and `орган` outright. Generation is for senses that
are rare *everywhere*, which no amount of mining will surface.

Only groups in the serving manifest are generated for. A group the API cannot
route to the model gains nothing from training data, and the manifest is also
where the glosses live that the request needs.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity, generate
from ukstress_ml.annotate import (
    FLASH_DEPLOYMENT,
    LABEL_DEPLOYMENT,
    VERIFY_DEPLOYMENT,
    Usage,
    load_config,
)

#: `label` and `verify` are two deployments on ONE Azure resource and share its
#: quota; only `flash` sits on a second resource. Measured, that quota admits
#: roughly one request in flight — 19 of 20 concurrent calls returned 429 — so
#: workers above 2 spend their time in backoff rather than generating. Real
#: parallelism comes from running one process per *resource*, not per
#: deployment, which is what `--shard` is for.
DEPLOYMENTS = {
    "label": LABEL_DEPLOYMENT,
    "verify": VERIFY_DEPLOYMENT,
    "flash": FLASH_DEPLOYMENT,
}


def load_counts(paths: list[Path]) -> dict[int, Counter[str]]:
    """Rows already held per (group, sense) across every training corpus."""
    have: dict[int, Counter[str]] = defaultdict(Counter)
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        for row in rows:
            sense = row.get("gold_sense") or row.get("sense_id")
            if sense is not None:
                have[int(row["group_id"])][str(sense)] += 1
    return have


def completed(raw_path: Path) -> set[str]:
    done: set[str] = set()
    if not raw_path.exists():
        return done
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("content"):
            done.add(f"{record['form']}|{record['sense_id']}|{record.get('stage', 'first')}")
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, action="append", default=None,
                        help="repeatable; defaults to the glossed inventory")
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_expanded.json"))
    parser.add_argument("--corpus", type=Path, action="append",
                        default=[Path("output/ml/silver_plus_generated_v2.json")])
    parser.add_argument("--raw", type=Path,
                        default=Path("output/ml/raw/generated_balanced.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/ml/generated_balanced.json"))
    parser.add_argument("--floor", type=int, default=12,
                        help="minimum rows every sense should reach")
    parser.add_argument("--min-share", type=float, default=0.25,
                        help="target share of the group for its weakest sense; "
                             "measured, the model scores ~1.00 above 0.20 and "
                             "degrades in proportion below it")
    parser.add_argument("--per-call", type=int, default=12)
    parser.add_argument("--forms-per-group", type=int, default=2,
                        help="inflected forms to write for; deficits are per group")
    parser.add_argument("--escalate-below", type=int, default=4,
                        help="retry on gpt-5.4 when a sense yields fewer than this")
    parser.add_argument("--limit", type=int,
                        help="stop after this many (form, sense) jobs; 0 plans and exits")
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--workers", type=int, default=2,
                        help="the shared resource admits ~1 request in flight; "
                             "above 2 the extra threads only queue and retry")
    parser.add_argument("--deployment", choices=sorted(DEPLOYMENTS), default="label")
    parser.add_argument("--shard", default="0/1",
                        help="i/n: take every n-th job, so one process per resource "
                             "can run without overlapping")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="re-harvest the raw responses already collected and exit")
    args = parser.parse_args()

    if args.rebuild_only:
        rebuild(args)
        return

    inventories = args.inventory or [Path("output/ml/ambiguous_forms_glossed.jsonl")]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]
    servable = {entry["group_id"] for entry in manifest.values()}

    forms: dict[str, ambiguity.AmbiguousForm] = {}
    for path in inventories:
        for form in ambiguity.load(path):
            if form.group_id in servable and len({c.signature for c in form.candidates}) > 1:
                forms.setdefault(form.form, form)

    # Deficits are a property of the group, so writing for every inflected form
    # of it would multiply the same shortfall by the size of the paradigm.
    chosen: dict[str, ambiguity.AmbiguousForm] = {}
    per_group: Counter[int] = Counter()
    for key in sorted(forms):
        form = forms[key]
        if per_group[form.group_id] >= args.forms_per_group:
            continue
        per_group[form.group_id] += 1
        chosen[key] = form

    have = load_counts(args.corpus)
    # A proportional target cannot be expressed as one global floor, so the
    # deficit is computed per group and handed to `plan_requests` as a
    # per-group floor via a synthetic "have" map.
    targets: dict[int, dict[str, int]] = {}
    for form in chosen.values():
        counts = have.get(form.group_id, Counter())
        total = sum(counts.values())
        floor = max(args.floor, int(args.min_share * total))
        targets[form.group_id] = {c.sense_id: counts.get(c.sense_id, 0)
                                  for c in form.candidates}
        targets[form.group_id]["__floor__"] = floor

    requests = []
    for form_key in sorted(chosen):
        form = chosen[form_key]
        counts = targets[form.group_id]
        floor = counts["__floor__"]
        for index, candidate in enumerate(form.candidates):
            shortfall = floor - counts.get(candidate.sense_id, 0)
            if shortfall <= 0:
                continue
            requests.append({
                "form": form_key,
                "group_id": form.group_id,
                "sense_id": candidate.sense_id,
                "signature": candidate.signature,
                "count": min(shortfall, args.per_call),
                "topic": generate.TOPICS[(len(requests) + index) % len(generate.TOPICS)],
            })
    print(f"servable multi-sense groups : {len(servable & set(per_group)):,}")
    print(f"forms written for           : {len(chosen):,}")
    print(f"deficient (form, sense) jobs: {len(requests):,}")
    print(f"sentences requested         : {sum(r['count'] for r in requests):,}", flush=True)
    index, _, count = args.shard.partition("/")
    shard, shards = int(index), int(count or 1)
    if shards > 1:
        requests = [r for i, r in enumerate(requests) if i % shards == shard]
        print(f"shard {shard}/{shards}: {len(requests):,} jobs", flush=True)
    if args.limit is not None:
        # `--limit 0` must mean "plan only". Treating it as falsy ran the whole
        # list instead, which is the opposite of what a dry run is for.
        requests = requests[:args.limit]
    if not requests:
        print("nothing to generate")
        return

    config = load_config(Path(".env"))
    chosen_deployment = DEPLOYMENTS[args.deployment]
    primary = chosen_deployment.client(config), chosen_deployment.resolved_model(config)
    escalate = FLASH_DEPLOYMENT.client(config), FLASH_DEPLOYMENT.resolved_model(config)
    print(f"deployment: {primary[1]}  workers {args.workers}", flush=True)
    usage = Usage()
    done = completed(args.raw)
    rejected: Counter[str] = Counter()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    finished = 0
    kept_total = 0

    def run_job(request: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
        """Generate for one (form, sense), escalating when the yield is thin."""
        form = chosen[request["form"]]
        sense = next(c for c in form.candidates if c.sense_id == request["sense_id"])
        kept: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        records: list[dict[str, Any]] = []
        for stage, (client, model) in (("first", primary), ("escalated", escalate)):
            if stage == "escalated" and len(kept) >= args.escalate_below:
                break
            if f"{form.form}|{sense.sense_id}|{stage}" in done:
                continue
            prompt = generate.build_prompt(form, sense, request["count"], request["topic"])
            content = generate.call_model(client, model, prompt, usage, args.max_tokens)
            records.append({"form": form.form, "sense_id": sense.sense_id,
                            "group_id": form.group_id, "stage": stage,
                            "content": content or None})
            if not content:
                continue
            harvested, batch = generate.harvest(content, form, sense.sense_id)
            reasons.update(batch)
            seen = {row["sentence"] for row in kept}
            kept += [row for row in harvested if row["sentence"] not in seen]
        with lock:
            for record in records:
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_handle.flush()
        return kept, reasons

    with args.raw.open("a", encoding="utf-8") as raw_handle, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, request) for request in requests]
        for future in as_completed(futures):
            try:
                kept, reasons = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"  job failed: {error}", flush=True)
                continue
            with lock:
                rejected.update(reasons)
                finished += 1
                kept_total += len(kept)
                if finished % 50 == 0 or finished == len(requests):
                    print(f"  {finished:,}/{len(requests):,}  kept {kept_total:,} sentences",
                          flush=True)

    rebuild(args)
    print(json.dumps(usage.snapshot(), indent=2))
    for reason, count in rejected.most_common(10):
        print(f"  rejected {count:>6,}  {reason}")


def rebuild(args: argparse.Namespace) -> None:
    """Re-harvest every raw response into the output corpus.

    Kept separate from the call loop so a change to `validate_sentence` costs a
    re-parse rather than a re-spend, which is the same contract the gloss and
    labelling stages use.
    """
    inventories = args.inventory or [Path("output/ml/ambiguous_forms_glossed.jsonl")]
    forms: dict[str, ambiguity.AmbiguousForm] = {}
    for path in inventories:
        for form in ambiguity.load(path):
            forms.setdefault(form.form, form)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if args.raw.exists():
        for line in args.raw.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            form = forms.get(str(record.get("form", "")))
            if form is None or not record.get("content"):
                continue
            harvested, _ = generate.harvest(str(record["content"]), form,
                                            str(record["sense_id"]))
            for row in harvested:
                if row["sentence"] in seen:
                    continue
                seen.add(row["sentence"])
                rows.append({**row, "corpus": "generated_balanced",
                             "source_tier": "generated"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    by_group = len({row["group_id"] for row in rows})
    print(f"\nkept {len(rows):,} sentences across {by_group:,} groups -> {args.output}")


if __name__ == "__main__":
    main()
