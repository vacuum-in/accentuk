"""L2 generation for starved senses, then blind verification of the result."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ukstress_ml import ambiguity, annotate, generate

CORPUS = Path("output/ml/corpus/v1")
RAW = Path("output/ml/raw")
OUT = Path("output/ml/generated")
FLOOR = 12
PER_CALL = 12

# Verification must use the module-level deployment and its own
# AZURE_OPENAI_*_VERIFY credentials. Declaring a local Deployment here that
# borrows the *labelling* endpoint variables both breaks credential resolution
# and quietly turns blind verification into self-verification.
VERIFY_DEPLOYMENT = annotate.VERIFY_DEPLOYMENT


def main() -> None:
    forms_all = {f.form: f for f in ambiguity.load(Path("output/ml/ambiguous_forms.jsonl"))}

    existing = []
    for split in ("train", "dev", "test", "test_unseen"):
        path = CORPUS / f"{split}.jsonl"
        if path.exists():
            existing += [json.loads(line) for line in path.open(encoding="utf-8")]

    have: dict[int, dict[str, int]] = defaultdict(Counter)
    anchors: dict[tuple[int, str], list[str]] = defaultdict(list)
    covered_forms: set[str] = set()
    for row in existing:
        have[row["group_id"]][row["sense_id"]] += 1
        covered_forms.add(row["form"])
        if len(anchors[(row["group_id"], row["sense_id"])]) < 3:
            anchors[(row["group_id"], row["sense_id"])].append(row["sentence"])

    forms = {key: forms_all[key] for key in sorted(covered_forms) if key in forms_all}
    requests = generate.plan_requests(
        forms,
        {gid: dict(counts) for gid, counts in have.items()},
        anchors,
        floor=FLOOR,
        per_call=PER_CALL,
        topics=annotate.TOPICS,
    )
    print(
        f"generation requests: {len(requests)} across "
        f"{len({r['group_id'] for r in requests})} groups, "
        f"target {sum(r['count'] for r in requests):,} sentences",
        flush=True,
    )

    usage = annotate.run_generation(
        requests=requests,
        forms=forms,
        deployment=annotate.LABEL_DEPLOYMENT,
        config=annotate.load_config(),
        raw_path=RAW / "generate_ukwiki.jsonl",
        workers=10,
        progress_every=25,
    )
    print("generation usage:", usage.snapshot(), flush=True)

    rows, stats = generate.parse_generated(RAW / "generate_ukwiki.jsonl", forms)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "generation_report.json").write_text(
        json.dumps(
            {"usage": usage.snapshot(), "stats": stats, "floor": FLOOR, "rows": len(rows)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    # Generated sentences face the same blind verification as mined ones.
    masked = [{**row, "masked": annotate.mask_sentence(row)} for row in rows]
    verify_usage = annotate.run_job(
        job="verify",
        rows=masked,
        forms=forms,
        deployment=VERIFY_DEPLOYMENT,
        config=annotate.load_config(),
        raw_path=RAW / "verify_generated.jsonl",
        batch_size=40,
        workers=10,
        max_tokens=12000,
        system=annotate.VERIFY_SYSTEM,
        prompt_builder=annotate.verify_prompt,
        progress_every=25,
    )
    print("verify usage:", verify_usage.snapshot(), flush=True)


if __name__ == "__main__":
    main()
