"""L1 sense labelling and L3 blind verification over mined candidates.

The two jobs run on different Azure resources, so they have independent quotas
and can run concurrently — and, as the specification requires, no model ever
confirms its own output. Verification is blind (target masked, no intended
label supplied), so it does not need the labelling pass to finish first.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ukstress_ml import ambiguity, annotate

_ap = argparse.ArgumentParser()
_ap.add_argument("--mined", type=Path, default=Path("output/ml/mined/ukwiki/candidates.jsonl"))
_ap.add_argument("--forms", type=Path, default=Path("output/ml/ambiguous_forms.jsonl"))
_ap.add_argument("--raw", type=Path, default=Path("output/ml/raw"))
_ap.add_argument("--tag", default="ukwiki")
_ap.add_argument("--cap-per-form", type=int, default=32)
_ap.add_argument("--floor-per-form", type=int, default=6)
_ap.add_argument("--max-forms", type=int, help="sample this many forms (evaluation pilot)")
_ARGS = _ap.parse_args()

MINED = _ARGS.mined
FORMS = _ARGS.forms
RAW = _ARGS.raw
TAG = _ARGS.tag
CAP_PER_FORM = _ARGS.cap_per_form
FLOOR_PER_FORM = _ARGS.floor_per_form
SEED = 20260815
BATCH_SIZE = 40
WORKERS = 10

# Verification uses the module-level deployment, which is configured with its
# own AZURE_OPENAI_*_VERIFY credentials. A local override here previously
# pointed verification at the *labelling* endpoint variables, which both breaks
# credential resolution and undermines the point of a blind second opinion.
VERIFY_DEPLOYMENT = annotate.VERIFY_DEPLOYMENT


def select_rows() -> tuple[list[dict], dict[str, object]]:
    forms = {
        f.form: f for f in ambiguity.load(FORMS) if f.complete
    }
    by_form: dict[str, list[dict]] = defaultdict(list)
    with MINED.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["form"] in forms:
                by_form[row["form"]].append(row)

    rng = random.Random(SEED)
    eligible = sorted(k for k, v in by_form.items() if len(v) >= FLOOR_PER_FORM)
    if _ARGS.max_forms and len(eligible) > _ARGS.max_forms:
        eligible = sorted(random.Random(SEED).sample(eligible, _ARGS.max_forms))
    keep = set(eligible)
    rows: list[dict] = []
    for form_key in sorted(by_form):
        if form_key not in keep:
            continue
        items = sorted(by_form[form_key], key=lambda r: r["sentence_id"])
        if len(items) < FLOOR_PER_FORM:
            continue
        rng.shuffle(items)
        rows.extend(items[:CAP_PER_FORM])
    return rows, forms


def main() -> None:
    rows, forms = select_rows()
    print(
        f"labelling+verifying {len(rows):,} sentences across "
        f"{len({r['form'] for r in rows})} forms "
        f"(cap={CAP_PER_FORM}, floor={FLOOR_PER_FORM})",
        flush=True,
    )
    config = annotate.load_config()
    masked = [{**row, "masked": annotate.mask_sentence(row)} for row in rows]

    def label_job() -> annotate.Usage:
        return annotate.run_job(
            job="label",
            rows=rows,
            forms=forms,
            deployment=annotate.LABEL_DEPLOYMENT,
            config=config,
            raw_path=RAW / f"label_{TAG}.jsonl",
            batch_size=BATCH_SIZE,
            workers=WORKERS,
            max_tokens=12000,
            system=annotate.LABEL_SYSTEM,
            prompt_builder=annotate.label_prompt,
            progress_every=25,
        )

    def verify_job() -> annotate.Usage:
        return annotate.run_job(
            job="verify",
            rows=masked,
            forms=forms,
            deployment=VERIFY_DEPLOYMENT,
            config=config,
            raw_path=RAW / f"verify_{TAG}.jsonl",
            batch_size=BATCH_SIZE,
            workers=WORKERS,
            max_tokens=12000,
            system=annotate.VERIFY_SYSTEM,
            prompt_builder=annotate.verify_prompt,
            progress_every=25,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        label_future = pool.submit(label_job)
        verify_future = pool.submit(verify_job)
        label_usage = label_future.result()
        verify_usage = verify_future.result()

    labels = annotate.parse_raw(RAW / f"label_{TAG}.jsonl")
    verifications = annotate.parse_raw(RAW / f"verify_{TAG}.jsonl")
    print(f"parsed labels={len(labels):,} verifications={len(verifications):,}", flush=True)

    (RAW / f"annotation_usage_{TAG}.json").write_text(
        json.dumps(
            {
                "label": {
                    "deployment": annotate.LABEL_DEPLOYMENT.name,
                    "endpoint_var": annotate.LABEL_DEPLOYMENT.endpoint_var,
                    **label_usage.snapshot(),
                },
                "verify": {
                    "deployment": VERIFY_DEPLOYMENT.name,
                    "endpoint_var": VERIFY_DEPLOYMENT.endpoint_var,
                    **verify_usage.snapshot(),
                },
                "cap_per_form": CAP_PER_FORM,
                "floor_per_form": FLOOR_PER_FORM,
                "batch_size": BATCH_SIZE,
                "workers": WORKERS,
                "seed": SEED,
                "rows_requested": len(rows),
                "labels_parsed": len(labels),
                "verifications_parsed": len(verifications),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
