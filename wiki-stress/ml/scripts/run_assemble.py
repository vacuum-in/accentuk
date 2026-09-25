"""Assemble the training corpus from mined and generated candidates.

Version is the first argument (default ``v2``). Generated rows carry their
intended sense by construction, but still pass every validator and the same
blind verification as mined rows.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from ukstress_ml import ambiguity, annotate, corpus

MINED = Path("output/ml/mined/ukwiki/candidates.jsonl")
GENERATED = Path("output/ml/generated/candidates.jsonl")
RAW = Path("output/ml/raw")
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"


def main() -> None:
    out = Path(f"output/ml/corpus/{VERSION}")
    forms = {f.form: f for f in ambiguity.load(Path("output/ml/ambiguous_forms.jsonl"))}
    inventory_manifest = json.loads(
        Path("output/ml/inventory_v1.manifest.json").read_text(encoding="utf-8")
    )
    coverage = json.loads(Path("output/ml/mined/ukwiki/coverage.json").read_text(encoding="utf-8"))

    with MINED.open(encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle]
    labels = annotate.parse_raw(RAW / "label_ukwiki.jsonl")
    verifications = annotate.parse_raw(RAW / "verify_ukwiki.jsonl")
    mined_count = len(candidates)

    generated_count = 0
    if GENERATED.exists():
        with GENERATED.open(encoding="utf-8") as handle:
            generated = [json.loads(line) for line in handle]
        generated_count = len(generated)
        candidates += generated
        # The intended sense is the label; it still has to survive validation
        # and blind verification.
        for row in generated:
            labels[row["sentence_id"]] = {
                "sense_id": row["gen_sense_id"],
                "confidence": None,
                "cue": row.get("cue", ""),
                "unclear": False,
                "deployment": row.get("deployment"),
                "form": row["form"],
            }
        verifications.update(annotate.parse_raw(RAW / "verify_generated.jsonl"))

    print(
        f"mined={mined_count:,} generated={generated_count:,} "
        f"labels={len(labels):,} verifications={len(verifications):,}",
        flush=True,
    )

    report = corpus.AssemblyReport()
    quarantine: list[dict] = []
    rows = corpus.build_rows(candidates, labels, verifications, forms, report, quarantine)
    print(f"validated rows: {len(rows):,}", flush=True)
    rows = corpus.deduplicate(rows, report, quarantine)
    print(f"after dedup: {len(rows):,}", flush=True)

    # Split first, then balance training only: the evaluation set has to keep
    # the natural sense distribution, because that is the distribution the
    # served model will meet.
    splits = corpus.split_natural_eval(rows, report)
    splits["train"] = corpus.balance(splits["train"], report)
    report.splits = {name: len(items) for name, items in splits.items()}
    print(
        "  ".join(f"{name}={len(items):,}" for name, items in splits.items()),
        flush=True,
    )
    origin = Counter(row["label_origin"] for row in rows)
    manifest = corpus.write(
        splits,
        quarantine,
        report,
        {
            "corpus_version": VERSION,
            "inventory_hash": inventory_manifest["content_hash"],
            "eval_design": (
                "no group held out; dev and test_natural are mined Wikipedia "
                "sentences from groups occurring with more than one sense in "
                "natural text; generated sentences are training-only; train "
                "balanced after splitting so evaluation keeps the natural skew"
            ),
            "label_origin": dict(origin),
            "sources": [
                {
                    "corpus": "ukwiki",
                    "licence": coverage["licence"],
                    "dump_sha256": coverage["dump_sha256"],
                    "scan_config": coverage["config"],
                },
                {"corpus": "generated", "licence": "generated", "rows": generated_count},
            ],
            "annotation": json.loads((RAW / "annotation_usage.json").read_text(encoding="utf-8"))
            if (RAW / "annotation_usage.json").exists()
            else {},
        },
        out,
    )
    print(json.dumps(manifest["report"], ensure_ascii=False, indent=2), flush=True)
    print("label origin:", dict(origin), flush=True)


if __name__ == "__main__":
    main()
