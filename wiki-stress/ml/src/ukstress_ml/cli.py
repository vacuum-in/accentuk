"""Command-line entry point for the homograph disambiguation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ukstress-ml")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inventory = subcommands.add_parser("inventory", help="freeze the homograph sense inventory")
    inventory.add_argument(
        "--source", type=Path, default=Path("output/homographs_precess_pilot_enriched2.csv")
    )
    inventory.add_argument("--output", type=Path, default=Path("output/ml"))

    ambiguity = subcommands.add_parser("ambiguity", help="compute the ambiguous form surface")
    ambiguity.add_argument("--inventory", type=Path, default=Path("output/ml/inventory_v1.jsonl"))
    ambiguity.add_argument("--output", type=Path, default=Path("output/ml"))

    triage = subcommands.add_parser(
        "triage-surface",
        help="classify the published ambiguous surface by why each form is ambiguous",
    )
    triage.add_argument("--database-url", required=True)
    triage.add_argument("--dataset-id", type=int)
    triage.add_argument(
        "--output", type=Path, default=Path("output/ml/ambiguous_surface_v2.jsonl")
    )

    mine = subcommands.add_parser("mine", help="scan a dump for ambiguous-form sentences")
    mine.add_argument("--dump", type=Path, required=True)
    mine.add_argument("--corpus", required=True)
    mine.add_argument("--licence", required=True)
    mine.add_argument("--forms", type=Path, default=Path("output/ml/ambiguous_forms.jsonl"))
    mine.add_argument("--output", type=Path, default=Path("output/ml/mined"))
    mine.add_argument("--cap-per-form", type=int, default=80)
    mine.add_argument("--seed", type=int, default=20260815)
    mine.add_argument("--max-pages", type=int)

    evaluate = subcommands.add_parser("evaluate", help="score a trained model")
    evaluate.add_argument("--model", type=Path, default=Path("output/ml/models/v1"))
    evaluate.add_argument("--corpus", type=Path, default=Path("output/ml/corpus/v1"))
    evaluate.add_argument(
        "--output", type=Path, default=Path("output/ml/models/v1/evaluation.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "inventory":
        from ukstress_ml import inventory

        senses, report = inventory.build(args.source)
        manifest = inventory.write(senses, report, args.output)
        json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    if args.command == "ambiguity":
        from ukstress_ml import ambiguity, inventory

        forms, surface_report = ambiguity.build(inventory.load(args.inventory))
        ambiguity.write(forms, surface_report, args.output)
        json.dump(surface_report, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    if args.command == "triage-surface":
        from ukstress_ml import triage

        rows = triage.read_ambiguous_surface(args.database_url, dataset_id=args.dataset_id)
        counts = triage.write_surface(args.output, triage.triage_forms(rows))
        manifest = triage.write_manifest(args.output, counts)
        json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        print()
        return 0

    if args.command == "mine":
        from ukstress_ml import ambiguity, mine

        forms = ambiguity.load(args.forms)
        manifest_path = Path(str(args.dump) + ".manifest.json")
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"sha256": ""}
        )
        reservoirs, stats = mine.mine(
            args.dump,
            forms,
            cap_per_form=args.cap_per_form,
            seed=args.seed,
            max_pages=args.max_pages,
        )
        coverage = mine.write(
            reservoirs,
            stats,
            forms,
            args.output / args.corpus,
            corpus=args.corpus,
            licence=args.licence,
            dump_sha256=str(manifest["sha256"]),
            config={"cap_per_form": args.cap_per_form, "seed": args.seed},
        )
        json.dump(coverage["stats"], sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    if args.command == "evaluate":
        from ukstress_ml import evaluate

        evaluate.evaluate(args.model, args.corpus, args.output)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
