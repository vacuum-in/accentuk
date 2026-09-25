"""Train the cross-encoder twice on identical data, differing only in the right-hand side.

The question is whether scoring a sentence against *labelled usages* of a sense
beats scoring it against the sense's dictionary *definition*.

It is asked as a matched pair, not against the shipped `v19-v10`, because the
corpus and inventory that produced `v19-v10` were never recorded — the training
script takes both as arguments and writes only `corpus_version` into its
manifest. Comparing a new run to it would confound the change under test with
whatever data pairing was used a week ago.

Motivation, measured: on the benchmark, forms with 20+ training rows score 73.8%
and forms the model never saw score 75.6%. Supervision is not reaching the
decision. The suspect is what the decision is made against — a definition
describes a sense, an exemplar shows it, and only the second shares vocabulary
with the sentence being judged.

    python ml/scripts/run_exemplar_ablation.py \
        --silver output/ml/silver_mined_v10.json \
        --inventory output/ml/ambiguous_forms_merged.jsonl

Each arm is roughly 3.5 h on an 8 GB card. Judge the result with
`uk-tts-frontend/scripts/benchmark.py --baseline`, not with `dev_group_macro`,
which pointed the wrong way three times on this project.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_finetune_inflected.py")


def arm(name: str, output: Path, exemplars: int, args: argparse.Namespace) -> dict:
    command = [
        sys.executable, str(SCRIPT),
        "--silver", str(args.silver),
        "--inventory", str(args.inventory),
        "--output", str(output),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--batch-rows", str(args.batch_rows),
    ]
    if exemplars:
        command += ["--exemplars", str(exemplars)]
    print(f"\n=== {name} ===\n{' '.join(command)}", flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=Path.cwd(), check=False)
    return {"arm": name, "output": str(output), "exemplars": exemplars,
            "returncode": result.returncode, "seconds": round(time.time() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, default=Path("output/ml/silver_mined_v10.json"))
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/ambiguous_forms_merged.jsonl"),
                        help="gloss inventory. Note ambiguous_surface_v*.jsonl is a "
                             "triage file, not an inventory, and has no candidates.")
    parser.add_argument("--out-root", type=Path, default=Path("output/ml/models"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--batch-rows", type=int, default=16)
    parser.add_argument("--exemplars", type=int, default=3,
                        help="usages per sense in the treatment arm")
    parser.add_argument("--report", type=Path,
                        default=Path("output/ml/exemplar_ablation.json"))
    args = parser.parse_args(argv)

    runs = [
        # Control first: if the machine dies overnight, the arm that reproduces
        # current behaviour is the one worth having finished.
        arm("control-definitions", args.out_root / "v21-definitions", 0, args),
        arm("treatment-exemplars", args.out_root / "v21-exemplars", args.exemplars, args),
    ]
    args.report.write_text(json.dumps({
        "silver": str(args.silver), "inventory": str(args.inventory),
        "seed": args.seed, "epochs": args.epochs, "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(runs, indent=2), flush=True)
    print(f"\nreport -> {args.report}")
    print("Next: serve each checkpoint and compare with\n"
          "  benchmark.py --save control.json    (control)\n"
          "  benchmark.py --baseline control.json (treatment)")
    return 0 if all(r["returncode"] == 0 for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
