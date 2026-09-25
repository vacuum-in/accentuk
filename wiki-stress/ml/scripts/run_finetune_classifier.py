"""Train tier 3 as classification and compare it, like for like, with the pair model.

Same corpus, same merged inventory, same split by form, same epochs. The only
change is the formulation: instead of scoring `(sentence, sense definition)`
pairs, the model pools the target span and picks a stress signature from the
form's own candidates.

    python ml/scripts/run_finetune_classifier.py \
        --silver output/ml/silver_mined_v10.json \
        --manifest output/ml/serving_manifest_v22.json \
        --output output/ml/models/v22-classifier

Judge it with `uk-tts-frontend/scripts/benchmark.py --baseline`, not with
`dev_group_macro`, which has pointed the wrong way three times on this project.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ukstress_ml import classifier


def split_by_form(rows: list[dict], holdout: float, seed: int) -> tuple[list, list]:
    """Hold out whole forms, never sentences.

    Splitting sentences would put other occurrences of the same form on both
    sides and measure memorisation. Held-out forms measure what matters: whether
    the model generalises to a form it was not trained on.
    """
    forms = sorted({row["form"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(forms)
    held = set(forms[: max(1, int(len(forms) * holdout))])
    train = [r for r in rows if r["form"] not in held]
    dev = [r for r in rows if r["form"] in held]
    return train, dev


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path,
                        default=Path("output/ml/silver_mined_v10.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_v22.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/ml/models/v22-classifier"))
    parser.add_argument("--base", default="xlm-roberta-base",
                        help="any encoder AutoModel can load; jhu-clsp/mmBERT-base "
                             "is the RoPE-based alternative worth trying next")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--holdout", type=float, default=0.25)
    args = parser.parse_args(argv)

    corpus = json.loads(args.silver.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]
    rows = classifier.prepare_rows(corpus, manifest)
    print(f"corpus {len(corpus):,} rows -> {len(rows):,} usable "
          f"({len(corpus)-len(rows):,} dropped: form or gold sense not in the manifest)",
          flush=True)
    print("gold signature distribution:",
          Counter(r["gold_signature"] for r in rows).most_common(), flush=True)

    train_rows, dev_rows = split_by_form(rows, args.holdout, args.seed)
    train_forms = len({r["form"] for r in train_rows})
    dev_forms = len({r["form"] for r in dev_rows})
    print(f"train {len(train_rows):,} rows ({train_forms:,} forms)  "
          f"dev {len(dev_rows):,} rows ({dev_forms:,} forms)", flush=True)

    config = classifier.ClassifierConfig(
        base_model=args.base, max_epochs=args.epochs, batch_rows=args.batch_rows,
        learning_rate=args.lr, seed=args.seed)
    run = classifier.train(train_rows, dev_rows, args.output, config=config,
                           corpus_version=f"{args.silver.stem}+classifier")
    print(f"trained: best_epoch={run.best_epoch} "
          f"dev_group_macro={run.best_dev_group_macro:.4f} "
          f"params={run.parameters/1e6:.1f}M", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
