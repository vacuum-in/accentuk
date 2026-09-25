"""Train and evaluate the pretrained cross-encoder on a corpus version."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml import ambiguity, corpus, crossencoder
from ukstress_ml.evaluate import baseline_results, majority_sense, summarize

# Usage: run_crossencoder.py [corpus_version] [base_model_or_checkpoint] [run_tag]
#
# `base` may be a Hub id or a local checkpoint directory. Passing a checkpoint
# continues training from those weights — verified to preserve the trained
# classifier head, not re-initialise it. Always pass a `run_tag` when resuming,
# so the continued run writes to its own directory instead of overwriting the
# checkpoint it started from.
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v3"
BASE = sys.argv[2] if len(sys.argv) > 2 else "xlm-roberta-base"
RUN_TAG = sys.argv[3] if len(sys.argv) > 3 else "xenc"


def load_glosses() -> dict[str, dict[str, str]]:
    glosses: dict[str, dict[str, str]] = {}
    for form in ambiguity.load(Path("output/ml/ambiguous_forms.jsonl")):
        for candidate in form.candidates:
            glosses[candidate.sense_id] = {
                "definition": candidate.definition,
                "stressed": candidate.stressed,
            }
    return glosses


def main() -> None:
    corpus_dir = Path(f"output/ml/corpus/{VERSION}")
    out = Path(f"output/ml/models/{VERSION}-{RUN_TAG}")
    if Path(BASE).is_dir() and Path(BASE).resolve().is_relative_to(out.resolve()):
        raise SystemExit(
            f"refusing to resume from {BASE} into {out}: the run would overwrite "
            "its own starting checkpoint. Pass a distinct run_tag."
        )
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    glosses = load_glosses()

    train_rows = corpus.load_split(corpus_dir / "train.jsonl")
    dev_rows = corpus.load_split(corpus_dir / "dev.jsonl")

    config = crossencoder.CrossEncoderConfig(base_model=BASE)
    run = crossencoder.train(
        train_rows,
        dev_rows,
        glosses,
        out,
        config=config,
        corpus_version=manifest["corpus_version"],
        inventory_hash=manifest["inventory_hash"],
    )
    print(
        f"best epoch {run.best_epoch}  dev group-macro {run.best_dev_group_macro:.4f}  "
        f"{run.seconds / 60:.1f} min",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(out / "checkpoint")
    model = AutoModelForSequenceClassification.from_pretrained(out / "checkpoint")
    device = crossencoder.select_device()
    model.to(device)
    collate = crossencoder.make_collate(tokenizer, config.max_length)
    default = majority_sense(train_rows)

    report: dict[str, object] = {
        "base_model": BASE,
        "corpus_version": VERSION,
        "parameters": run.parameters,
        "device": str(device),
        "splits": {},
    }
    for split in ("dev", "test_natural"):
        path = corpus_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = corpus.load_split(path)
        loader = DataLoader(
            crossencoder.PairDataset(rows, glosses),
            batch_size=config.batch_rows,
            shuffle=False,
            collate_fn=collate,
        )
        with torch.no_grad():
            results = crossencoder.predict(model, loader, device)
        base = baseline_results(rows, default)
        model_summary = summarize(split, rows, results, default)
        baseline_summary = summarize(split, rows, base, default)
        report["splits"][split] = {  # type: ignore[index]
            "model": model_summary,
            "majority_baseline": baseline_summary,
        }
        print(
            f"{split:<13} macro {model_summary['group_macro_accuracy']:.4f} "
            f"(baseline {baseline_summary['group_macro_accuracy']:.4f})  "
            f"multi-sense {model_summary['multi_sense_group_macro']:.4f} "
            f"(baseline {baseline_summary['multi_sense_group_macro']:.4f})  "
            f"minority-recall {model_summary.get('minority_sense_recall', 0):.4f} "
            f"on {model_summary.get('minority_rows', 0)} rows",
            flush=True,
        )

    (out / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
