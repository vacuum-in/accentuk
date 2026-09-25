"""Release report: metrics, calibration, abstention threshold, limitations.

Produces the artifacts the specification gates a release on — per-group results,
a calibration table over the decision margin, the abstention threshold derived
from it, and a named list of groups the model resolves poorly. No figure here is
estimated; all come from scoring the held-out natural evaluation set.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml import corpus, crossencoder
from ukstress_ml.evaluate import baseline_results, calibration, majority_sense, summarize

sys.path.insert(0, str(Path(__file__).parent))
from run_crossencoder import load_glosses

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "models/v3-xenc")
CORPUS_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "output/ml/corpus/v3")
GROUP_FLOOR = 0.6
TARGET_PRECISION = 0.95


def abstention_threshold(
    results: list[dict[str, Any]], target_precision: float
) -> dict[str, Any]:
    """Lowest margin at which answering reaches ``target_precision``.

    Below it the API keeps the dictionary's default sense instead of the model's
    choice, so the served answer is either right or explicitly deferred.
    """
    finite = sorted(
        (r for r in results if r["margin"] != float("inf")),
        key=lambda r: r["margin"],
        reverse=True,
    )
    best: dict[str, Any] = {
        "threshold": None,
        "precision": 0.0,
        "coverage": 0.0,
        "answered": 0,
    }
    correct = 0
    for index, result in enumerate(finite, start=1):
        correct += bool(result["correct"])
        precision = correct / index
        if precision >= target_precision:
            best = {
                "threshold": round(result["margin"], 4),
                "precision": round(precision, 4),
                "coverage": round(index / len(finite), 4),
                "answered": index,
            }
    return best


def per_group(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        grouped[row["group_id"]].append((row, result))
    out = []
    for group_id, pairs in grouped.items():
        correct = sum(1 for _, result in pairs if result["correct"])
        senses = {row["sense_id"] for row, _ in pairs}
        out.append(
            {
                "group_id": group_id,
                "form": pairs[0][0]["form"],
                "rows": len(pairs),
                "senses_present": len(senses),
                "accuracy": round(correct / len(pairs), 4),
            }
        )
    return sorted(out, key=lambda item: (item["accuracy"], -item["rows"]))


def main() -> None:
    glosses = load_glosses()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "checkpoint")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "checkpoint")
    device = crossencoder.select_device()
    model.to(device)
    collate = crossencoder.make_collate(tokenizer, 192)

    train_rows = corpus.load_split(CORPUS_DIR / "train.jsonl")
    default = majority_sense(train_rows)
    rows = corpus.load_split(CORPUS_DIR / "test_natural.jsonl")

    loader = DataLoader(
        crossencoder.PairDataset(rows, glosses),
        batch_size=16,
        shuffle=False,
        collate_fn=collate,
    )
    with torch.no_grad():
        results = crossencoder.predict(model, loader, device)

    model_summary = summarize("test_natural", rows, results, default)
    baseline_summary = summarize("test_natural", rows, baseline_results(rows, default), default)
    groups = per_group(rows, results)
    weak = [g for g in groups if g["accuracy"] < GROUP_FLOOR]
    threshold = abstention_threshold(results, TARGET_PRECISION)

    training_run = json.loads((MODEL_DIR / "training_run.json").read_text(encoding="utf-8"))
    corpus_manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))

    gates = {
        "beats_majority_baseline": (
            model_summary["group_macro_accuracy"] > baseline_summary["group_macro_accuracy"]
        ),
        "multi_sense_macro_at_least_0.80": model_summary["multi_sense_group_macro"] >= 0.80,
        "minority_recall_above_zero": model_summary.get("minority_sense_recall", 0) > 0,
        "abstention_threshold_found": threshold["threshold"] is not None,
    }

    report = {
        "model_dir": str(MODEL_DIR),
        "base_model": training_run.get("base_model"),
        "parameters": training_run.get("parameters"),
        "corpus_version": corpus_manifest.get("corpus_version"),
        "inventory_hash": corpus_manifest.get("inventory_hash"),
        "best_epoch": training_run.get("best_epoch"),
        "training_minutes": round(training_run.get("seconds", 0) / 60, 1),
        "test_natural": {"model": model_summary, "majority_baseline": baseline_summary},
        "calibration": calibration(results),
        "abstention": {"target_precision": TARGET_PRECISION, **threshold},
        "groups_evaluated": len(groups),
        "groups_below_floor": {
            "floor": GROUP_FLOOR,
            "count": len(weak),
            "groups": weak[:60],
        },
        "release_gates": gates,
        "release_eligible": all(gates.values()),
    }
    (MODEL_DIR / "release_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"test_natural macro {model_summary['group_macro_accuracy']:.4f} "
        f"(baseline {baseline_summary['group_macro_accuracy']:.4f})\n"
        f"multi-sense  {model_summary['multi_sense_group_macro']:.4f} "
        f"(baseline {baseline_summary['multi_sense_group_macro']:.4f})\n"
        f"minority recall {model_summary.get('minority_sense_recall', 0):.4f} "
        f"on {model_summary.get('minority_rows', 0)} rows\n"
        f"abstention: margin >= {threshold['threshold']} gives "
        f"{threshold['precision']} precision at {threshold['coverage']} coverage\n"
        f"groups below {GROUP_FLOOR}: {len(weak)} of {len(groups)}\n"
        f"release gates: {gates}\nrelease eligible: {all(gates.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
