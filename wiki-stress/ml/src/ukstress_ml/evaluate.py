"""Evaluation under constrained candidate scoring.

Never a single aggregate figure: group-macro accuracy is the headline, and the
default-sense baseline is reported beside it so the number has a scale.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import MarianMTModel

from ukstress_ml.tokenizer import SourceTokenizer, TargetVocab
from ukstress_ml.train import TrainConfig, group_macro_accuracy, score_candidates, select_device


def majority_sense(train_rows: list[dict[str, Any]]) -> dict[int, str]:
    """The most frequent sense per group in training — the baseline to beat."""
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        counts[row["group_id"]][row["sense_id"]] += 1
    return {group: counter.most_common(1)[0][0] for group, counter in counts.items()}


def baseline_results(rows: list[dict[str, Any]], default: dict[int, str]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        predicted = default.get(row["group_id"])
        if predicted is None:
            candidates = sorted(row["candidates"], key=lambda c: c["sense_id"])
            predicted = candidates[0]["sense_id"]
        results.append(
            {
                "predicted": predicted,
                "gold": row["sense_id"],
                "group_id": row["group_id"],
                "correct": predicted == row["sense_id"],
                "margin": 0.0,
            }
        )
    return results


def summarize(
    name: str,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    default: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Headline plus the two slices that actually show context reading.

    Groups represented by a single sense in a split are answered correctly by
    memorising a prior, so they inflate every aggregate. Minority-sense recall
    is the complement: the majority baseline scores exactly 0 on those rows, so
    anything above 0 is context being read.
    """
    correct = sum(1 for r in results if r["correct"])

    senses_in_group: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        senses_in_group[row["group_id"]].add(row["sense_id"])
    multi = [
        (row, result)
        for row, result in zip(rows, results, strict=True)
        if len(senses_in_group[row["group_id"]]) > 1
    ]

    summary = {
        "split": name,
        "rows": len(results),
        "groups": len({r["group_id"] for r in results}),
        "group_macro_accuracy": round(group_macro_accuracy(results), 4),
        "micro_accuracy": round(correct / max(1, len(results)), 4),
        "multi_sense_rows": len(multi),
        "multi_sense_group_macro": round(
            group_macro_accuracy([result for _, result in multi]), 4
        ),
        "multi_sense_micro": round(
            sum(1 for _, result in multi if result["correct"]) / max(1, len(multi)), 4
        ),
    }

    if default is not None:
        minority = [
            (row, result)
            for row, result in zip(rows, results, strict=True)
            if default.get(row["group_id"]) not in (None, row["sense_id"])
        ]
        summary["minority_rows"] = len(minority)
        summary["minority_sense_recall"] = round(
            sum(1 for _, result in minority if result["correct"]) / max(1, len(minority)), 4
        )
    return summary


def calibration(results: list[dict[str, Any]], bins: int = 10) -> list[dict[str, Any]]:
    finite = [r for r in results if r["margin"] != float("inf")]
    if not finite:
        return []
    margins = sorted(r["margin"] for r in finite)
    edges = [margins[int(len(margins) * i / bins)] for i in range(bins)] + [margins[-1] + 1e-9]
    table = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        bucket = [r for r in finite if low <= r["margin"] < high]
        if not bucket:
            continue
        table.append(
            {
                "margin_low": round(low, 4),
                "margin_high": round(high, 4),
                "rows": len(bucket),
                "accuracy": round(sum(1 for r in bucket if r["correct"]) / len(bucket), 4),
            }
        )
    return table


def worst_groups(
    rows: list[dict[str, Any]], results: list[dict[str, Any]], limit: int = 25
) -> list[dict[str, Any]]:
    by_group: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        by_group[row["group_id"]].append((row, result))
    scored = []
    for group_id, pairs in by_group.items():
        correct = sum(1 for _, result in pairs if result["correct"])
        scored.append(
            {
                "group_id": group_id,
                "form": pairs[0][0]["form"],
                "rows": len(pairs),
                "accuracy": round(correct / len(pairs), 4),
            }
        )
    return sorted(scored, key=lambda item: (item["accuracy"], -item["rows"]))[:limit]


def evaluate(
    model_dir: Path,
    corpus_dir: Path,
    output_path: Path,
    *,
    splits: tuple[str, ...] = ("dev", "test_natural"),
) -> dict[str, Any]:
    from ukstress_ml import corpus as corpus_module

    config = TrainConfig()
    source = SourceTokenizer.load(model_dir)
    target = TargetVocab.load(model_dir / "target_vocab.json")
    device = select_device()
    model = MarianMTModel.from_pretrained(model_dir / "checkpoint").to(device)

    train_rows = corpus_module.load_split(corpus_dir / "train.jsonl")
    default = majority_sense(train_rows)

    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "corpus_dir": str(corpus_dir),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "splits": {},
    }
    for split in splits:
        path = corpus_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = corpus_module.load_split(path)
        if not rows:
            continue
        with torch.no_grad():
            results = score_candidates(
                model, rows, source, target, config, device, batch_size=config.eval_batch_size
            )
        base = baseline_results(rows, default)
        report["splits"][split] = {
            "model": summarize(split, rows, results, default),
            "majority_baseline": summarize(split, rows, base, default),
            "calibration": calibration(results),
            "worst_groups": worst_groups(rows, results),
        }
        model_summary = report["splits"][split]["model"]
        baseline_summary = report["splits"][split]["majority_baseline"]
        print(
            f"{split:<12} macro {model_summary['group_macro_accuracy']:.4f} "
            f"(baseline {baseline_summary['group_macro_accuracy']:.4f})  "
            f"multi-sense {model_summary['multi_sense_group_macro']:.4f} "
            f"(baseline {baseline_summary['multi_sense_group_macro']:.4f})  "
            f"minority-recall {model_summary.get('minority_sense_recall', 0):.4f} "
            f"on {model_summary.get('minority_rows', 0)} rows",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
