"""Serving latency benchmark for the disambiguation model.

Measures CPU inference only, because that is the declared serving target, and
reports fp32 against dynamically int8-quantized weights. Every figure comes from
this run on this machine; nothing is pre-populated.

Run it on an otherwise idle machine — a number measured while training saturates
the accelerator is not a serving figure.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml import corpus, crossencoder

sys.path.insert(0, str(Path(__file__).parent))
from run_crossencoder import load_glosses

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "models/v3-xenc")
CORPUS_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "output/ml/corpus/v3")
SPANS_PER_REQUEST = (1, 2, 4, 8)
REPEATS = 40
WARMUP = 5


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def measure(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    glosses: dict[str, dict[str, str]],
    spans: int,
) -> dict[str, float]:
    """Latency of one request carrying ``spans`` ambiguous tokens."""
    collate = crossencoder.make_collate(tokenizer, 192)
    device = torch.device("cpu")
    samples: list[float] = []

    for iteration in range(WARMUP + REPEATS):
        batch_rows = rows[(iteration * spans) % max(1, len(rows) - spans) :][:spans]
        loader = DataLoader(
            crossencoder.PairDataset(batch_rows, glosses),
            batch_size=spans,
            shuffle=False,
            collate_fn=collate,
        )
        started = time.perf_counter()
        with torch.no_grad():
            crossencoder.predict(model, loader, device)
        elapsed = (time.perf_counter() - started) * 1000.0
        if iteration >= WARMUP:
            samples.append(elapsed)

    return {
        "spans_per_request": spans,
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(_percentile(samples, 0.95), 2),
        "mean_ms": round(statistics.mean(samples), 2),
        "per_span_p50_ms": round(statistics.median(samples) / spans, 2),
    }


def main() -> None:
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    glosses = load_glosses()
    rows = corpus.load_split(CORPUS_DIR / "test_natural.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "checkpoint")

    report: dict[str, Any] = {
        "model_dir": str(MODEL_DIR),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
        },
        "torch": torch.__version__,
        "repeats": REPEATS,
        "warmup": WARMUP,
        "variants": {},
    }

    # Dynamic quantization needs an explicit backend; the default is "none" on
    # this build, which fails with NoQEngine at prepack time.
    engines = list(torch.backends.quantized.supported_engines)
    for candidate in ("fbgemm", "qnnpack", "onednn"):
        if candidate in engines:
            torch.backends.quantized.engine = candidate
            break
    report["quantization_engine"] = torch.backends.quantized.engine
    report["supported_engines"] = engines

    for variant in ("fp32", "int8-dynamic"):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "checkpoint")
        model.eval()
        if variant == "int8-dynamic":
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except RuntimeError as error:
                # Never lose the fp32 measurements to a quantization failure.
                report["variants"][variant] = {"error": str(error)}
                print(f"{variant}: unavailable — {error}", flush=True)
                continue
        size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)
        measurements = [measure(model, tokenizer, rows, glosses, n) for n in SPANS_PER_REQUEST]
        report["variants"][variant] = {
            "parameter_bytes_mb": round(size_mb, 1),
            "measurements": measurements,
        }
        for entry in measurements:
            print(
                f"{variant:<13} spans={entry['spans_per_request']:<2} "
                f"p50={entry['p50_ms']:>8.2f} ms  p95={entry['p95_ms']:>8.2f} ms  "
                f"per-span p50={entry['per_span_p50_ms']:>7.2f} ms",
                flush=True,
            )

    output = MODEL_DIR / "benchmark.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwritten to {output}", flush=True)


if __name__ == "__main__":
    main()
