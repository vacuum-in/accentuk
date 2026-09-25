"""ONNX Runtime serving benchmark for the disambiguation model.

`RESULTS.md` §6 reports torch dynamic int8 as slower than fp32 and names an
optimised runtime as one of the three levers never tried. This measures it.

Two properties keep the comparison honest:

* It reuses `crossencoder.make_collate`, so tokenization, padding and the
  candidate-scoring layout are identical to the torch benchmark. A latency win
  from quietly encoding shorter inputs would not be a runtime win.
* It verifies the ONNX logits against eager torch on the same batches before
  reporting any timing. A faster model that scores differently is not a faster
  model, and dynamic int8 changes numerics -- so the maximum absolute deviation
  and the argmax agreement rate are reported alongside the milliseconds.

Run on an otherwise idle machine.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml import corpus, crossencoder

sys.path.insert(0, str(Path(__file__).parent))
from run_crossencoder import load_glosses

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "models/v3-xenc")
CORPUS_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "output/ml/corpus/v3")
ARTIFACT_DIR = Path(sys.argv[3] if len(sys.argv) > 3 else "output/ml/onnx/v3-xenc")
SPANS_PER_REQUEST = (1, 2, 4, 8)
REPEATS = 40
WARMUP = 5
MAX_LENGTH = 192
#: Match the torch benchmark's thread count so the two are comparable.
THREADS = 8


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def export(checkpoint: Path, destination: Path) -> Path:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    if (destination / "model.onnx").exists():
        return destination
    model = ORTModelForSequenceClassification.from_pretrained(checkpoint, export=True)
    model.save_pretrained(destination)
    return destination


def quantize(source: Path, destination: Path) -> Path | None:
    """Dynamic int8 via ONNX Runtime, using the widest ISA this CPU supports."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    if (destination / "model_quantized.onnx").exists():
        return destination
    flags = Path("/proc/cpuinfo").read_text(encoding="utf-8") if Path("/proc/cpuinfo").exists() else ""
    if "avx512_vnni" in flags:
        config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    elif "avx512f" in flags:
        config = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)
    else:
        config = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    try:
        ORTQuantizer.from_pretrained(source).quantize(
            save_dir=destination, quantization_config=config
        )
    except Exception as error:  # noqa: BLE001 - never lose the fp32 numbers
        print(f"int8 quantization unavailable: {error}", flush=True)
        return None
    return destination


def _session(model_path: Path) -> Any:
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


def _run(session: Any, encoded: dict[str, Any]) -> np.ndarray:
    feeds = {
        node.name: encoded[node.name].numpy()
        for node in session.get_inputs()
        if node.name in encoded
    }
    return session.run(None, feeds)[0]


def verify(session: Any, reference: Any, batches: list[dict[str, Any]]) -> dict[str, float]:
    """Compare ONNX logits with eager torch on identical batches."""
    deviations: list[float] = []
    agree = 0
    total = 0
    for batch in batches:
        encoded = batch["encoded"]
        onnx_logits = np.asarray(_run(session, dict(encoded))).squeeze(-1).reshape(-1)
        with torch.no_grad():
            torch_logits = (
                reference(**{k: v for k, v in encoded.items()}).logits.squeeze(-1).reshape(-1)
            )
        torch_np = torch_logits.detach().cpu().numpy()
        deviations.append(float(np.max(np.abs(onnx_logits - torch_np))))
        offset = 0
        for width in batch["spans"]:
            if width > 1:
                total += 1
                if int(np.argmax(onnx_logits[offset : offset + width])) == int(
                    np.argmax(torch_np[offset : offset + width])
                ):
                    agree += 1
            offset += width
    return {
        "max_abs_logit_deviation": round(max(deviations), 5) if deviations else 0.0,
        "argmax_agreement": round(agree / total, 4) if total else 1.0,
        "spans_compared": total,
    }


def measure(session: Any, batches: list[dict[str, Any]], spans: int) -> dict[str, float]:
    samples: list[float] = []
    for iteration in range(WARMUP + REPEATS):
        batch = batches[iteration % len(batches)]
        started = time.perf_counter()
        _run(session, dict(batch["encoded"]))
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


def _batches(rows: list[dict[str, Any]], glosses: Any, tokenizer: Any, spans: int) -> list[dict]:
    collate = crossencoder.make_collate(tokenizer, MAX_LENGTH)
    loader = DataLoader(
        crossencoder.PairDataset(rows, glosses),
        batch_size=spans,
        shuffle=False,
        collate_fn=collate,
    )
    return [batch for _, batch in zip(range(WARMUP + REPEATS), loader, strict=False)]


def main() -> None:
    glosses = load_glosses()
    rows = corpus.load_split(CORPUS_DIR / "test_natural.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "checkpoint")

    fp32_dir = export(MODEL_DIR / "checkpoint", ARTIFACT_DIR / "fp32")
    int8_dir = quantize(fp32_dir, ARTIFACT_DIR / "int8")

    reference = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "checkpoint")
    reference.eval()

    report: dict[str, Any] = {
        "model_dir": str(MODEL_DIR),
        "runtime": "onnxruntime",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": THREADS,
        },
        "max_length": MAX_LENGTH,
        "repeats": REPEATS,
        "warmup": WARMUP,
        "variants": {},
    }
    import onnxruntime

    report["onnxruntime"] = onnxruntime.__version__

    variants: list[tuple[str, Path | None]] = [
        ("onnx-fp32", fp32_dir / "model.onnx"),
        ("onnx-int8-dynamic", (int8_dir / "model_quantized.onnx") if int8_dir else None),
    ]
    for name, path in variants:
        if path is None or not path.exists():
            report["variants"][name] = {"error": "artifact not produced"}
            continue
        session = _session(path)
        checked = verify(session, reference, _batches(rows, glosses, tokenizer, 2)[:8])
        measurements = [
            measure(session, _batches(rows, glosses, tokenizer, n), n) for n in SPANS_PER_REQUEST
        ]
        report["variants"][name] = {
            "file_bytes_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "verification": checked,
            "measurements": measurements,
        }
        print(
            f"{name}: max|Δlogit|={checked['max_abs_logit_deviation']} "
            f"argmax_agreement={checked['argmax_agreement']} "
            f"({checked['spans_compared']} spans)",
            flush=True,
        )
        for entry in measurements:
            print(
                f"{name:<18} spans={entry['spans_per_request']:<2} "
                f"p50={entry['p50_ms']:>8.2f} ms  p95={entry['p95_ms']:>8.2f} ms  "
                f"per-span p50={entry['per_span_p50_ms']:>7.2f} ms",
                flush=True,
            )

    output = MODEL_DIR / "benchmark_onnx.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwritten to {output}", flush=True)


if __name__ == "__main__":
    main()
