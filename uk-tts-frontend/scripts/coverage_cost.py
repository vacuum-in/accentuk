"""What does serving coverage the model never trained on actually cost?

55.7% of the forms in the serving manifest (8,535 of 15,332) never appear in
the training corpus, and on the benchmark they score no worse than forms with
20+ training rows — 75.6% against 73.8%. So the manifest is an *inventory* of
forms that have glosses, not a claim about what the model can do.

That leaves one question worth answering: is the unused coverage free? A larger
manifest is a larger candidate set to score, which is latency and memory. This
measures both, so the manifest can be trimmed on evidence or left alone on it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def trained_forms(corpus: Path) -> set[str]:
    return {row["form"] for row in json.loads(corpus.read_text(encoding="utf-8"))}


def split_manifest(manifest: Path, corpus: Path) -> tuple[dict, dict]:
    """Return the manifest as-is and one restricted to trained forms."""
    full = json.loads(manifest.read_text(encoding="utf-8"))
    trained = trained_forms(corpus)
    restricted = dict(full)
    restricted["forms"] = {f: e for f, e in full["forms"].items() if f in trained}
    return full, restricted


def candidate_load(manifest: dict, sentences: list[str]) -> dict[str, Any]:
    """How many candidates the model would be asked to score for this text.

    This is the quantity a larger manifest actually inflates: every extra
    covered form in a sentence is another (sentence, gloss) pair through the
    encoder.
    """
    forms = manifest["forms"]
    pairs = hits = 0
    for sentence in sentences:
        for word in sentence.lower().split():
            word = word.strip(".,;:!?«»\"'()—–")
            entry = forms.get(word)
            if entry:
                hits += 1
                pairs += len(entry["candidates"])
    return {"covered_tokens": hits, "candidate_pairs": pairs}


def measure(api: str, sentences: list[str], repeats: int) -> dict[str, float]:
    import httpx

    client = httpx.Client(base_url=api, timeout=180)
    timings = []
    for _ in range(repeats):
        for sentence in sentences:
            started = time.perf_counter()
            client.post("/v1/stress", json={"text": sentence})
            timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {
        "requests": len(timings),
        "p50_ms": round(statistics.median(timings), 1),
        "p99_ms": round(timings[int(0.99 * (len(timings) - 1))], 1),
        "mean_ms": round(statistics.fmean(timings), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("/home/devops/wiki-stress/output/ml/serving_manifest_v22.json"))
    parser.add_argument("--corpus", type=Path,
                        default=Path("/home/devops/wiki-stress/output/ml/silver_mined_v10.json"))
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    import csv

    data = args.benchmark / "lexical_stress_benchmark" / "data" / "lexical_stress_dataset.csv"
    sentences = [r["StressedSentence"].replace("+", "")
                 for r in csv.DictReader(data.open(encoding="utf-8"))][: args.limit]

    full, restricted = split_manifest(args.manifest, args.corpus)
    report: dict[str, Any] = {
        "manifest_forms": len(full["forms"]),
        "trained_forms": len(restricted["forms"]),
        "untrained_forms": len(full["forms"]) - len(restricted["forms"]),
        "manifest_bytes": args.manifest.stat().st_size,
        "restricted_bytes": len(json.dumps(restricted, ensure_ascii=False).encode()),
        "sentences": len(sentences),
        "load_full": candidate_load(full, sentences),
        "load_restricted": candidate_load(restricted, sentences),
    }
    report["extra_pairs_per_sentence"] = round(
        (report["load_full"]["candidate_pairs"]
         - report["load_restricted"]["candidate_pairs"]) / max(len(sentences), 1), 2)
    report["latency_as_served"] = measure(args.api, sentences, args.repeats)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
