"""Score an LLM directly on the benchmark's heteronym tokens.

Every retrain lands between 79.7% and 81.2% heteronym accuracy regardless of how
much data it sees, so the cross-encoder is saturated. The labelling pipeline
resolves this same ambiguity hundreds of thousands of times with two models
agreeing, which is evidence the task is not intrinsically hard for an LLM. This
measures that directly: same prompt as labelling, scored against benchmark gold
under Rule 5.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from ukstress.normalizer import lookup_key

from ukstress_ml import ambiguity, annotate

WORD = re.compile(r"[^\W\d_]+(?:['’ʼ+-][^\W\d_+]*)*", re.UNICODE)
VOWELS = frozenset("аеєиіїоуюя")


def marks(token: str) -> tuple[set[int], int]:
    """Ordinals of stressed vowels, and how many vowels the token has."""
    out: list[int] = []
    count = 0
    for char in unicodedata.normalize("NFD", token).lower():
        if char == "+":
            out.append(count - 1)
            continue
        if char in VOWELS:
            count += 1
    return set(out), count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_v19.json"))
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/inventory_v18_merged.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw/llm_bench.jsonl"))
    parser.add_argument("--deployment", choices=("label", "verify", "flash"), default="label")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data = args.benchmark / "lexical_stress_benchmark" / "data"
    frame = pd.read_csv(data / "lexical_stress_dataset.csv")
    gold = [str(s) for s in frame["StressedSentence"]]
    source = [str(s) for s in frame["Source"]]
    plain = [g.replace("+", "") for g in gold]

    heteronyms = set()
    with (data / "heteronyms_list.csv").open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            for cell in row:
                cell = cell.strip().replace("+", "")
                if cell:
                    heteronyms.add(lookup_key(cell))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]
    forms = {f.form: f for f in ambiguity.load(args.inventory)}

    # One row per benchmark heteronym token that the manifest can offer senses
    # for. Tokens the gold left unmarked, and monosyllables, are out of scope
    # exactly as they are for the maintainers' scorer.
    rows: list[dict[str, Any]] = []
    truth: dict[str, tuple[set[int], str, str]] = {}
    for index, (sentence, marked) in enumerate(zip(plain, gold, strict=True)):
        gold_tokens = WORD.findall(marked)
        plain_tokens = list(WORD.finditer(sentence))
        if len(gold_tokens) != len(plain_tokens):
            continue
        for match, gold_token in zip(plain_tokens, gold_tokens, strict=True):
            surface = match.group(0)
            key = lookup_key(surface)
            stresses, vowels = marks(gold_token)
            if key not in heteronyms or vowels < 2 or not stresses:
                continue
            if key not in manifest or key not in forms:
                continue
            row_id = f"{index}:{match.start()}"
            rows.append({
                "sentence_id": row_id,
                "sentence": sentence,
                "form": key,
                "surface": surface,
                "start": match.start(),
                "end": match.end(),
            })
            truth[row_id] = (stresses, source[index], surface)
    if args.limit:
        rows = rows[: args.limit]
    print(f"benchmark heteronym tokens the manifest covers: {len(rows):,}", flush=True)

    config = annotate.load_config()
    deployment = {"label": annotate.LABEL_DEPLOYMENT,
                  "verify": annotate.VERIFY_DEPLOYMENT,
                  "flash": annotate.FLASH_DEPLOYMENT}[args.deployment]
    usage = annotate.run_job(
        job="label",
        rows=rows,
        forms=forms,
        deployment=deployment,
        config=config,
        raw_path=args.raw,
        batch_size=args.batch_size,
        workers=args.workers,
        max_tokens=8000,
        system=annotate.LABEL_SYSTEM,
        prompt_builder=annotate.label_prompt,
        progress_every=10,
    )
    print(json.dumps(annotate.usage_dict(usage) if hasattr(annotate, "usage_dict")
                     else {"calls": getattr(usage, "calls", None)}, indent=2), flush=True)

    # Score: the chosen sense's signature must be a subset of gold's stresses.
    answers: dict[str, str] = {}
    with args.raw.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("content") or ""
            payload = re.sub(r"^```(?:json)?|```$", "", payload.strip(), flags=re.MULTILINE).strip()
            try:
                items = json.loads(payload).get("items", [])
            except json.JSONDecodeError:
                continue
            ids = record.get("sentence_ids") or record.get("ids") or []
            for item in items:
                position = item.get("i")
                if not isinstance(position, int) or position >= len(ids):
                    continue
                if item.get("unclear"):
                    continue
                answers[str(ids[position])] = str(item.get("sense_id", ""))

    by_source: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    unanswered = 0
    for row in rows:
        row_id = row["sentence_id"]
        stresses, src, surface = truth[row_id]
        sense = answers.get(row_id)
        if sense is None:
            unanswered += 1
            continue
        entry = manifest[row["form"]]
        signature = next((c["signature"] for c in entry["candidates"]
                          if c["sense_id"] == sense), None)
        if signature is None:
            unanswered += 1
            continue
        predicted = {int(part) for part in signature.split("|") if part.isdigit()}
        for bucket in (src, "УСЬОГО"):
            by_source[bucket][0] += 1
            by_source[bucket][1] += int(bool(predicted) and predicted <= stresses)
    print(f"\nunanswered (unclear or unmapped): {unanswered:,}")
    print(f"{'source':14}{'scored':>9}{'correct':>9}{'accuracy':>10}")
    for bucket in sorted(by_source, key=lambda b: -by_source[b][0]):
        total, hit = by_source[bucket]
        print(f"{bucket:14}{total:>9}{hit:>9}{hit / total:>9.1%}")


if __name__ == "__main__":
    main()
