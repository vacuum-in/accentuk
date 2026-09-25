"""Compare the two tier-3 formulations on the tokens the model tier owns.

Not the pipeline metric: this isolates tier 3. For every benchmark heteronym
token whose form the manifest covers, ask both the served pipeline and the
signature classifier, and score both against gold.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import unicodedata
from pathlib import Path

ACUTE = "́"
#: `й` occupies a vowel ordinal in the lexicon's signature convention even
#: though it is not a syllable. Verified against the manifest: with it counted,
#: 29,710 of 30,712 stressed spellings reproduce exactly; without it, `байко́ва`
#: (signature 2) renders as `байкова́`.
VOWELS = "аеєиіїоуюяй"


def apply_signature(word: str, signature: str) -> str:
    """Place the acute on the signature-th vowel, the lexicon's own convention."""
    index = int(signature.split("|")[0])
    seen = -1
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", word):
        out.append(ch)
        if ch.lower() in VOWELS:
            seen += 1
            if seen == index:
                out.append(ACUTE)
    return unicodedata.normalize("NFC", "".join(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("/home/devops/wiki-stress/output/ml/serving_manifest_v22.json"))
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, "/home/devops/wiki-stress/ml/src")
    import httpx
    import torch
    from benchmark import to_plus_notation
    from transformers import AutoTokenizer
    from ukstress_ml.classifier import (
        SIGNATURES,
        SignatureClassifier,
        candidate_mask,
        masked_logits,
    )

    data = args.benchmark / "lexical_stress_benchmark" / "data"
    rows = [r["StressedSentence"] for r in
            csv.DictReader((data / "lexical_stress_dataset.csv").open(encoding="utf-8"))]
    heteronyms = {r["Heteronym"] for r in
                  csv.DictReader((data / "heteronyms_list.csv").open(encoding="utf-8"))}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    model = SignatureClassifier(str(args.checkpoint)).eval()
    model.head.load_state_dict(
        torch.load(args.checkpoint / "head.pt", map_location="cpu"))

    api = httpx.Client(base_url=args.api, timeout=180)
    pipe_ok = clf_ok = seen = 0
    differ: collections.Counter[str] = collections.Counter()
    by_class: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    from ukstress_ml.morphology import SpacyMorphologyTier
    from ukstress_ml.triage import classify_readings
    tier = SpacyMorphologyTier(
        model_path="/home/devops/wiki-stress/models/uk_core_news_sm")
    klass_cache: dict[str, str] = {}

    def klass(form: str) -> str:
        if form not in klass_cache:
            readings = tier.readings(form)
            klass_cache[form] = (classify_readings(readings)[0] if readings
                                 else "not_in_trie")
        return klass_cache[form]
    for index, gold_sentence in enumerate(rows, 1):
        plain = gold_sentence.replace("+", "")
        served = api.post("/v1/stress",
                          json={"text": plain, "include_tokens": True}).json()
        gold_words = gold_sentence.lower().split()
        got_words = to_plus_notation(served["text"]).lower().split()
        if len(gold_words) != len(got_words):
            continue
        # Pair API tokens to whitespace words by character offset, never by
        # position. The API skips standalone punctuation, so a single dash
        # desynchronises the two lists for the rest of the sentence.
        word_at = {}
        cursor = 0
        for position, word in enumerate(plain.split()):
            cursor = plain.index(word, cursor)
            for offset in range(cursor, cursor + len(word)):
                word_at[offset] = position
            cursor += len(word)
        for token in served["tokens"]:
            position = word_at.get(token["start"])
            if position is None or position >= len(gold_words):
                continue
            gold, got = gold_words[position], got_words[position]
            form = gold.replace("+", "")
            if form not in heteronyms or form not in manifest:
                continue
            candidates = [{"signature": c["signature"]}
                          for c in manifest[form]["candidates"]]
            encoded = tokenizer(plain, truncation=True, max_length=192,
                                return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping")[0].tolist()
            span = torch.zeros(1, len(offsets), dtype=torch.bool)
            for j, (a, b) in enumerate(offsets):
                if a != b and a < token["end"] and b > token["start"]:
                    span[0, j] = True
            if not span.any():
                span = encoded["attention_mask"].bool()
            with torch.inference_mode():
                logits = masked_logits(
                    model(encoded["input_ids"], encoded["attention_mask"], span),
                    candidate_mask([{"candidates": candidates}]))
            predicted = apply_signature(token["text"], SIGNATURES[int(logits.argmax(-1))])
            classifier_right = to_plus_notation(predicted).lower() == gold
            pipeline_right = gold == got
            seen += 1
            pipe_ok += pipeline_right
            clf_ok += classifier_right
            if pipeline_right != classifier_right:
                differ[form] += 1
            bucket = by_class[klass(form)]
            bucket[0] += 1
            bucket[1] += pipeline_right
            bucket[2] += classifier_right
        if index % 200 == 0:
            print(f"  {index}/{len(rows)} sentences", flush=True)

    report = {
        "tokens": seen,
        "pipeline_correct": pipe_ok,
        "classifier_correct": clf_ok,
        "pipeline_accuracy": round(100 * pipe_ok / max(seen, 1), 2),
        "classifier_accuracy": round(100 * clf_ok / max(seen, 1), 2),
        "disagreements": sum(differ.values()),
        "forms": differ.most_common(15),
        "by_class": {k: {"tokens": v[0],
                         "pipeline": round(100 * v[1] / v[0], 1),
                         "classifier": round(100 * v[2] / v[0], 1)}
                     for k, v in sorted(by_class.items())},
    }
    print(f"\nmanifest-covered heteronym tokens: {seen}")
    print(f"  served pipeline : {pipe_ok:4}  {report['pipeline_accuracy']:5.1f}%")
    print(f"  classifier      : {clf_ok:4}  {report['classifier_accuracy']:5.1f}%")
    print(f"  differ on       : {report['disagreements']} tokens")
    print("  forms:", report["forms"])
    print("\n  by ambiguity class:")
    for name, row in report["by_class"].items():
        print(f"    {name:16} {row['tokens']:4} tok   "
              f"pipeline {row['pipeline']:5.1f}%   classifier {row['classifier']:5.1f}%")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
