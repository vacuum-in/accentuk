#!/usr/bin/env python3
"""Compare local accentuk package outputs with the live Go stress service."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from accentuk import Accentor

ACCENTS = "\u0300\u0301\u0342\u0306"


def strip_accents(text: str) -> str:
    # Remove combining acute, grave, perispomeni, but keep short accent on й if decomposed
    # In NFD: й is \u0438\u0306, don't strip \u0306 from й/й!
    # A cleaner way is: normalize to NFC, strip \u0300\u0301\u0342, keep NFC.
    nfc = unicodedata.normalize("NFC", text)
    res = re.sub(r"[\u0300\u0301\u0342]", "", nfc)
    return unicodedata.normalize("NFC", res)


def call_go_service(url: str, text: str, combiner: bool = False) -> dict | None:
    req = Request(
        url,
        data=json.dumps({"text": text, "combiner": combiner}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        return None


def run_comparison(
    go_url: str = "http://localhost:8080/v1/stress",
    asset_dir: str | Path | None = None,
    sample_file: Path | None = None,
    combiner: bool = False,
    max_sentences: int = 50,
) -> int:
    print(f"Connecting to Go service at {go_url}...")
    probe = call_go_service(go_url, "Мова.", combiner=combiner)
    if not probe:
        print(f"Error: Unable to connect to Go service at {go_url}.")
        return 1

    print("Go service responded successfully.")
    print(f"Initializing accentuk.Accentor with asset_dir={asset_dir}...")
    engine = Accentor(asset_dir=asset_dir, enable_combiner=combiner)

    # Collect test sentences
    sentences: list[str] = [
        "Мова.",
        "Мої сестри.",
        "Замок стоїть на горі.",
        "Він відчинив двері ключем і побачив замок.",
        "Вона має ручку.",
        "Маю велику надію на краще.",
        "Без нього ми б не впоралися.",
        "По-перше, це гарно.",
        "Спеціально-науковий підхід.",
        "123 книги та 5 яблук.",
        "Київ — столиця України.",
        "Державний університет.",
        "П'ять хвилин перерви.",
    ]

    if sample_file and sample_file.is_file():
        count = 0
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("—"):
                    continue
                plain = strip_accents(line)
                if len(plain) > 5 and plain not in sentences:
                    sentences.append(plain)
                    count += 1
                    if count >= max_sentences:
                        break

    print(f"Comparing {len(sentences)} sentences (combiner={combiner})...\n")

    total_sentences = len(sentences)
    exact_text_matches = 0
    total_tokens = 0
    exact_token_matches = 0
    token_status_matches = 0
    discrepancies: list[dict] = []

    for s_idx, sentence in enumerate(sentences, 1):
        go_resp = call_go_service(go_url, sentence, combiner=combiner)
        if not go_resp:
            print(f"Failed to query Go service for sentence #{s_idx}")
            continue

        acc_res = engine.stress(sentence, combiner=combiner)

        go_text = go_resp.get("text", "")
        py_text = acc_res.text

        is_text_match = go_text == py_text
        if is_text_match:
            exact_text_matches += 1

        go_tokens = go_resp.get("tokens", [])
        py_tokens = acc_res.tokens

        if len(go_tokens) != len(py_tokens):
            discrepancies.append({
                "sentence": sentence,
                "reason": f"Token count mismatch: Go={len(go_tokens)}, Py={len(py_tokens)}",
            })
            continue

        for gt, pt in zip(go_tokens, py_tokens):
            total_tokens += 1
            text_match = gt["text"] == pt.text and gt["output_text"] == pt.output_text
            status_match = gt["status"] == pt.status
            go_cands = gt.get("candidates") or []
            py_cands = pt.candidates or []
            cand_match = go_cands == py_cands


            if text_match:
                exact_token_matches += 1
            if status_match:
                token_status_matches += 1

            if not (text_match and status_match and cand_match):
                discrepancies.append({
                    "sentence": sentence,
                    "token": pt.text,
                    "go": {"output": gt.get("output_text"), "status": gt.get("status"), "cands": gt.get("candidates")},
                    "py": {"output": pt.output_text, "status": pt.status, "cands": pt.candidates},
                })

    print("=" * 60)
    print("PARITY VERIFICATION REPORT")
    print("=" * 60)
    print(f"Sentences tested:      {total_sentences}")
    print(f"Exact text match:      {exact_text_matches}/{total_sentences} ({exact_text_matches / total_sentences * 100:.1f}%)")
    print(f"Total tokens compared: {total_tokens}")
    print(f"Token text match:      {exact_token_matches}/{total_tokens} ({exact_token_matches / total_tokens * 100:.1f}%)")
    print(f"Token status match:    {token_status_matches}/{total_tokens} ({token_status_matches / total_tokens * 100:.1f}%)")
    print(f"Discrepancy count:     {len(discrepancies)}")

    if discrepancies:
        print("\nFirst discrepancies:")
        for d in discrepancies[:10]:
            print(" ", json.dumps(d, ensure_ascii=False))

    return 0 if len(discrepancies) == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Go service and local accentuk")
    parser.add_argument("--url", default="http://localhost:8080/v1/stress", help="Go service URL")
    parser.add_argument("--assets", default="/tmp/accentuk-bundle-test", help="Path to assets")
    parser.add_argument("--gold", type=Path, default=Path("gold_modern_reviewed_1.txt"))
    parser.add_argument("--combiner", action="store_true", help="Test with combiner enabled")
    parser.add_argument("--limit", type=int, default=50, help="Max gold sentences")
    args = parser.parse_args()

    sys.exit(run_comparison(
        go_url=args.url,
        asset_dir=args.assets,
        sample_file=args.gold,
        combiner=args.combiner,
        max_sentences=args.limit,
    ))
