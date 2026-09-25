"""What does the token classifier do to the lang-uk benchmark?

The classifier was scored inside its own corpus — 76.6% on forms it never saw,
against 58.2% for answering candidates[0]. That says it learned context. It
does not say whether it beats the pipeline it is meant to sit beside, on the
sentences the pipeline is actually judged on.

This takes the live API's answers to the benchmark (with each token's tier and
its candidate readings), asks the classifier about every token the lexicon
reads two ways, and scores three sentences with the maintainers' own harness:
the pipeline as it stands, the pipeline with the classifier overriding every
ambiguous token, and the pipeline with the classifier overriding only the
tiers that are known to be weak on heteronyms.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukstress_ml.corpus import apply_signature  # noqa: E402

ACUTE = "́"
TOKEN = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’]+")


def plus_form(surface: str, signature: str) -> str:
    """The benchmark's notation: `+` after the stressed vowel."""
    marked = unicodedata.normalize("NFD", apply_signature(surface, signature)) \
        .replace(ACUTE, "+")
    # Back to NFC, or й and ї stay decomposed and the harness sees a different
    # word — that mistake cost 26 points of word accuracy on the first run.
    return unicodedata.normalize("NFC", marked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("output/ml/live_review8.json"),
                        help="the --save output of run_live_bench.py")
    parser.add_argument("--model", type=Path, default=Path("models/tok-v2"))
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--override", default="dictionary_default,morphology,suffix",
                        help="tiers the classifier overrides in the third variant")
    parser.add_argument("--coverage", type=Path,
                        default=Path("output/ml/corpus/tok-v2/train.jsonl"),
                        help="the classifier's training rows; a form absent "
                             "from them is one it is guessing on, and it must "
                             "not override the pipeline there")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="classifier confidence a decision needs before it "
                             "may override the pipeline")
    parser.add_argument("--llm", help="pick with a causal LLM zero-shot instead of the classifier")
    parser.add_argument("--llm-adapter", type=Path, help="a LoRA adapter for --llm")
    parser.add_argument("--glosses", type=Path, help="sense lexicon for option descriptions (--llm)")
    parser.add_argument("--allow", type=Path,
                        help="a JSON list of forms the classifier may answer on, in "
                             "place of the min-seen gate")
    parser.add_argument("--min-seen", type=int, default=10,
                        help="training rows a form needs before the classifier "
                             "may override the pipeline on it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("output/ml/bench_tok_v2.json"))
    args = parser.parse_args()

    import torch
    import transformers
    from torch import nn

    if not args.llm:
        meta = args.model / "resolver.json"
        if not meta.exists():
            meta = args.model / "config.json"   # the first tok-v2 clobbered it
        config = json.loads(meta.read_text(encoding="utf-8"))
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        if "model_type" in json.loads((args.model / "config.json").read_text(encoding="utf-8")):
            encoder = transformers.AutoModel.from_pretrained(args.model)
        else:
            from safetensors.torch import load_file
            encoder = transformers.AutoModel.from_pretrained(config["base"])
            encoder.load_state_dict(load_file(args.model / "model.safetensors"), strict=False)
        encoder = encoder.to(args.device).eval()
        width = encoder.config.hidden_size
        head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                             nn.Linear(width, config["max_candidates"])).to(args.device)
        head.load_state_dict(torch.load(args.model / "head.pt", map_location=args.device))
        head.eval()

    llm = None
    if args.llm:
        from ukstress_ml.llm_picker import LLMPicker
        llm = LLMPicker(args.llm, glosses=args.glosses, batch=1, adapter=args.llm_adapter)

    def pick(sentence: str, start: int, end: int, candidates: list[str]) -> str:
        if llm is not None:
            d = llm.resolve([{"index": 0, "sentence": sentence, "start": start, "end": end,
                              "form": sentence[start:end], "candidates": candidates}])[0]
            return d["signature"], d["confidence"]
        return pick_classifier(sentence, start, end, candidates)

    def pick_classifier(sentence: str, start: int, end: int, candidates: list[str]) -> str:
        got = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=160,
                        return_offsets_mapping=True)
        offsets = got.pop("offset_mapping")[0].tolist()
        hit = [t for t, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
        mask = torch.zeros(1, len(offsets), dtype=torch.bool)
        mask[0, hit or [0]] = True
        allowed = torch.zeros(1, config["max_candidates"], dtype=torch.bool)
        for c in candidates:
            if c.isdigit() and int(c) < config["max_candidates"]:
                allowed[0, int(c)] = True
        with torch.no_grad():
            hidden = encoder(**{k: v.to(args.device) for k, v in got.items()}).last_hidden_state
            pooled = (hidden * mask.to(args.device).unsqueeze(-1)).sum(1) / mask.sum()
            logits = head(pooled).masked_fill(~allowed.to(args.device), float("-inf"))
            confidence = float(torch.softmax(logits, -1).max())
        return str(int(logits.argmax())), confidence

    rows = json.loads(args.review.read_text(encoding="utf-8"))
    override = set(args.override.split(","))
    seen: collections.Counter[str] = collections.Counter()
    if args.coverage.exists():
        for line in args.coverage.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if len(r["candidates"]) > 1:
                    seen[r["form"].lower()] += 1
    allow = set(json.loads(args.allow.read_text(encoding="utf-8"))) if args.allow else None
    rule = f"classifier over {args.override}, forms seen {args.min_seen}+, conf ≥{args.min_confidence}"
    variants = {"pipeline": {}, "classifier on every ambiguous token": {},
                f"classifier over {args.override}": {}, rule: {}}
    agree = collections.Counter()
    asked = 0

    for row in rows:
        plain, got = row["plain"], row["got"]
        variants["pipeline"][plain] = got
        words_all = got.split()
        words_weak = got.split()
        words_rule = got.split()
        tokens = [t for t in row.get("tokens", []) if t.get("status")]
        cursor = 0
        for token in tokens:
            candidates = [c for c in token.get("candidates", []) if c.isdigit()]
            if len(set(candidates)) < 2:
                continue
            bare = unicodedata.normalize("NFC", token["text"].lower())
            # Find this token's word in the stressed sentence, in order.
            while cursor < len(words_all):
                word = unicodedata.normalize(
                    "NFC", "".join(TOKEN.findall(words_all[cursor].replace("+", ""))).lower())
                if word == bare:
                    break
                cursor += 1
            if cursor >= len(words_all):
                break
            chosen, confidence = pick(plain, token["start"], token["end"], candidates)
            asked += 1
            try:
                replacement = plus_form(token["text"], chosen)
            except Exception:  # noqa: BLE001
                cursor += 1
                continue
            original = words_all[cursor]
            stripped = original.replace("+", "")
            at = stripped.find(token["text"])
            if at < 0:
                cursor += 1
                continue
            rebuilt = stripped[:at] + replacement + stripped[at + len(token["text"]):]
            words_all[cursor] = rebuilt
            if token["status"] in override:
                words_weak[cursor] = rebuilt
                eligible = bare in allow if allow is not None else seen.get(bare, 0) >= args.min_seen
                if eligible and confidence >= args.min_confidence:
                    words_rule[cursor] = rebuilt
            pipeline_sig = None
            out = unicodedata.normalize("NFD", token.get("output_text", ""))
            if ACUTE in out:
                ordinal = -1
                for ch in out:
                    if ch.lower() in "аеєиіїоуюяй":
                        ordinal += 1
                    if ch == ACUTE:
                        pipeline_sig = str(ordinal)
                        break
            agree["same" if pipeline_sig == chosen else "differ"] += 1
            cursor += 1
        variants["classifier on every ambiguous token"][plain] = " ".join(words_all)
        variants[f"classifier over {args.override}"][plain] = " ".join(words_weak)
        variants[rule][plain] = " ".join(words_rule)

    print(f"{asked:,} ambiguous tokens asked; classifier agrees with the pipeline on "
          f"{agree['same']:,}, differs on {agree['differ']:,}")

    sys.path.insert(0, str(args.benchmark))
    cwd = os.getcwd()
    os.chdir(args.benchmark / "lexical_stress_benchmark" / "examples")
    report = {}
    try:
        from lexical_stress_benchmark import evaluate_stressification
        for name, resolved in variants.items():
            accuracies = evaluate_stressification(
                lambda text, r=resolved: r.get(text, text), show_progress=False,
                raise_on_mismatch=False)
            report[name] = dict(zip(["sentence", "word", "heteronym", "unambiguous",
                                     "macro_f1_heteronyms"], accuracies.values()))
    finally:
        os.chdir(cwd)

    print(f"\n{'variant':<60}{'word':>8}{'heteronym':>11}{'sentence':>10}{'F1 het':>8}")
    for name, scores in report.items():
        print(f"{name:<60}{100 * scores['word']:>7.2f}%{100 * scores['heteronym']:>10.2f}%"
              f"{100 * scores['sentence']:>9.2f}%{100 * scores['macro_f1_heteronyms']:>7.2f}%")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
