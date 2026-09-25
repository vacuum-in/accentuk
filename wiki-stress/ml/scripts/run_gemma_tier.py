"""Ask a local Gemma about the words the confident policy leaves unmarked.

Measured per tier on lang-uk, the pipeline's certainty is very unevenly spread:
a single-reading lexicon hit is right 99.0% of the time and the prepositional
rule 97.4%, while the dictionary default is right 77.3%, the suffix fallback
71.5%, and the model below a margin of 2 is right 49.0% — worse than a coin.
A dataset for audio wants the first kind and none of the last, so the confident
policy marks 78.0% of multi-vowel tokens at 99.00% and leaves the rest bare.

This asks whether a local instruction model can fill some of that gap. The bar
is not the model's raw accuracy: on exactly those left-behind tokens the
current pipeline is already right 76.6% of the time, so anything below that is
a step back, and anything below about 95% is useless for a mode whose whole
point is that what it marks can be trusted.

The model is given the sentence and the candidate readings and asked to choose
one. It is never allowed to invent a reading: the answer is matched back to the
candidate list and anything else counts as an abstention.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
import unicodedata
from pathlib import Path

REPO = "google/gemma-4-E4B-it-qat-q4_0-gguf"
FILENAME = "gemma-4-E4B_q4_0-it.gguf"

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’+\-]+")
VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")

#: What the confident policy keeps. Everything else is what this tier is for.
TRUSTED = frozenset({"prepositional", "counted_form", "aspect"})


def vowels(word: str) -> int:
    return sum(1 for character in word if character in VOWELS)


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", (value or "").replace("+", "́")).lower()


def confident(token: dict) -> bool:
    """Does the confident policy already mark this token?"""
    status = token.get("status")
    if status in TRUSTED:
        return True
    # A single-reading lexicon hit: decided, with nothing to decide between.
    return status == "stressed" and len(token.get("candidates") or []) < 2


def build_prompt(sentence: str, surface: str, options: list[str]) -> str:
    listed = "\n".join(f"{index}. {option}" for index, option in enumerate(options, 1))
    return (
        "Ти визначаєш наголос в українському тексті.\n\n"
        f"Речення: {sentence}\n"
        f"Слово: {surface}\n\n"
        f"Можливі наголоси:\n{listed}\n\n"
        "Який варіант правильний у цьому реченні? "
        "Відповідай ЛИШЕ цифрою."
    )



CODEX_PROMPT = """Ти визначаєш наголос в українському тексті.

Для кожного пронумерованого випадку обери, який варіант наголосу правильний
САМЕ в цьому реченні. Зважай на значення слова та граматику, а не на те, який
наголос трапляється частіше.

{cases}

Виведи РІВНО валідний JSON, без пояснень і без markdown:
[{{"n": 1, "choice": 2}}, ...]
де "choice" — номер обраного варіанта.
"""

MASKED_PROMPT = """Ти визначаєш наголос в українському тексті.

У кожному реченні одне слово замінено на ___. Обери, який із варіантів
правильно стоїть на цьому місці. Біля варіантів, де це відомо, наведено
значення — спирайся насамперед на нього та на граматику речення, а не на те,
який наголос трапляється частіше.

{cases}

Виведи РІВНО валідний JSON, без пояснень і без markdown:
[{{"n": 1, "choice": 2}}, ...]
де "choice" — номер обраного варіанта.
"""


def run_codex_batches(args, asked: list[dict]) -> int:
    """Ask the Codex CLI about many tokens per turn.

    One turn per token would spend about twenty seconds of process startup on
    four seconds of work, so the cases are batched and answered by number.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ukstress_ml.browser import apply_signature

    def options_for(item: dict) -> list[str]:
        out = []
        for signature in item["signatures"]:
            try:
                out.append(apply_signature(item["surface"], signature))
            except Exception:  # noqa: BLE001 - an unusable signature is not an option
                out.append(item["surface"])
        return out

    glosses = load_glosses(args.glosses)
    print(f"forms with glosses: {len(glosses):,}", flush=True)

    results: list[dict] = []
    tally: collections.Counter[str] = collections.Counter()
    started = time.time()
    for start in range(0, len(asked), args.codex_batch):
        batch = asked[start:start + args.codex_batch]
        blocks = []
        for number, item in enumerate(batch, 1):
            options = options_for(item)
            senses = glosses.get(
                unicodedata.normalize("NFC", item["surface"]).lower(), {})
            listed = describe(options, item["signatures"], senses)
            if args.prompt_style == "masked":
                # Naming the word separately leaves the model to guess which
                # occurrence is meant when it appears twice, and it invites an
                # answer about the word rather than about this position in this
                # sentence. A blank has exactly one referent.
                # Named apart from the loop variable on purpose: calling both
                # `start` shadowed the batch offset and the progress line then
                # reported a character position as a token count.
                span_start, span_end = item.get("start", -1), item.get("end", -1)
                sentence = item["sentence"]
                if 0 <= span_start < span_end <= len(sentence):
                    masked = sentence[:span_start] + "___" + sentence[span_end:]
                else:
                    masked = re.sub(rf"(?<!\w){re.escape(item['surface'])}(?!\w)",
                                    "___", sentence, count=1)
                blocks.append(f"{number}. {masked}\n   Варіанти:\n{listed}")
            else:
                blocks.append(f'{number}. Речення: {item["sentence"]}\n'
                              f'   Слово: {item["surface"]}\n   Варіанти:\n{listed}')
        template = MASKED_PROMPT if args.prompt_style == "masked" else CODEX_PROMPT
        reply = run_codex_cli(template.format(cases="\n".join(blocks)), args)
        picks = {}
        try:
            parsed = json.loads(reply[reply.index("["): reply.rindex("]") + 1])
            picks = {int(row["n"]): int(row["choice"]) for row in parsed
                     if isinstance(row, dict) and "n" in row and "choice" in row}
        except Exception as error:  # noqa: BLE001 - a bad batch is an abstention
            print(f"  batch at {start} unparsable: {error}", flush=True)

        for number, item in enumerate(batch, 1):
            options = options_for(item)
            index = picks.get(number, 0) - 1
            picked = options[index] if 0 <= index < len(options) else None
            correct = picked is not None and nfc(picked) == nfc(item["gold"])
            served_correct = nfc(item["served"] or "") == nfc(item["gold"])
            tally["asked"] += 1
            tally["abstained" if picked is None else ("right" if correct else "wrong")] += 1
            tally["served_right"] += served_correct
            results.append({**item, "options": options, "reply": str(picks.get(number)),
                            "picked": picked, "correct": correct,
                            "served_correct": served_correct})
        done = start + len(batch)
        print(f"  {done}/{len(asked)}  {done / (time.time() - started):.2f} tok/s  "
              f"codex {tally['right']}/{done - tally['abstained']}  "
              f"pipeline {tally['served_right']}/{done}", flush=True)

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    answered = tally["asked"] - tally["abstained"]
    print(f"\nasked {tally['asked']}, abstained {tally['abstained']}")
    if answered:
        print(f"codex:    {tally['right']:>4}/{answered} = {100 * tally['right'] / answered:.1f}%")
    print(f"pipeline: {tally['served_right']:>4}/{tally['asked']} = "
          f"{100 * tally['served_right'] / tally['asked']:.1f}%  (the bar to beat)")
    print(f"-> {args.out}")
    return 0


def run_codex_cli(prompt: str, args) -> str:
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as handle:
        out = Path(handle.name)
    try:
        subprocess.run(
            ["codex", "exec", "--skip-git-repo-check",
             "-c", f"model={args.codex_model}",
             "-c", "model_reasoning_effort=medium",
             "--output-last-message", str(out), prompt],
            check=False, capture_output=True, timeout=900,
        )
        return out.read_text(encoding="utf-8")
    finally:
        out.unlink(missing_ok=True)



def load_glosses(path: Path) -> dict[str, dict[str, str]]:
    """form -> signature -> definition, from the serving manifest.

    The cross-encoder scores `(sentence, gloss)` pairs, so the glosses are the
    signal the pipeline itself uses to tell readings apart. Handing them to a
    general model turns "which accent" — a question about spelling — into
    "which sense", which is the question the sentence can actually answer.
    Coverage is not universal: a form outside the manifest gets bare readings.
    """
    if not path.is_file():
        return {}
    forms = json.loads(path.read_text(encoding="utf-8")).get("forms", {})
    out: dict[str, dict[str, str]] = {}
    for form, entry in forms.items():
        senses = {}
        for candidate in entry.get("candidates", []):
            definition = (candidate.get("definition") or "").strip()
            if definition:
                senses[str(candidate.get("signature"))] = definition
        if senses:
            out[unicodedata.normalize("NFC", form).lower()] = senses
    return out


def describe(options: list[str], signatures: list[str],
             senses: dict[str, str], width: int = 110) -> str:
    """The options as a list, each with its gloss where one exists."""
    lines = []
    for index, (option, signature) in enumerate(zip(options, signatures, strict=False), 1):
        gloss = senses.get(str(signature), "")
        if len(gloss) > width:
            gloss = gloss[:width].rsplit(" ", 1)[0] + "…"
        lines.append(f"     {index}) {option}" + (f" — {gloss}" if gloss else ""))
    return "\n".join(lines)

def excluded_forms(path: Path) -> set[str]:
    """Forms held out of the model queue by hand.

    The triage covers free variation, which has no right answer to find. This
    covers the other case: a form whose alternative reading belongs to a lexeme
    nobody writes, so the question looks open and is not. `світи́` is the plural
    of `світ` and the imperative of `світити` at the same accent; only the
    genitive singular of a rare feminine noun differs, and offering that as an
    option invites a model to invent a distinction the sentence never had.
    """
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(unicodedata.normalize("NFC", json.loads(line)["form"]).lower())
    return out

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=Path("output/ml/live_tokens.json"),
                        help="run_live_bench --save output, with per-token detail")
    parser.add_argument("--model", type=Path,
                        help="path to the gguf; downloaded from the hub if omitted")
    parser.add_argument("--limit", type=int, default=200,
                        help="tokens to ask about; the whole set is 2,080")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1200,
                        help="room for the reasoning as well as the answer")
    parser.add_argument("--reasoning", default="high",
                        choices=("low", "medium", "high"),
                        help="how long the model may think before answering")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--api",
                        help="OpenAI-compatible endpoint, e.g. LM Studio's "
                             "http://localhost:1234/v1 — used instead of loading "
                             "the gguf in-process, which is what gets the model "
                             "onto the GPU here: there is no CUDA wheel for this "
                             "Python and llama.cpp ships no Linux CUDA binary")
    parser.add_argument("--api-model", default="gemma-4-e4b-it-qat",
                        help="model id as the server reports it")
    parser.add_argument("--triage", type=Path,
                        default=Path("output/ml/ambiguous_surface_v4.jsonl"),
                        help="surface triage; free-variation forms are dropped")
    parser.add_argument("--glosses", type=Path,
                        default=Path("output/ml/models/v19-v10/serving_manifest.json"),
                        help="serving manifest; its definitions are shown beside "
                             "each option so the model is asked about sense "
                             "rather than about spelling")
    parser.add_argument("--exclude", type=Path,
                        default=Path("ml/data/excluded_forms.jsonl"),
                        help="forms held out by hand, with the reason recorded")
    parser.add_argument("--keep-free-variation", action="store_true",
                        help="ask about forms where both readings are acceptable "
                             "and the reference simply picked one — measuring a "
                             "model there measures the annotator, not the model")
    parser.add_argument("--prompt-style", default="named",
                        choices=("named", "masked"),
                        help="how the target is pointed at: named beside the "
                             "sentence, or blanked out inside it")
    parser.add_argument("--codex", action="store_true",
                        help="ask through the Codex CLI instead of a local server")
    parser.add_argument("--codex-model", default="gpt-5.6-luna")
    parser.add_argument("--codex-batch", type=int, default=20,
                        help="tokens per Codex turn; one turn per token would "
                             "spend twenty seconds of startup on four of work")
    parser.add_argument("--out", type=Path, default=Path("output/ml/gemma_tier.json"))
    args = parser.parse_args()

    rows = json.loads(args.tokens.read_text(encoding="utf-8"))
    asked: list[dict] = []
    for row in rows:
        gold_words: dict[str, list[str]] = collections.defaultdict(list)
        for word in WORD.findall(row["gold"]):
            gold_words[nfc(word).replace("́", "")].append(word)
        for token in row.get("tokens", []):
            surface = token["text"]
            if vowels(surface) < 2 or confident(token):
                continue
            candidates = token.get("candidates") or []
            if len(candidates) < 2:
                continue
            matching = gold_words.get(nfc(surface))
            if not matching:
                continue
            asked.append({
                "sentence": row["plain"], "surface": surface,
                "start": token.get("start", -1), "end": token.get("end", -1),
                "gold": matching.pop(0), "served": token.get("output_text"),
                "status": token["status"], "signatures": candidates,
            })
    print(f"tokens the confident policy leaves bare: {len(asked):,}", flush=True)

    # Free variation has no right answer to find: both readings are correct and
    # the reference recorded one of them. Asking a model about those measures
    # the annotator's coin flip, and it is 172 of the 636 disagreements.
    if not args.keep_free_variation and args.triage.is_file():
        free = excluded_forms(args.exclude)
        for line in args.triage.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("triage_class") == "free_variation":
                free.add(row["form_normalized"])
        before = len(asked)
        asked = [item for item in asked
                 if unicodedata.normalize("NFC", item["surface"]).lower() not in free]
        print(f"dropped {before - len(asked):,} free-variation tokens, "
              f"{len(asked):,} left", flush=True)

    asked = asked[: args.limit]

    if args.codex:
        return run_codex_batches(args, asked)

    if args.api:
        import httpx

        client = httpx.Client(base_url=args.api.rstrip("/"), timeout=120.0)

        def ask(prompt: str) -> str:
            body = client.post("/chat/completions", json={
                "model": args.api_model,
                "messages": [{"role": "user", "content": prompt}],
                # This model reasons before answering and the reasoning is
                # billed against the same budget: at 4 tokens it spent all of
                # them thinking and returned an empty string 200 times out of
                # 200. Measured, the reasoning runs about 270 tokens.
                "max_tokens": args.max_tokens, "temperature": 0.0,
                "reasoning_effort": args.reasoning,
            })
            body.raise_for_status()
            message = body.json()["choices"][0]["message"]
            return message.get("content") or message.get("reasoning_content") or ""
    else:
        from llama_cpp import Llama

        if args.model:
            model = Llama(model_path=str(args.model), n_ctx=1024, n_threads=args.threads,
                          n_gpu_layers=args.gpu_layers, verbose=False)
        else:
            model = Llama.from_pretrained(repo_id=REPO, filename=FILENAME, n_ctx=1024,
                                          n_threads=args.threads,
                                          n_gpu_layers=args.gpu_layers, verbose=False)

        def ask(prompt: str) -> str:
            return model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens, temperature=0.0,
            )["choices"][0]["message"]["content"]

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ukstress_ml.browser import apply_signature

    results = []
    tally: collections.Counter[str] = collections.Counter()
    started = time.time()
    for index, item in enumerate(asked, 1):
        options = []
        for signature in item["signatures"]:
            try:
                options.append(apply_signature(item["surface"], signature))
            except Exception:  # noqa: BLE001 - an unusable signature is not an option
                options.append(item["surface"])
        prompt = build_prompt(item["sentence"], item["surface"], options)
        reply = ask(prompt)
        # The answer is the last digit: reasoning text is full of numbers and
        # the choice is what the model settles on at the end.
        digits = re.findall(r"\d", reply or "")
        choice = int(digits[-1]) - 1 if digits else -1
        picked = options[choice] if 0 <= choice < len(options) else None

        correct = picked is not None and nfc(picked) == nfc(item["gold"])
        served_correct = nfc(item["served"] or "") == nfc(item["gold"])
        tally["asked"] += 1
        tally["abstained" if picked is None else ("right" if correct else "wrong")] += 1
        tally["served_right"] += served_correct
        results.append({**item, "options": options, "reply": reply,
                        "picked": picked, "correct": correct,
                        "served_correct": served_correct})
        if index % 25 == 0:
            rate = index / (time.time() - started)
            print(f"  {index}/{len(asked)}  {rate:.2f} tok/s  "
                  f"gemma {tally['right']}/{tally['asked'] - tally['abstained']}  "
                  f"pipeline {tally['served_right']}/{tally['asked']}", flush=True)

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    answered = tally["asked"] - tally["abstained"]
    print(f"\nasked {tally['asked']}, abstained {tally['abstained']}")
    if answered:
        print(f"gemma:    {tally['right']:>4}/{answered} = {100 * tally['right'] / answered:.1f}%")
    print(f"pipeline: {tally['served_right']:>4}/{tally['asked']} = "
          f"{100 * tally['served_right'] / tally['asked']:.1f}%  (the bar to beat)")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
