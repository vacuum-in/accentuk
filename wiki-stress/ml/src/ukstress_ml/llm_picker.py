"""Stress as a fixed-choice decision on a causal LLM, no training.

The Jev-style path: the sentence and the word go into a chat prompt that
lists the possible stressed spellings as lettered options and asks for the
letter; the model runs once, and the softmax over the option letters'
logits at the first answer position is the decision and its probability.
Nothing is generated. Glosses, where the sense lexicon has them, ride along
as the option's description.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

VOWELS = "аеєиіїоуюяАЕЄИІЇОУЮЯ"
ACUTE = "́"
LETTERS = "ABCDEFGH"

SYSTEM = ("Ти — фахівець з української орфоепії. Визначаєш наголос слова за контекстом. "
          "Відповідаєш лише однією літерою варіанта.")


def stressed(form: str, signature: str) -> str:
    target = int(signature) if signature.isdigit() else -1
    out, seen = [], -1
    for ch in unicodedata.normalize("NFC", form):
        out.append(ch)
        if ch in VOWELS:
            seen += 1
            if seen == target:
                out.append(ACUTE)
    return unicodedata.normalize("NFC", "".join(out))


def load_glosses(path: Path) -> dict[str, dict[str, str]]:
    """form -> signature -> 'pos — definition' from the sense lexicon."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = json.loads(line)
        if not f.get("complete"):
            continue
        for c in f["candidates"]:
            sig = str(c.get("signature", ""))
            if sig.isdigit():
                out.setdefault(f["form"], {}).setdefault(sig, f"{c.get('pos', '')} — {c.get('definition') or ''}".strip(" —"))
    return out


def build_prompt(tokenizer, glosses: dict, sentence: str, start: int, end: int,
                 form: str, candidates: list[str]) -> str:
    word = sentence[start:end]
    marked = sentence[:start] + "«" + word + "»" + sentence[end:]
    lines = []
    for i, sig in enumerate(candidates):
        gloss = glosses.get(form.lower(), {}).get(sig)
        lines.append(f"{LETTERS[i]} = {stressed(word, sig)}" + (f" ({gloss})" if gloss else ""))
    user = (f"Речення: {marked}\n\nЯкий наголос має слово «{word}» у цьому реченні?\n"
            + "\n".join(lines) + "\n\nВідповідь — лише літера.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def letter_ids(tokenizer) -> list[int]:
    out = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label {letter!r} is not one token: {ids}")
        out.append(ids[0])
    return out


class LLMPicker:
    def __init__(self, model_name: str, *, device: str = "cuda", glosses: Path | None = None,
                 batch: int = 16, max_length: int = 512, adapter: Path | None = None):
        import torch
        import transformers
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model = self.model.to(device).eval()
        self.device, self.batch, self.max_length = device, batch, max_length
        self.glosses = load_glosses(glosses) if glosses else {}
        self.letter_ids = letter_ids(self.tokenizer)

    def prompt(self, sentence: str, start: int, end: int, form: str, candidates: list[str]) -> str:
        return build_prompt(self.tokenizer, self.glosses, sentence, start, end, form, candidates)

    def _score(self, chunk: list[dict], orders: list[list[str]]) -> list[dict[str, float]]:
        torch = self.torch
        prompts = [self.prompt(t["sentence"], t["start"], t["end"], t["form"], order)
                   for t, order in zip(chunk, orders)]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_length).to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out = []
        for order, row in zip(orders, logits):
            probs = torch.softmax(row[self.letter_ids[:len(order)]], dim=0)
            out.append({c: float(p) for c, p in zip(order, probs)})
        return out

    def resolve(self, targets: list[dict]) -> list[dict]:
        """targets: index, sentence, start, end, form, candidates -> index, signature, confidence.

        The letters carry a position bias (the smoke test picked A four times
        out of four), so every item is scored in its given order and in the
        reversed one, and the two distributions are averaged per reading."""
        out = []
        for at in range(0, len(targets), self.batch):
            chunk = targets[at:at + self.batch]
            forward = [t["candidates"][:len(LETTERS)] for t in chunk]
            backward = [list(reversed(c)) for c in forward]
            a = self._score(chunk, forward)
            b = self._score(chunk, backward)
            for t, cands, pa, pb in zip(chunk, forward, a, b):
                probs = {c: (pa[c] + pb[c]) / 2 for c in cands}
                best = max(probs, key=probs.get)
                out.append({"index": t["index"], "signature": best,
                            "confidence": probs[best], "probabilities": probs})
        return out
