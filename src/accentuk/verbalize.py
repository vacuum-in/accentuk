"""Verbalization: Gemma-3-270M on CTranslate2 (default), Marian MT on PyTorch or CTranslate2, passthrough."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class PassthroughVerbalizer:
    """Return text unchanged."""

    name = "passthrough"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def verbalize(self, texts: list[str]) -> list[str]:
        return list(texts)


class MarianVerbalizer:
    """Local PyTorch Marian MT sequence-to-sequence verbalizer."""

    name = "marian"

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_source_tokens: int = 384,
        max_new_tokens: int = 512,
        num_beams: int = 1,
    ) -> None:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        ckpt_path = Path(checkpoint)
        if not (ckpt_path / "config.json").is_file():
            raise FileNotFoundError(f"No Marian verbalizer checkpoint found at {ckpt_path}")

        self.device = torch.device(device)
        self._torch = torch
        self.batch_size = batch_size
        self.max_source_tokens = max_source_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.tokenizer = MarianTokenizer.from_pretrained(ckpt_path, local_files_only=True)
        self.model = (
            MarianMTModel.from_pretrained(ckpt_path, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.model.config.use_cache = True

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def verbalize(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        outputs: list[str] = []
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_source_tokens,
                ).to(self.device)
                source_tokens = int(encoded["attention_mask"].sum(dim=1).max())
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=min(self.max_new_tokens, max(128, source_tokens * 3)),
                    num_beams=self.num_beams,
                    do_sample=False,
                )
                outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


class CT2MarianVerbalizer:
    """High-performance CTranslate2 INT8/FP32 Marian MT verbalizer."""

    name = "ct2_marian"

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str = "cpu",
        compute_type: str = "default",
        batch_size: int = 16,
        max_source_tokens: int = 384,
        max_new_tokens: int = 512,
    ) -> None:
        import ctranslate2
        from transformers import MarianTokenizer

        ckpt_path = Path(checkpoint)
        if not (ckpt_path / "model.bin").is_file():
            raise FileNotFoundError(f"No CTranslate2 model.bin found at {ckpt_path}")

        self.device = device
        self.batch_size = batch_size
        self.max_source_tokens = max_source_tokens
        self.max_new_tokens = max_new_tokens

        self.tokenizer = MarianTokenizer.from_pretrained(ckpt_path, local_files_only=True)
        self.translator = ctranslate2.Translator(
            str(ckpt_path),
            device=self.device,
            compute_type=compute_type,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def verbalize(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        outputs: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = [
                self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer(s, truncation=True, max_length=self.max_source_tokens)["input_ids"]
                )
                for s in batch
            ]
            res = self.translator.translate_batch(
                tokens,
                beam_size=1,
                max_decoding_length=self.max_new_tokens,
                suppress_sequences=[["<s>"]],
            )
            outputs.extend([
                self.tokenizer.decode(
                    self.tokenizer.convert_tokens_to_ids(r.hypotheses[0]),
                    skip_special_tokens=True,
                )
                for r in res
            ])
        return outputs


class CT2GemmaVerbalizer:
    """skypro1111/gemma-3-270m-uk-verbalizer converted to CTranslate2, run as its card prescribes: one sentence
    per prompt, greedy, digit tokens banned, and the model's own two deterministic steps first (bare digit strings
    read digit by digit without the model, long numbers grouped by three). Those steps ship with the model as
    digit_router.py and bignum_spacing.py and are loaded from its directory.

    batch_size defaults to 1: batched INT8 decoding changes some outputs ("3 квітня" came out as "три перше
    квітня" in a batch of four, as "три квітня" alone, as in transformers), and the published accuracy is the
    one-sentence figure."""

    name = "ct2_gemma"
    max_source_tokens = 96

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 1,
        max_new_tokens: int = 192,
    ) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        ckpt_path = Path(checkpoint)
        if not (ckpt_path / "model.bin").is_file():
            raise FileNotFoundError(f"No CTranslate2 model.bin found at {ckpt_path}")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
        self.generator = ctranslate2.Generator(str(ckpt_path), device=device, compute_type=compute_type)
        vocab = self.tokenizer.get_vocab()
        self._banned = [[t] for t in vocab if any(c.isdigit() for c in t)]
        self._end = [t for t in ("<eos>", "<end_of_turn>") if t in vocab]
        self._route = _load_helper(ckpt_path / "digit_router.py", "route")
        self._spacing = _load_helper(ckpt_path / "bignum_spacing.py", "add_spacing")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _prompt(self, text: str) -> list[str]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids  # transformers 5 returns a BatchEncoding
        return self.tokenizer.convert_ids_to_tokens(ids)

    def verbalize(self, texts: list[str]) -> list[str]:
        outputs: list[str | None] = []
        pending: list[int] = []
        prompts: list[list[str]] = []
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                outputs.append(text)
                continue
            if self._route is not None:
                handled, routed = self._route(text)
                if handled:
                    outputs.append(routed)
                    continue
            if self._spacing is not None:
                text = self._spacing(text)
            outputs.append(None)
            pending.append(i)
            prompts.append(self._prompt(text))
        if prompts:
            results = self.generator.generate_batch(
                prompts,
                max_batch_size=self.batch_size,
                max_length=self.max_new_tokens,
                sampling_topk=1,
                include_prompt_in_result=False,
                end_token=self._end,
                suppress_sequences=self._banned,
            )
            for i, r in zip(pending, results, strict=True):
                outputs[i] = self.tokenizer.decode(r.sequences_ids[0], skip_special_tokens=True).strip()
        return [o if o is not None else "" for o in outputs]


def _load_helper(path: Path, attr: str):
    if not path.is_file():
        log.warning("Verbalizer helper %s is missing; its preprocessing step is skipped.", path.name)
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"accentuk_verbalizer_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


def _is_gemma(path: Path) -> bool:
    return (path / "model.bin").is_file() and (path / "chat_template.jinja").is_file()


def load_verbalizer(
    checkpoint: Path | str | None,
    *,
    mode: str = "auto",
    device: str = "cpu",
    batch_size: int | None = None,
) -> CT2GemmaVerbalizer | MarianVerbalizer | CT2MarianVerbalizer | PassthroughVerbalizer:
    """Load verbalizer based on requested mode and available assets. Gemma (CT2) serves both modes when present."""
    if not checkpoint:
        return PassthroughVerbalizer()
    ckpt_path = Path(checkpoint)
    if _is_gemma(ckpt_path):
        try:
            return CT2GemmaVerbalizer(ckpt_path, device=device, batch_size=batch_size or 1)
        except Exception as err:
            log.warning("Failed to load the Gemma CT2 verbalizer: %s", err)
            return PassthroughVerbalizer()

    # Detect CT2 candidates
    ct2_candidate: Path | None = None
    if (ckpt_path / "model.bin").is_file():
        ct2_candidate = ckpt_path
    elif (ckpt_path / "verbalizer-ct2-int8" / "model.bin").is_file():
        ct2_candidate = ckpt_path / "verbalizer-ct2-int8"
    elif (ckpt_path / "ct2_models" / "verbalizer-ct2-int8" / "model.bin").is_file():
        ct2_candidate = ckpt_path / "ct2_models" / "verbalizer-ct2-int8"

    # Detect PyTorch candidates
    pt_candidate: Path | None = None
    if (ckpt_path / "config.json").is_file() and not (ckpt_path / "model.bin").is_file():
        pt_candidate = ckpt_path
    elif (ckpt_path / "verbalizer" / "config.json").is_file() and not (ckpt_path / "verbalizer" / "model.bin").is_file():
        pt_candidate = ckpt_path / "verbalizer"

    if mode in ("light", "auto") and ct2_candidate:
        try:
            return CT2MarianVerbalizer(
                ct2_candidate,
                device=device,
                compute_type="int8",
                batch_size=batch_size or 16,
            )
        except Exception as err:
            log.warning("Failed to load CT2 verbalizer: %s", err)

    if mode in ("full", "auto") and pt_candidate:
        try:
            return MarianVerbalizer(pt_candidate, device=device, batch_size=batch_size or 8)
        except Exception as err:
            log.warning("Failed to load PyTorch verbalizer: %s", err)

    if mode == "auto" and ct2_candidate:
        try:
            return CT2MarianVerbalizer(
                ct2_candidate,
                device=device,
                compute_type="int8",
                batch_size=batch_size or 16,
            )
        except Exception:
            pass

    return PassthroughVerbalizer()
