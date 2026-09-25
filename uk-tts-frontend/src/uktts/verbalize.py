"""Stage 1 — verbalization: digits, dates, symbols and Latin words to Ukrainian words.

The model owns the whole transformation. Nothing here converts, expands,
repairs, or validates text: a chunk goes in unchanged and the model's output
comes back raw. Quality is changed by choosing a checkpoint, not by patching
its output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .config import VerbalizerConfig


class Verbalizer(Protocol):
    """What the pipeline needs from a verbalization stage."""

    name: str

    def count_tokens(self, text: str) -> int: ...

    def verbalize(self, texts: list[str]) -> list[str]: ...


class PassthroughVerbalizer:
    """Return text unchanged — used when verbalization is switched off."""

    name = "passthrough"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def verbalize(self, texts: list[str]) -> list[str]:
        return list(texts)


class MarianVerbalizer:
    """Marian sequence-to-sequence verbalizer loaded from a local checkpoint."""

    name = "marian"

    def __init__(self, config: VerbalizerConfig) -> None:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        checkpoint = config.checkpoint
        if checkpoint is None:
            raise FileNotFoundError(
                "no verbalizer checkpoint configured: set UKTTS_VERBALIZER_CHECKPOINT "
                "to a Marian checkpoint directory, pass --checkpoint, or turn the "
                "stage off with --no-verbalize (UKTTS_VERBALIZE=0)"
            )
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"no verbalizer checkpoint at {checkpoint}")

        self.config = config
        self._torch = torch
        self.device = self._resolve_device(config.device)
        self.tokenizer = MarianTokenizer.from_pretrained(checkpoint)
        self.model = MarianMTModel.from_pretrained(checkpoint).to(self.device).eval()
        self.model.config.use_cache = True

    @staticmethod
    def _resolve_device(requested: str) -> str:
        import torch

        if requested != "auto":
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def checkpoint_id(self) -> dict[str, str]:
        """Identify the served checkpoint in health output and per-request metadata."""

        assert self.config.checkpoint is not None  # set in __init__ or it raised
        manifest = self.config.checkpoint / "numeric_pipeline_manifest.json"
        identity = {"checkpoint": str(self.config.checkpoint), "device": self.device}
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for key in ("numeric_policy_version", "tokenizer_hash"):
                if key in payload:
                    identity[key] = str(payload[key])
        return identity

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def verbalize(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        outputs: list[str] = []
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                batch = texts[start : start + self.config.batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_source_tokens,
                ).to(self.device)
                # A verbalized sentence is longer than its source — `14:30`
                # becomes four words. Budget generously off the longest input
                # so nothing is cut off mid-number.
                source_tokens = int(encoded["attention_mask"].sum(dim=1).max())
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=min(self.config.max_new_tokens, max(128, source_tokens * 3)),
                    num_beams=self.config.num_beams,
                    do_sample=False,
                )
                outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


def load(config: VerbalizerConfig) -> Verbalizer:
    return MarianVerbalizer(config) if config.enabled else PassthroughVerbalizer()


def default_checkpoint_exists(path: Path) -> bool:
    return (path / "config.json").is_file()
