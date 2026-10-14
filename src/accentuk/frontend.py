"""Optional local TTS frontend pipeline: verbalization + stress marking."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .engine import Accentor
from .route import needs_verbalization
from .segment import Chunk, fit_to_window, join, split_sentences
from .text import StressToken
from .verbalize import (
    CT2SpanVerbalizer,
    CT2GemmaVerbalizer,
    CT2MarianVerbalizer,
    MarianVerbalizer,
    PassthroughVerbalizer,
    load_verbalizer,
)


@dataclass(frozen=True)
class FrontendResult:
    """The result of full TTS preparation (verbalization + stress marking)."""

    source: str
    normalized: str
    verbalized: str
    text: str
    tokens: list[StressToken] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset_id: int = 2
    model_version: str = ""
    coverage_manifest_hash: str = ""
    mode: str = ""


class Frontend:
    """Local TTS preparation pipeline for Ukrainian text."""

    def __init__(
        self,
        asset_dir: Path | str | None = None,
        *,
        mode: str = "auto",
        device: str = "cpu",
        verbalize: bool = True,
        engine: Accentor | None = None,
    ) -> None:
        self.mode = mode
        self.device = device
        self._owns_engine = engine is None
        self.engine = engine or Accentor(asset_dir=asset_dir, mode=mode, device=device)
        self.asset_dir = self.engine.asset_dir
        self._closed = False

        self.verbalizer: CT2SpanVerbalizer | CT2GemmaVerbalizer | MarianVerbalizer | CT2MarianVerbalizer | PassthroughVerbalizer
        if verbalize:
            candidates = [
                self.asset_dir / "ct2_models" / "verbalizer-spans-ct2-int8",
                self.asset_dir / "verbalizer-spans-ct2-int8",
                self.asset_dir / "ct2_models" / "gemma-verbalizer-ct2-int8",
                self.asset_dir / "gemma-verbalizer-ct2-int8",
                self.asset_dir / "verbalizer-ct2-int8",
                self.asset_dir / "ct2_models" / "verbalizer-ct2-int8",
                self.asset_dir / "verbalizer",
            ]
            verbalizer_path = self.asset_dir / "verbalizer"
            for c in candidates:
                if (c / "model.bin").is_file() or (c / "config.json").is_file():
                    verbalizer_path = c
                    break

            self.verbalizer = load_verbalizer(verbalizer_path, mode=mode, device=device)
        else:
            self.verbalizer = PassthroughVerbalizer()

    def _plan(self, text: str, warnings: list[str]) -> list[Chunk]:
        chunks = split_sentences(text)
        if isinstance(self.verbalizer, PassthroughVerbalizer):
            return chunks
        return fit_to_window(
            chunks,
            self.verbalizer.count_tokens,
            self.verbalizer.max_source_tokens,
            warnings,
        )

    def prepare(
        self,
        text: str,
        *,
        on_ambiguity: str = "default",
        combiner: bool | None = None,
    ) -> FrontendResult:
        """Prepare a single input text for TTS."""
        return self.prepare_many([text], on_ambiguity=on_ambiguity, combiner=combiner)[0]

    def prepare_many(
        self,
        texts: list[str],
        *,
        on_ambiguity: str = "default",
        combiner: bool | None = None,
    ) -> list[FrontendResult]:
        """Prepare multiple texts for TTS in batch."""
        if self._closed:
            raise RuntimeError("Frontend is closed.")

        warnings: list[list[str]] = [[] for _ in texts]
        normalized = [unicodedata.normalize("NFC", text) for text in texts]
        plans = [self._plan(t, warnings[i]) for i, t in enumerate(normalized)]

        # Collect speakable chunks that need verbalization
        pending: list[str] = []
        slots: list[tuple[int, int]] = []
        for row, plan in enumerate(plans):
            for pos, chunk in enumerate(plan):
                if not chunk.speakable:
                    continue
                if not needs_verbalization(chunk.text):
                    continue
                slots.append((row, pos))
                pending.append(chunk.text)

        if pending:
            produced = self.verbalizer.verbalize(pending)
            for (row, pos), output in zip(slots, produced, strict=True):
                plans[row][pos] = Chunk(output, True)

        results: list[FrontendResult] = []
        for row, plan in enumerate(plans):
            verbalized = join(plan)
            # Stress chunk by chunk, aggregating offsets against verbalized text
            pieces: list[str] = []
            tokens: list[StressToken] = []
            offset = 0
            for chunk in plan:
                if not chunk.speakable:
                    pieces.append(chunk.text)
                    offset += len(chunk.text)
                    continue
                res = self.engine.stress(
                    chunk.text, on_ambiguity=on_ambiguity, combiner=combiner
                )
                pieces.append(res.text)
                for tok in res.tokens:
                    tokens.append(
                        replace(tok, start=tok.start + offset, end=tok.end + offset)
                    )
                warnings[row].extend(res.warnings)
                offset += len(chunk.text)

            final_text = "".join(pieces)
            results.append(
                FrontendResult(
                    source=texts[row],
                    normalized=normalized[row],
                    verbalized=verbalized,
                    text=final_text,
                    tokens=tokens,
                    warnings=warnings[row],
                    dataset_id=self.engine.dataset_id,
                    model_version=self.engine.model_version,
                    coverage_manifest_hash=self.engine.coverage_manifest_hash,
                    mode=self.mode,
                )
            )

        return results

    def close(self) -> None:
        """Close the engine if owned by Frontend."""
        self._closed = True
        if self._owns_engine:
            self.engine.close()

    def __enter__(self) -> Frontend:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
