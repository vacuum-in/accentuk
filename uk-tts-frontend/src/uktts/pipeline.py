"""The unified frontend: raw text in, TTS-ready stressed text out.

    source ──▶ NFC ──▶ chunk ──▶ verbalize (Marian) ──▶ stress (Go API) ──▶ output

Verbalization runs first on purpose. A stress lexicon has no entry for `14:30`
or `1 500 грн`, so stressing before verbalization leaves every number the
verbalizer later produces unstressed, and the voice guesses. Verbalizing first
hands the stress tier ordinary Ukrainian words it can actually look up.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field, replace

from . import route, segment
from . import stress as stress_stage
from . import verbalize as verbalize_stage
from .config import Config
from .segment import Chunk
from .stress import Stresser, StressToken
from .verbalize import Verbalizer

# Which stages a request runs. The order never changes — only whether a stage
# is skipped — because stressing before verbalizing leaves every word the
# verbalizer produces unstressed.
MODES = ("both", "verbalize", "stress")


@dataclass(frozen=True)
class Prepared:
    """One prepared input, with every intermediate stage kept for inspection."""

    source: str
    normalized: str
    verbalized: str
    text: str
    chunks: int = 0
    tokens: tuple[StressToken, ...] = ()
    warnings: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "verbalized": self.verbalized,
            "text": self.text,
            "chunks": self.chunks,
            "warnings": list(self.warnings),
            "timings_ms": self.timings_ms,
            "tokens": [
                {
                    "start": token.start,
                    "end": token.end,
                    "text": token.text,
                    "output_text": token.output_text,
                    "status": token.status,
                }
                for token in self.tokens
            ],
        }


class Pipeline:
    """Owns one loaded verbalizer and one stress client for the process lifetime."""

    def __init__(
        self,
        config: Config | None = None,
        verbalizer: Verbalizer | None = None,
        stresser: Stresser | None = None,
    ) -> None:
        self.config = config or Config.from_env()
        self.verbalizer = verbalizer or verbalize_stage.load(self.config.verbalizer)
        self.stresser = stresser or stress_stage.load(self.config.stress)

    def ready(self) -> dict[str, object]:
        ready, detail = self.stresser.ready()
        status: dict[str, object] = {
            "ready": ready,
            "verbalizer": self.verbalizer.name,
            "stress": {"backend": self.stresser.name, "ready": ready, "detail": detail},
        }
        identity = getattr(self.verbalizer, "checkpoint_id", None)
        if identity is not None:
            status["checkpoint"] = identity
        return status

    def _plan(self, text: str, warnings: list[str], verbalize: bool) -> list[Chunk]:
        chunks = segment.split_sentences(text)
        if not verbalize:
            # The window belongs to the verbalizer. With that stage skipped,
            # re-cutting a long sentence would only fragment the context the
            # stress model reads.
            return chunks
        return segment.fit_to_window(
            chunks,
            self.verbalizer.count_tokens,
            self.config.verbalizer.max_source_tokens,
            warnings,
        )

    def _stress(
        self, plan: list[Chunk], on_ambiguity: str | None, warnings: list[str],
        combiner: bool | None = None,
    ) -> tuple[str, tuple[StressToken, ...]]:
        """Stress each sentence in its own call.

        The stress API passes the whole request text to the contextual model as
        that word's sentence, and the cross-encoder was trained on one sentence.
        Handed a paragraph it answers confidently and wrongly: `Не вистачає
        руки` alone comes back `руки́` at margin 0.68, and inside a four-sentence
        block `ру́ки` at margin 1.56. One call per sentence keeps every word in
        the context it was scored against.

        Token offsets stay relative to the verbalized text, so the caller sees
        one coherent set of spans no matter how many calls it took.
        """

        pieces: list[str] = []
        tokens: list[StressToken] = []
        offset = 0
        for chunk in plan:
            if not chunk.speakable:
                pieces.append(chunk.text)
                offset += len(chunk.text)
                continue
            stressed = (self.stresser.stress(chunk.text, on_ambiguity) if combiner is None
                        else self.stresser.stress(chunk.text, on_ambiguity, combiner=combiner))
            pieces.append(stressed.text)
            tokens.extend(
                replace(token, start=token.start + offset, end=token.end + offset)
                for token in stressed.tokens
            )
            warnings.extend(stressed.warnings)
            offset += len(chunk.text)
        return "".join(pieces), tuple(tokens)

    def prepare(
        self, text: str, on_ambiguity: str | None = None, mode: str = "both",
        combiner: bool | None = None,
    ) -> Prepared:
        return self.prepare_many([text], on_ambiguity, mode, combiner)[0]

    def prepare_many(
        self, texts: list[str], on_ambiguity: str | None = None, mode: str = "both",
        combiner: bool | None = None,
    ) -> list[Prepared]:
        """Prepare several inputs, verbalizing every chunk of all of them in one pass.

        `mode` selects which stages run: `both`, `verbalize` to see what the
        normalizer produced before anything stresses it, or `stress` to stress
        text that is already spelled out.
        """

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        verbalize = mode in ("both", "verbalize")
        stress = mode in ("both", "stress")

        warnings: list[list[str]] = [[] for _ in texts]
        normalized = [unicodedata.normalize("NFC", text) for text in texts]

        started = time.perf_counter()
        plans = [
            self._plan(text, warnings[index], verbalize)
            for index, text in enumerate(normalized)
        ]
        plan_ms = (time.perf_counter() - started) * 1000

        verbalize_ms = 0.0
        routing = self.config.verbalizer.route
        if verbalize:
            pending: list[str] = []
            slots: list[tuple[int, int]] = []
            for row, plan in enumerate(plans):
                for position, chunk in enumerate(plan):
                    # A chunk with nothing to spell out is passed through. Sent
                    # anyway, the checkpoint rewrites roughly a quarter of plain
                    # Ukrainian sentences into something else — see route.py.
                    if not chunk.speakable:
                        continue
                    if routing and not route.needs_verbalization(chunk.text):
                        continue
                    slots.append((row, position))
                    pending.append(chunk.text)

            started = time.perf_counter()
            produced = self.verbalizer.verbalize(pending)
            verbalize_ms = (time.perf_counter() - started) * 1000
            if len(produced) != len(pending):  # pragma: no cover - contract breach
                raise RuntimeError(
                    f"verbalizer returned {len(produced)} outputs for {len(pending)} chunks"
                )

            for (row, position), output in zip(slots, produced, strict=True):
                plans[row][position] = Chunk(output, True)

        results: list[Prepared] = []
        for row, plan in enumerate(plans):
            verbalized = segment.join(plan)
            started = time.perf_counter()
            if stress:
                text, tokens = self._stress(plan, on_ambiguity, warnings[row], combiner)
            else:
                text, tokens = verbalized, ()
            stress_ms = (time.perf_counter() - started) * 1000
            results.append(
                Prepared(
                    source=texts[row],
                    normalized=normalized[row],
                    verbalized=verbalized,
                    text=text,
                    chunks=sum(1 for chunk in plan if chunk.speakable),
                    tokens=tokens,
                    warnings=tuple(warnings[row]),
                    timings_ms={
                        "chunk": round(plan_ms / max(len(texts), 1), 2),
                        "verbalize": round(verbalize_ms / max(len(texts), 1), 2),
                        "stress": round(stress_ms, 2),
                    },
                )
            )
        return results
