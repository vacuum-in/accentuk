"""The token classifier as a serving tier.

One softmax over the lexicon's candidate readings for a span, read off the
encoder's hidden state at that span. It never sees a gloss and cannot invent a
reading: the answer is always one of the caller's candidates.

It answers only for forms on its coverage list, and only where the pipeline
would otherwise fall to the dictionary default. Both limits are measured, not
cautious. On forms its training corpus never contained it scores 64% where the
dictionary default scores 89%; on forms it saw ten or more times it scores
93.8% where the default scores 78.5% (Common Voice, gold from audio, 19,089
ambiguous tokens). The coverage list is therefore the whole safety of the tier,
and it is loaded here, not in the caller.

Trained on audiobooks: see docs/lessons.md for how, and RESULTS.md for the
numbers on all three tests.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TokenResolver:
    def __init__(self, model_dir: Path, *, min_seen: int | None = None,
                 max_length: int = 160, window_chars: int = 300) -> None:
        import torch
        import transformers
        from torch import nn

        self.model_dir = model_dir
        meta = json.loads((model_dir / "resolver.json").read_text(encoding="utf-8"))
        coverage = json.loads((model_dir / "coverage.json").read_text(encoding="utf-8"))
        self.model_version = str(coverage.get("model_version") or model_dir.name)
        self.min_seen = int(min_seen if min_seen is not None else coverage.get("min_seen", 10))
        self.forms: dict[str, int] = {f: int(n) for f, n in coverage["forms"].items()}
        self.max_candidates = int(meta["max_candidates"])
        self.max_length = max_length
        # About 300 characters each side is what 160 subword tokens hold of
        # Ukrainian prose, and more context than the classifier was trained on.
        self.window_chars = window_chars

        self._torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        self.encoder = transformers.AutoModel.from_pretrained(model_dir).eval()
        width = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, self.max_candidates))
        self.head.load_state_dict(torch.load(model_dir / "head.pt", map_location="cpu"))
        self.head.eval()
        # Inference is CPU in the container; torch's own intra-op threads are
        # enough and a second request must not interleave with the first.
        self._lock = threading.Lock()
        log.info("token resolver %s: %d forms, %d eligible at %d+",
                 self.model_version, len(self.forms),
                 sum(1 for n in self.forms.values() if n >= self.min_seen), self.min_seen)

    def covers(self, form: str) -> bool:
        return self.forms.get(form, 0) >= self.min_seen

    def resolve(self, targets: list[dict[str, Any]], *,
                with_probabilities: bool = False, ignore_coverage: bool = False) -> list[dict[str, Any]]:
        """Decide every target whose form is covered; skip the rest silently.

        Each target: index, sentence, start, end, form, candidates. Each
        decision: index, signature, confidence. Candidates that are not a
        single vowel ordinal ("0|1" on a compound) are not classes and are
        ignored; a target with fewer than two usable candidates is skipped.
        """
        torch = self._torch
        wanted = []
        for t in targets:
            # the combiner weighs the classifier on every form, and knows
            # itself how far to trust it on one it rarely saw
            if not ignore_coverage and not self.covers(t["form"]):
                continue
            usable = [c for c in t["candidates"] if c.isdigit() and int(c) < self.max_candidates]
            if len(set(usable)) < 2:
                continue
            wanted.append((t, usable))
        if not wanted:
            return []

        # The encoder reads at most max_length subword tokens. Truncating the
        # sentence from the start silently dropped every span past the cut:
        # its mask fell back to the CLS token and every such word got the same
        # low-confidence guess. Each target gets a window of text around its
        # own span instead, and a span the tokenizer still cannot place is
        # declined rather than answered.
        windows = []
        for t, _ in wanted:
            left = max(0, t["start"] - self.window_chars)
            right = min(len(t["sentence"]), t["end"] + self.window_chars)
            windows.append((t["sentence"][left:right], t["start"] - left, t["end"] - left))
        got = self.tokenizer([w for w, _, _ in windows], return_tensors="pt",
                             padding=True, truncation=True, max_length=self.max_length,
                             return_offsets_mapping=True)
        offsets = got.pop("offset_mapping")
        masks = torch.zeros(len(wanted), offsets.shape[1], dtype=torch.bool)
        allowed = torch.zeros(len(wanted), self.max_candidates, dtype=torch.bool)
        placed = []
        for i, ((t, usable), (_, start, end)) in enumerate(zip(wanted, windows)):
            hit = [k for k, (a, b) in enumerate(offsets[i].tolist())
                   if a < end and b > start and b > a]
            placed.append(bool(hit))
            masks[i, hit or [0]] = True
            for c in usable:
                allowed[i, int(c)] = True
        with self._lock, torch.inference_mode():
            hidden = self.encoder(**got).last_hidden_state
            weights = masks.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
            logits = self.head(pooled).masked_fill(~allowed, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
        picks = logits.argmax(-1).tolist()
        out = []
        for i, ((t, usable), pick) in enumerate(zip(wanted, picks, strict=True)):
            if not placed[i]:
                continue
            decision = {"index": t["index"], "signature": str(pick),
                        "confidence": round(float(probabilities[i, pick]), 4)}
            if with_probabilities:
                # every usable candidate, for a combiner that weighs the tiers
                decision["probabilities"] = {c: round(float(probabilities[i, int(c)]), 4) for c in usable}
            out.append(decision)
        return out

    def describe(self) -> dict[str, Any]:
        return {"model_version": self.model_version, "forms": len(self.forms),
                "eligible": sum(1 for n in self.forms.values() if n >= self.min_seen),
                "min_seen": self.min_seen}


def load_token_resolver(model_dir: str | None) -> TokenResolver | None:
    """None when the tier is not configured or its files are absent. A missing
    tier is the pipeline as it was; a broken one must not take the service
    down with it."""
    if not model_dir:
        return None
    directory = Path(model_dir)
    if not (directory / "head.pt").is_file() or not (directory / "coverage.json").is_file():
        log.warning("token resolver disabled: %s has no head.pt/coverage.json", directory)
        return None
    try:
        return TokenResolver(directory)
    except Exception as error:  # noqa: BLE001
        log.error("token resolver disabled: %s", error)
        return None
