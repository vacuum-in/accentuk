"""Token classifier inference tier supporting PyTorch and CTranslate2."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TokenResolver:
    """XLM-R token classifier over ambiguous words in context (PyTorch)."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        min_seen: int | None = None,
        max_length: int = 160,
        window_chars: int = 300,
        device: str = "cpu",
    ) -> None:
        import torch
        import transformers
        from torch import nn

        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "resolver.json").read_text(encoding="utf-8"))
        coverage = json.loads((self.model_dir / "coverage.json").read_text(encoding="utf-8"))
        self.model_version = str(coverage.get("model_version") or self.model_dir.name)
        self.min_seen = int(min_seen if min_seen is not None else coverage.get("min_seen", 10))
        self.forms: dict[str, int] = {f: int(n) for f, n in coverage["forms"].items()}
        self.max_candidates = int(meta["max_candidates"])
        self.max_length = max_length
        self.window_chars = window_chars
        self.device = torch.device(device)

        self._torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.encoder = transformers.AutoModel.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(self.device).eval()
        width = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, self.max_candidates)
        )
        self.head.load_state_dict(
            torch.load(self.model_dir / "head.pt", map_location=self.device)
        )
        self.head.to(self.device).eval()
        self._lock = threading.Lock()

    def covers(self, form: str) -> bool:
        return self.forms.get(form, 0) >= self.min_seen

    def resolve(
        self,
        targets: list[dict[str, Any]],
        *,
        with_probabilities: bool = False,
        ignore_coverage: bool = False,
    ) -> list[dict[str, Any]]:
        """Decide every target whose form is covered; skip the rest silently."""
        torch = self._torch
        wanted = []
        for t in targets:
            if not ignore_coverage and not self.covers(t["form"]):
                continue
            usable = [c for c in t["candidates"] if c.isdigit() and int(c) < self.max_candidates]
            if len(set(usable)) < 2:
                continue
            wanted.append((t, usable))
        if not wanted:
            return []

        windows = []
        for t, _ in wanted:
            left = max(0, t["start"] - self.window_chars)
            right = min(len(t["sentence"]), t["end"] + self.window_chars)
            windows.append((t["sentence"][left:right], t["start"] - left, t["end"] - left))

        got = self.tokenizer(
            [w for w, _, _ in windows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = got.pop("offset_mapping")
        masks = torch.zeros(len(wanted), offsets.shape[1], dtype=torch.bool, device=self.device)
        allowed = torch.zeros(
            len(wanted), self.max_candidates, dtype=torch.bool, device=self.device
        )
        placed = []
        for i, ((t, usable), (_, start, end)) in enumerate(zip(wanted, windows)):
            hit = [
                k
                for k, (a, b) in enumerate(offsets[i].tolist())
                if a < end and b > start and b > a
            ]
            placed.append(bool(hit))
            masks[i, hit or [0]] = True
            for c in usable:
                allowed[i, int(c)] = True

        got = {k: v.to(self.device) for k, v in got.items()}
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
            decision = {
                "index": t["index"],
                "signature": str(pick),
                "confidence": round(float(probabilities[i, pick]), 4),
            }
            if with_probabilities:
                decision["probabilities"] = {
                    c: round(float(probabilities[i, int(c)]), 4) for c in usable
                }
            out.append(decision)
        return out


class CT2TokenResolver:
    """High-performance CTranslate2 INT8/FP32 XLM-R token classifier."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        min_seen: int | None = None,
        max_length: int = 160,
        window_chars: int = 300,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        import ctranslate2
        import torch
        import transformers
        from torch import nn

        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "resolver.json").read_text(encoding="utf-8"))
        coverage = json.loads((self.model_dir / "coverage.json").read_text(encoding="utf-8"))
        self.model_version = str(coverage.get("model_version") or self.model_dir.name)
        self.min_seen = int(min_seen if min_seen is not None else coverage.get("min_seen", 10))
        self.forms: dict[str, int] = {f: int(n) for f, n in coverage["forms"].items()}
        self.max_candidates = int(meta["max_candidates"])
        self.max_length = max_length
        self.window_chars = window_chars
        self.device = device
        self._torch = torch

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.encoder = ctranslate2.Encoder(
            str(self.model_dir), device=device, compute_type=compute_type
        )
        width = 768
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, self.max_candidates)
        )
        # Load weights from head.pt or head.npz
        head_pt = self.model_dir / "head.pt"
        head_npz = self.model_dir / "head.npz"
        if not head_pt.is_file() and (self.model_dir.parent / "tok-v5" / "head.pt").is_file():
            head_pt = self.model_dir.parent / "tok-v5" / "head.pt"

        if head_pt.is_file():
            self.head.load_state_dict(torch.load(head_pt, map_location="cpu"))
        elif head_npz.is_file():
            import numpy as np
            npz = np.load(head_npz)
            self.head.load_state_dict({
                "0.weight": torch.from_numpy(npz["0.weight"]),
                "0.bias": torch.from_numpy(npz["0.bias"]),
                "2.weight": torch.from_numpy(npz["2.weight"]),
                "2.bias": torch.from_numpy(npz["2.bias"]),
            })
        else:
            # a CT2 conversion without its own head: the PyTorch tok-v5 of the same bundle has it
            for fallback in (model_path.parent / "tok-v5" / "head.pt", model_path.parent.parent / "tok-v5" / "head.pt"):
                if fallback.is_file():
                    self.head.load_state_dict(torch.load(fallback, map_location="cpu"))
                    break
        self.head.eval()
        self._lock = threading.Lock()

    def covers(self, form: str) -> bool:
        return self.forms.get(form, 0) >= self.min_seen

    def resolve(
        self,
        targets: list[dict[str, Any]],
        *,
        with_probabilities: bool = False,
        ignore_coverage: bool = False,
    ) -> list[dict[str, Any]]:
        """Decide every target whose form is covered; skip the rest silently."""
        import numpy as np
        torch = self._torch

        wanted = []
        for t in targets:
            if not ignore_coverage and not self.covers(t["form"]):
                continue
            usable = [c for c in t["candidates"] if c.isdigit() and int(c) < self.max_candidates]
            if len(set(usable)) < 2:
                continue
            wanted.append((t, usable))
        if not wanted:
            return []

        windows = []
        for t, _ in wanted:
            left = max(0, t["start"] - self.window_chars)
            right = min(len(t["sentence"]), t["end"] + self.window_chars)
            windows.append((t["sentence"][left:right], t["start"] - left, t["end"] - left))

        encoded = self.tokenizer(
            [w for w, _, _ in windows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        lens = encoded["attention_mask"].sum(dim=1).tolist()
        tokens = [
            self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][i][: lens[i]])
            for i in range(len(wanted))
        ]

        max_len = max(len(t) for t in tokens)
        masks = torch.zeros(len(wanted), max_len, dtype=torch.bool)
        allowed = torch.zeros(len(wanted), self.max_candidates, dtype=torch.bool)
        placed = []
        for i, ((t, usable), (_, start, end)) in enumerate(zip(wanted, windows)):
            hit = [
                k
                for k, (a, b) in enumerate(offsets[i].tolist())
                if a < end and b > start and b > a and k < max_len
            ]
            placed.append(bool(hit))
            masks[i, hit or [0]] = True
            for c in usable:
                allowed[i, int(c)] = True

        with self._lock:
            out = self.encoder.forward_batch(tokens)
            hidden = torch.from_numpy(np.array(out.last_hidden_state))
            weights = masks.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
            with torch.inference_mode():
                logits = self.head(pooled).masked_fill(~allowed, float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)

        picks = logits.argmax(-1).tolist()
        out_list = []
        for i, ((t, usable), pick) in enumerate(zip(wanted, picks, strict=True)):
            if not placed[i]:
                continue
            decision = {
                "index": t["index"],
                "signature": str(pick),
                "confidence": round(float(probabilities[i, pick]), 4),
            }
            if with_probabilities:
                decision["probabilities"] = {
                    c: round(float(probabilities[i, int(c)]), 4) for c in usable
                }
            out_list.append(decision)
        return out_list


def load_token_resolver(
    model_dir: Path | str | None,
    *,
    mode: str = "auto",
    device: str = "cpu",
) -> TokenResolver | CT2TokenResolver | None:
    """Load TokenResolver or CT2TokenResolver respecting mode (light/full/auto)."""
    if not model_dir:
        return None
    directory = Path(model_dir)

    ct2_candidate: Path | None = None
    if (directory / "model.bin").is_file():
        ct2_candidate = directory
    elif (directory / "tok-v5-ct2-int8" / "model.bin").is_file():
        ct2_candidate = directory / "tok-v5-ct2-int8"
    elif (directory / "ct2_models" / "tok-v5-ct2-int8" / "model.bin").is_file():
        ct2_candidate = directory / "ct2_models" / "tok-v5-ct2-int8"

    pt_candidate: Path | None = None
    if (directory / "head.pt").is_file() and not (directory / "model.bin").is_file():
        pt_candidate = directory
    elif (directory / "tok-v5" / "head.pt").is_file():
        pt_candidate = directory / "tok-v5"

    if mode in ("light", "auto") and ct2_candidate:
        try:
            return CT2TokenResolver(ct2_candidate, device=device, compute_type="int8")
        except Exception as err:
            log.warning("Failed to load CT2 token resolver: %s", err)

    if mode in ("full", "auto") and pt_candidate:
        try:
            return TokenResolver(pt_candidate, device=device)
        except Exception as err:
            log.warning("Failed to load PyTorch token resolver: %s", err)

    if mode == "auto" and ct2_candidate:
        try:
            return CT2TokenResolver(ct2_candidate, device=device, compute_type="int8")
        except Exception:
            pass

    return None
