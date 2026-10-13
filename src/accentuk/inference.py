"""Local candidate-constrained cross-encoder and target inference supporting PyTorch and CTranslate2."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ServingError
from .normalizer import lookup_key

log = logging.getLogger(__name__)

OPEN_MARK = "⟦"
CLOSE_MARK = "⟧"
MAX_LENGTH = 192


@dataclass(frozen=True)
class Target:
    sentence: str
    start: int
    end: int
    form: str
    candidates: tuple[str, ...]


def validate_target(target: Target, manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Enforce preconditions: offsets must land on requested form, form in manifest, candidates match."""
    if not (0 <= target.start < target.end <= len(target.sentence)):
        raise ServingError("target offsets are outside the sentence")
    if lookup_key(target.sentence[target.start : target.end]) != target.form:
        raise ServingError("target offsets do not identify the requested form")
    entry = manifest.get("forms", {}).get(target.form)
    if entry is None:
        raise ServingError("form is outside frozen training coverage")
    allowed = sorted(entry["signatures"])
    if sorted(target.candidates) != allowed:
        raise ServingError("database candidates differ from model inventory")
    return entry, allowed


def _extract_pairs_from_targets(
    targets: list[Target], manifest: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], list[str], list[str], list[int], list[list[str]], list[int]]:
    results: dict[int, dict[str, Any]] = {}
    left: list[str] = []
    right: list[str] = []
    widths: list[int] = []
    signatures: list[list[str]] = []
    valid_indices: list[int] = []

    for index, target in enumerate(targets):
        if not (0 <= target.start < target.end <= len(target.sentence)):
            results[index] = {"index": index, "status": "model_ineligible", "signature": "", "scores": None, "margin": None}
            continue
        if lookup_key(target.sentence[target.start : target.end]) != target.form:
            results[index] = {"index": index, "status": "model_ineligible", "signature": "", "scores": None, "margin": None}
            continue
        entry = manifest.get("forms", {}).get(target.form)
        if entry is None:
            results[index] = {"index": index, "status": "model_ineligible", "signature": "", "scores": None, "margin": None}
            continue
        allowed = sorted(entry.get("signatures", []))
        if sorted(target.candidates) != allowed:
            results[index] = {"index": index, "status": "candidate_mismatch", "signature": "", "scores": None, "margin": None}
            continue

        marked = (
            target.sentence[: target.start]
            + OPEN_MARK
            + target.sentence[target.start : target.end]
            + CLOSE_MARK
            + target.sentence[target.end :]
        )
        row_signatures: list[str] = []
        for candidate in entry["candidates"]:
            signature = candidate["signature"]
            if signature not in target.candidates:
                continue
            gloss = f"{candidate['stressed']}: {candidate['definition']}".strip(": ")
            left.append(marked)
            right.append(gloss)
            row_signatures.append(signature)
        if set(row_signatures) != set(allowed):
            results[index] = {"index": index, "status": "candidate_mismatch", "signature": "", "scores": None, "margin": None}
            continue

        widths.append(len(row_signatures))
        signatures.append(row_signatures)
        valid_indices.append(index)

    return results, left, right, widths, signatures, valid_indices


def _rank_scores_into_results(
    results: dict[int, dict[str, Any]],
    valid_indices: list[int],
    widths: list[int],
    signatures: list[list[str]],
    logits: list[float],
    threshold: float,
    total_count: int,
) -> list[dict[str, Any]]:
    offset = 0
    for idx, (width, row_signatures) in enumerate(zip(widths, signatures, strict=True)):
        orig_index = valid_indices[idx]
        scores: dict[str, float] = {}
        for signature, score in zip(row_signatures, logits[offset : offset + width], strict=True):
            scores[signature] = max(scores.get(signature, float("-inf")), float(score))
        ranked = sorted(scores, key=scores.get, reverse=True)
        margin = scores[ranked[0]] - scores[ranked[1]] if len(ranked) > 1 else 0.0
        results[orig_index] = {
            "index": orig_index,
            "signature": ranked[0] if ranked else "",
            "scores": scores,
            "margin": margin,
            "status": "selected" if margin >= threshold else "low_margin",
        }
        offset += width

    return [results[i] for i in range(total_count)]


class ContextualStressModel:
    """Local PyTorch XLM-R cross-encoder for disambiguating heteronyms with sense glosses."""

    def __init__(
        self,
        model_dir: Path | str,
        manifest_path: Path | str | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = Path(model_dir)
        manifest_file = Path(manifest_path) if manifest_path else model_path / "serving_manifest.json"
        if not manifest_file.is_file():
            if (model_path.parent / "serving_manifest.json").is_file():
                manifest_file = model_path.parent / "serving_manifest.json"
            else:
                raise FileNotFoundError(f"serving_manifest.json not found at {manifest_file}")

        content = manifest_file.read_bytes()
        self.manifest: dict[str, Any] = json.loads(content)
        self.manifest_hash = hashlib.sha256(content).hexdigest()
        self.model_version = str(self.manifest.get("model_version", "unknown"))
        self.inventory_hash = str(self.manifest.get("inventory_hash", ""))
        self.threshold = float(self.manifest.get("threshold", 0.0))

        checkpoint = model_path / "checkpoint" if (model_path / "checkpoint").is_dir() else model_path
        self.device = torch.device(device)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

    def metadata(self) -> tuple[str, str]:
        return self.model_version, self.manifest_hash

    def resolve(self, targets: list[Target]) -> list[dict[str, Any]]:
        """Score each target sense in context and return candidate decisions."""
        results, left, right, widths, signatures, valid_indices = _extract_pairs_from_targets(
            targets, self.manifest
        )

        if left:
            encoded = self.tokenizer(
                left,
                right,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            moved = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                values = self.model(**moved).logits.squeeze(-1).detach().cpu().tolist()
            logits = values if isinstance(values, list) else [float(values)]
            return _rank_scores_into_results(
                results, valid_indices, widths, signatures, logits, self.threshold, len(targets)
            )

        return [results[i] for i in range(len(targets))]


class CT2ContextualStressModel:
    """High-performance CTranslate2 INT8/FP32 XLM-R cross-encoder."""

    def __init__(
        self,
        model_dir: Path | str,
        manifest_path: Path | str | None = None,
        *,
        device: str = "cpu",
        compute_type: str = "default",
        batch_size: int = 16,
    ) -> None:
        import ctranslate2
        import numpy as np
        from transformers import AutoTokenizer

        model_path = Path(model_dir)
        manifest_file = Path(manifest_path) if manifest_path else model_path / "serving_manifest.json"
        if not manifest_file.is_file():
            if (model_path.parent / "serving_manifest.json").is_file():
                manifest_file = model_path.parent / "serving_manifest.json"
            elif (model_path.parent.parent / "serving_manifest.json").is_file():
                manifest_file = model_path.parent.parent / "serving_manifest.json"
            else:
                raise FileNotFoundError(f"serving_manifest.json not found at {manifest_file}")

        content = manifest_file.read_bytes()
        self.manifest: dict[str, Any] = json.loads(content)
        self.manifest_hash = hashlib.sha256(content).hexdigest()
        self.model_version = str(self.manifest.get("model_version", "unknown"))
        self.inventory_hash = str(self.manifest.get("inventory_hash", ""))
        self.threshold = float(self.manifest.get("threshold", 0.0))

        self.device = device
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = ctranslate2.Encoder(
            str(model_path),
            device=self.device,
            compute_type=compute_type,
        )

        head_npz = model_path / "classifier_head.npz"
        if head_npz.is_file():
            data = np.load(head_npz)
            self.dense_w = data["dense_w"]
            self.dense_b = data["dense_b"]
            self.out_w = data["out_w"]
            self.out_b = data["out_b"]
        else:
            # Load from checkpoint fallback
            ckpt = model_path / "checkpoint" if (model_path / "checkpoint").is_dir() else model_path
            from transformers import AutoModelForSequenceClassification

            pt = AutoModelForSequenceClassification.from_pretrained(ckpt, local_files_only=True).eval()
            self.dense_w = pt.classifier.dense.weight.detach().numpy()
            self.dense_b = pt.classifier.dense.bias.detach().numpy()
            self.out_w = pt.classifier.out_proj.weight.detach().numpy()
            self.out_b = pt.classifier.out_proj.bias.detach().numpy()

    def metadata(self) -> tuple[str, str]:
        return self.model_version, self.manifest_hash

    def resolve(self, targets: list[Target]) -> list[dict[str, Any]]:
        """Score each target sense in context and return candidate decisions."""
        import numpy as np

        results, left, right, widths, signatures, valid_indices = _extract_pairs_from_targets(
            targets, self.manifest
        )

        if left:
            logits: list[float] = []
            total = len(left)
            for i in range(0, total, self.batch_size):
                b_left = left[i : i + self.batch_size]
                b_right = right[i : i + self.batch_size]
                enc = self.tokenizer(
                    b_left,
                    b_right,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                lens = enc["attention_mask"].sum(dim=1).tolist()
                tokens = [
                    self.tokenizer.convert_ids_to_tokens(enc["input_ids"][j][: lens[j]])
                    for j in range(len(b_left))
                ]
                out = self.encoder.forward_batch(tokens)
                hidden = np.array(out.last_hidden_state)[:, 0, :]
                h = np.tanh(np.dot(hidden, self.dense_w.T) + self.dense_b)
                vals = (np.dot(h, self.out_w.T) + self.out_b).squeeze(-1)
                if vals.ndim == 0:
                    logits.append(float(vals))
                else:
                    logits.extend(vals.tolist())

            return _rank_scores_into_results(
                results, valid_indices, widths, signatures, logits, self.threshold, len(targets)
            )

        return [results[i] for i in range(len(targets))]


def load_contextual_model(
    model_dir: Path | str,
    manifest_path: Path | str | None = None,
    *,
    mode: str = "auto",
    device: str = "cpu",
) -> ContextualStressModel | CT2ContextualStressModel | None:
    """Load contextual stress model respecting mode (light/full/auto)."""
    p = Path(model_dir)

    ct2_candidate: Path | None = None
    if (p / "model.bin").is_file():
        ct2_candidate = p
    elif (p / "stress-model-ct2-int8" / "model.bin").is_file():
        ct2_candidate = p / "stress-model-ct2-int8"
    elif (p / "ct2_models" / "stress-model-ct2-int8" / "model.bin").is_file():
        ct2_candidate = p / "ct2_models" / "stress-model-ct2-int8"

    pt_candidate: Path | None = None
    if ((p / "config.json").is_file() or (p / "checkpoint").is_dir()) and not (p / "model.bin").is_file():
        pt_candidate = p
    elif (p / "stress-model" / "checkpoint").is_dir() or (p / "stress-model" / "config.json").is_file():
        pt_candidate = p / "stress-model"

    if mode in ("light", "auto") and ct2_candidate:
        try:
            return CT2ContextualStressModel(ct2_candidate, manifest_path, device=device, compute_type="int8")
        except Exception as err:
            log.warning("Failed to load CT2 stress model: %s", err)

    if mode in ("full", "auto") and pt_candidate:
        try:
            return ContextualStressModel(pt_candidate, manifest_path, device=device)
        except Exception as err:
            log.warning("Failed to load PyTorch stress model: %s", err)

    if mode == "auto" and ct2_candidate:
        try:
            return CT2ContextualStressModel(ct2_candidate, manifest_path, device=device, compute_type="int8")
        except Exception:
            pass

    return None
