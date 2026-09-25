"""Torch-backed variable-length vowel ranker with lexical candidate masks."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("the vowel ranker requires `uv sync --extra training`") from error


@dataclass(frozen=True)
class VowelBatch:
    ssl_features: np.ndarray
    prosodic_features: np.ndarray
    valid_mask: np.ndarray
    targets: np.ndarray | None = None


def make_vowel_batch(
    ssl_sequences: list[np.ndarray],
    prosody_sequences: list[np.ndarray],
    *,
    targets: list[int] | None = None,
) -> VowelBatch:
    if not ssl_sequences or len(ssl_sequences) != len(prosody_sequences):
        raise ValueError("SSL and prosody sequences must be non-empty and have matching batch size")
    if targets is not None and len(targets) != len(ssl_sequences):
        raise ValueError("targets must match batch size")
    lengths = [item.shape[0] for item in ssl_sequences]
    ssl_dim = ssl_sequences[0].shape[1]
    prosody_dim = prosody_sequences[0].shape[1]
    if any(item.ndim != 2 or item.shape[1] != ssl_dim for item in ssl_sequences):
        raise ValueError("SSL sequences must have matching feature dimensions")
    if any(
        item.ndim != 2 or item.shape != (length, prosody_dim)
        for item, length in zip(prosody_sequences, lengths, strict=True)
    ):
        raise ValueError("prosody sequence shapes must match SSL vowel counts")
    maximum = max(lengths)
    ssl = np.zeros((len(lengths), maximum, ssl_dim), dtype=np.float32)
    prosody = np.zeros((len(lengths), maximum, prosody_dim), dtype=np.float32)
    valid = np.zeros((len(lengths), maximum), dtype=bool)
    for index, length in enumerate(lengths):
        if length < 1:
            raise ValueError("each word must have at least one vowel")
        ssl[index, :length] = ssl_sequences[index]
        prosody[index, :length] = prosody_sequences[index]
        valid[index, :length] = True
    target_values = None if targets is None else np.asarray(targets, dtype=np.int64)
    return VowelBatch(ssl, prosody, valid, target_values)


class VowelStressRanker:
    def __init__(
        self,
        *,
        ssl_size: int,
        prosody_size: int,
        hidden_size: int = 256,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        ffn_size: int = 768,
    ) -> None:
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        torch = _torch()
        nn = torch.nn

        class Module(nn.Module):  # type: ignore[name-defined, misc]
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(ssl_size + prosody_size, hidden_size)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=attention_heads,
                    dim_feedforward=ffn_size,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, ssl: Any, prosody: Any, valid: Any) -> Any:
                joined = self.projection(torch.cat((ssl, prosody), dim=-1))
                encoded = self.transformer(joined, src_key_padding_mask=~valid)
                return self.head(encoded).squeeze(-1)

        self.module: Any = Module()
        self.architecture = {
            "ssl_size": ssl_size,
            "prosody_size": prosody_size,
            "hidden_size": hidden_size,
            "transformer_layers": transformer_layers,
            "attention_heads": attention_heads,
            "ffn_size": ffn_size,
        }

    def parameters(self) -> Any:
        return self.module.parameters()

    def logits(self, batch: VowelBatch, *, device: str = "cpu") -> Any:
        torch = _torch()
        return self.module(
            torch.as_tensor(batch.ssl_features, device=device),
            torch.as_tensor(batch.prosodic_features, device=device),
            torch.as_tensor(batch.valid_mask, device=device),
        )

    def probabilities(
        self, batch: VowelBatch, *, candidate_mask: np.ndarray | None = None, device: str = "cpu"
    ) -> np.ndarray:
        torch = _torch()
        logits = self.logits(batch, device=device)
        mask = batch.valid_mask if candidate_mask is None else candidate_mask & batch.valid_mask
        masked = apply_candidate_mask(logits, torch.as_tensor(mask, device=device))
        return np.asarray(torch.softmax(masked, dim=-1).detach().cpu().numpy(), dtype=np.float32)


def apply_candidate_mask(logits: Any, candidate_mask: Any) -> Any:
    if logits.shape != candidate_mask.shape:
        raise ValueError("candidate mask must match logits shape")
    if not bool(candidate_mask.any(dim=-1).all()):
        raise ValueError("each word requires at least one eligible candidate")
    return logits.masked_fill(~candidate_mask, float("-inf"))


def masked_cross_entropy(logits: Any, candidate_mask: Any, targets: Any) -> Any:
    torch = _torch()
    masked = apply_candidate_mask(logits, candidate_mask)
    if not bool(candidate_mask.gather(1, targets.unsqueeze(1)).all()):
        raise ValueError("gold target must be an eligible candidate")
    return torch.nn.functional.cross_entropy(masked, targets)


def configure_ssl_trainability(encoder: Any, *, trainable_last_n: int) -> None:
    if trainable_last_n < 0:
        raise ValueError("trainable_last_n must be non-negative")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if trainable_last_n == 0:
        return
    layers = getattr(getattr(encoder, "encoder", encoder), "layers", None)
    if layers is None:
        raise ValueError("SSL encoder must expose encoder.layers for selective unfreezing")
    for layer in list(layers)[-trainable_last_n:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def save_ranker_checkpoint(
    path: str | Path, ranker: VowelStressRanker, *, metadata: dict[str, str]
) -> Path:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "architecture": ranker.architecture,
                "metadata": metadata,
                "state_dict": ranker.module.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_ranker_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[VowelStressRanker, dict[str, str]]:
    torch = _torch()
    payload: Any = torch.load(Path(path), map_location=device, weights_only=False)
    ranker = VowelStressRanker(**payload["architecture"])
    ranker.module.load_state_dict(payload["state_dict"])
    ranker.module.to(device)
    return ranker, {str(key): str(value) for key, value in payload["metadata"].items()}
