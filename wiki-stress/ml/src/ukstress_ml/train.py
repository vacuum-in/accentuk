"""Marian training for marked-span homograph stress selection.

Trained from scratch: the label space is unrelated to any translation task, so a
pretrained checkpoint's embeddings would be discarded anyway.

Checkpoint selection uses development **group-macro accuracy** under constrained
candidate scoring, not loss and not a translation metric — the model exists to
choose between candidates, so that is what selects it.
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import MarianConfig, MarianMTModel

from ukstress_ml.tokenizer import BOS, EOS, PAD, SourceTokenizer, TargetVocab


@dataclass
class TrainConfig:
    max_source_length: int = 128
    max_target_length: int = 32
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 2
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 1024
    dropout: float = 0.1
    source_vocab_size: int = 8000
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 5e-4
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    max_epochs: int = 30
    patience: int = 5
    seed: int = 20260815
    grad_clip: float = 1.0


class SpanDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        source: SourceTokenizer,
        target: TargetVocab,
        config: TrainConfig,
    ) -> None:
        self.rows = rows
        self.source = source
        self.target = target
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "input_ids": self.source.encode(row["source"], self.config.max_source_length),
            "labels": self.target.encode(row["target"], self.config.max_target_length),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    source_len = max(len(item["input_ids"]) for item in batch)
    target_len = max(len(item["labels"]) for item in batch)
    input_ids = torch.full((len(batch), source_len), PAD, dtype=torch.long)
    attention = torch.zeros((len(batch), source_len), dtype=torch.long)
    labels = torch.full((len(batch), target_len), -100, dtype=torch.long)
    decoder_input = torch.full((len(batch), target_len), PAD, dtype=torch.long)

    for position, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[position, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[position, : len(ids)] = 1
        target_ids = item["labels"]
        labels[position, : len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)
        shifted = [BOS, *target_ids[:-1]]
        decoder_input[position, : len(shifted)] = torch.tensor(shifted, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "decoder_input_ids": decoder_input,
    }


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: TrainConfig, source_vocab: int, target_vocab: int) -> MarianMTModel:
    marian = MarianConfig(
        vocab_size=source_vocab,
        decoder_vocab_size=target_vocab,
        share_encoder_decoder_embeddings=False,
        d_model=config.d_model,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        encoder_attention_heads=config.encoder_attention_heads,
        decoder_attention_heads=config.decoder_attention_heads,
        encoder_ffn_dim=config.encoder_ffn_dim,
        decoder_ffn_dim=config.decoder_ffn_dim,
        dropout=config.dropout,
        max_position_embeddings=max(config.max_source_length, config.max_target_length) + 8,
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=BOS,
        decoder_start_token_id=BOS,
        forced_eos_token_id=None,
    )
    return MarianMTModel(marian)


@torch.no_grad()
def score_candidates(
    model: MarianMTModel,
    rows: list[dict[str, Any]],
    source: SourceTokenizer,
    target: TargetVocab,
    config: TrainConfig,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Force-decode every candidate and return the highest-scoring one per row.

    Generation is never used, so a form outside the candidate set cannot be
    produced.
    """
    from ukstress_ml.corpus import EncodingError, apply_signature

    model.eval()
    expanded: list[tuple[int, str, list[int], list[int]]] = []
    for index, row in enumerate(rows):
        source_ids = source.encode(row["source"], config.max_source_length)
        for candidate in row["candidates"]:
            try:
                stressed = apply_signature(row["surface"], candidate["signature"])
            except EncodingError:
                continue
            expanded.append(
                (
                    index,
                    candidate["sense_id"],
                    source_ids,
                    target.encode(stressed, config.max_target_length),
                )
            )

    scores: dict[int, list[tuple[float, str]]] = {index: [] for index in range(len(rows))}
    for start in range(0, len(expanded), batch_size):
        chunk = expanded[start : start + batch_size]
        batch = collate(
            [{"input_ids": item[2], "labels": item[3]} for item in chunk]
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
        ).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        labels = batch["labels"]
        mask = labels != -100
        gathered = log_probs.gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        totals = (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        for position, item in enumerate(chunk):
            scores[item[0]].append((float(totals[position]), item[1]))

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        ranked = sorted(scores[index], reverse=True)
        if not ranked:
            results.append({"predicted": None, "margin": 0.0, "correct": False})
            continue
        best_score, best_sense = ranked[0]
        margin = best_score - ranked[1][0] if len(ranked) > 1 else float("inf")
        results.append(
            {
                "predicted": best_sense,
                "gold": row["sense_id"],
                "group_id": row["group_id"],
                "margin": margin,
                "correct": best_sense == row["sense_id"],
            }
        )
    return results


def group_macro_accuracy(results: list[dict[str, Any]]) -> float:
    by_group: dict[int, list[bool]] = {}
    for result in results:
        if result.get("gold") is None:
            continue
        by_group.setdefault(result["group_id"], []).append(bool(result["correct"]))
    if not by_group:
        return 0.0
    return sum(sum(v) / len(v) for v in by_group.values()) / len(by_group)


@dataclass
class TrainingRun:
    corpus_version: str
    inventory_hash: str
    config: dict[str, Any]
    source_checksum: str
    target_checksum: str
    device: str
    library_versions: dict[str, str]
    hardware: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_dev_group_macro: float = 0.0
    parameters: int = 0
    seconds: float = 0.0


def train(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    config: TrainConfig | None = None,
    corpus_version: str = "",
    inventory_hash: str = "",
) -> TrainingRun:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = SourceTokenizer.train(
        [row["source"] for row in train_rows], output_dir, vocab_size=config.source_vocab_size
    )
    target = TargetVocab.build([row["target"] for row in train_rows])
    target.save(output_dir / "target_vocab.json")

    device = select_device()
    model = build_model(config, source.vocab_size, target.vocab_size).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    loader = DataLoader(
        SpanDataset(train_rows, source, target, config),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config.label_smoothing)

    run = TrainingRun(
        corpus_version=corpus_version,
        inventory_hash=inventory_hash,
        config=asdict(config),
        source_checksum=source.checksum(),
        target_checksum=target.checksum(),
        device=str(device),
        library_versions={"torch": torch.__version__, "transformers": transformers.__version__},
        hardware={"platform": platform.platform(), "machine": platform.machine()},
        parameters=parameters,
    )
    print(
        f"device={device} params={parameters / 1e6:.1f}M "
        f"src_vocab={source.vocab_size} tgt_vocab={target.vocab_size} "
        f"train={len(train_rows):,} dev={len(dev_rows):,}",
        flush=True,
    )

    step = 0
    best = -1.0
    since_improvement = 0
    started = time.time()

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            step += 1
            scale = min(1.0, step / max(1, config.warmup_steps)) * min(
                1.0, math.sqrt(max(1, config.warmup_steps) / max(1, step))
            )
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * scale

            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
            ).logits
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
            batches += 1

        results = score_candidates(
            model, dev_rows, source, target, config, device, batch_size=config.eval_batch_size
        )
        macro = group_macro_accuracy(results)
        micro = sum(1 for r in results if r["correct"]) / max(1, len(results))
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            "dev_group_macro": macro,
            "dev_micro": micro,
            "seconds": time.time() - started,
        }
        run.history.append(entry)
        print(
            f"epoch {epoch:>2}  loss {entry['train_loss']:.4f}  "
            f"dev_macro {macro:.4f}  dev_micro {micro:.4f}  "
            f"{entry['seconds'] / 60:.1f} min",
            flush=True,
        )

        if macro > best:
            best = macro
            run.best_epoch = epoch
            run.best_dev_group_macro = macro
            since_improvement = 0
            model.save_pretrained(output_dir / "checkpoint")
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    run.seconds = time.time() - started
    (output_dir / "training_run.json").write_text(
        json.dumps(asdict(run), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run


def nfd(value: str) -> str:
    return unicodedata.normalize("NFD", value)
