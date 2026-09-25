"""Pretrained cross-encoder over (sentence, gloss) pairs.

Two Marian runs from scratch reached minority-sense recall of 0.18 without
glosses and 0.21 with them. Adding gloss text barely helped because a randomly
initialised model has no Ukrainian semantics with which to match a context
against a definition — the glosses were simply more unfamiliar tokens.

This model starts from a pretrained multilingual encoder, so the matching
ability is already there and the corpus only has to teach the task. For each
candidate sense it scores the pair (sentence with the target span marked,
gloss of that sense) and takes a softmax across the candidates of the row, so
training optimises the decision that is actually made at inference.
"""

from __future__ import annotations

import contextlib
import json
import platform
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml.corpus import CLOSE_MARK, OPEN_MARK


@dataclass
class CrossEncoderConfig:
    base_model: str = "xlm-roberta-base"
    max_length: int = 192
    batch_rows: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_epochs: int = 14
    #: Steps between progress lines; 0 silences them.
    progress_every: int = 500
    patience: int = 3
    seed: int = 20260816
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    # bf16 autocast and gradient accumulation exist so a larger encoder fits an
    # 8 GB card: xlm-roberta-large in fp32 at batch_rows=16 will not. Keep
    # `amp_dtype` None to reproduce every run made before they were added.
    amp_dtype: str | None = None
    accumulate: int = 1


def marked_sentence(row: dict[str, Any]) -> str:
    sentence, start, end = row["sentence"], row["start"], row["end"]
    return f"{sentence[:start]}{OPEN_MARK}{sentence[start:end]}{CLOSE_MARK}{sentence[end:]}"


def candidate_glosses(
    row: dict[str, Any], glosses: dict[str, dict[str, str]]
) -> list[tuple[str, str]]:
    """Return ``(sense_id, gloss text)`` for each candidate, in stable order."""
    out: list[tuple[str, str]] = []
    for candidate in sorted(row["candidates"], key=lambda c: c["sense_id"]):
        info = glosses.get(candidate["sense_id"], {})
        text = info.get("definition") or ""
        stressed = info.get("stressed") or candidate.get("stressed") or ""
        out.append((candidate["sense_id"], f"{stressed}: {text}".strip(": ")))
    return out


def build_exemplars(
    rows: Iterable[dict[str, Any]], per_sense: int = 3, seed: int = 20260831
) -> dict[str, list[str]]:
    """Collect labelled usages of each sense, to be scored against instead of a definition.

    Measured motivation: forms with 20+ training rows score 73.8% on the
    benchmark and forms the model never saw score 75.6%. Supervision is not
    reaching the decision, and the suspect is what the decision is made
    *against* — a lexicographic definition, which shares little surface with the
    sentence being judged. A definition describes a sense; an exemplar shows it.

    Sentences are marked at the target span exactly as the input side is, so
    both halves of the pair look alike to the encoder.
    """
    pool: dict[str, list[str]] = {}
    for row in rows:
        # Raw corpus rows carry `gold_sense`; rows prepared by the training
        # script carry `sense_id`. Accept either, so this can be built from
        # whichever is at hand.
        sense = row.get("sense_id") or row.get("gold_sense")
        if not sense:
            continue
        pool.setdefault(sense, []).append(marked_sentence(row))
    rng = random.Random(seed)
    chosen: dict[str, list[str]] = {}
    for sense, sentences in pool.items():
        # Deterministic per sense, so two runs of the same corpus agree.
        rng.seed(f"{seed}:{sense}")
        rng.shuffle(sentences)
        chosen[sense] = sentences[:per_sense]
    return chosen


def candidate_exemplars(
    row: dict[str, Any],
    glosses: dict[str, dict[str, str]],
    exemplars: dict[str, list[str]],
    separator: str = " | ",
) -> list[tuple[str, str]]:
    """Right-hand side built from real usages, falling back to the definition.

    A sense with no exemplars — 55.7% of served coverage has no training data at
    all — keeps its gloss, so this never removes information, only adds it.
    """
    out: list[tuple[str, str]] = []
    for candidate in sorted(row["candidates"], key=lambda c: c["sense_id"]):
        sense = candidate["sense_id"]
        info = glosses.get(sense, {})
        stressed = info.get("stressed") or candidate.get("stressed") or ""
        # Never let the row being judged appear among its own evidence.
        usages = [s for s in exemplars.get(sense, ())
                  if s != marked_sentence(row)]
        if usages:
            body = separator.join(usages)
        else:
            body = info.get("definition") or ""
        out.append((sense, f"{stressed}: {body}".strip(": ")))
    return out


class PairDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], glosses: dict[str, dict[str, str]],
                 exemplars: dict[str, list[str]] | None = None) -> None:
        self.rows = rows
        self.glosses = glosses
        #: None keeps the definition-only behaviour every earlier run used.
        self.exemplars = exemplars

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pairs = (candidate_glosses(row, self.glosses) if self.exemplars is None
                 else candidate_exemplars(row, self.glosses, self.exemplars))
        gold = next((i for i, (sense, _) in enumerate(pairs) if sense == row["sense_id"]), -1)
        return {
            "sentence": marked_sentence(row),
            "pairs": pairs,
            "gold": gold,
            "group_id": row["group_id"],
            "sense_id": row["sense_id"],
        }


def make_collate(tokenizer: Any, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        left: list[str] = []
        right: list[str] = []
        spans: list[int] = []
        golds: list[int] = []
        for item in batch:
            spans.append(len(item["pairs"]))
            golds.append(item["gold"])
            for _, gloss in item["pairs"]:
                left.append(item["sentence"])
                right.append(gloss)
        encoded = tokenizer(
            left,
            right,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            "encoded": encoded,
            "spans": spans,
            "golds": torch.tensor(golds, dtype=torch.long),
            "meta": batch,
        }

    return collate


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _listwise_loss(logits: torch.Tensor, spans: list[int], golds: torch.Tensor) -> torch.Tensor:
    """Softmax across each row's candidates, cross-entropy against the gold one."""
    losses = []
    offset = 0
    for position, width in enumerate(spans):
        chunk = logits[offset : offset + width].view(1, -1)
        gold = golds[position].view(1)
        if gold.item() >= 0:
            losses.append(torch.nn.functional.cross_entropy(chunk, gold.to(chunk.device)))
        offset += width
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def predict(
    model: Any,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []
    for batch in loader:
        encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
        logits = model(**encoded).logits.squeeze(-1)
        offset = 0
        for position, width in enumerate(batch["spans"]):
            chunk = logits[offset : offset + width]
            order = torch.argsort(chunk, descending=True)
            meta = batch["meta"][position]
            best = int(order[0])
            margin = (
                float(chunk[order[0]] - chunk[order[1]]) if width > 1 else float("inf")
            )
            results.append(
                {
                    "predicted": meta["pairs"][best][0],
                    "gold": meta["sense_id"],
                    "group_id": meta["group_id"],
                    "correct": meta["pairs"][best][0] == meta["sense_id"],
                    "margin": margin,
                }
            )
            offset += width
    return results


@dataclass
class CrossEncoderRun:
    base_model: str
    corpus_version: str
    inventory_hash: str
    config: dict[str, Any]
    device: str
    library_versions: dict[str, str]
    hardware: dict[str, Any]
    parameters: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_dev_group_macro: float = 0.0
    seconds: float = 0.0


def train(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    glosses: dict[str, dict[str, str]],
    output_dir: Path,
    *,
    config: CrossEncoderConfig | None = None,
    corpus_version: str = "",
    inventory_hash: str = "",
    exemplars: dict[str, list[str]] | None = None,
) -> CrossEncoderRun:
    from ukstress_ml.train import group_macro_accuracy

    config = config or CrossEncoderConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=1
    )
    device = select_device()
    model.to(device)

    collate = make_collate(tokenizer, config.max_length)
    train_loader = DataLoader(
        PairDataset(train_rows, glosses, exemplars),
        batch_size=config.batch_rows,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        PairDataset(dev_rows, glosses, exemplars),
        batch_size=config.batch_rows,
        shuffle=False,
        collate_fn=collate,
    )

    steps = max(1, len(train_loader) * config.max_epochs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(steps * config.warmup_ratio), steps
    )

    run = CrossEncoderRun(
        base_model=config.base_model,
        corpus_version=corpus_version,
        inventory_hash=inventory_hash,
        config=asdict(config),
        device=str(device),
        library_versions={"torch": torch.__version__, "transformers": transformers.__version__},
        hardware={"platform": platform.platform(), "machine": platform.machine()},
        parameters=sum(p.numel() for p in model.parameters()),
    )
    print(
        f"device={device} base={config.base_model} params={run.parameters / 1e6:.0f}M "
        f"train={len(train_rows):,} dev={len(dev_rows):,}",
        flush=True,
    )

    best = -1.0
    since = 0
    started = time.time()
    for epoch in range(config.max_epochs):
        model.train()
        total = 0.0
        seen = 0
        amp = (
            torch.autocast(device_type=device.type, dtype=getattr(torch, config.amp_dtype))
            if config.amp_dtype
            else contextlib.nullcontext()
        )
        for index, batch in enumerate(train_loader):
            encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
            with amp:
                logits = model(**encoded).logits.squeeze(-1)
                loss = _listwise_loss(logits, batch["spans"], batch["golds"])
            (loss / config.accumulate).backward()
            if (index + 1) % config.accumulate == 0 or index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            seen += 1

            # An epoch on this data is hours long, and without a line inside it
            # a stalled run is indistinguishable from a slow one until it ends.
            if config.progress_every and (index + 1) % config.progress_every == 0:
                done = time.time() - started
                rate = (index + 1) / max(done, 1e-9)
                left = (len(train_loader) - index - 1) / max(rate, 1e-9)
                print(f"  epoch {epoch:>2}  step {index + 1:>6}/{len(train_loader)}  "
                      f"loss {total / seen:.4f}  {rate:.1f} step/s  "
                      f"{left / 60:.0f} min left in epoch", flush=True)

        results = predict(model, dev_loader, device)
        macro = group_macro_accuracy(results)
        micro = sum(1 for r in results if r["correct"]) / max(1, len(results))
        entry = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "dev_group_macro": macro,
            "dev_micro": micro,
            "seconds": time.time() - started,
        }
        run.history.append(entry)
        print(
            f"epoch {epoch:>2}  loss {entry['train_loss']:.4f}  dev_macro {macro:.4f}  "
            f"dev_micro {micro:.4f}  {entry['seconds'] / 60:.1f} min",
            flush=True,
        )

        if macro > best:
            best = macro
            run.best_epoch = epoch
            run.best_dev_group_macro = macro
            since = 0
            model.save_pretrained(output_dir / "checkpoint")
            tokenizer.save_pretrained(output_dir / "checkpoint")
        else:
            since += 1
            if since >= config.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    run.seconds = time.time() - started
    (output_dir / "training_run.json").write_text(
        json.dumps(asdict(run), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run
