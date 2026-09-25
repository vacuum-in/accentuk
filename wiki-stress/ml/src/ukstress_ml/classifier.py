"""Tier 3 as classification: predict the stressed vowel, not a matching definition.

The cross-encoder scores `(sentence, sense definition)` pairs. Two measurements
say that formulation is the ceiling rather than the data:

* Forms with 20+ training rows score 73.8% on the benchmark; forms the model has
  never seen score 75.6%. Per-form supervision updates a similarity function
  shared by all 15,332 forms and never becomes that form's own boundary.
* On semantic homographs -- the only thing gloss matching is built for -- it
  scores 69.1%, its worst class.

RUAccent, which reports 0.9637 on Russian homographs, uses "a transformer
encoder with a linear layer on the head" and never matched against definitions.
This module is that: encode the sentence, pool the target span, project to the
small global vocabulary of stress signatures, mask to the form's candidates.

Two deliberate differences from the pair model:

*No span markers.* The pair model wraps the target in `⟦ ⟧`, which XLM-R's
tokenizer maps to `<unk>`. Here the span is located through the tokenizer's
offset mapping instead, so nothing is inserted into the text and two occurrences
of the same word in one sentence are separable -- the failure RUAccent lists as
its own limitation.

*Masking, not enumeration.* A form's candidates select which logits compete, so
the label is the answer itself and the model is never asked to score a reading
the lexicon does not offer.
"""

from __future__ import annotations

import json
import platform
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

#: Every stress signature the lexicon uses, in a fixed order. Seven values cover
#: the whole manifest; `0|1` is the wordlist's "either is acceptable" notation
#: and is kept so a form carrying it can still be represented.
SIGNATURES: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "0|1")
SIGNATURE_INDEX = {s: i for i, s in enumerate(SIGNATURES)}
NEGATIVE = torch.finfo(torch.float32).min


@dataclass
class ClassifierConfig:
    base_model: str = "xlm-roberta-base"
    max_length: int = 192
    batch_rows: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_epochs: int = 2
    patience: int = 3
    seed: int = 20260901
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    amp_dtype: str | None = None
    accumulate: int = 1


@dataclass
class ClassifierRun:
    base_model: str
    corpus_version: str
    config: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_dev_group_macro: float = 0.0
    seconds: float = 0.0
    parameters: int = 0
    device: str = ""


class SignatureClassifier(nn.Module):
    """Encoder plus a linear head over the signature vocabulary."""

    def __init__(self, base_model: str, labels: int = len(SIGNATURES)) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                span_mask: torch.Tensor) -> torch.Tensor:
        states = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        # Mean-pool the tokens covering the target word. A sentence-level vector
        # could not tell two occurrences of the same form apart.
        weights = span_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.head(self.dropout(pooled))


def candidate_mask(rows: Sequence[dict[str, Any]]) -> torch.Tensor:
    """True where a signature is one this form actually offers."""
    mask = torch.zeros(len(rows), len(SIGNATURES), dtype=torch.bool)
    for i, row in enumerate(rows):
        for candidate in row["candidates"]:
            index = SIGNATURE_INDEX.get(candidate["signature"])
            if index is not None:
                mask[i, index] = True
    return mask


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Push every signature the form does not offer out of the softmax."""
    return logits.masked_fill(~mask.to(logits.device), NEGATIVE)


class SpanDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def make_collate(tokenizer: Any, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["sentence"] for row in batch],
            truncation=True, max_length=max_length, padding=True,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        span = torch.zeros(offsets.shape[:2], dtype=torch.bool)
        for i, row in enumerate(batch):
            start, end = row["start"], row["end"]
            for j, (a, b) in enumerate(offsets[i].tolist()):
                # (0, 0) marks a special token; overlap decides membership.
                if a != b and a < end and b > start:
                    span[i, j] = True
            if not span[i].any():
                # Truncation cut the target off. Fall back to the whole
                # sequence rather than dividing by zero; the row is still
                # counted, so the loss reflects that the input was unusable.
                span[i] = encoded["attention_mask"][i].bool()
        golds = torch.tensor(
            [SIGNATURE_INDEX[row["gold_signature"]] for row in batch], dtype=torch.long
        )
        return {"encoded": dict(encoded), "span": span,
                "mask": candidate_mask(batch), "golds": golds, "meta": batch}

    return collate


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: SignatureClassifier, loader: DataLoader,
             device: torch.device) -> dict[str, Any]:
    """Micro accuracy, and macro over homograph groups so a frequent form
    cannot hide a rare one."""
    model.eval()
    correct = total = 0
    by_group: dict[Any, list[int]] = {}
    with torch.inference_mode():
        for batch in loader:
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            logits = masked_logits(
                model(encoded["input_ids"], encoded["attention_mask"],
                      batch["span"].to(device)),
                batch["mask"],
            )
            picked = logits.argmax(dim=-1).cpu()
            for row, got, want in zip(batch["meta"], picked.tolist(),
                                      batch["golds"].tolist(), strict=True):
                hit = int(got == want)
                correct += hit
                total += 1
                by_group.setdefault(row.get("group_id"), []).append(hit)
    macro = (sum(sum(v) / len(v) for v in by_group.values()) / len(by_group)
             if by_group else 0.0)
    return {"micro": correct / total if total else 0.0, "group_macro": macro,
            "rows": total}


def train(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]],
          output_dir: Path, *, config: ClassifierConfig | None = None,
          corpus_version: str = "") -> ClassifierRun:
    config = config or ClassifierConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = SignatureClassifier(config.base_model).to(device)
    collate = make_collate(tokenizer, config.max_length)

    train_loader = DataLoader(SpanDataset(train_rows), batch_size=config.batch_rows,
                              shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(SpanDataset(dev_rows), batch_size=config.batch_rows,
                            shuffle=False, collate_fn=collate)

    steps = max(1, len(train_loader) * config.max_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(steps * config.warmup_ratio), steps)
    loss_fn = nn.CrossEntropyLoss()

    run = ClassifierRun(base_model=config.base_model, corpus_version=corpus_version,
                        config=asdict(config), device=str(device),
                        parameters=sum(p.numel() for p in model.parameters()))
    started = time.time()
    best = -1.0
    for epoch in range(config.max_epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            logits = masked_logits(
                model(encoded["input_ids"], encoded["attention_mask"],
                      batch["span"].to(device)),
                batch["mask"],
            )
            loss = loss_fn(logits, batch["golds"].to(device)) / config.accumulate
            loss.backward()
            running += loss.detach().item() * config.accumulate
            if (step + 1) % config.accumulate == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        scores = evaluate(model, dev_loader, device)
        run.history.append({"epoch": epoch, "train_loss": running / max(len(train_loader), 1),
                            "dev_micro": scores["micro"],
                            "dev_group_macro": scores["group_macro"],
                            "seconds": time.time() - started})
        print(f"epoch {epoch:2}  loss {running/max(len(train_loader),1):.4f}  "
              f"dev_macro {scores['group_macro']:.4f}  dev_micro {scores['micro']:.4f}  "
              f"{(time.time()-started)/60:.1f} min", flush=True)
        if scores["group_macro"] > best:
            best = scores["group_macro"]
            run.best_epoch, run.best_dev_group_macro = epoch, best
            model.encoder.save_pretrained(output_dir / "checkpoint")
            tokenizer.save_pretrained(output_dir / "checkpoint")
            torch.save(model.head.state_dict(), output_dir / "checkpoint" / "head.pt")

    run.seconds = time.time() - started
    (output_dir / "training_run.json").write_text(
        json.dumps({**asdict(run), "signatures": list(SIGNATURES),
                    "platform": platform.platform()},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run


def prepare_rows(rows: Iterable[dict[str, Any]],
                 manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach each row's candidate signatures and its gold signature.

    A row whose gold sense is not among the form's candidates is dropped: the
    masked softmax could never select it, so keeping it would train the model
    against an unreachable target.
    """
    out = []
    for row in rows:
        entry = manifest.get(row["form"])
        if not entry:
            continue
        # Rows mined for forms with no dictionary senses carry the signature
        # directly; corpus rows carry a sense that has to be looked up.
        gold = row.get("gold_signature") or next(
            (c["signature"] for c in entry["candidates"]
             if c["sense_id"] == row.get("gold_sense")), None)
        if gold is None or gold not in SIGNATURE_INDEX:
            continue
        if gold not in {c["signature"] for c in entry["candidates"]}:
            continue
        out.append({**row, "gold_signature": gold,
                    "candidates": [{"signature": c["signature"]}
                                   for c in entry["candidates"]]})
    return out
