"""Manual ambiguous-occurrence records kept separate from automatic labels."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ukstress.lexicon.interface import StressLexicon
from ukstress.text import normalize_transcript


class ManualGoldOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    target_word: str = Field(min_length=1)
    selected_vowel_index: int = Field(ge=0)
    context_original: str = Field(min_length=1)
    context_normalized: str = Field(min_length=1)
    speaker_id: str | None = None
    annotator_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def target_is_normalized(self) -> ManualGoldOccurrence:
        if self.target_word != normalize_transcript(self.target_word):
            raise ValueError("target_word must be normalized")
        return self


def import_manual_gold(path: str | Path) -> list[ManualGoldOccurrence]:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".jsonl":
        raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    elif suffix == ".csv":
        with source.open(encoding="utf-8", newline="") as stream:
            raw = list(csv.DictReader(stream))
    else:
        raise ValueError("manual gold input must be CSV or JSONL")
    return [ManualGoldOccurrence.model_validate(item) for item in raw]


def validate_manual_gold(records: list[ManualGoldOccurrence], lexicon: StressLexicon) -> None:
    seen_record_ids: set[str] = set()
    seen_contexts: set[tuple[str | None, str, str]] = set()
    for record in records:
        if record.record_id in seen_record_ids:
            raise ValueError(f"duplicate manual-gold record_id: {record.record_id}")
        seen_record_ids.add(record.record_id)
        candidates = lexicon.lookup(record.target_word)
        if record.selected_vowel_index not in {candidate.vowel_index for candidate in candidates}:
            raise ValueError("selected stress is not a lexicon candidate")
        context_key = (record.speaker_id, record.target_word, record.context_normalized)
        if context_key in seen_contexts:
            raise ValueError("duplicate speaker/form/context manual-gold annotation")
        seen_contexts.add(context_key)


def manual_gold_report(records: list[ManualGoldOccurrence]) -> dict[str, object]:
    forms: dict[str, Counter[int]] = defaultdict(Counter)
    speakers: Counter[str] = Counter()
    for record in records:
        forms[record.target_word][record.selected_vowel_index] += 1
        if record.speaker_id:
            speakers[record.speaker_id] += 1
    return {
        "records": len(records),
        "forms": {word: dict(sorted(variants.items())) for word, variants in sorted(forms.items())},
        "speakers": dict(sorted(speakers.items())),
    }
