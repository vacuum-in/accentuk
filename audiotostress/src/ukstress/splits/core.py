"""Stable splits and reports that prevent speaker/form leakage."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ukstress.datasets import FeatureExample
from ukstress.provenance.fingerprints import content_fingerprint

SplitMap = dict[str, list[str]]


@dataclass(frozen=True)
class SplitReport:
    strategy: str
    seed: int
    record_counts: dict[str, int]
    group_counts: dict[str, int]
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def deterministic_split(
    record_ids: list[str],
    *,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> SplitMap:
    _validate_fractions(validation_fraction, test_fraction)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record_ids must be unique")
    assigned: SplitMap = {"train": [], "validation": [], "test": []}
    for record_id in sorted(record_ids):
        value = _fraction(seed, record_id)
        split = (
            "test"
            if value < test_fraction
            else "validation"
            if value < test_fraction + validation_fraction
            else "train"
        )
        assigned[split].append(record_id)
    return assigned


def speaker_held_out_split(
    examples: list[FeatureExample],
    *,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> SplitMap:
    return _grouped_split(
        examples, lambda item: item.speaker_id, "speaker", seed, validation_fraction, test_fraction
    )


def target_word_held_out_split(
    examples: list[FeatureExample],
    *,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> SplitMap:
    return _grouped_split(
        examples,
        lambda item: item.target_word,
        "target_word",
        seed,
        validation_fraction,
        test_fraction,
    )


def _grouped_split(
    examples: list[FeatureExample],
    group: Callable[[FeatureExample], str | None],
    strategy: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> SplitMap:
    _validate_fractions(validation_fraction, test_fraction)
    groups: dict[str, list[str]] = defaultdict(list)
    for item in examples:
        key = group(item)
        if not key:
            raise ValueError(f"{strategy} split requires every record to have a group value")
        groups[key].append(item.record_id)
    group_splits = deterministic_split(
        list(groups),
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    result: SplitMap = {name: [] for name in group_splits}
    for split, identifiers in group_splits.items():
        result[split] = sorted(
            record_id for identifier in identifiers for record_id in groups[identifier]
        )
    validate_group_leakage(result, examples, group)
    return result


def validate_group_leakage(
    splits: SplitMap, examples: list[FeatureExample], group: Callable[[FeatureExample], str | None]
) -> None:
    by_id = {item.record_id: item for item in examples}
    seen: dict[str, str] = {}
    for split, record_ids in splits.items():
        for record_id in record_ids:
            item = by_id.get(record_id)
            if item is None:
                raise ValueError(f"split references unknown record {record_id}")
            key = group(item)
            if not key:
                raise ValueError("record has no leakage group")
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError(f"leakage group {key!r} appears in {previous} and {split}")


def balance_by_lexeme(
    examples: list[FeatureExample], *, maximum_per_lexeme: int
) -> list[FeatureExample]:
    if maximum_per_lexeme < 1:
        raise ValueError("maximum_per_lexeme must be positive")
    counts: Counter[str] = Counter()
    selected: list[FeatureExample] = []
    for item in sorted(examples, key=lambda value: (value.target_word, value.record_id)):
        if counts[item.target_word] >= maximum_per_lexeme:
            continue
        selected.append(item)
        counts[item.target_word] += 1
    return selected


def split_report(
    splits: SplitMap,
    examples: list[FeatureExample],
    *,
    strategy: str,
    seed: int,
    group: Callable[[FeatureExample], str | None],
) -> SplitReport:
    by_id = {item.record_id: item for item in examples}
    record_counts = {name: len(ids) for name, ids in splits.items()}
    group_counts = {
        name: len({group(by_id[record_id]) for record_id in ids}) for name, ids in splits.items()
    }
    fingerprint = content_fingerprint(
        {"strategy": strategy, "seed": seed, "splits": splits}, namespace="split-report"
    )
    return SplitReport(strategy, seed, record_counts, group_counts, fingerprint)


def _fraction(seed: int, key: str) -> float:
    digest = content_fingerprint({"seed": seed, "key": key}, namespace="split-assignment")
    return int(digest.removeprefix("sha256:")[:16], 16) / 2**64


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must leave training data")
