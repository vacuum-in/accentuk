"""Import a flat accented wordlist as a secondary stress source.

The Wiktionary parser produces lexemes with paradigms and provenance. A flat
wordlist carries none of that: it is one inflected surface form per line, with
the stress already marked. It is nonetheless the asset that unblocks paradigm
coverage, because it enumerates forms the Wiktionary dump never contained.

Two properties of the input drive the encoding here, and both are checked
rather than assumed (see `audit_wordlist`):

* Every line is a single token — no spaces, no hyphens. A form carrying more
  than one acute is therefore recording *alternative* stresses of one token,
  not one stress per member of a compound. Each acute is split into its own
  `stress_variant`, which is what makes the form read as `ambiguous` through
  `stress_lookup` instead of silently resolving to one arbitrary reading.
* The source marks stress with U+00B4 ACUTE ACCENT, while the rest of this
  codebase canonicalizes to U+0301 COMBINING ACUTE ACCENT. The conversion
  happens once, here, before any normalizer call.

Conflict policy is **Wiktionary wins**: a form already carried by the primary
source is not touched, so importing this wordlist cannot change any stress that
serves correctly today. Only forms absent from the primary source are added.
Disagreements are not silently dropped — they are written to a conflict report.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ukstress.deduplicator import lexeme_key, stable_natural_key
from ukstress.normalizer import ACUTE, canonical_stressed_form, lookup_key, stress_signature
from ukstress.validator import classify_candidate

#: The spacing acute the wordlist uses, which NFD does not decompose.
SPACING_ACUTE = "´"

#: Namespaces wordlist lexemes inside the `lexeme` unique key so they can never
#: collide with a Wiktionary lexeme for the same spelling.
SOURCE_SECTION = "StressedWordlist"
SOURCE_KIND = "stressed_wordlist"

#: Below the Wiktionary parser's 0.95, so `stress_lookup`'s
#: `confidence DESC, source_rank` ordering puts Wiktionary first.
DEFAULT_CONFIDENCE = 0.90


def to_combining_acute(value: str) -> str:
    """Replace the wordlist's spacing acute with the canonical combining one."""
    return value.replace(SPACING_ACUTE, ACUTE)


def split_stress_alternatives(canonical: str) -> list[str]:
    """Return one single-stress spelling per acute in a multi-acute token.

    `за́мо́к` records two readings of one token, so it expands to `за́мок` and
    `замо́к`. Callers must only pass single tokens; for a compound such as
    `а́льфа-ро́зпад` the marks are simultaneous, not alternative, and splitting
    would invent readings that do not exist.
    """
    positions = [index for index, character in enumerate(canonical) if character == ACUTE]
    if len(positions) <= 1:
        return [canonical] if positions else []
    base = canonical.replace(ACUTE, "")
    alternatives = []
    for rank, position in enumerate(positions):
        insert_at = position - rank
        alternatives.append(base[:insert_at] + ACUTE + base[insert_at:])
    return alternatives


@dataclass(frozen=True)
class WordlistForm:
    """One surface form with every stress reading the wordlist records."""

    form_normalized: str
    stressed_forms: tuple[str, ...]

    @property
    def is_ambiguous(self) -> bool:
        return len({stress_signature(form) for form in self.stressed_forms}) > 1


@dataclass
class WordlistStats:
    """Counters for every line, so nothing is dropped without being counted."""

    lines: int = 0
    blank: int = 0
    unstressed: int = 0
    rejected: int = 0
    accepted_forms: int = 0
    ambiguous_forms: int = 0
    duplicate_lines: int = 0
    skipped_present_in_primary: int = 0
    conflicts: int = 0
    agreements: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        payload = {
            key: value for key, value in self.__dict__.items() if isinstance(value, int)
        }
        return payload


def audit_wordlist(path: Path) -> dict[str, int]:
    """Verify the single-token assumption the alternative-splitting relies on."""
    multi_token = 0
    hyphenated = 0
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            total += 1
            if any(character.isspace() for character in value):
                multi_token += 1
            if "-" in value:
                hyphenated += 1
    return {"forms": total, "multi_token": multi_token, "hyphenated": hyphenated}


def iter_wordlist_forms(path: Path, stats: WordlistStats) -> Iterator[WordlistForm]:
    """Yield deduplicated, validated forms in file order.

    Lines that normalize to the same lookup key (apostrophe variants, casing)
    are merged, and their stress readings unioned, rather than emitted twice.
    """
    seen: dict[str, set[str]] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stats.lines += 1
            raw = line.strip()
            if not raw:
                stats.blank += 1
                continue
            converted = to_combining_acute(raw)
            if ACUTE not in converted:
                stats.unstressed += 1
                continue
            canonical = canonical_stressed_form(converted)
            alternatives = split_stress_alternatives(canonical)
            accepted: list[str] = []
            for alternative in alternatives:
                validation = classify_candidate(alternative)
                if validation.classification != "valid":
                    stats.rejected += 1
                    for reason in validation.reasons or ("unknown",):
                        stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + 1
                    continue
                accepted.append(validation.canonical_form)
            if not accepted:
                continue
            key = lookup_key(accepted[0])
            if key in seen:
                stats.duplicate_lines += 1
            else:
                seen[key] = set()
                order.append(key)
            seen[key].update(accepted)

    for key in order:
        yield WordlistForm(form_normalized=key, stressed_forms=tuple(sorted(seen[key])))


def _signatures(stressed_forms: Iterable[str]) -> set[str]:
    return {stress_signature(form) for form in stressed_forms}


def build_form_records(
    form: WordlistForm,
    *,
    dataset_key: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Return the staging records for one wordlist form.

    A flat wordlist carries no lemma or paradigm, so each surface form becomes
    its own single-form lexeme. `stressed_lemma` stays null when the form is
    ambiguous: choosing one reading there would be inventing the answer the
    model tier exists to supply.
    """
    lexeme_natural_key = lexeme_key(
        dataset_key,
        form.form_normalized,
        "unknown",
        "0",
        form.form_normalized,
        SOURCE_SECTION,
    )
    word_form_natural_key = stable_natural_key(
        "word_form", lexeme_natural_key, form.form_normalized, ""
    )
    ambiguous = len(_signatures(form.stressed_forms)) > 1
    source_rank = round((1.0 - confidence) * 100)

    lexeme: dict[str, object] = {
        "natural_key": lexeme_natural_key,
        "lemma": form.form_normalized,
        "lemma_normalized": form.form_normalized,
        "stressed_lemma": None if ambiguous else form.stressed_forms[0],
        "part_of_speech": "unknown",
        "sense_key": "0",
        "source_title": form.form_normalized,
        "source_section": SOURCE_SECTION,
        "is_multiword": False,
        "is_obsolete": False,
        "confidence": confidence,
    }
    word_form: dict[str, object] = {
        "natural_key": word_form_natural_key,
        "lexeme_natural_key": lexeme_natural_key,
        "form": form.stressed_forms[0],
        "form_normalized": form.form_normalized,
        "grammatical_tags": [],
        "morphology_key": "",
        "is_lemma": True,
        "is_variant": False,
        "confidence": confidence,
        "source_rank": source_rank,
    }
    variants: list[dict[str, object]] = [
        {
            "natural_key": stable_natural_key(
                "stress_variant", word_form_natural_key, stressed, stress_signature(stressed)
            ),
            "word_form_natural_key": word_form_natural_key,
            "stressed_form": stressed,
            "stress_signature": stress_signature(stressed),
            "variant_type": "primary",
            "confidence": confidence,
        }
        for stressed in form.stressed_forms
    ]
    source_ref: dict[str, object] = {
        "natural_key": stable_natural_key(
            "source_ref", lexeme_natural_key, word_form_natural_key, SOURCE_KIND
        ),
        "lexeme_natural_key": lexeme_natural_key,
        "word_form_natural_key": word_form_natural_key,
        "page_id": None,
        "revision_id": None,
        "page_title": form.form_normalized,
        "source_kind": SOURCE_KIND,
        "source_section": SOURCE_SECTION,
        "source_fragment": " ".join(form.stressed_forms),
        "source_offset": None,
    }
    return lexeme, word_form, variants, source_ref


def normalized_acute(value: str) -> str:
    """Return NFD text for comparison regardless of which acute the source used."""
    return unicodedata.normalize("NFD", to_combining_acute(value))
