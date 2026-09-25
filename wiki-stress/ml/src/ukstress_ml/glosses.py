"""Author the missing sense definitions that gate coverage extension.

The cross-encoder scores `(sentence, gloss)` pairs. A candidate whose gloss is
empty is scored against an empty string, so it cannot win on merit no matter how
clear the context is. After paradigm expansion, 11,670 of 27,131 candidates
(43%) are in that state, across 790 groups — which is `RESULTS.md` §7's
"1,900 senses still lack a definition ... this gates any coverage extension",
measured at paradigm scale.

Glosses are authored **per group**, never per form: every inflected form of a
group shares its senses, so 790 calls fix 7,319 forms.

Two rules this module enforces, both from `PLAN_MARIAN_HOMOGRAPHS.md` §4.2:

* A gloss is a *label definition* — the thing the model is scored against — so
  an LLM-written one is marked `needs_review` and never silently becomes ground
  truth.
* The gloss must describe the sense without naming the stress. A gloss that
  says "наголос на другому складі" leaks the answer into the model's input and
  would train it to read the gloss rather than the context.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACUTE = "́"

#: Phrasing that describes the *stress* rather than the meaning. A gloss
#: containing any of these leaks the label.
#: "склад" is deliberately NOT matched on its own. It means *syllable*, but it
#: equally means *composition* — "належати до складу" ("to belong to the
#: composition") is an ordinary definition, and matching the bare word rejected
#: 1 of every 78 groups for describing stress when it did nothing of the kind.
#: Only an ordinal before it ("на першому складі") is actually metalinguistic.
_METALINGUISTIC = re.compile(
    r"наголос|наголош|вимовля"
    r"|(перш|друг|трет|четверт|останн|передостанн)\w*\s+склад",
    re.IGNORECASE,
)

SYSTEM = (
    "Ти — український лексикограф. Пишеш стислі тлумачення значень слів "
    "для навчального корпусу. Відповідаєш лише валідним JSON."
)


@dataclass(frozen=True)
class GlossTask:
    group_id: int
    missing: tuple[dict[str, Any], ...]
    known_siblings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AuthoredGloss:
    group_id: int
    sense_id: str
    definition: str
    review_status: str = "llm_proposed"


class GlossError(ValueError):
    """Raised when an authored gloss violates the corpus contract."""


def build_prompt(task: GlossTask) -> str:
    missing = "\n".join(
        f"  {m['sense_id']} — форма: {m['stressed_example']} — частина мови: {m.get('pos') or 'невідомо'}"
        for m in task.missing
    )
    siblings = (
        "\n".join(f"  {s['sense_id']} — {s['definition']}" for s in task.known_siblings)
        or "  (немає)"
    )
    return f"""Група омографів {task.group_id}: одне написання, різні значення.

Значення з відомим тлумаченням:
{siblings}

Значення БЕЗ тлумачення (потрібно написати):
{missing}

Для кожного значення без тлумачення напиши стисле тлумачення українською.
Правила:
- 5–20 слів, називає ЗНАЧЕННЯ (що слово означає), а не вимову.
- НЕ згадуй наголос, склади чи вимову — тлумачення має описувати зміст.
- Якщо в групі є інші значення, тлумачення має чітко відрізняти це значення від них.
- Якщо значення неможливо визначити, постав "unclear": true.

Формат (лише JSON):
{{"items": [{{"sense_id": "...", "definition": "...", "unclear": false}}]}}"""


def validate_gloss(definition: str, *, surface_forms: Iterable[str] = ()) -> str:
    """Return the cleaned gloss or raise if it violates the contract."""
    # A gloss that spells a word with its acute is not describing the stress,
    # it is just quoting the word -- but leaving the mark in would hand the
    # model the answer it is supposed to infer from context. Strip it and keep
    # the definition; only *metalinguistic* glosses are actually unusable.
    text = " ".join(
        unicodedata.normalize("NFC", unicodedata.normalize("NFD", definition).replace(ACUTE, ""))
        .split()
    )
    if not text:
        raise GlossError("empty gloss")
    if len(text) < 8:
        raise GlossError(f"gloss too short: {text!r}")
    if _METALINGUISTIC.search(text):
        raise GlossError(f"gloss describes stress rather than meaning: {text!r}")
    # Naming the target inside a definition is fine; a "definition" that is only
    # the word, however many times it is repeated, defines nothing.
    words = unicodedata.normalize("NFD", text).replace(ACUTE, "").lower().split()
    for surface in surface_forms:
        bare = unicodedata.normalize("NFD", surface).replace(ACUTE, "").lower()
        if not bare or len(bare) <= 3:
            continue
        remainder = [word for word in words if word.strip(".,;:!?()") != bare]
        if len("".join(remainder)) < 8:
            raise GlossError("gloss restates the word without defining it")
    return text


def parse_response(task: GlossTask, content: str) -> tuple[list[AuthoredGloss], list[str]]:
    """Parse one response, returning accepted glosses and rejection reasons."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return [], ["response contained no JSON object"]
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError as error:
        return [], [f"invalid JSON: {error}"]

    wanted = {str(m["sense_id"]): m for m in task.missing}
    accepted: list[AuthoredGloss] = []
    rejected: list[str] = []
    for item in payload.get("items", []):
        sense_id = str(item.get("sense_id", ""))
        if sense_id not in wanted:
            rejected.append(f"{sense_id}: not a requested sense")
            continue
        if item.get("unclear"):
            rejected.append(f"{sense_id}: model declined (unclear)")
            continue
        try:
            definition = validate_gloss(
                str(item.get("definition", "")),
                surface_forms=[str(wanted[sense_id].get("stressed_example", ""))],
            )
        except GlossError as error:
            rejected.append(f"{sense_id}: {error}")
            continue
        accepted.append(AuthoredGloss(task.group_id, sense_id, definition))
    missing_ids = {m["sense_id"] for m in task.missing} - {a.sense_id for a in accepted}
    rejected.extend(f"{sense_id}: no usable item returned" for sense_id in sorted(missing_ids)
                    if not any(sense_id in r for r in rejected))
    return accepted, rejected


def load_tasks(path: Path) -> Iterator[GlossTask]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            yield GlossTask(
                group_id=int(data["group_id"]),
                missing=tuple(data["missing"]),
                known_siblings=tuple(data.get("known_siblings", ())),
            )


def apply_glosses(
    inventory_path: Path, glosses: Iterable[AuthoredGloss], output_path: Path
) -> dict[str, int]:
    """Fill authored glosses into an inventory, marking them for review."""
    by_key = {(g.group_id, g.sense_id): g for g in glosses}
    filled = untouched = still_missing = 0
    with inventory_path.open(encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            for candidate in row["candidates"]:
                if candidate.get("definition"):
                    untouched += 1
                    continue
                gloss = by_key.get((int(row["group_id"]), str(candidate["sense_id"])))
                if gloss is None:
                    still_missing += 1
                    continue
                candidate["definition"] = gloss.definition
                candidate["review_status"] = gloss.review_status
                filled += 1
            row["complete"] = all(c["definition"] for c in row["candidates"])
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"filled": filled, "already_present": untouched, "still_missing": still_missing}
