"""L2 generation for senses the corpus starves, and its validation.

Mining Ukrainian Wikipedia produced an almost single-sense corpus: 78% of
training groups carried only one sense, so a model can win by memorising
form -> dominant stress and never learn to read context. Generation exists to
supply the minority sense that natural text does not.

Generated sentences are validated here and still pass blind verification
afterwards, exactly like mined ones.
"""

from __future__ import annotations

import json
import random
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ukstress.deduplicator import stable_natural_key
from ukstress.normalizer import ACUTE, lookup_key

from ukstress_ml.ambiguity import AmbiguousForm
from ukstress_ml.mine import _WORD

# Sentences that talk *about* the word instead of using it leak the answer.
_METALINGUISTIC = re.compile(
    r"наголос|означа[єют]|у значенні|це слово|слово \"|слово «|омограф|"
    r"вимовля|лексем|словник",
    re.IGNORECASE,
)


class Rejected(ValueError):
    pass


def plan_requests(
    forms: dict[str, AmbiguousForm],
    have: dict[int, dict[str, int]],
    anchors: dict[tuple[int, str], list[str]],
    *,
    floor: int = 12,
    per_call: int = 12,
    topics: Iterable[str],
) -> list[dict[str, Any]]:
    """One request per starved (group, sense), capped at ``per_call`` sentences."""
    topic_cycle = list(topics)
    requests: list[dict[str, Any]] = []
    index = 0
    for form_key in sorted(forms):
        form = forms[form_key]
        counts = have.get(form.group_id, {})
        for candidate in form.candidates:
            shortfall = floor - counts.get(candidate.sense_id, 0)
            if shortfall <= 0:
                continue
            requests.append(
                {
                    "form": form_key,
                    "group_id": form.group_id,
                    "sense_id": candidate.sense_id,
                    "signature": candidate.signature,
                    "count": min(shortfall, per_call),
                    "topic": topic_cycle[index % len(topic_cycle)],
                    "anchors": anchors.get((form.group_id, candidate.sense_id), []),
                }
            )
            index += 1
    return requests


def validate_sentence(
    text: str,
    cue: str,
    form: AmbiguousForm,
    sense_id: str,
) -> tuple[str, int, int]:
    """Return ``(sentence, start, end)`` or raise ``Rejected``."""
    sentence = " ".join(text.split())
    if not sentence:
        raise Rejected("empty")
    if ACUTE in unicodedata.normalize("NFD", sentence):
        raise Rejected("carries_stress_marks")
    if _METALINGUISTIC.search(sentence):
        raise Rejected("metalinguistic")

    matches = [m for m in _WORD.finditer(sentence) if lookup_key(m.group(0)) == form.form]
    if len(matches) != 1:
        raise Rejected("target_not_exactly_once")

    token_count = len(_WORD.findall(sentence))
    if not 6 <= token_count <= 40:
        raise Rejected("length")

    # A sibling's stressed spelling appearing verbatim would hand over the answer.
    normalized = lookup_key(sentence)
    for candidate in form.candidates:
        if candidate.sense_id == sense_id:
            continue
        if lookup_key(candidate.stressed) != form.form and lookup_key(
            candidate.stressed
        ) in normalized.split():
            raise Rejected("sibling_form_present")

    if not cue.strip():
        raise Rejected("cue_absent")

    match = matches[0]
    return sentence, match.start(), match.end()


def parse_generated(
    raw_path: Path,
    forms: dict[str, AmbiguousForm],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse persisted generation responses into corpus-shaped rows."""
    from ukstress_ml.annotate import _extract_json

    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {"batches": 0, "proposed": 0, "accepted": 0}
    seen: set[str] = set()

    if not raw_path.exists():
        return rows, stats

    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            stats["batches"] += 1
            content = record.get("content")
            if not content:
                stats["no_content"] = stats.get("no_content", 0) + 1
                continue
            parsed = _extract_json(content)
            if not parsed or not isinstance(parsed.get("sentences"), list):
                stats["unparseable"] = stats.get("unparseable", 0) + 1
                continue

            form = forms.get(record["form"])
            spec = record.get("spec") or {}
            sense_id = spec.get("sense_id")
            if form is None or sense_id is None:
                continue

            for item in parsed["sentences"]:
                if not isinstance(item, dict):
                    continue
                stats["proposed"] += 1
                text = str(item.get("text", ""))
                cue = str(item.get("cue", ""))
                try:
                    sentence, start, end = validate_sentence(text, cue, form, sense_id)
                except Rejected as error:
                    key = f"rejected_{error.args[0]}"
                    stats[key] = stats.get(key, 0) + 1
                    continue

                sentence_id = stable_natural_key("generated_sentence", sentence, form.form)
                if sentence_id in seen:
                    stats["rejected_duplicate"] = stats.get("rejected_duplicate", 0) + 1
                    continue
                seen.add(sentence_id)

                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "sentence": sentence,
                        "form": form.form,
                        "surface": sentence[start:end],
                        "start": start,
                        "end": end,
                        "group_id": form.group_id,
                        "gen_sense_id": sense_id,
                        "cue": cue,
                        "register": str(item.get("register", "")),
                        "topic": spec.get("topic", ""),
                        "corpus": "generated",
                        "licence": "generated",
                        "source_tier": "generated",
                        "deployment": record.get("deployment"),
                    }
                )
                stats["accepted"] += 1
    return rows, stats

#: Topics are cycled across requests so a group's sentences do not all come
#: back from the same domain, which produced near-duplicate contexts the model
#: could memorise instead of reading.
TOPICS = ("побут", "історія", "техніка", "новини", "художня література",
          "природа", "наука")

SYSTEM = (
    "Ти — укладач корпусу для навчання моделі наголошування. "
    "Пишеш природні українські речення. Відповідаєш лише валідним JSON."
)


def build_prompt(form: AmbiguousForm, sense: Any, count: int, topic: str) -> str:
    """Ask for `count` sentences using `form` in exactly one of its senses.

    The sibling senses are listed and forbidden. Without that the model writes
    for whichever sense is more frequent, which is the imbalance generation
    exists to correct.
    """
    others = "\n".join(
        f"  - {c.definition}" for c in form.candidates if c.sense_id != sense.sense_id
    )
    return f"""Словоформа: {form.form}
Потрібне значення: {sense.definition}
Інші значення того самого написання (їх НЕ використовуй):
{others or "  (немає)"}

Напиши {count} різних речень, у яких «{form.form}» вжите САМЕ в потрібному значенні.

Правила:
- Форма «{form.form}» зустрічається в реченні РІВНО один раз, без знаків наголосу.
- Контекст має однозначно вказувати на потрібне значення.
- Не пояснюй слово, не вживай слів «наголос», «означає», «у значенні».
- 8–25 слів, різні синтаксичні конструкції. Тема: {topic}.
- "cue" — слова з речення (дослівно), які визначають значення.

Формат (лише JSON):
{{"sentences": [{{"text": "...", "cue": "..."}}]}}"""


def call_model(client: Any, model: str, prompt: str, usage: Any,
               max_tokens: int, attempts: int = 7) -> str:
    """One generation call, returning "" rather than raising.

    A dropped request costs sentences for one sense; raising would cost the
    whole run, and the caller records the shortfall either way.

    The backoff is exponential with jitter and deliberately patient. These
    deployments rate-limit hard — 12% of calls were being abandoned under a
    3-attempt, 5-second-linear policy — and an abandoned call is not a slower
    run but a sense that stays at zero rows, which is exactly the deficit
    generation exists to close.
    """
    for attempt in range(attempts):
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
            if model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = 0.9
            response = client.chat.completions.create(**kwargs)
            if response.usage:
                usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            if attempt == attempts - 1:
                usage.fail()
                return ""
            time.sleep(min(2 ** attempt, 60) + random.random() * 2)
    return ""


def harvest(content: str, form: AmbiguousForm,
            sense_id: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Validate one response, returning kept rows and a reason tally."""
    rejected: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        rejected["no_json"] += 1
        return kept, rejected
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        rejected["bad_json"] += 1
        return kept, rejected
    for item in payload.get("sentences", []):
        try:
            sentence, start, end = validate_sentence(
                str(item.get("text", "")), str(item.get("cue", "")), form, sense_id)
        except Rejected as reason:
            rejected[str(reason)] += 1
            continue
        kept.append({"sentence": sentence, "start": start, "end": end,
                     "cue": str(item.get("cue", "")), "form": form.form,
                     "group_id": form.group_id, "gold_sense": sense_id})
    return kept, rejected
