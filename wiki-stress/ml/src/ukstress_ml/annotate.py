"""Azure AI Foundry annotation: sense labelling and blind verification.

Two jobs, deliberately separate and on different deployments, so a model never
confirms its own output:

* **label** — reads a mined sentence and picks the inventory sense it uses.
  Declining (``unclear``) is a first-class answer.
* **verify** — reads the same sentence with the target token **masked** and no
  intended label, and picks a sense. If it cannot recover the label, the context
  did not disambiguate, whoever wrote it.

Every raw response is written before parsing, so a parser change costs a
re-parse rather than a re-spend.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI

from ukstress_ml.ambiguity import AmbiguousForm

MASK = "▁▁▁"
_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


@dataclass(frozen=True)
class Deployment:
    """One Azure AI Foundry deployment, resolved from the environment.

    Variable names are tried in order, because a Foundry project may be
    configured under either the `AZURE_OPENAI_*` names this pipeline was first
    written against or the `AZURE_FOUNDRY_OPENAI_*` names the portal now emits.
    `model_var` lets the deployed model name come from the environment too --
    the literal in `name` is only a default, and the real deployment is rarely
    called what the code guessed.
    """

    name: str
    endpoint_var: str
    key_var: str
    fallback_endpoint_vars: tuple[str, ...] = ()
    fallback_key_vars: tuple[str, ...] = ()
    model_var: str | None = None

    @staticmethod
    def _lookup(config: dict[str, str | None], *names: str) -> str | None:
        for candidate in names:
            value = config.get(candidate) or os.environ.get(candidate)
            if value:
                return value
        return None

    def resolved_model(self, config: dict[str, str | None]) -> str:
        """The deployment name actually sent as the `model` field."""
        if self.model_var:
            override = self._lookup(config, self.model_var)
            if override:
                return override
        return self.name

    def client(self, config: dict[str, str | None]) -> OpenAI:
        endpoint = self._lookup(config, self.endpoint_var, *self.fallback_endpoint_vars)
        key = self._lookup(config, self.key_var, *self.fallback_key_vars)
        if not endpoint or not key:
            expected = " or ".join((self.endpoint_var, *self.fallback_endpoint_vars))
            expected_key = " or ".join((self.key_var, *self.fallback_key_vars))
            raise RuntimeError(
                f"no credentials for deployment {self.name!r}: set {expected} "
                f"and {expected_key}"
            )
        return OpenAI(base_url=endpoint.rstrip("/"), api_key=key, timeout=180.0, max_retries=3)


# The V4-Flash deployment on the initaitesting resource returns 429 at even
# modest concurrency, so labelling runs on the resource that has headroom.
# Labelling and verification still use different deployments.
#: Names the Foundry portal emits, tried after the originals.
_FOUNDRY_ENDPOINT = ("AZURE_FOUNDRY_OPENAI_BASE_URL",)
_FOUNDRY_KEY = ("AZURE_FOUNDRY_OPENAI_API_KEY",)

LABEL_DEPLOYMENT = Deployment(
    name="DeepSeek-V4-Pro",
    endpoint_var="AZURE_OPENAI_ENDPOINT",
    key_var="AZURE_OPENAI_API_KEY",
    fallback_endpoint_vars=_FOUNDRY_ENDPOINT,
    fallback_key_vars=_FOUNDRY_KEY,
    model_var="AZURE_OPENAI_DEPLOYMENT",
)
FLASH_DEPLOYMENT = Deployment(
    name="DeepSeek-V4-Flash",
    endpoint_var="AZURE_OPENAI_ENDPOINT_INITAITESTING",
    key_var="AZURE_OPENAI_API_KEY_INITAITESTING",
    model_var="AZURE_OPENAI_DEPLOYMENT_INITAITESTING",
)
# Verification must not run on the deployment that produced the labels, or a
# model confirms its own output. `run_job` refuses when they resolve equal.
#
# Deliberately no `_FOUNDRY_*` fallback here: if it shared the label
# deployment's fallback, a single-deployment environment would resolve both to
# the same model and blind verification would quietly become self-verification.
# Failing to find credentials is the correct outcome until a second deployment
# is configured — blind verify rejected 12.9% of labels and was the strongest
# filter in the corpus build.
VERIFY_DEPLOYMENT = Deployment(
    name="DeepSeek-V3.2",
    endpoint_var="AZURE_OPENAI_ENDPOINT_VERIFY",
    key_var="AZURE_OPENAI_API_KEY_VERIFY",
    model_var="AZURE_OPENAI_DEPLOYMENT_VERIFY",
)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    def fail(self) -> None:
        with self._lock:
            self.failures += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failures": self.failures,
        }


def load_config(env_path: Path = Path(".env")) -> dict[str, str | None]:
    return dotenv_values(env_path)


def _sense_block(form: AmbiguousForm, *, with_stress: bool = True) -> str:
    lines = []
    for candidate in form.candidates:
        head = candidate.stressed if with_stress else candidate.sense_id
        definition = candidate.definition or "(без тлумачення)"
        lines.append(f"  {candidate.sense_id} — {head} — {candidate.pos} — {definition}")
    return "\n".join(lines)


LABEL_SYSTEM = (
    "Ти — лінгвіст-анотатор українського корпусу. Визначаєш, у якому значенні "
    "вжите багатозначне слово в реченні. Відповідаєш лише валідним JSON."
)

VERIFY_SYSTEM = (
    "Ти — лінгвіст-анотатор українського корпусу. У реченні одне слово приховане "
    f"маркером {MASK}. Визначаєш, яке зі значень підходить у цьому контексті. "
    "Відповідаєш лише валідним JSON."
)


def label_prompt(form: AmbiguousForm, sentences: list[dict[str, Any]]) -> str:
    numbered = "\n".join(
        f"{i}. {row['sentence']}" for i, row in enumerate(sentences)
    )
    return f"""Написання: {form.form}
Значення (id — наголошена форма — частина мови — тлумачення):
{_sense_block(form)}

Речення:
{numbered}

Для кожного речення визнач, яке значення вжите.
Якщо контекст НЕ дозволяє однозначно визначити значення — постав "unclear": true.
"cue" — слова з речення, які визначають вибір (дослівно з речення).

Формат відповіді (лише JSON, без пояснень):
{{"items": [{{"i": 0, "sense_id": "...", "confidence": 0.0-1.0, "cue": "...", "unclear": false}}]}}"""


def verify_prompt(form: AmbiguousForm, sentences: list[dict[str, Any]]) -> str:
    numbered = "\n".join(
        f"{i}. {row['masked']}" for i, row in enumerate(sentences)
    )
    return f"""Приховане слово має написання: {form.form}
Можливі значення (id — частина мови — тлумачення):
{_sense_block(form, with_stress=False)}

Речення (приховане слово позначене {MASK}):
{numbered}

Для кожного речення визнач, яке значення стоїть на місці {MASK}.
Якщо контекст не дозволяє визначити — "unclear": true.

Формат відповіді (лише JSON, без пояснень):
{{"items": [{{"i": 0, "sense_id": "...", "confidence": 0.0-1.0, "unclear": false}}]}}"""


GENERATE_SYSTEM = (
    "Ти — укладач корпусу для навчання моделі наголошування. Пишеш природні "
    "українські речення. Відповідаєш лише валідним JSON."
)

TOPICS = (
    "побут",
    "історія",
    "техніка",
    "новини",
    "художня література",
    "природа",
    "місто",
    "робота",
)


def generate_prompt(form: AmbiguousForm, request: list[dict[str, Any]]) -> str:
    """One call per (form, target sense). ``request`` carries a single spec."""
    spec = request[0]
    anchors = spec.get("anchors") or []
    anchor_block = ""
    if anchors:
        listed = "\n".join(f"  - {text}" for text in anchors[:3])
        anchor_block = f"\nПриклади реального вжитку цього значення (наслідуй стиль):\n{listed}\n"

    return f"""Написання: {form.form}
Значення (id — наголошена форма — частина мови — тлумачення):
{_sense_block(form)}

Цільове значення: {spec["sense_id"]}
Тема для різноманітності: {spec["topic"]}
{anchor_block}
Напиши {spec["count"]} різних українських речень, у яких слово "{form.form}" вжите
САМЕ у значенні {spec["sense_id"]}.

Правила:
- Кожне речення містить слово "{form.form}" РІВНО один раз, без знаків наголосу.
- Контекст має однозначно вказувати на значення {spec["sense_id"]}, а не на інші значення.
- Не пояснюй слово, не пиши про наголос чи про значення — просто вживай слово.
- Різні синтаксичні конструкції, різна довжина (8–25 слів).
- "cue" — слова саме з цього речення, які роблять значення однозначним.

Формат відповіді (лише JSON):
{{"sentences": [{{"text": "...", "cue": "...", "register": "neutral"}}]}}"""


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, list):
        return {"items": parsed}
    return parsed if isinstance(parsed, dict) else None


def _call(
    client: OpenAI,
    deployment: str,
    system: str,
    prompt: str,
    usage: Usage,
    *,
    max_tokens: int,
    temperature: float,
    attempts: int = 6,
) -> tuple[str | None, dict[str, int]]:
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as error:  # noqa: BLE001 -- transient service errors are retried
            # A 429 means the deployment is saturated, so back off much harder
            # than for an ordinary transient failure; retrying immediately just
            # burns the attempt budget and loses the batch.
            rate_limited = "429" in str(error) or "RateLimit" in type(error).__name__
            delay = (6.0 * (attempt + 1)) if rate_limited else (2.0 * (attempt + 1))
            time.sleep(delay + random.uniform(0, 2.0))
            continue
        counts = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        usage.add(counts["prompt_tokens"], counts["completion_tokens"])
        return response.choices[0].message.content, counts
    usage.fail()
    return None, {"prompt_tokens": 0, "completion_tokens": 0}


def _batches(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _completed_keys(raw_path: Path) -> set[str]:
    """Batch keys that returned usable content.

    A batch whose call failed is recorded so the attempt is auditable, but it
    must not count as done — otherwise a transient service error silently
    removes those sentences from the corpus for good.
    """
    if not raw_path.exists():
        return set()
    keys: set[str] = set()
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("content"):
                keys.add(record.get("batch_key", ""))
    keys.discard("")
    return keys


def run_job(
    *,
    job: str,
    rows: list[dict[str, Any]],
    forms: dict[str, AmbiguousForm],
    deployment: Deployment,
    config: dict[str, str | None],
    raw_path: Path,
    batch_size: int = 12,
    workers: int = 8,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    prompt_builder: Callable[[AmbiguousForm, list[dict[str, Any]]], str] | None = None,
    system: str = "",
    progress_every: int = 25,
) -> Usage:
    """Run one annotation job, resuming from persisted raw responses."""
    if job == "verify":
        # Compare *resolved* names: two Deployment objects with different
        # literals can still point at one model once the environment is
        # applied, and that collision is exactly what makes verification
        # worthless.
        verifier = deployment.resolved_model(config)
        labeller = LABEL_DEPLOYMENT.resolved_model(config)
        if verifier == labeller:
            raise RuntimeError(
                f"blind verification would run on {verifier!r}, the same model that "
                "produced the labels — a model confirming its own output is not "
                "verification. Configure a second deployment via "
                "AZURE_OPENAI_ENDPOINT_VERIFY / AZURE_OPENAI_API_KEY_VERIFY / "
                "AZURE_OPENAI_DEPLOYMENT_VERIFY."
            )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(raw_path)
    client = deployment.client(config)
    model_name = deployment.resolved_model(config)
    usage = Usage()

    by_form: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_form[row["form"]].append(row)

    tasks: list[tuple[str, str, list[dict[str, Any]]]] = []
    for form_key in sorted(by_form):
        for index, batch in enumerate(_batches(by_form[form_key], batch_size)):
            batch_key = f"{job}:{form_key}:{index}"
            if batch_key not in done:
                tasks.append((batch_key, form_key, batch))

    if not tasks:
        return usage

    write_lock = threading.Lock()
    handle = raw_path.open("a", encoding="utf-8")
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for batch_key, form_key, batch in tasks:
                form = forms[form_key]
                assert prompt_builder is not None
                prompt = prompt_builder(form, batch)
                futures[
                    pool.submit(
                        _call,
                        client,
                        model_name,
                        system,
                        prompt,
                        usage,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                ] = (batch_key, form_key, batch)

            for future in as_completed(futures):
                batch_key, form_key, batch = futures[future]
                content, counts = future.result()
                record = {
                    "batch_key": batch_key,
                    "job": job,
                    "form": form_key,
                    "deployment": model_name,
                    "sentence_ids": [row["sentence_id"] for row in batch],
                    "content": content,
                    "usage": counts,
                }
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
                    if progress_every and completed % progress_every == 0:
                        snapshot = usage.snapshot()
                        print(
                            f"  {job}: {completed}/{len(tasks)} batches, "
                            f"{snapshot['prompt_tokens']:,}+{snapshot['completion_tokens']:,} tokens, "
                            f"{snapshot['failures']} failures",
                            flush=True,
                        )
    finally:
        handle.close()
    return usage


def run_generation(
    *,
    requests: list[dict[str, Any]],
    forms: dict[str, AmbiguousForm],
    deployment: Deployment,
    config: dict[str, str | None],
    raw_path: Path,
    workers: int = 10,
    max_tokens: int = 6000,
    temperature: float = 0.9,
    progress_every: int = 25,
) -> Usage:
    """One call per starved (group, sense), resuming from persisted responses."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(raw_path)
    client = deployment.client(config)
    model_name = deployment.resolved_model(config)
    usage = Usage()

    pending = [
        request
        for request in requests
        if f"generate:{request['form']}:{request['sense_id']}" not in done
    ]
    if not pending:
        return usage

    write_lock = threading.Lock()
    handle = raw_path.open("a", encoding="utf-8")
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for request in pending:
                prompt = generate_prompt(forms[request["form"]], [request])
                futures[
                    pool.submit(
                        _call,
                        client,
                        model_name,
                        GENERATE_SYSTEM,
                        prompt,
                        usage,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                ] = request

            for future in as_completed(futures):
                request = futures[future]
                content, counts = future.result()
                with write_lock:
                    handle.write(
                        json.dumps(
                            {
                                "batch_key": f"generate:{request['form']}:{request['sense_id']}",
                                "job": "generate",
                                "form": request["form"],
                                "spec": request,
                                "deployment": model_name,
                                "content": content,
                                "usage": counts,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    completed += 1
                    if progress_every and completed % progress_every == 0:
                        snapshot = usage.snapshot()
                        print(
                            f"  generate: {completed}/{len(pending)} calls, "
                            f"{snapshot['prompt_tokens']:,}+"
                            f"{snapshot['completion_tokens']:,} tokens, "
                            f"{snapshot['failures']} failures",
                            flush=True,
                        )
    finally:
        handle.close()
    return usage


def parse_raw(raw_path: Path) -> dict[str, dict[str, Any]]:
    """Parse persisted raw responses into per-sentence results."""
    results: dict[str, dict[str, Any]] = {}
    if not raw_path.exists():
        return results
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            content = record.get("content")
            if not content:
                continue
            parsed = _extract_json(content)
            if not parsed or not isinstance(parsed.get("items"), list):
                continue
            sentence_ids = record["sentence_ids"]
            for item in parsed["items"]:
                if not isinstance(item, dict):
                    continue
                index = item.get("i")
                if not isinstance(index, int) or not 0 <= index < len(sentence_ids):
                    continue
                results[sentence_ids[index]] = {
                    "sense_id": item.get("sense_id"),
                    "confidence": item.get("confidence"),
                    "cue": item.get("cue", ""),
                    "unclear": bool(item.get("unclear", False)),
                    "deployment": record["deployment"],
                    "form": record["form"],
                }
    return results


def mask_sentence(row: dict[str, Any]) -> str:
    return row["sentence"][: row["start"]] + MASK + row["sentence"][row["end"] :]
