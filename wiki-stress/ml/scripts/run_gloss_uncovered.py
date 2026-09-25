"""Author glosses for ambiguous forms the serving manifest does not cover.

`serving.py` refuses any form absent from `manifest["forms"]`, and the Go API
answers a refused ambiguous form with the *unstressed* token. On the gold set
that class — 200 of 1000 rows after the trie recovery — scored 0.0000 before a
dictionary fallback was added and still cannot reach the model. It is the
largest remaining block of rows the pipeline is structurally unable to get
right.

The blocker is glosses, not the model. A manifest entry needs a definition per
candidate because the cross-encoder scores `(sentence, gloss)` pairs, and the
existing gloss pipeline reads them off Wiktionary senses. These forms have no
Wiktionary group: they come from the accented wordlist, and the trie records
their readings as tags and accent positions, with no meanings attached.

So the glosses are authored per *form*, from its stressed variants, rather than
per group as `run_glosses.py` does. That costs one call per form instead of one
per paradigm, which is the price of covering forms Wiktionary never grouped.
Every response is written raw before parsing, so a parser change costs a
re-parse rather than a re-spend, and completed forms are skipped on resume.

Each variant is presented with the trie's own tag sets for that accent, which
is the evidence the labeller would otherwise invent. Without them a pilot
derived `коли́` from `кола` ("circle") — the reading the trie files under
`ко́ли` — and glossed it accordingly.

A gloss is assembled from three requested parts rather than asked for whole:
a grammatical label, a lemma, and a meaning, joined as
`"<label>, від: <lemma>. <meaning>"`. Two reasons. It is the shape the shipped
manifest already uses, so these entries are in the distribution the model was
trained on rather than a new one. And it is what makes *grammatical* homographs
servable at all: `ві́йни` and `війни́` mean the same thing and differ only in
case and number, so a purely semantic pair of glosses comes back as two
paraphrases of "armed conflict" and is rejected as indistinguishable. The
grammatical label is the part that actually separates them, and the context
sentence is where the cross-encoder can find it.

The gloss contract from `ukstress_ml.glosses` is enforced unchanged: no gloss
may describe the stress, and two candidates may not share a gloss — a form
whose senses come back indistinguishable is dropped rather than served, because
the cross-encoder would be choosing between identical inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ukstress_ml.annotate import (
    FLASH_DEPLOYMENT,
    LABEL_DEPLOYMENT,
    VERIFY_DEPLOYMENT,
    Usage,
    load_config,
)
from ukstress_ml.glosses import GlossError, validate_gloss

#: Throughput here is a per-deployment quota, not a client-side one: raising
#: workers from 6 to 24 on one deployment moved the rate not at all and only
#: turned latency into 429s. Sharding the target list across deployments on
#: different resources is what actually adds capacity.
DEPLOYMENTS = {
    "label": LABEL_DEPLOYMENT,
    "flash": FLASH_DEPLOYMENT,
    "verify": VERIFY_DEPLOYMENT,
}

SYSTEM = (
    "Ти — український лексикограф. Пишеш стислі тлумачення значень слів "
    "для навчального корпусу. Відповідаєш лише валідним JSON."
)

#: New groups start above the Wiktionary-derived space so a form authored here
#: can never collide with, or be mistaken for, a curated group.
GROUP_ID_BASE = 1_000_000

#: Enough for two glosses and their grammar labels with room to spare.
MAX_OUTPUT_TOKENS = 600


def build_prompt(form: str, variants: list[dict[str, Any]]) -> str:
    lines = []
    for variant in variants:
        readings = ", ".join(variant.get("readings") or []) or "невідомо"
        lines.append(f"  {variant['id']} — {variant['stressed']}\n"
                     f"      граматичні розбори зі словника: {readings}")
    listing = "\n".join(lines)
    return f"""Слово «{form}» пишеться однаково, але має різні наголоси, і від наголосу
залежить значення.

Варіанти (з граматичними розборами, які словник приписує саме цьому наголосу):
{listing}

Для КОЖНОГО варіанта напиши:
- "grammar": коротку граматичну характеристику саме цієї форми, 2–5 слів
  (напр. "іменник, родовий відмінок однини", "дієслово, наказовий спосіб",
  "прикметник, жіночий рід");
- "lemma": початкову форму слова (називний однини для іменників, інфінітив
  для дієслів), без знаку наголосу;
- "definition": стисле тлумачення українською, 5–20 слів, яке пояснює ЗНАЧЕННЯ.

Правила:
- Спирайся на подані граматичні розбори: вони кажуть, яка саме частина мови і
  яка форма відповідає цьому наголосу. НЕ приписуй варіанту значення, яке
  суперечить його розборам.
- НЕ згадуй наголос, склади чи вимову — опис має стосуватися граматики та
  змісту, а не звучання.
- Якщо варіанти означають одне й те саме і різняться лише формою, це нормально:
  тоді їх розрізняє саме "grammar", і воно має бути різним.
- Не відсилай в тлумаченні до іншого варіанта ("те саме, що …").
- Якщо варіант не відповідає жодному реальному українському слову, постав
  "unclear": true для нього.

Формат (лише JSON):
{{"items": [{{"id": "a", "grammar": "...", "lemma": "...", "definition": "...",
              "unclear": false}}]}}"""


def parse_response(form: str, variants: list[dict[str, str]],
                   content: str) -> tuple[list[dict[str, Any]], str | None]:
    """Return authored candidates, or a reason the form cannot be served."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return [], "response contained no JSON object"
    try:
        payload = json.loads(content[start:end + 1])
    except json.JSONDecodeError as error:
        return [], f"invalid JSON: {error}"

    wanted = {v["id"]: v for v in variants}
    authored: dict[str, dict[str, Any]] = {}
    for item in payload.get("items", []):
        key = str(item.get("id", ""))
        if key not in wanted or item.get("unclear"):
            continue
        grammar = " ".join(str(item.get("grammar", "")).split()).rstrip(".,;")
        lemma = " ".join(str(item.get("lemma", "")).split()).rstrip(".,;")
        meaning = " ".join(str(item.get("definition", "")).split())
        if not grammar or not meaning:
            continue
        composed = f"{grammar}, від: {lemma}. {meaning}" if lemma else f"{grammar}. {meaning}"
        try:
            definition = validate_gloss(composed,
                                        surface_forms=[wanted[key]["stressed"], form])
        except GlossError:
            continue
        # Exactly `ukstress_ml.ambiguity.Candidate`'s fields and no others:
        # the inventory loader constructs it with `Candidate(**c)`, so an extra
        # key here is a TypeError at manifest build time. The lemma is not
        # dropped — it is inside `definition`, which is what the model reads.
        authored[key] = {
            "sense_id": f"{form}.{key}",
            "stressed": wanted[key]["stressed"],
            "signature": wanted[key]["signature"],
            "pos": grammar,
            "definition": definition,
            "priority": None,
            "review_status": "llm_proposed",
        }

    if len(authored) < len(variants):
        return [], "not every variant received a usable gloss"
    definitions = {c["definition"].strip() for c in authored.values()}
    if len(definitions) < len(authored):
        # Identical glosses give the cross-encoder identical inputs to choose
        # between; the pick would be noise served at whatever confidence the
        # scores happened to differ by.
        return [], "glosses do not distinguish the senses"
    return [authored[v["id"]] for v in variants], None


def completed(raw_path: Path) -> set[str]:
    done: set[str] = set()
    if not raw_path.exists():
        return done
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("content"):
                done.add(str(record["form"]))
    return done


def load_targets(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path,
                        default=Path("output/ml/uncovered_targets.json"))
    parser.add_argument("--raw", type=Path,
                        default=Path("output/ml/raw/glosses_uncovered.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/uncovered_glossed.jsonl"))
    parser.add_argument("--limit", type=int,
                        help="stop after this many forms; 0 rebuilds the inventory and exits")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--deployment", choices=sorted(DEPLOYMENTS), default="label")
    parser.add_argument("--shard", default="0/1",
                        help="i/n: take every n-th target starting at i, so several "
                             "deployments can run the same list without overlapping")
    parser.add_argument("--also-raw", type=Path, action="append", default=[],
                        help="other shards' raw files, read when assembling the inventory")
    args = parser.parse_args()

    index, _, count = args.shard.partition("/")
    shard, shards = int(index), int(count or 1)

    targets = load_targets(args.targets)
    raw_paths = [args.raw, *args.also_raw]
    done: set[str] = set()
    for path in raw_paths:
        done |= completed(path)
    todo = [t for i, t in enumerate(targets)
            if i % shards == shard and t["form"] not in done]
    if args.limit is not None:
        # `--limit 0` must mean "re-harvest what is already collected". Treating
        # it as falsy started a full 11,618-form run instead, which is the
        # opposite of a rebuild.
        todo = todo[:args.limit]
    if not todo:
        write_inventory(targets, raw_paths, args.out)
        return
    print(f"targets: {len(targets):,}  already done: {len(done):,}  to ask: {len(todo):,}",
          flush=True)

    config = load_config(Path(".env"))
    deployment = DEPLOYMENTS[args.deployment]
    client = deployment.client(config)
    model = deployment.resolved_model(config)
    print(f"deployment: {model}  shard {shard}/{shards}", flush=True)
    limits: dict[str, Any] = {"temperature": 0.2, "max_tokens": MAX_OUTPUT_TOKENS}
    limits_lock = threading.Lock()
    usage = Usage()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    started = time.time()
    finished = 0

    def ask(target: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(target["form"], target["variants"])
        # The deployment returns 429 well before the client's own retries help,
        # and a dropped form is a permanent hole in coverage rather than a
        # slower run, so back off and keep the slot.
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, **limits)
            except Exception as error:
                text = str(error)
                # Newer deployments reject `max_tokens` and `temperature`
                # outright rather than ignoring them. `limits` is shared, so the
                # retry must not depend on *this* thread being the one that
                # removed the offending key — the first version did, and every
                # sibling call already in flight raised instead of retrying,
                # which failed the whole shard.
                if "nsupported parameter" in text or "unsupported_parameter" in text:
                    with limits_lock:
                        if "max_tokens" in text and "max_tokens" in limits:
                            limits.pop("max_tokens")
                            limits["max_completion_tokens"] = MAX_OUTPUT_TOKENS
                        elif "temperature" in text and "temperature" in limits:
                            limits.pop("temperature")
                    if attempt < 5:
                        continue
                    raise
                if "429" not in text or attempt == 5:
                    raise
                time.sleep(2 ** attempt + random.random())
                continue
            if response.usage is not None:
                usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            return {"form": target["form"], "content": response.choices[0].message.content}
        raise RuntimeError("unreachable")

    with args.raw.open("a", encoding="utf-8") as raw, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ask, t): t for t in todo}
        for future in as_completed(futures):
            target = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001
                usage.fail()
                record = {"form": target["form"], "content": None, "error": str(error)}
            with lock:
                raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw.flush()
                finished += 1
                if finished % 100 == 0 or finished == len(todo):
                    rate = finished / max(time.time() - started, 1e-6)
                    print(f"  {finished:,}/{len(todo):,}  {rate:.1f}/s  "
                          f"eta {(len(todo)-finished)/max(rate,1e-6)/60:.1f} min", flush=True)

    write_inventory(targets, raw_paths, args.out)
    print(json.dumps(usage.snapshot(), indent=2))


def write_inventory(targets: list[dict[str, Any]], raw_paths: list[Path],
                    out_path: Path) -> None:
    """Turn raw responses into inventory rows `build_expanded_manifest` accepts."""
    by_form = {t["form"]: t for t in targets}
    responses: dict[str, str] = {}
    for raw_path in raw_paths:
        if not raw_path.exists():
            continue
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("content"):
                    responses[str(record["form"])] = str(record["content"])

    written = 0
    rejected: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for index, form in enumerate(sorted(responses)):
            target = by_form.get(form)
            if target is None:
                continue
            candidates, reason = parse_response(form, target["variants"], responses[form])
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            out.write(json.dumps({
                "group_id": GROUP_ID_BASE + index,
                "form": form,
                "feats": "surface",
                "candidates": candidates,
                "paradigm_source": "trie_surface",
                "complete": True,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nglossed forms written: {written:,}  -> {out_path}")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {count:>6,}  {reason}")


if __name__ == "__main__":
    main()
