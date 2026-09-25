"""Single streaming pass over a corpus for ambiguous-form sentences.

Two stages, for speed. A C-speed Aho-Corasick automaton over raw wikitext
rejects the large majority of pages without touching them further; only pages
that survive are stripped to prose, split into sentences, and matched exactly
using the project's canonical lookup key.

Per-form reservoir sampling happens during the pass, so a form occurring tens of
thousands of times never reaches disk that many times.

`mine()` reads a MediaWiki dump; `mine_documents()` takes any stream of
`(document_id, text)` and applies the *same* prefilter, sentence splitter,
matcher and rejection rules. Wikipedia is an encyclopedia, and an encyclopedia
starves exactly the senses this project needs: 78% of mined groups came back
single-sense, and the model's remaining errors are all on groups whose minority
sense never appeared. A corpus of fiction, news and speech carries those senses
in natural text, which is worth more than generating them.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import ahocorasick
from ukstress.deduplicator import stable_natural_key
from ukstress.dump_reader import stream_pages
from ukstress.normalizer import ACUTE, CANONICAL_APOSTROPHE, lookup_key

from ukstress_ml.ambiguity import AmbiguousForm

APOSTROPHES = "'`‘’ʻʼ＇"
_UK_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
_WORD = re.compile(f"[{_UK_LETTERS}a-zA-Z]+(?:[{APOSTROPHES}‐-―-][{_UK_LETTERS}]+)*")

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF = re.compile(r"<ref[^>]*?/>|<ref.*?</ref>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_HEADING = re.compile(r"^\s*=+.*?=+\s*$", re.MULTILINE)
_LIST_LINE = re.compile(r"^\s*[*#:;|!].*$", re.MULTILINE)
_FILE_LINK = re.compile(
    r"\[\[(?:File|Image|Файл|Зображення|Категорія|Category)\s*:[^\]]*\]\]", re.IGNORECASE
)
_PIPED_LINK = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
_PLAIN_LINK = re.compile(r"\[\[([^\]|]*)\]\]")
_EXT_LINK = re.compile(r"\[(?:https?|//)\S*\s+([^\]]*)\]")
_BARE_EXT_LINK = re.compile(r"\[(?:https?|//)\S*\]")
_BOLD_ITALIC = re.compile(r"'{2,5}")
_WS = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{2,}")

# Ukrainian abbreviations that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    ["тис", "млн", "млрд", "грн", "див", "напр", "табл", "мал", "рис", "ст", "стор", "с", "р", "рр", "вв", "ім", "акад", "проф", "доц", "буд", "кв", "вул", "просп", "пров", "смт", "обл", "р-н", "км", "см", "мм", "кг", "мг", "г", "л", "мл", "год", "хв", "сек", "грам", "англ", "укр", "рос", "нім", "фр", "лат", "гр", "перекл", "ред", "упоряд", "вид", "т", "ч", "зб", "як", "напр", "приблизно", "ін", "т.д", "т.п", "тобто"]
)

_SENT_END = re.compile(r"([.!?…]+)([\"»)\]]*)(\s+)")


def strip_to_prose(wikitext: str) -> str:
    """Remove markup, leaving running prose paragraphs."""
    text = _COMMENT.sub(" ", wikitext)
    text = _REF.sub(" ", text)
    text = _TABLE.sub(" ", text)
    text = _FILE_LINK.sub(" ", text)

    # Templates nest; strip innermost braces repeatedly rather than with one regex.
    for _ in range(8):
        replaced, count = re.subn(r"\{\{[^{}]*\}\}", " ", text)
        text = replaced
        if not count:
            break
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)

    text = _PIPED_LINK.sub(r"\1", text)
    text = _PLAIN_LINK.sub(r"\1", text)
    text = _EXT_LINK.sub(r"\1", text)
    text = _BARE_EXT_LINK.sub(" ", text)
    text = _HEADING.sub("\n", text)
    text = _LIST_LINE.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = _BOLD_ITALIC.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    text = _WS.sub(" ", text)
    return _BLANK_LINES.sub("\n", text)


def split_sentences(text: str) -> Iterator[str]:
    """Split prose into sentences, holding abbreviations together."""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if len(paragraph) < 20:
            continue
        start = 0
        for match in _SENT_END.finditer(paragraph):
            end = match.end(2)
            candidate = paragraph[start:end].strip()
            head = candidate.rstrip(".!?…\"»)]").split()
            last = head[-1].lower().strip("(«\"") if head else ""
            if match.group(1) == "." and last in _ABBREVIATIONS:
                continue
            following = paragraph[match.end() : match.end() + 1]
            if following and not (following.isupper() or following.isdigit() or following in "«\""):
                continue
            if candidate:
                yield candidate
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            yield tail


def _prefilter_variants(form: str) -> set[str]:
    """Surface spellings of a canonical form as they appear in raw wikitext."""
    composed = unicodedata.normalize("NFC", form)
    variants = {composed}
    if CANONICAL_APOSTROPHE in composed:
        variants |= {composed.replace(CANONICAL_APOSTROPHE, a) for a in APOSTROPHES}
    return variants


def build_prefilter(forms: list[AmbiguousForm]) -> ahocorasick.Automaton:
    automaton = ahocorasick.Automaton()
    for form in forms:
        for variant in _prefilter_variants(form.form):
            automaton.add_word(variant, form.form)
    automaton.make_automaton()
    return automaton


@dataclass
class Reservoir:
    """Uniform sample of fixed size over an unbounded stream."""

    cap: int
    seen: int = 0
    items: list[dict[str, object]] = field(default_factory=list)

    def offer(self, item: dict[str, object], rng: random.Random) -> None:
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
            return
        index = rng.randrange(self.seen)
        if index < self.cap:
            self.items[index] = item


@dataclass
class MineStats:
    pages: int = 0
    pages_prefiltered: int = 0
    sentences_examined: int = 0
    hits: int = 0
    kept: int = 0
    rejected_length: int = 0
    rejected_stress_marks: int = 0
    rejected_multi_form: int = 0
    rejected_script_ratio: int = 0


def _ukrainian_ratio(sentence: str) -> float:
    letters = [c for c in sentence if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in _UK_LETTERS) / len(letters)


def _harvest_sentence(
    sentence: str,
    by_form: dict[str, AmbiguousForm],
    reservoirs: dict[str, Reservoir],
    rng: random.Random,
    stats: MineStats,
    provenance: dict[str, object],
    *,
    min_tokens: int,
    max_tokens: int,
) -> None:
    """Apply every acceptance rule to one sentence, sampling it if it passes."""
    stats.sentences_examined += 1
    matches: list[tuple[str, int, int]] = []
    for token_match in _WORD.finditer(sentence):
        key = lookup_key(token_match.group(0))
        if key in by_form:
            matches.append((key, token_match.start(), token_match.end()))
    if not matches:
        return
    stats.hits += 1

    if len(matches) > 1:
        # Two target forms in one sentence would give the row two labels.
        stats.rejected_multi_form += 1
        return
    if ACUTE in unicodedata.normalize("NFD", sentence):
        stats.rejected_stress_marks += 1
        return
    token_count = len(_WORD.findall(sentence))
    if not min_tokens <= token_count <= max_tokens:
        stats.rejected_length += 1
        return
    if _ukrainian_ratio(sentence) < 0.9:
        stats.rejected_script_ratio += 1
        return

    key, start, end = matches[0]
    form = by_form[key]
    reservoirs[key].offer(
        {
            "sentence_id": stable_natural_key("mined_sentence", sentence, key),
            "sentence": sentence,
            "form": key,
            "surface": sentence[start:end],
            "start": start,
            "end": end,
            "group_id": form.group_id,
            **provenance,
        },
        rng,
    )
    stats.kept += 1


def mine_documents(
    documents: Iterator[tuple[str, str]],
    forms: list[AmbiguousForm],
    *,
    cap_per_form: int = 400,
    seed: int = 20260815,
    min_tokens: int = 6,
    max_tokens: int = 40,
    max_documents: int | None = None,
    progress_every: int = 100_000,
) -> tuple[dict[str, Reservoir], MineStats]:
    """Mine plain-text documents, applying the dump miner's rules unchanged."""
    by_form = {form.form: form for form in forms}
    automaton = build_prefilter(forms)
    reservoirs: dict[str, Reservoir] = defaultdict(lambda: Reservoir(cap=cap_per_form))
    rng = random.Random(seed)
    stats = MineStats()

    for document_id, text in documents:
        stats.pages += 1
        if max_documents is not None and stats.pages > max_documents:
            break
        if progress_every and stats.pages % progress_every == 0:
            print(f"  docs={stats.pages:,} prefiltered={stats.pages_prefiltered:,} "
                  f"kept={stats.kept:,}", flush=True)
        try:
            next(automaton.iter(text.lower()))
        except StopIteration:
            continue
        stats.pages_prefiltered += 1

        provenance: dict[str, object] = {
            "page_id": None, "revision_id": None, "page_title": document_id,
        }
        for sentence in split_sentences(text):
            _harvest_sentence(sentence, by_form, reservoirs, rng, stats, provenance,
                              min_tokens=min_tokens, max_tokens=max_tokens)

    return dict(reservoirs), stats


def mine(
    dump_path: Path,
    forms: list[AmbiguousForm],
    *,
    cap_per_form: int = 400,
    seed: int = 20260815,
    min_tokens: int = 6,
    max_tokens: int = 40,
    max_pages: int | None = None,
    progress_every: int = 50_000,
) -> tuple[dict[str, Reservoir], MineStats]:
    by_form = {form.form: form for form in forms}
    automaton = build_prefilter(forms)
    reservoirs: dict[str, Reservoir] = defaultdict(lambda: Reservoir(cap=cap_per_form))
    rng = random.Random(seed)
    stats = MineStats()

    for page in stream_pages(dump_path, max_page_bytes=4_000_000):
        stats.pages += 1
        if max_pages is not None and stats.pages > max_pages:
            break
        if progress_every and stats.pages % progress_every == 0:
            print(
                f"  pages={stats.pages:,} prefiltered={stats.pages_prefiltered:,} "
                f"kept={stats.kept:,}",
                flush=True,
            )
        if page.redirect_target is not None:
            continue

        lowered = page.wikitext.lower()
        try:
            next(automaton.iter(lowered))
        except StopIteration:
            continue
        stats.pages_prefiltered += 1

        provenance = {
            "page_id": page.page_id,
            "revision_id": page.revision_id,
            "page_title": page.title,
        }
        prose = strip_to_prose(page.wikitext)
        for sentence in split_sentences(prose):
            _harvest_sentence(sentence, by_form, reservoirs, rng, stats, provenance,
                              min_tokens=min_tokens, max_tokens=max_tokens)

    return dict(reservoirs), stats


def write(
    reservoirs: dict[str, Reservoir],
    stats: MineStats,
    forms: list[AmbiguousForm],
    output_dir: Path,
    *,
    corpus: str,
    licence: str,
    dump_sha256: str,
    config: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for key in sorted(reservoirs):
            for item in reservoirs[key].items:
                item = dict(item)
                item["corpus"] = corpus
                item["licence"] = licence
                item["source_tier"] = "wiki"
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1

    per_form = {key: reservoirs[key].seen for key in sorted(reservoirs)}
    starved = sorted(form.form for form in forms if reservoirs.get(form.form) is None)
    coverage: dict[str, object] = {
        "corpus": corpus,
        "licence": licence,
        "dump_sha256": dump_sha256,
        "config": config,
        "stats": vars(stats),
        "rows_written": rows,
        "forms_with_candidates": len(reservoirs),
        "forms_total": len(forms),
        "forms_starved": len(starved),
        "starved_forms": starved,
        "occurrences_seen_per_form": per_form,
    }
    (output_dir / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return coverage
