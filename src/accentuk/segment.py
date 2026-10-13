"""Lossless chunking of input text."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

BOUNDARY = re.compile(r"[.!?…]+[\"'»”’\)\]]*")
WHITESPACE = re.compile(r"\s+")
LAST_TOKEN = re.compile(r"\S+$")
CLAUSE = re.compile(r"(?<=[;:])\s+|\s+[—–-]\s+|(?<=,)\s+")

ABBREVIATIONS = frozenset(
    """
    р рр ст стст в вв м с сс сел смт вул просп бульв пров пл наб обл
    р-н тис млн млрд грн коп дол євро шт зам вид перекл упоряд
    див напр тобто т ім акад проф доц канд докт ред гол зав
    буд кв корп каб оф под пн пд сх зх
    хв год сек кг мг км см мм кв куб
    """.split()
)


@dataclass(frozen=True)
class Chunk:
    """One slice of the input. Only speakable slices reach the model."""

    text: str
    speakable: bool


def _emit(chunks: list[Chunk], value: str) -> None:
    if not value:
        return
    if not value.strip():
        chunks.append(Chunk(value, False))
        return
    lead = len(value) - len(value.lstrip())
    tail = len(value.rstrip())
    if lead:
        chunks.append(Chunk(value[:lead], False))
    chunks.append(Chunk(value[lead:tail], True))
    if tail < len(value):
        chunks.append(Chunk(value[tail:], False))


def _is_sentence_end(text: str, match: re.Match[str], resume: int) -> bool:
    if resume >= len(text):
        return True
    token = LAST_TOKEN.search(text[: match.start()])
    if token is not None:
        word = token.group().strip("\"'«»“”‘’()[]")
        if word.lower() in ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha() and word.isupper():
            return False
    following = text[resume]
    if following.islower():
        return False
    return True


def _split_line(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    position = 0
    for match in BOUNDARY.finditer(text):
        if match.start() < position:
            continue
        end = match.end()
        whitespace = WHITESPACE.match(text, end)
        resume = whitespace.end() if whitespace is not None else end
        gap = text[end:resume]
        if not gap and resume < len(text):
            continue
        if not _is_sentence_end(text, match, resume):
            continue
        _emit(chunks, text[position:end])
        if gap:
            chunks.append(Chunk(gap, False))
        position = resume
    _emit(chunks, text[position:])
    return chunks


def split_sentences(text: str) -> list[Chunk]:
    """Split text into sentence and whitespace chunks without altering it."""
    chunks: list[Chunk] = []
    for part in re.split(r"(\n+)", text):
        if not part:
            continue
        if part.startswith("\n"):
            chunks.append(Chunk(part, False))
            continue
        chunks.extend(_split_line(part))
    return chunks


def _pieces(text: str) -> list[str]:
    parts: list[str] = []
    position = 0
    for match in CLAUSE.finditer(text):
        if match.end() <= position:
            continue
        parts.append(text[position : match.end()])
        position = match.end()
    if position < len(text):
        parts.append(text[position:])
    return parts or [text]


def _words(text: str) -> list[str]:
    parts = re.split(r"(\s+)", text)
    merged: list[str] = []
    for part in parts:
        if not part:
            continue
        if merged and part.isspace():
            merged[-1] += part
        else:
            merged.append(part)
    return merged or [text]


def _pack(parts: Iterable[str], count: Callable[[str], int], budget: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in parts:
        candidate = current + part
        if current and count(candidate) > budget:
            packed.append(current)
            current = part
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def fit_to_window(
    chunks: list[Chunk],
    count: Callable[[str], int],
    budget: int,
    warnings: list[str] | None = None,
) -> list[Chunk]:
    """Re-cut speakable chunks that do not fit the model's source window."""
    result: list[Chunk] = []
    for chunk in chunks:
        if not chunk.speakable or count(chunk.text) <= budget:
            result.append(chunk)
            continue
        packed = _pack(_pieces(chunk.text), count, budget)
        if any(count(part) > budget for part in packed):
            packed = _pack(_words(chunk.text), count, budget)
        if warnings is not None:
            warnings.append(
                f"sentence of {count(chunk.text)} tokens exceeds the {budget}-token "
                f"source window; verbalized as {len(packed)} chunks"
            )
        for part in packed:
            _emit(result, part)
    return result


def join(chunks: Iterable[Chunk]) -> str:
    return "".join(chunk.text for chunk in chunks)
