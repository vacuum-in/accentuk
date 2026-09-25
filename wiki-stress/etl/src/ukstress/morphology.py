"""Restricted, data-only expansion of Ukrainian inflection templates."""

from __future__ import annotations

import bz2
import re
from dataclasses import dataclass
from pathlib import Path

import mwparserfromhell
from lxml import etree
from mwparserfromhell.nodes import Template

from ukstress.normalizer import ACUTE, validate_stress

_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
_SUPPORTED_PREFIXES = (
    "імен uk",
    "прикм uk",
    "дієсл uk",
    "займ uk",
    "числ uk",
    "присл uk",
)
_COLLECTORS = {
    "inflection-імен-uk": "noun",
    "відмінки-всі-к": "adjective",
    "відмінки-всі": "adjective",
    "дієсл-блок": "verb",
}
_REDIRECT = re.compile(
    r"(?:#ПЕРЕНАПРАВЛЕННЯ|#REDIRECT)\s*\[\[Шаблон:(?P<target>[^\]|]+)",
    re.IGNORECASE,
)
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_NOINCLUDE_TAG = re.compile(r"</?noinclude>", re.IGNORECASE)
_OPTIONAL = re.compile(r"\(([^()]*)\)")


@dataclass(frozen=True)
class MorphologicalForm:
    stressed_form: str
    part_of_speech: str
    grammatical_tags: tuple[str, ...]


def _normalized_name(value: str) -> str:
    return "-".join(
        _HTML_TAG.sub("", value).replace("_", " ").casefold().split()
    )


def _template_arguments(template: Template) -> dict[str, str]:
    values: dict[str, str] = {}
    positional = 1
    for parameter in template.params:
        name = str(parameter.name).strip()
        if not parameter.showkey:
            name = str(positional)
            positional += 1
        values[name] = str(parameter.value).strip()
    return values


def _expand_arguments(value: str, arguments: dict[str, str]) -> str:
    code = mwparserfromhell.parse(value)
    for _ in range(12):
        nodes = code.filter_arguments(recursive=True)
        if not nodes:
            break
        changed = False
        for node in reversed(nodes):
            name = str(node.name).strip()
            replacement = arguments.get(name)
            if replacement is None:
                replacement = "" if node.default is None else str(node.default)
            try:
                code.replace(node, replacement)
                changed = True
            except ValueError:
                continue
        if not changed:
            break
    return str(code)


def _expand_functions(value: str) -> str:
    code = mwparserfromhell.parse(value)
    for _ in range(8):
        changed = False
        for template in reversed(code.filter_templates(recursive=True)):
            name = str(template.name).strip()
            replacement: str | None = None
            if name.casefold().startswith("#if:"):
                condition = name.partition(":")[2].strip()
                then = str(template.get(1).value) if template.has(1) else ""
                otherwise = str(template.get(2).value) if template.has(2) else ""
                replacement = then if condition else otherwise
            elif name.casefold().startswith("#ifeq:"):
                left = name.partition(":")[2].strip()
                right = str(template.get(1).value).strip() if template.has(1) else ""
                then = str(template.get(2).value) if template.has(2) else ""
                otherwise = str(template.get(3).value) if template.has(3) else ""
                replacement = then if left == right else otherwise
            elif _normalized_name(name) in {"основа", "основа1"}:
                replacement = str(template.get(1).value) if template.has(1) else ""
            if replacement is not None:
                try:
                    code.replace(template, replacement)
                    changed = True
                except ValueError:
                    continue
        if not changed:
            break
    return str(code)


def _optional_variants(value: str) -> set[str]:
    match = _OPTIONAL.search(value)
    if match is None:
        return {value}
    before = value[: match.start()]
    after = value[match.end() :]
    return _optional_variants(before + after) | _optional_variants(
        before + match.group(1) + after
    )


def _tags(field: str) -> tuple[str, ...]:
    normalized = " ".join(field.casefold().split())
    direct: list[str] = []
    cases = {
        "nom": "nominative",
        "gen": "genitive",
        "dat": "dative",
        "acc": "accusative",
        "ins": "instrumental",
        "loc": "locative",
        "voc": "vocative",
        "н": "nominative",
        "р": "genitive",
        "д": "dative",
        "з": "accusative",
        "о": "instrumental",
        "м": "locative",
    }
    first = re.split(r"[-\s(]", normalized, maxsplit=1)[0]
    if first in cases:
        direct.append(cases[first])
    if "-sg" in normalized or "одн" in normalized:
        direct.append("singular")
    if "-pl" in normalized or "мн" in normalized:
        direct.append("plural")
    for marker, tag in (
        ("чол", "masculine"),
        ("жін", "feminine"),
        ("сер", "neuter"),
        ("іст", "animate"),
        ("неіст", "inanimate"),
        ("мин.", "past"),
        ("майб.", "future"),
        ("наказ.", "imperative"),
    ):
        if marker in normalized:
            direct.append(tag)
    for marker, tag in (
        ("я ", "first_person"),
        ("ми ", "first_person"),
        ("ти ", "second_person"),
        ("ви ", "second_person"),
        ("він", "third_person"),
        ("вони", "third_person"),
    ):
        if normalized.startswith(marker):
            direct.append(tag)
    if normalized.startswith(("я ", "ти ", "він")):
        direct.append("singular")
    if normalized.startswith(("ми ", "ви ", "вони")):
        direct.append("plural")
    return tuple(dict.fromkeys([*direct, f"source_field:{normalized}"]))


class MorphologyCatalog:
    """Immutable catalog of supported template definitions from the same dump."""

    def __init__(self, definitions: dict[str, str]) -> None:
        self._definitions = definitions

    @classmethod
    def from_dump(cls, path: Path) -> MorphologyCatalog:
        definitions: dict[str, str] = {}
        with bz2.open(path, "rb") as source:
            for _, page in etree.iterparse(source, events=("end",), tag=f"{_NS}page"):
                try:
                    if page.findtext(f"{_NS}ns") != "10":
                        continue
                    title = page.findtext(f"{_NS}title") or ""
                    if not title.startswith("Шаблон:"):
                        continue
                    name = title.removeprefix("Шаблон:")
                    if not name.casefold().startswith(_SUPPORTED_PREFIXES):
                        continue
                    revision = page.find(f"{_NS}revision")
                    text = "" if revision is None else revision.findtext(f"{_NS}text") or ""
                    definitions[_normalized_name(name)] = text
                finally:
                    page.clear()
                    while page.getprevious() is not None:
                        del page.getparent()[0]
        return cls(definitions)

    def extract(self, invocation: Template) -> list[MorphologicalForm]:
        name = _normalized_name(str(invocation.name))
        definition = self._resolve_definition(name)
        if definition is None:
            return []
        arguments = _template_arguments(invocation)
        results: list[MorphologicalForm] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        code = mwparserfromhell.parse(_NOINCLUDE_TAG.sub("", definition))
        for collector in code.filter_templates(recursive=True):
            part_of_speech = _COLLECTORS.get(_normalized_name(str(collector.name)))
            if part_of_speech is None:
                continue
            for parameter in collector.params:
                field = str(parameter.name).strip()
                if not parameter.showkey or field in {
                    "склади",
                    "коментар",
                    "hide-text",
                    "шаблон-кат",
                }:
                    continue
                expanded = _expand_functions(
                    _expand_arguments(str(parameter.value), arguments)
                )
                for raw_markup in _BREAK.split(expanded):
                    raw = mwparserfromhell.parse(raw_markup).strip_code(
                        normalize=True, collapse=True
                    )
                    cleaned = raw.strip().replace("·", "").rstrip("*").strip()
                    for variant in _optional_variants(cleaned):
                        variant = variant.strip()
                        grammar = _tags(field)
                        key = (variant, grammar)
                        if (
                            variant
                            and ACUTE in variant
                            and validate_stress(variant).classification == "valid"
                            and key not in seen
                        ):
                            seen.add(key)
                            results.append(
                                MorphologicalForm(
                                    variant, part_of_speech, grammar
                                )
                            )
        return results

    def _resolve_definition(self, name: str) -> str | None:
        definition = self._definitions.get(name)
        visited: set[str] = set()
        while definition is not None and name not in visited:
            visited.add(name)
            redirect = _REDIRECT.search(definition)
            if redirect is None:
                return definition
            name = _normalized_name(redirect.group("target"))
            definition = self._definitions.get(name)
        return None
