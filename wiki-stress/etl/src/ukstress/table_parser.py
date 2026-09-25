"""Conservative extraction from supported MediaWiki table cells."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ukstress.normalizer import validate_stress

_LINK = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>|\[\d+\]", re.IGNORECASE)
_NOTE = re.compile(r"\{\{[^{}]*\}\}")


@dataclass(frozen=True)
class TableForm:
    stressed_form: str
    row: int
    column: int


def _clean(cell: str) -> str:
    value = _LINK.sub(r"\1", cell)
    value = _REF.sub("", value)
    value = _NOTE.sub("", value)
    return value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")


def extract_table_forms(wikitext: str) -> list[TableForm]:
    """Extract valid stressed values from simple supported tables without splitting hyphens."""
    forms: list[TableForm] = []
    for table in re.findall(r"\{\|(.*?)\|\}", wikitext, flags=re.DOTALL):
        row = 0
        for line in table.splitlines():
            if line.startswith("|-"):
                row += 1
                continue
            if not line.startswith("|"):
                continue
            for column, cell in enumerate(line[1:].split("||")):
                for variant in re.split(r"\n|\s*/\s*|\s*,\s*", _clean(cell)):
                    candidate = variant.strip()
                    valid_stressed = (
                        candidate
                        and "\u0301" in candidate
                        and validate_stress(candidate).classification == "valid"
                    )
                    if valid_stressed:
                        forms.append(TableForm(candidate, row, column))
    return forms
