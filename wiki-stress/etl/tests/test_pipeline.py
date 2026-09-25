from __future__ import annotations

import bz2
from pathlib import Path

from ukstress.pipeline import parse_dump, validate_staging_directory
from ukstress.staging import read_jsonl_zst


def _dump(path: Path) -> None:
    xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page><title>мова</title><ns>0</ns><id>1</id><revision><id>11</id>
<timestamp>2026-01-01T00:00:00Z</timestamp>
<text>==Українська==\n===Іменник===\n{{uk-noun|мо́ва}}\n'''мо́ва'''</text>
</revision></page>
<page><title>мовонька</title><ns>0</ns><id>2</id><redirect title="мова"/>
<revision><id>12</id><text>#REDIRECT [[мова́]]</text></revision></page>
</mediawiki>"""
    path.write_bytes(bz2.compress(xml.encode()))


def test_parse_dump_writes_relational_staging_and_ignores_redirect_text(tmp_path: Path) -> None:
    dump = tmp_path / "dump.xml.bz2"
    _dump(dump)
    summary = parse_dump(dump, tmp_path / "output", workers=2, dataset_key="fixture")

    counts = validate_staging_directory(summary.output_dir)
    assert counts["lexemes"] == 1
    assert counts["stress_variants"] == 1
    variants = read_jsonl_zst(summary.output_dir / "stress_variants.jsonl.zst")
    assert variants[0]["stressed_form"] == "мо́ва"
    assert summary.statistics["aliases"] == 1
