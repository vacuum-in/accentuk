import bz2
from pathlib import Path

import pytest
from lxml import etree

from ukstress.dump_reader import DumpPage, RejectedPage, parse_pages_parallel, stream_pages


def _write_dump(path: Path, pages: str) -> None:
    document = (
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
        f"{pages}</mediawiki>"
    )
    path.write_bytes(bz2.compress(document.encode()))


def _page(identifier: int, title: str, text: str, namespace: int = 0) -> str:
    return (
        f"<page><title>{title}</title><ns>{namespace}</ns><id>{identifier}</id>"
        f"<revision><id>{identifier + 100}</id><timestamp>2026-01-01T00:00:00Z</timestamp>"
        f"<text>{text}</text></revision></page>"
    )


def test_streams_main_namespace_metadata_and_skips_other_namespaces(tmp_path: Path) -> None:
    dump = tmp_path / "fixture.xml.bz2"
    _write_dump(dump, _page(1, "мо́ва", "text") + _page(2, "Template:x", "ignored", 10))

    assert list(stream_pages(dump)) == [
        DumpPage(1, "мо́ва", None, 101, "2026-01-01T00:00:00Z", "text")
    ]


def test_rejects_oversized_page_and_continues(tmp_path: Path) -> None:
    dump = tmp_path / "fixture.xml.bz2"
    _write_dump(dump, _page(1, "велика", "x" * 20) + _page(2, "мала", "ok"))
    rejected: list[RejectedPage] = []

    pages = list(stream_pages(dump, max_page_bytes=10, on_rejected=rejected.append))

    assert [page.title for page in pages] == ["мала"]
    assert rejected[0].reason == "page exceeds byte limit"


def test_malformed_xml_fails(tmp_path: Path) -> None:
    dump = tmp_path / "broken.xml.bz2"
    dump.write_bytes(bz2.compress(b"<mediawiki>"))
    with pytest.raises(etree.XMLSyntaxError):
        list(stream_pages(dump))


def test_parallel_parse_is_deterministic_across_worker_counts(tmp_path: Path) -> None:
    dump = tmp_path / "fixture.xml.bz2"
    _write_dump(dump, "".join(_page(index, f"слово{index}", "text") for index in range(1, 9)))

    one_worker = list(parse_pages_parallel(stream_pages(dump), 1, lambda page: page.page_id))
    many_workers = list(parse_pages_parallel(stream_pages(dump), 4, lambda page: page.page_id))
    assert one_worker == many_workers == list(range(1, 9))
