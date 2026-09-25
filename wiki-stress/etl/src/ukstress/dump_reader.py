"""Bounded-memory MediaWiki XML page streaming."""

from __future__ import annotations

import bz2
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


@dataclass(frozen=True)
class DumpPage:
    page_id: int
    title: str
    redirect_target: str | None
    revision_id: int
    revision_timestamp: str | None
    wikitext: str


@dataclass(frozen=True)
class RejectedPage:
    title: str
    reason: str


def _page_id(page: etree._Element) -> int:
    identifier = page.findtext(f"{_NS}id")
    if identifier is None:
        raise ValueError("page has no ID")
    return int(identifier)


def stream_pages(
    path: Path,
    *,
    max_page_bytes: int = 2_000_000,
    max_title_characters: int = 512,
    on_rejected: Callable[[RejectedPage], None] | None = None,
) -> Iterator[DumpPage]:
    """Yield main-namespace pages while clearing every processed XML element."""
    with bz2.open(path, "rb") as source:
        for _, page in etree.iterparse(source, events=("end",), tag=f"{_NS}page"):
            try:
                namespace = page.findtext(f"{_NS}ns")
                title = page.findtext(f"{_NS}title") or ""
                revision = page.find(f"{_NS}revision")
                wikitext = "" if revision is None else revision.findtext(f"{_NS}text") or ""
                if namespace != "0":
                    continue
                if len(title) > max_title_characters:
                    if on_rejected is not None:
                        on_rejected(
                            RejectedPage(title[:max_title_characters], "title exceeds limit")
                        )
                    continue
                if len(wikitext.encode("utf-8")) > max_page_bytes:
                    if on_rejected is not None:
                        on_rejected(RejectedPage(title, "page exceeds byte limit"))
                    continue
                if revision is None:
                    raise ValueError("main-namespace page has no revision")
                revision_id = revision.findtext(f"{_NS}id")
                if revision_id is None:
                    raise ValueError("revision has no ID")
                redirect = page.find(f"{_NS}redirect")
                yield DumpPage(
                    page_id=_page_id(page),
                    title=title,
                    redirect_target=None if redirect is None else redirect.get("title"),
                    revision_id=int(revision_id),
                    revision_timestamp=revision.findtext(f"{_NS}timestamp"),
                    wikitext=wikitext,
                )
            finally:
                page.clear()
                while page.getprevious() is not None:
                    del page.getparent()[0]


def parse_pages_parallel[T](
    pages: Iterator[DumpPage], worker_count: int, parser: Callable[[DumpPage], T]
) -> Iterator[T]:
    """Use a bounded in-flight queue while preserving source order deterministically."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="ukstress-parse"
    ) as executor:
        pending: dict[int, Future[T]] = {}
        next_index = 0
        for index, page in enumerate(pages):
            pending[index] = executor.submit(parser, page)
            if len(pending) >= worker_count * 2:
                yield pending.pop(next_index).result()
                next_index += 1
        while pending:
            yield pending.pop(next_index).result()
            next_index += 1
