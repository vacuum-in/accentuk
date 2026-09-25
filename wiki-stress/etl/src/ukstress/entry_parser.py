"""Page-level extraction orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from ukstress.dump_reader import DumpPage
from ukstress.normalizer import lookup_key


@dataclass(frozen=True)
class RedirectAlias:
    alias: str
    alias_normalized: str
    target_title: str
    target_normalized: str
    page_id: int
    revision_id: int


def redirect_alias(page: DumpPage) -> RedirectAlias | None:
    """Convert MediaWiki redirect metadata into an alias without parsing redirect text."""
    if page.redirect_target is None:
        return None
    return RedirectAlias(
        alias=page.title,
        alias_normalized=lookup_key(page.title),
        target_title=page.redirect_target,
        target_normalized=lookup_key(page.redirect_target),
        page_id=page.page_id,
        revision_id=page.revision_id,
    )

