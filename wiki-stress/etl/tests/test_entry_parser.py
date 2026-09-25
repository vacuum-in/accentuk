from ukstress.dump_reader import DumpPage
from ukstress.entry_parser import RedirectAlias, redirect_alias


def test_redirect_becomes_alias_without_parsing_redirect_wikitext() -> None:
    page = DumpPage(
        page_id=42,
        title="п'ять",
        redirect_target="п’ять",
        revision_id=99,
        revision_timestamp="2026-01-01T00:00:00Z",
        wikitext="#REDIRECT [[ви́гаданий стрес]]",
    )

    assert redirect_alias(page) == RedirectAlias(
        alias="п'ять",
        alias_normalized="пʼять",
        target_title="п’ять",
        target_normalized="пʼять",
        page_id=42,
        revision_id=99,
    )


def test_non_redirect_has_no_alias() -> None:
    page = DumpPage(1, "мо́ва", None, 2, None, "'''мо́ва'''")
    assert redirect_alias(page) is None

