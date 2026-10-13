from accentuk.route import needs_verbalization


def test_needs_verbalization():
    # Plain text should bypass verbalizer
    assert not needs_verbalization("сидів спокійно")
    assert not needs_verbalization("на березі річки")
    assert not needs_verbalization("довгі коси спадали на плечі")

    # Digits, dates, currency, symbols need verbalization
    assert needs_verbalization("2024 р.")
    assert needs_verbalization("о 14:30")
    assert needs_verbalization("1 500 грн")
    assert needs_verbalization("50%")
    assert needs_verbalization("№ 5")
    assert needs_verbalization("USD")

    # Abbreviations with dot
    assert needs_verbalization("кілька тис. осіб")

    # Heading in all capitals should NOT trigger verbalization
    assert not needs_verbalization("ІСТОРИЧНИЙ НАРИС")
