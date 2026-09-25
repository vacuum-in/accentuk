import pytest

from uktts.route import needs_verbalization

PLAIN = [
    "Авторів собака сидів спокійно.",
    "На дубі, ялиці та березі розкинулося довге гілля.",
    "Три сестри прийшли додому.",
    "Він не має руки.",
    "«Цитата» — і тире, й лапки, й апостроф: сім'я.",
    "Києво-Могилянська академія.",
]

NEEDS = [
    "Зустріч о 14:30 коштує 1 500 грн.",
    "Компанія Apple випустила iPhone 15 Pro.",
    "Температура від -5 до +12 °C.",
    "Ціна зросла на 50%.",
    "Подія 2024 р. відбулась.",
    "Кілька тис. осіб зібралися.",
    "Делегація ООН прибула вчора.",
    "Формула H₂O.",
    "Розділ XX Конституції.",
    "Пиши на пошту: скринька@приклад.",
]


@pytest.mark.parametrize("text", PLAIN)
def test_plain_ukrainian_is_passed_through(text: str) -> None:
    assert not needs_verbalization(text)


@pytest.mark.parametrize("text", NEEDS)
def test_anything_to_spell_out_goes_to_the_model(text: str) -> None:
    assert needs_verbalization(text)


def test_ordinary_punctuation_is_not_a_symbol() -> None:
    assert not needs_verbalization("Слово, ще слово; і третє — крапка.")


def test_an_abbreviation_without_a_period_is_not_enough() -> None:
    # `тис` as an ordinary word is not the abbreviation `тис.`
    assert not needs_verbalization("Він тис руку міцно.")


def test_a_single_capital_letter_is_not_an_abbreviation() -> None:
    assert not needs_verbalization("Слово на початку речення.")


def test_an_all_capitals_heading_is_not_an_acronym() -> None:
    # `ІСТОРИЧНИЙ НАРИС` came back from the model as `і ес те оРИЧНИЙ НАРИС`.
    assert not needs_verbalization("ІСТОРИЧНИЙ НАРИС")
    assert not needs_verbalization("РОЗДІЛ ПЕРШИЙ")


def test_an_acronym_inside_ordinary_text_still_routes() -> None:
    assert needs_verbalization("Нота про твердження президента СДА")


def test_an_initial_is_not_an_abbreviation_to_expand() -> None:
    # The model dropped the initial from `С. Смеречинський` entirely.
    assert not needs_verbalization("У праці С. Смеречинський зупиняється на цьому.")
