import pytest

from uktts.segment import Chunk, fit_to_window, join, split_sentences

TEXTS = [
    "",
    "   ",
    "Привіт.",
    "Привіт! Як справи? Добре…",
    "Уроки 25 травня 2024 р. о 14:30 коштують 1 500 грн.",
    "Ціна 3.5 млн грн. Це багато.",
    "Т. Шевченко народився 1814 р. у селі Моринці.",
    "Перший рядок\nДругий рядок\n\nТретій.",
    "Див. напр. праці проф. Іваненка. Вони відомі.",
    "«Цитата.» Далі текст.",
    "Сайт www.example.com працює. Заходьте.",
]


@pytest.mark.parametrize("text", TEXTS)
def test_split_is_lossless(text: str) -> None:
    assert join(split_sentences(text)) == text


@pytest.mark.parametrize("text", TEXTS)
def test_fit_is_lossless(text: str) -> None:
    chunks = fit_to_window(split_sentences(text), lambda value: len(value.split()), 3)
    assert join(chunks) == text


def test_model_input_never_carries_surrounding_whitespace() -> None:
    for chunk in split_sentences("  Привіт світ.  Ще речення.  "):
        if chunk.speakable:
            assert chunk.text == chunk.text.strip()


def test_abbreviation_period_does_not_end_a_sentence() -> None:
    chunks = split_sentences("Подія 2024 р. о 14:30 відбулась.")
    assert [chunk.text for chunk in chunks if chunk.speakable] == [
        "Подія 2024 р. о 14:30 відбулась."
    ]


def test_initial_does_not_end_a_sentence() -> None:
    chunks = split_sentences("Т. Шевченко писав вірші.")
    assert [chunk.text for chunk in chunks if chunk.speakable] == ["Т. Шевченко писав вірші."]


def test_decimal_point_does_not_end_a_sentence() -> None:
    chunks = split_sentences("Курс 41.5 гривні.")
    assert [chunk.text for chunk in chunks if chunk.speakable] == ["Курс 41.5 гривні."]


def test_newline_ends_a_line_without_terminal_punctuation() -> None:
    chunks = split_sentences("Заголовок розділу\nПерший абзац тексту")
    assert [chunk.text for chunk in chunks if chunk.speakable] == [
        "Заголовок розділу",
        "Перший абзац тексту",
    ]


def test_blank_line_separates_paragraphs() -> None:
    text = "Абзац перший.\n\nАбзац другий."
    chunks = split_sentences(text)
    assert join(chunks) == text
    assert [chunk.text for chunk in chunks if chunk.speakable] == [
        "Абзац перший.",
        "Абзац другий.",
    ]


def test_terminal_punctuation_starts_a_new_sentence() -> None:
    chunks = split_sentences("Одне речення. Друге речення.")
    assert [chunk.text for chunk in chunks if chunk.speakable] == [
        "Одне речення.",
        "Друге речення.",
    ]


def test_long_sentence_is_cut_at_clause_seams() -> None:
    text = "Раз, два, три, чотири, п'ять, шість."
    warnings: list[str] = []
    chunks = fit_to_window(
        split_sentences(text), lambda value: len(value.split()), 2, warnings
    )
    assert join(chunks) == text
    assert all(len(c.text.split()) <= 2 for c in chunks if c.speakable)
    assert warnings and "source window" in warnings[0]


def test_unbreakable_run_falls_back_to_word_boundaries() -> None:
    text = "слово " * 20
    chunks = fit_to_window([Chunk(text.strip(), True)], lambda v: len(v.split()), 3)
    assert join(chunks) == text.strip()
    assert all(len(c.text.split()) <= 3 for c in chunks if c.speakable)
