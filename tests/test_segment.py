from accentuk.segment import fit_to_window, join, split_sentences


def test_split_sentences_lossless():
    texts = [
        "Це перше речення. А це друге! І третє?",
        "2024 р. о 14:30 відбудеться зустріч.\nНовий рядок.",
        "Т. Шевченко написав Кобзар.",
        "Просто текст без крапки",
        "  Пробіли на початку та в кінці.  ",
    ]
    for text in texts:
        chunks = split_sentences(text)
        assert join(chunks) == text


def test_abbreviation_not_sentence_end():
    text = "2024 р. у Києві."
    chunks = split_sentences(text)
    # Shouldn't split at р.
    speakable = [c.text for c in chunks if c.speakable]
    assert len(speakable) == 1
    assert speakable[0] == "2024 р. у Києві."


def test_fit_to_window():
    text = "Це дуже довге речення, яке має підрядну частину; а також іншу частину."
    chunks = split_sentences(text)

    # Budget of 5 words
    def count(s: str) -> int:
        return len(s.split())

    fitted = fit_to_window(chunks, count, budget=5)
    assert join(fitted) == text
    assert len(fitted) >= len(chunks)
