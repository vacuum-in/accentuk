from accentuk.spans import assemble, detect_spans


def test_spans_are_found_by_shape_and_prose_is_kept():
    assert [s.text for s in detect_spans("На 2 поверсі є 3 кімнати та 12 дверей.")] == ["2 поверсі є", "3 кімнати та", "12 дверей"]
    assert [s.text for s in detect_spans("Як Вічний Жид[93], і в мене також.")] == ["[93]"]
    assert detect_spans("Він відчинив замок ключем.") == []


def test_assemble_copies_everything_outside_the_spans():
    t = "Як Вічний Жид[93], і в мене також."
    assert assemble(t, detect_spans(t), [""]) == "Як Вічний Жид, і в мене також."
    t = "У графі постав Так/Ні для пунктів."
    sp = detect_spans(t)
    assert [x.text for x in sp] == ["/Ні"]
    assert assemble(t, sp, ["слеш Ні"]) == "У графі постав Так слеш Ні для пунктів."
