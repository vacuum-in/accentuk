import importlib.util
import unicodedata
from pathlib import Path

import pytest


def test_package_exists():
    assert importlib.util.find_spec('accentuk') is not None


def test_normalization_vectors():
    from accentuk.normalizer import canonical_stressed_form, lookup_key
    ref = Path('wiki-stress/etl/src/ukstress/normalizer.py')
    if not ref.is_file():
        import pytest
        pytest.skip('needs the accentuk-api checkout (wiki-stress/etl) next to this repository')
    spec = importlib.util.spec_from_file_location('reference_normalizer', ref)
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for value in ('мо́ва', "п'ять", 'п’ять', 'пів—яблука', 'ҐАНОК', 'райо́н', 'ї́жа'):
        assert lookup_key(value) == module.lookup_key(value)
        assert canonical_stressed_form(value) == module.canonical_stressed_form(value)


def test_signatures_preserve_surface():
    from accentuk.text import apply_signature
    for raw, sig, expected in [
        ('район', '2', 'райо́н'),
        ('війна', '2', 'війна́'),
        ("м'ясо", '0', "м'я́со"),
        ('їжа', '0', 'ї́жа'),
        ('держава', '0|1', 'де́ржава'),
        ('пів-яблука', '0:0|1:0', 'пі́в-я́блука'),
        ('МОВА', '0', 'МО́ВА'),
        ('пів-яблука', '1:0', 'пів-я́блука'),
        ("п'ятьох", '1', "п'ятьо́х"),
        ('їхати', '0', 'ї́хати'),
        ("здоров'я", '1', "здоро́в'я"),
        ("сім'я", '0', "сі́м'я"),
        ("полум'ям", '0', "по́лум'ям"),
        ('дівчина', '0|1', 'ді́вчина'),
        ('років', '0|1', 'ро́ків'),
        ('району', '2', 'райо́ну'),
        ('їжак', '0', 'ї́жак'),
    ]:
        assert apply_signature(raw, sig) == expected
    raw = unicodedata.normalize('NFD', 'їжа')
    assert apply_signature(raw, '0') == raw[:2] + '\u0301' + raw[2:]


def test_apply_signature_invalid():
    from accentuk.text import apply_signature
    with pytest.raises(ValueError):
        apply_signature('мо́ва', '0')  # already has acute
    with pytest.raises(ValueError):
        apply_signature('мова', '')  # empty signature
    with pytest.raises(ValueError):
        apply_signature('мова', '-1')  # negative ordinal
    with pytest.raises(ValueError):
        apply_signature('мова', '0:-1')  # negative ordinal
    with pytest.raises(ValueError):
        apply_signature('мова', 'abc')  # invalid format
    with pytest.raises(ValueError):
        apply_signature('мова', '5')  # signature does not fit surface (too few vowels)


def test_word_spans():
    from accentuk.text import word_tokens
    text = "🦉 'Мої сестри.' м’ясо пів—яблука мо́ва -"
    tokens = word_tokens(text)
    words = ['Мої', 'сестри', 'м’ясо', 'пів—яблука', 'мо́ва']
    assert [(t.start, t.end, t.text) for t in tokens] == [
        (text.index(w), text.index(w) + len(w), w) for w in words
    ]
    # Slices match exactly
    for t in tokens:
        assert text[t.start:t.end] == t.text


def test_by_capitalisation_go_vectors():
    from accentuk.text import by_capitalisation
    proper = {"русі\x001": True}
    both = ["1", "0"]
    # Lowercase русі should be the common noun
    got = by_capitalisation(both, proper, "русі", False, form="русі")
    assert got == ["0"]

    # Capitalised mid-sentence Русі should be the proper noun
    got = by_capitalisation(both, proper, "Русі", False, form="русі")
    assert got == ["1"]

    # Capital opening a sentence says nothing
    got = by_capitalisation(both, proper, "Русі", True, form="русі")
    assert got == ["1", "0"]

    # Partial narrowing must reorder, not drop
    three = ["2", "1", "0"]
    three_proper = {"форма\x002": True}
    got = by_capitalisation(three, three_proper, "форма", False, form="форма")
    assert got == ["1", "0", "2"]


def test_sentence_initial_go_vectors():
    from accentuk.text import sentence_initial
    cases = [
        ("Русі", "Русі", True),
        ("Він жив у Київській Русі.", "Русі", False),
        ("Кінець. Русі багато.", "Русі", True),
        ("— «Русі", "Русі", True),
    ]
    for text, word, expected in cases:
        at = text.index(word)
        assert sentence_initial(text, at) == expected


def test_signature_of_output_go_vectors():
    from accentuk.text import signature_of_output
    cases = {
        "за́мок": "0",
        "замо́к": "1",
        "райо́н": "2",
        "Сестри́": "1",
        "замок": "",
    }
    for output, want in cases.items():
        assert signature_of_output(output) == want
