
from ukstress_ml.ambiguity import build as build_ambiguity
from ukstress_ml.ambiguity import is_variant_notation
from ukstress_ml.inventory import Sense, parse_definition, part_of_speech


def _sense(sense_id: str, group: int, spelling: str, stressed: str, signature: str, **kw) -> Sense:
    return Sense(
        sense_id=sense_id,
        group_id=group,
        spelling=spelling,
        stressed=stressed,
        signature=signature,
        pos=kw.get("pos", "noun"),
        gram_style=kw.get("gram_style", "іменник чоловічого роду"),
        definition=kw.get("definition", "тлумачення"),
        register="",
        contrast="",
        comment="",
        priority=None,
    )


def test_definition_list_strings_are_parsed() -> None:
    definition, register = parse_definition('["Перше значення. ","Друге значення. "]')
    assert definition.startswith("Перше значення.")
    assert register == ""


def test_register_label_is_separated_from_the_gloss() -> None:
    definition, register = parse_definition("Розмовне слово чи вираз Пестити кого-небудь.")
    assert register == "Розмовне слово чи вираз"
    assert definition == "Пестити кого-небудь."


def test_part_of_speech_mapping() -> None:
    assert part_of_speech("іменник чоловічого роду, істота") == "noun"
    assert part_of_speech("дієслово доконаного виду") == "verb"
    assert part_of_speech("прикметник") == "adjective"
    assert part_of_speech("") == "unknown"


def test_variant_notation_is_distinguished_from_compounds() -> None:
    assert is_variant_notation(_sense("1.a", 1, "батьківщина", "ба́тьківщи́на", "0|2"))
    assert not is_variant_notation(_sense("2.a", 2, "альфа-розпад", "а́льфа-ро́зпад", "0:0|1:0"))
    assert not is_variant_notation(_sense("3.a", 3, "замок", "за́мок", "0"))


def test_ambiguous_surface_keeps_colliding_senses_and_drops_variant_groups() -> None:
    senses = [
        _sense("1.a", 1, "замок", "за́мок", "0"),
        _sense("1.b", 1, "замок", "замо́к", "1"),
        _sense("2.a", 2, "батьківщина", "ба́тьківщи́на", "0|2"),
        _sense("2.b", 2, "батьківщина", "батьківщи́на", "2"),
    ]
    forms, report = build_ambiguity(senses)
    assert [f.form for f in forms] == ["замок"]
    assert report["groups_excluded_variant_notation"] == 1
    assert len(forms[0].candidates) == 2
