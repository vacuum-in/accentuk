from __future__ import annotations

import json

from uktts.cli import main


def test_passthrough_mode_needs_neither_stage(capsys) -> None:
    assert main(["prepare", "--no-verbalize", "--no-stress", "Текст 5."]) == 0
    assert capsys.readouterr().out == "Текст 5.\n"


def test_json_output_carries_every_stage(capsys) -> None:
    assert main(["prepare", "--no-verbalize", "--no-stress", "--json", "Слово."]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "Слово."
    assert payload["verbalized"] == "Слово."
    assert payload["text"] == "Слово."


def test_each_line_can_be_prepared_separately(tmp_path, capsys) -> None:
    path = tmp_path / "input.txt"
    path.write_text("Перше.\nДруге.\n", encoding="utf-8")
    assert main(["prepare", "--no-verbalize", "--no-stress", "--lines", "--file", str(path)]) == 0
    assert capsys.readouterr().out == "Перше.\nДруге.\n"


def test_a_file_is_one_input_unless_lines_is_given(tmp_path, capsys) -> None:
    path = tmp_path / "input.txt"
    path.write_text("Перше.\nДруге.\n", encoding="utf-8")
    assert main(["prepare", "--no-verbalize", "--no-stress", "--file", str(path)]) == 0
    assert capsys.readouterr().out == "Перше.\nДруге.\n\n"


def test_ready_reports_a_disabled_stack_as_ready(capsys) -> None:
    assert main(["ready", "--no-verbalize", "--no-stress"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_an_unreachable_stress_service_exits_two_with_a_hint(capsys) -> None:
    code = main(
        ["prepare", "--no-verbalize", "--stress-url", "http://127.0.0.1:1", "Слово."]
    )
    assert code == 2
    assert "--no-stress" in capsys.readouterr().err
