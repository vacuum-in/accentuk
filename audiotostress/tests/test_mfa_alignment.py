from pathlib import Path
from typing import Any

import numpy as np

from ukstress.alignment import MFAFineAligner
from ukstress.datasets import WordAlignment

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 0.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.5
        intervals: size = 3
        intervals [1]:
            xmin = 0.00
            xmax = 0.08
            text = "z"
        intervals [2]:
            xmin = 0.08
            xmax = 0.20
            text = "a"
        intervals [3]:
            xmin = 0.30
            xmax = 0.42
            text = "o"
'''


def test_mfa_adapter_runs_align_one_and_maps_vowels_to_absolute_offsets(tmp_path: Path) -> None:
    commands: list[tuple[list[str], Path]] = []

    def runner(command: Any, cwd: Path) -> None:
        commands.append((list(command), cwd))
        (cwd / "target.TextGrid").write_text(TEXTGRID, encoding="utf-8")

    backend = MFAFineAligner(
        dictionary_path=tmp_path / "uk.dict",
        acoustic_model_path=tmp_path / "uk.zip",
        command_runner=runner,
    )
    word = WordAlignment(
        token="замок",
        normalized_token="замок",
        start_s=1.0,
        end_s=1.5,
        backend="whisperx",
    )
    result = backend.align(np.zeros(32_000, dtype=np.float32), 16_000, word, "замок")

    assert commands[0][0][0:2] == ["mfa", "align_one"]
    assert commands[0][0][3].endswith("target.lab")
    assert [(v.grapheme, v.phone, v.start_s, v.end_s) for v in result.vowels] == [
        ("а", "a", 1.08, 1.2),
        ("о", "o", 1.3, 1.42),
    ]
    assert result.quality == 1.0
    assert result.rejection_reasons == []


def test_mfa_textgrid_parser_returns_empty_for_missing_phone_tier(tmp_path: Path) -> None:
    from ukstress.alignment.mfa_backend import parse_textgrid_phone_intervals

    path = tmp_path / "empty.TextGrid"
    path.write_text('name = "words"\n', encoding="utf-8")
    assert parse_textgrid_phone_intervals(path) == []
