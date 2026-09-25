"""MFA ``align_one`` adapter for phone and vowel intervals."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from ukstress.alignment.fine_interface import FineAlignmentResult
from ukstress.alignment.fine_validation import validate_vowel_intervals
from ukstress.datasets import VowelInterval, WordAlignment
from ukstress.lexicon.stress import UKRAINIAN_VOWELS
from ukstress.text import normalize_transcript

CommandRunner = Callable[[Sequence[str], Path], None]


class MFAFineAligner:
    """Run MFA on one aligned word and convert its phone tier to vowel intervals.

    MFA is intentionally an external dependency.  The command runner is injectable so CI and
    callers can test TextGrid parsing without installing MFA or downloading acoustic models.
    """

    def __init__(
        self,
        *,
        dictionary_path: str | Path,
        acoustic_model_path: str | Path,
        executable: str = "mfa",
        vowel_phones: Sequence[str] = ("a", "e", "i", "o", "u", "ɪ", "ɛ", "ɔ", "ʊ"),
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.dictionary_path = Path(dictionary_path)
        self.acoustic_model_path = Path(acoustic_model_path)
        self.executable = executable
        self.vowel_phones = frozenset(phone.casefold() for phone in vowel_phones)
        self._command_runner = command_runner or self._run_command

    @property
    def backend_id(self) -> str:
        return "mfa"

    @property
    def model_id(self) -> str:
        return str(self.acoustic_model_path)

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        word: WordAlignment,
        normalized_word: str,
    ) -> FineAlignmentResult:
        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError("MFA fine alignment requires a mono 16-kHz waveform")
        target = normalize_transcript(normalized_word)
        if not target or target != word.normalized_token:
            raise ValueError("normalized_word must match the word alignment token")
        start_sample = round(word.start_s * sample_rate)
        end_sample = round(word.end_s * sample_rate)
        if start_sample < 0 or end_sample > len(waveform) or start_sample >= end_sample:
            raise ValueError("word alignment is outside the supplied waveform")

        with tempfile.TemporaryDirectory(prefix="ukstress-mfa-") as temporary:
            root = Path(temporary)
            sound_path = root / "target.wav"
            text_path = root / "target.lab"
            output_path = root / "target.TextGrid"
            sf.write(
                sound_path,
                np.ascontiguousarray(waveform[start_sample:end_sample]),
                sample_rate,
            )
            text_path.write_text(f"{target}\n", encoding="utf-8")
            self._command_runner(
                [
                    self.executable,
                    "align_one",
                    str(sound_path),
                    str(text_path),
                    str(self.dictionary_path),
                    str(self.acoustic_model_path),
                    str(output_path),
                ],
                root,
            )
            phones = parse_textgrid_phone_intervals(output_path)

        vowels = self._vowels_from_phones(phones, target, word.start_s)
        result = FineAlignmentResult(
            backend=self.backend_id,
            model_id=self.model_id,
            target_word=target,
            word_start_s=word.start_s,
            word_end_s=word.end_s,
            vowels=vowels,
            quality=1.0 if vowels else 0.0,
        )
        validation = validate_vowel_intervals(
            vowels,
            target,
            word_start_s=word.start_s,
            word_end_s=word.end_s,
        )
        return result.with_rejections(validation.reasons)

    def _vowels_from_phones(
        self, phones: list[tuple[str, float, float]], target: str, offset_s: float
    ) -> list[VowelInterval]:
        graphemes = [character for character in target if character in UKRAINIAN_VOWELS]
        vowels: list[VowelInterval] = []
        for phone, start, end in phones:
            if phone.casefold() not in self.vowel_phones:
                continue
            index = len(vowels)
            if index >= len(graphemes):
                break
            vowels.append(
                VowelInterval(
                    vowel_index=index,
                    grapheme=graphemes[index],
                    phone=phone,
                    start_s=offset_s + start,
                    end_s=offset_s + end,
                )
            )
        return vowels

    @staticmethod
    def _run_command(command: Sequence[str], cwd: Path) -> None:
        subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


_INTERVAL_RE = re.compile(
    r"xmin\s*=\s*(?P<start>[-+0-9.eE]+).*?"
    r"xmax\s*=\s*(?P<end>[-+0-9.eE]+).*?"
    r"text\s*=\s*\"(?P<text>[^\"]*)\"",
    re.DOTALL,
)
_TIER_RE = re.compile(
    r"item \[\d+\]:.*?name\s*=\s*\"(?P<name>[^\"]+)\".*?"
    r"intervals:\s*size\s*=\s*\d+\s*(?P<body>.*?)(?=\n\s*item \[|\Z)",
    re.DOTALL,
)


def parse_textgrid_phone_intervals(path: str | Path) -> list[tuple[str, float, float]]:
    """Parse MFA long TextGrid phone intervals without adding a TextGrid dependency."""

    text = Path(path).read_text(encoding="utf-8")
    for tier in _TIER_RE.finditer(text):
        if tier.group("name").casefold() not in {"phones", "phone"}:
            continue
        intervals: list[tuple[str, float, float]] = []
        for interval in _INTERVAL_RE.finditer(tier.group("body")):
            start = float(interval.group("start"))
            end = float(interval.group("end"))
            label = interval.group("text").strip()
            if label and end > start:
                intervals.append((label, start, end))
        return intervals
    return []
