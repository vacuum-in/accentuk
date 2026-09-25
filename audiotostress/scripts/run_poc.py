"""End-to-end proof of concept: can a recording tell us where the stress is?

The pipeline had been built up to word boundaries and stopped there. This runs
the whole chain on one slice of an audiobook — transcribe, align, refine to
vowels, measure prosody — and scores the answer against the stress lexicon on
words the lexicon names unambiguously. Those are the only words with a truth to
compare against: for an ambiguous form the dictionary has a guess, not an
answer, and counting it would score the guess.

The book text is used as the forced-alignment transcript instead of the ASR
output where the two agree well enough to match, because alignment against the
words actually spoken is what makes the vowel boundaries trustworthy; ASR
errors move them.

Proof of concept only. It reports agreement rates and a handful of examples,
never bulk text.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

VOWELS = "аеєиіїоуюяй"
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def load_slice(path: Path, start_s: float, seconds: float) -> tuple[np.ndarray, int]:
    """Decode one window as mono 16 kHz without reading the whole file."""
    info = sf.info(str(path))
    start = int(start_s * info.samplerate)
    frames = int(seconds * info.samplerate)
    data, rate = sf.read(str(path), start=start, frames=frames,
                         dtype="float32", always_2d=True)
    waveform = data.mean(axis=1)
    if rate != 16_000:
        length = int(round(len(waveform) * 16_000 / rate))
        waveform = np.interp(
            np.linspace(0.0, len(waveform) - 1, length, dtype=np.float64),
            np.arange(len(waveform), dtype=np.float64),
            waveform,
        ).astype(np.float32)
        rate = 16_000
    return np.ascontiguousarray(waveform), rate


def book_words(pdf: Path) -> list[str]:
    """The book as a word list, with the line-wrap breaks repaired.

    Extraction splits about one word in twenty across a line; left alone those
    halves never match anything and the alignment transcript drifts.
    """
    import pypdf

    reader = pypdf.PdfReader(str(pdf))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"([а-яїієґ])-\n([а-яїієґ])", r"\1\2", text)
    text = re.sub(r"([а-яїієґ])\n([а-яїієґ])", r"\1\2", text)
    return WORD.findall(text)


def locate(spoken: list[str], book: list[str]) -> tuple[int, float] | None:
    """Where in the book this passage is, by matching the opening words."""
    needle = [w.lower() for w in spoken[:12]]
    if len(needle) < 6:
        return None
    lowered = [w.lower() for w in book]
    joined = " ".join(lowered)
    probe = " ".join(needle[:6])
    position = joined.find(probe)
    if position >= 0:
        return joined[:position].count(" "), 1.0
    matcher = difflib.SequenceMatcher(None, lowered, needle, autojunk=False)
    block = max(matcher.get_matching_blocks(), key=lambda b: b.size, default=None)
    if block is None or block.size < 4:
        return None
    return block.a - block.b, block.size / len(needle)


def lexicon_ordinal(trie, word: str) -> int | None:
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value

    value = _trie_value(trie, unicodedata.normalize("NFC", word))
    if value is None:
        return None
    positions = {tuple(a) for _, a in _parse_dictionary_value(value[0])}
    if len(positions) != 1:
        return None
    accents = next(iter(positions))
    if len(accents) != 1:
        return None
    ordinal = -1
    for index, character in enumerate(unicodedata.normalize("NFC", word), 1):
        if character.lower() in VOWELS:
            ordinal += 1
        if index == accents[0]:
            return ordinal
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("/mnt/c/lmfiles/11.22.63.mp3"))
    parser.add_argument("--pdf", type=Path,
                        default=Path("/mnt/c/lmfiles/11-22-63-65957258.pdf"))
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts/audio_runs/poc_11_22_63"))
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    waveform, rate = load_slice(args.audio, args.start, args.seconds)
    print(f"audio: {len(waveform) / rate:.0f}s from {args.start:.0f}s", flush=True)

    from ukstress.asr.whisper_backend import WhisperCompatibleBackend
    from ukstress.config.models import ASRConfig

    asr = WhisperCompatibleBackend(ASRConfig(device=args.device, compute_type="int8"))
    result = asr.transcribe(waveform, rate)
    spoken = WORD.findall(" ".join(s.text for s in result.segments))
    print(f"transcribed {len(spoken)} words", flush=True)

    book = book_words(args.pdf)
    found = locate(spoken, book)
    print(f"book words: {len(book):,} | passage located: "
          f"{'yes, offset %d (%.0f%% match)' % found if found else 'no'}", flush=True)

    # Align against the book's wording when the passage was found, the ASR's
    # otherwise: a wrong word puts the vowel boundaries in the wrong place.
    transcript = (" ".join(book[found[0]: found[0] + len(spoken)])
                  if found and found[1] >= 0.5 else " ".join(spoken))

    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import WhisperXCharFineAligner
    from ukstress.config.models import AlignmentConfig

    config = AlignmentConfig(device=args.device, return_char_alignments=True)
    words = WhisperXWordAligner(config).align(waveform, rate, transcript).words
    print(f"aligned {len(words)} words", flush=True)

    fine = WhisperXCharFineAligner(config)
    from ukstress.features.prosody import extract_prosodic_features

    import marisa_trie

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)

    refined = []
    for word in words:
        if sum(1 for c in word.normalized_token if c in VOWELS) < 2:
            continue
        try:
            result_fine = fine.align(waveform, rate, word, word.normalized_token)
        except Exception:  # noqa: BLE001 - one word cannot stop the pass
            continue
        if len(result_fine.vowels) < 2:
            continue
        refined.append((word, result_fine))
    print(f"words with two or more timed vowels: {len(refined)}", flush=True)

    every = [v for _, r in refined for v in r.vowels]
    scored = []
    for word, result_fine in refined:
        features = extract_prosodic_features(
            waveform, rate, result_fine.vowels,
            word_start_s=word.start_s, word_end_s=word.end_s,
            utterance_vowels=every,
        )
        durations = [v.end_s - v.start_s for v in result_fine.vowels]
        expected = lexicon_ordinal(trie, word.normalized_token)
        scored.append({
            "token": word.token,
            "expected": expected,
            "by_duration": int(np.argmax(durations)),
            "durations_ms": [round(d * 1000) for d in durations],
            "features": features,
            "quality": result_fine.quality,
        })

    (args.out / "scored.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=1), encoding="utf-8")
    known = [s for s in scored if s["expected"] is not None]
    print(f"\nunambiguous in the lexicon: {len(known)}")
    if known:
        hit = sum(1 for s in known if s["by_duration"] == s["expected"])
        print(f"longest vowel is the stressed one: {hit}/{len(known)} = "
              f"{100 * hit / len(known):.1f}%")
        chance = sum(1 / len(s["durations_ms"]) for s in known) / len(known)
        print(f"chance level for these words:       {100 * chance:.1f}%")
    print(f"-> {args.out}/scored.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
