"""Score the vowels acoustically and check the answer against the lexicon.

This is the step the whole audio idea rests on: if the vowel a speaker actually
stressed can be read off duration, energy and pitch, then a recording is a
source of stress labels, and the label does not depend on any dictionary having
recorded the word.

So the measurement has to be against words whose stress is not in doubt. Only
forms the stress trie gives exactly one reading for are scored — for anything
ambiguous there is no ground truth here to compare with, and counting those
would measure the dictionary's guess rather than the speaker's mouth.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from ukstress.datasets.models import VowelInterval  # noqa: E402
from ukstress.features.prosody import extract_prosodic_features  # noqa: E402

VOWELS = "аеєиіїоуюяй"


def load_mono_16k(path: Path, seconds: float) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = data.mean(axis=1)
    if rate != 16_000:
        length = int(round(len(waveform) * 16_000 / rate))
        waveform = np.interp(
            np.linspace(0.0, len(waveform) - 1, length, dtype=np.float64),
            np.arange(len(waveform), dtype=np.float64),
            waveform,
        ).astype(np.float32)
        rate = 16_000
    return np.ascontiguousarray(waveform[: int(seconds * rate)]), rate


def lexicon_ordinal(trie, word: str) -> int | None:
    """The vowel the dictionary stresses, or None when it does not say plainly.

    None covers both a missing word and an ambiguous one. The trie's accent is a
    1-based character position into the composed form, so it is converted to a
    vowel ordinal here — the two are different counts and conflating them marks
    the wrong syllable.
    """
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value

    value = _trie_value(trie, unicodedata.normalize("NFC", word))
    if value is None:
        return None
    positions = {tuple(accents) for _, accents in _parse_dictionary_value(value[0])}
    if len(positions) != 1:
        return None
    accents = next(iter(positions))
    if len(accents) != 1:
        return None
    composed = unicodedata.normalize("NFC", word)
    ordinal = -1
    for index, character in enumerate(composed, 1):
        if character.lower() in VOWELS:
            ordinal += 1
        if index == accents[0]:
            return ordinal
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("artifacts/audio_runs/napivdykyj_00_demo/first_minute"))
    parser.add_argument("--audio", type=Path,
                        default=Path("/mnt/c/lmfiles/napivdykyj_00_demo.mp3"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    args = parser.parse_args()

    import marisa_trie

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)

    words = json.loads((args.run / "vowels.json").read_text(encoding="utf-8"))["words"]
    waveform, rate = load_mono_16k(args.audio, args.seconds)
    every = [VowelInterval.model_validate(v) for row in words for v in row["vowels"]]

    scored, agree = [], 0
    for row in words:
        vowels = [VowelInterval.model_validate(v) for v in row["vowels"]]
        if len(vowels) < 2:
            continue  # one vowel carries the stress by default; nothing to decide
        features = extract_prosodic_features(
            waveform, rate, vowels,
            word_start_s=row["word_start_s"], word_end_s=row["word_end_s"],
            utterance_vowels=every,
        )
        durations = [v.end_s - v.start_s for v in vowels]
        energies = [float(f.get("energy_ratio") or f.get("rms_ratio") or 0.0)
                    for f in features]
        by_duration = int(np.argmax(durations))
        # Duration and loudness together, equally weighted: neither alone is
        # what a listener hears as stress, and this is the simplest combination
        # that uses both without a trained weight to justify.
        combined = int(np.argmax([d / max(durations) + e / (max(energies) or 1.0)
                                  for d, e in zip(durations, energies, strict=True)]))
        expected = lexicon_ordinal(trie, row["token"].lower())
        scored.append({
            "token": row["token"], "expected": expected,
            "by_duration": by_duration, "combined": combined,
            "durations_ms": [round(d * 1000) for d in durations],
            "quality": row["quality"],
        })
        if expected is not None and combined == expected:
            agree += 1

    known = [s for s in scored if s["expected"] is not None]
    out = args.run / "acoustic_stress.json"
    out.write_text(json.dumps(scored, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"multi-vowel words scored: {len(scored)}")
    print(f"of those the trie names unambiguously: {len(known)}")
    if known:
        dur = sum(1 for s in known if s["by_duration"] == s["expected"])
        print(f"  duration alone agrees:      {dur}/{len(known)} = "
              f"{100 * dur / len(known):.1f}%")
        print(f"  duration + energy agrees:   {agree}/{len(known)} = "
              f"{100 * agree / len(known):.1f}%")
        print("\nexamples:")
        for s in known[:12]:
            mark = "ok " if s["combined"] == s["expected"] else "MISS"
            print(f"  {mark} {s['token']:<14} lexicon {s['expected']}  "
                  f"acoustic {s['combined']}  durations(ms) {s['durations_ms']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
