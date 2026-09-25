"""Run the character-level aligner over a word alignment and keep the vowels.

The demo run stopped at word boundaries. Stress lives inside the word, so what
the pipeline needs next is a timed interval per vowel — which
`WhisperXCharFineAligner` already produces and which nothing was calling. This
wires the two together over an existing run and reports how often the vowels
actually land.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from ukstress.alignment.whisperx_char_backend import WhisperXCharFineAligner  # noqa: E402
from ukstress.config.models import AlignmentConfig  # noqa: E402
from ukstress.datasets.models import WordAlignment  # noqa: E402

VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")


def load_mono_16k(path: Path, seconds: float) -> tuple[np.ndarray, int]:
    """Read the head of a file as mono 16 kHz, which is what the aligner takes."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("artifacts/audio_runs/napivdykyj_00_demo/first_minute"))
    parser.add_argument("--audio", type=Path,
                        default=Path("/mnt/c/lmfiles/napivdykyj_00_demo.mp3"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="0 means every word")
    args = parser.parse_args()

    alignment = json.loads((args.run / "alignment.json").read_text(encoding="utf-8"))
    words = [WordAlignment.model_validate(row) for row in alignment["words"]]
    if args.limit:
        words = words[: args.limit]
    print(f"words to refine: {len(words)}", flush=True)

    waveform, rate = load_mono_16k(args.audio, args.seconds)
    print(f"audio: {len(waveform) / rate:.1f}s mono @ {rate} Hz", flush=True)

    aligner = WhisperXCharFineAligner(
        AlignmentConfig(device=args.device, return_char_alignments=True))

    results: list[dict] = []
    failures: list[dict] = []
    for index, word in enumerate(words, 1):
        expected = sum(1 for character in word.normalized_token if character in VOWELS)
        if expected == 0:
            continue
        try:
            fine = aligner.align(waveform, rate, word, word.normalized_token)
        except Exception as error:  # noqa: BLE001 - one word cannot stop the pass
            failures.append({"token": word.token, "error": str(error)[:120]})
            continue
        results.append({
            "token": word.token,
            "word_start_s": fine.word_start_s,
            "word_end_s": fine.word_end_s,
            "quality": fine.quality,
            "expected_vowels": expected,
            "found_vowels": len(fine.vowels),
            "vowels": [v.model_dump() for v in fine.vowels],
            "rejections": list(getattr(fine, "rejections", []) or []),
        })
        if index % 25 == 0:
            print(f"  {index}/{len(words)}  kept {len(results)}  failed {len(failures)}",
                  flush=True)

    out = args.run / "vowels.json"
    out.write_text(json.dumps({"words": results, "failures": failures},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    complete = [r for r in results if r["found_vowels"] == r["expected_vowels"]]
    print(f"\nrefined {len(results)} words, {len(failures)} failed")
    if results:
        print(f"all vowels found: {len(complete)}/{len(results)} "
              f"({100 * len(complete) / len(results):.1f}%)")
        mean = sum(r["quality"] for r in results) / len(results)
        print(f"mean vowel-alignment quality: {mean:.3f}")
    for row in failures[:5]:
        print(f"  failed: {row['token']}: {row['error']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
