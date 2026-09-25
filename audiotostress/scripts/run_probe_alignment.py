"""Why are the book vowel boundaries noise?

Without any model: the longest vowel in a word guesses its stress 63.4% of the
time on Common Voice against 37.6% chance, because a stressed Ukrainian vowel
is longer. On every mined book that baseline sits one point above chance, which
says the boundaries are not measurements.

The suspect is the transcript. The miner cuts a ten-minute window into sixty-
second slices by proportion and pads the word list by 35% on each side so the
true words are certainly inside — but forced alignment has no way to leave a
word out. It must place all of them, so a slice of sixty seconds is made to
hold roughly a hundred seconds of text and every boundary slides.

This aligns the same slices three ways and scores each with the same baseline:
as mined, with no padding, and against what Whisper hears in that slice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402

from run_mine_book import slice_audio  # noqa: E402
from run_mine_commonvoice import VOWELS, lexicon_ordinal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--slice", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import marisa_trie
    import torch
    import transformers

    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.config.models import AlignmentConfig

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)
    aligner = WhisperXWordAligner(AlignmentConfig(device=args.device,
                                                  return_char_alignments=True))
    import whisperx
    asr = whisperx.load_model("large-v3", args.device, compute_type="float16",
                              language="uk")

    anchors = [json.loads(line) for line in
               args.anchors.read_text(encoding="utf-8").splitlines() if line.strip()]
    anchors = [a for a in anchors if "word_start" in a][2:2 + args.windows]

    tallies = {name: [0, 0, 0.0] for name in ("as mined", "no padding", "whisper")}

    def measure(name: str, waveform, text: str) -> None:
        if not text.strip():
            return
        try:
            result, raw = aligner.align_with_chars(waveform, 16_000, text)
        except Exception as error:  # noqa: BLE001
            print(f"  {name}: {str(error)[:60]}", flush=True)
            return
        chars = sorted((c for c in _raw_chars(raw or {})
                        if c.get("start") is not None and c.get("end") is not None),
                       key=lambda c: float(c["start"]))
        for word in result.words:
            form = word.normalized_token
            expected = sum(1 for c in form if c in VOWELS)
            if expected < 2:
                continue
            inside = [c for c in chars
                      if word.start_s - 0.02 <= float(c["start"])
                      and float(c["end"]) <= word.end_s + 0.02]
            vowels = _vowels_from_whisperx_chars(inside, form)
            if len(vowels) != expected:
                continue
            durations = [v.end_s - v.start_s for v in vowels]
            if len(set(round(d, 4) for d in durations)) == 1:
                continue
            gold = lexicon_ordinal(trie, form)
            if gold is None or not 0 <= gold < len(durations):
                continue
            tallies[name][0] += int(max(range(len(durations)),
                                        key=lambda i: durations[i]) == gold)
            tallies[name][1] += 1
            tallies[name][2] += 1 / len(durations)

    for anchor in anchors:
        span = anchor.get("seconds") or 600.0
        words = anchor["text"].split()
        pieces = max(1, int(round(span / args.slice)))
        share = len(words) / pieces
        for piece in range(pieces):
            offset = piece * span / pieces
            waveform = slice_audio(args.audio, anchor["start_s"] + offset,
                                   span / pieces)
            padded = " ".join(words[max(0, int(piece * share - share * 0.35)):
                                    min(len(words), int((piece + 1) * share + share * 0.35))])
            exact = " ".join(words[int(piece * share):
                                   min(len(words), int((piece + 1) * share))])
            measure("as mined", waveform, padded)
            measure("no padding", waveform, exact)
            heard = asr.transcribe(waveform, batch_size=8, language="uk")
            measure("whisper", waveform,
                    " ".join(s["text"].strip() for s in heard["segments"]))
        print(f"window {anchor['window']} done", flush=True)

    print("\n| transcript | words | longest vowel | chance |")
    print("| --- | ---: | ---: | ---: |")
    for name, (hits, total, chance) in tallies.items():
        if total:
            print(f"| {name} | {total:,} | {hits / total:.1%} | {chance / total:.1%} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
