"""Mine stress labels from an anchored audiobook.

The anchors say which words of the book are spoken in which ten-minute window.
This aligns those words — the book's, not the ASR's — against the audio, finds
the vowels, and keeps what the lexicon can label. It is the Common Voice miner
with one substitution: the transcript comes from the anchor instead of from the
clip's metadata.

Two things it records that the Common Voice miner did not, because books are
where they matter:

* **every ambiguous form in context**, with the book position, so a homograph
  can be found again and its readings compared across books;
* **every out-of-vocabulary form**, which is 6.7% of multi-vowel words here
  against almost none in Common Voice, and is the only training material a
  stress placer will ever have for the words it actually meets.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402

from run_mine_commonvoice import VOWELS, lexicon_ordinal  # noqa: E402

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def slice_audio(path: Path, start: float, seconds: float) -> np.ndarray:
    """Decode one window with ffmpeg.

    Not soundfile: it cannot seek into compressed audio and decodes from the
    beginning, which on a fifteen-hour book costs minutes per window near the
    end. ffmpeg seeks the container.
    """
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start}", "-t", f"{seconds}",
         "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--book", required=True, help="slug recorded on every row")
    parser.add_argument("--windows", type=int, default=0, help="0 = all anchored")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--slice", type=float, default=60.0,
                        help="align in slices this long. Ten minutes is the "
                             "right unit for anchoring and the wrong one for "
                             "alignment: wav2vec2 runs a dynamic program over "
                             "every frame against every character, and a "
                             "600-second window took 8 GB of GPU memory and "
                             "produced nothing in ten minutes")
    parser.add_argument("--margin", type=float, default=0.35,
                        help="extra text given to each slice on both sides, as "
                             "a fraction of its share. The split is "
                             "proportional and therefore approximate; the "
                             "margin lets the aligner find the true boundary")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import marisa_trie
    import torch
    import transformers

    from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value
    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.config.models import AlignmentConfig
    from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)

    def readings(word: str) -> int:
        value = _trie_value(trie, unicodedata.normalize("NFC", word))
        if value is None:
            return 0
        return len({tuple(a) for _, a in _parse_dictionary_value(value[0])})

    anchors = [json.loads(line) for line in
               args.anchors.read_text(encoding="utf-8").splitlines() if line.strip()]
    anchors = [a for a in anchors if "word_start" in a][args.skip:]
    if args.windows:
        anchors = anchors[:args.windows]
    print(f"{len(anchors)} anchored windows", flush=True)

    aligner = WhisperXWordAligner(AlignmentConfig(device=args.device,
                                                  return_char_alignments=True))
    encoder = HuggingFaceSpeechEncoder(
        args.ssl_model, device=args.device,
        processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
        model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())

    labelled = ambiguous = unknown = 0
    started = time.time()
    handle = args.out.open("a", encoding="utf-8")
    try:
        for position, anchor in enumerate(anchors, 1):
            span = anchor.get("seconds", 600.0)
            words = anchor["text"].split()
            pieces = max(1, int(round(span / args.slice)))
            for piece in range(pieces):
                share = len(words) / pieces
                low = max(0, int(piece * share - share * args.margin))
                high = min(len(words), int((piece + 1) * share + share * args.margin))
                text = " ".join(words[low:high])
                if not text.strip():
                    continue
                offset = piece * span / pieces
                try:
                    waveform = slice_audio(args.audio,
                                           anchor["start_s"] + offset, span / pieces)
                    result, raw = aligner.align_with_chars(waveform, 16_000, text)
                    frames = encoder.encode(waveform, 16_000)
                except Exception as error:  # noqa: BLE001 - one slice is not fatal
                    print(f"  window {anchor['window']}.{piece}: {str(error)[:60]}",
                          flush=True)
                    continue
                chars = sorted((c for c in _raw_chars(raw or {})
                                if c.get("start") is not None and c.get("end") is not None),
                               key=lambda c: float(c["start"]))
                with torch.no_grad():
                  for word in result.words:
                    form = word.normalized_token
                    expected = sum(1 for c in form if c in VOWELS)
                    if expected < 2:
                        continue
                    inside = [c for c in chars
                              if word.start_s - 0.02 <= float(c["start"])
                              and float(c["end"]) <= word.end_s + 0.02]
                    vowels = _vowels_from_whisperx_chars(inside, form)
                    if len(vowels) < 2 or len(vowels) / expected < args.min_quality:
                        continue
                    durations = [round((v.end_s - v.start_s) * 1000) for v in vowels]
                    if len(set(durations)) == 1:
                        continue  # the aligner's fallback span, not a measurement
                    pooled = np.asarray(mean_pool_intervals(
                        frames, vowels, frame_shift_s=encoder.frame_shift_s),
                        dtype=np.float32)
                    kind = readings(form)
                    row = {
                        "book": args.book, "window": anchor["window"],
                        "doc": anchor["doc"], "paragraph": anchor["paragraph"],
                        "token": word.token, "form": form,
                        "start_s": round(anchor["start_s"] + offset + word.start_s, 3),
                        "durations_ms": durations,
                        "quality": round(len(vowels) / expected, 3),
                        "readings": kind,
                        "label": lexicon_ordinal(trie, form) if kind == 1 else None,
                        # Vowel intervals, so a second pass can pool the
                        # encoder over exactly these spans without realigning.
                        # Embeddings themselves are not stored: 1024 floats per
                        # vowel would be gigabytes of JSON, and the audio is
                        # still here.
                        "vowel_spans": [[round(v.start_s, 4), round(v.end_s, 4)]
                                        for v in vowels],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    labelled += row["label"] is not None
                    ambiguous += kind > 1
                    unknown += kind == 0
            handle.flush()
            if position % 5 == 0 or position == len(anchors):
                print(f"  {position}/{len(anchors)} windows | labelled {labelled:,} | "
                      f"ambiguous {ambiguous:,} | OOV {unknown:,} | "
                      f"{(time.time()-started)/60:.1f} min", flush=True)
    finally:
        handle.close()
    print(f"\nlabelled {labelled:,} | ambiguous {ambiguous:,} | OOV {unknown:,} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
