"""Mine stress labels from a recording, one window at a time, resumably.

The proof of concept established that no single acoustic feature finds the
stressed vowel: duration lands 51.1%, F0 median 56.5%, energy 42.0%, against a
38.4% chance level on the same words. That is the expected shape — a listener
hears stress as a combination, and RUAccent trained a classifier over the whole
vector rather than thresholding one number. Training one needs rows, and this
produces them.

Two filters decide what is worth keeping, and both come from watching the POC
fail:

* WhisperX gives up on individual words ("backtrack failed, resorting to
  original") and falls back to the word-level span. The vowels it then reports
  are not measurements, and they are recognisable: every vowel comes out the
  same length. Those rows are dropped.
* Only forms the stress trie names unambiguously carry a label. For anything
  else the dictionary has a guess, and training on a guess teaches the guess.

Rows append to JSONL so a long run survives being stopped, and each window is
recorded even when it yields nothing, so a resume does not redo it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

VOWELS = "аеєиіїоуюяй"
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def load_slice(path: Path, start_s: float, seconds: float) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    data, rate = sf.read(str(path), start=int(start_s * info.samplerate),
                         frames=int(seconds * info.samplerate),
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
    # A decoded WAV, not the mp3: soundfile cannot seek into compressed audio
    # and decodes from the start instead, at about 9.7 ms per second of offset.
    # Mining walks forward, so every window paid for all the audio before it —
    # 117 s of a 312 s window at the three-hour mark, and 19 minutes near the
    # end of this book. See scripts/run_decode_audio.py.
    parser.add_argument("--audio", type=Path,
                        default=Path("artifacts/11.22.63.16k.wav"))
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--window", type=float, default=120.0)
    parser.add_argument("--windows", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ssl", action="store_true",
                        help="also pool wav2vec2 frames over each vowel")
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
                        help="the same checkpoint the aligner uses")
    parser.add_argument("--asr-model", default="large-v3",
                        help="only has to be good enough to place the passage "
                             "when --book is given; alignment then uses the "
                             "book's own words")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--slow", action="store_true",
                        help="re-align every word individually, as the first "
                             "version did; kept only to reproduce old numbers")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/mined/rows.jsonl"))
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_path = args.out.with_suffix(".windows.json")

    done: set[float] = set()
    if done_path.is_file():
        done = set(json.loads(done_path.read_text(encoding="utf-8")))
        print(f"resuming: {len(done)} windows already processed", flush=True)

    import marisa_trie

    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        WhisperXCharFineAligner,
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.asr.whisper_backend import WhisperCompatibleBackend
    from ukstress.config.models import AlignmentConfig, ASRConfig
    from ukstress.features.prosody import extract_prosodic_features, f0_track

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)
    asr = WhisperCompatibleBackend(ASRConfig(
        model_id=args.asr_model, device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8"))
    config = AlignmentConfig(device=args.device, return_char_alignments=True)
    word_aligner = WhisperXWordAligner(config)
    fine = WhisperXCharFineAligner(config)

    encoder = None
    ssl_blocks: list[np.ndarray] = []
    if args.ssl:
        from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals

        # A feature extractor, not AutoProcessor: this checkpoint ships a
        # KenLM decoder that AutoProcessor insists on loading, and the decoder
        # is for transcription. Frame embeddings need neither it nor the two
        # extra dependencies it drags in.
        import transformers

        encoder = HuggingFaceSpeechEncoder(
            args.ssl_model, device=args.device,
            processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
            model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())

    kept = 0
    started = time.time()
    handle = args.out.open("a", encoding="utf-8")
    try:
        for index in range(args.windows):
            offset = args.start + index * args.window
            if offset in done:
                continue
            try:
                waveform, rate = load_slice(args.audio, offset, args.window)
            except Exception as error:  # noqa: BLE001 - a bad window is not fatal
                print(f"  {offset:.0f}s unreadable: {error}", flush=True)
                continue
            if len(waveform) < rate:
                break

            stage = time.time()
            transcript = " ".join(
                s.text for s in asr.transcribe(waveform, rate).segments).strip()
            asr_s = time.time() - stage
            if not transcript:
                done.add(offset)
                continue
            raw = None
            stage = time.time()
            try:
                if args.slow:
                    words = word_aligner.align(waveform, rate, transcript).words
                else:
                    result, raw = word_aligner.align_with_chars(waveform, rate, transcript)
                    words = result.words
            except Exception as error:  # noqa: BLE001
                print(f"  {offset:.0f}s alignment failed: {str(error)[:70]}", flush=True)
                done.add(offset)
                continue
            # The characters come from the same pass as the words, so a window
            # costs one forward pass instead of one per word.
            align_s = time.time() - stage
            stage = time.time()
            chars = sorted(
                (c for c in _raw_chars(raw or {})
                 if c.get("start") is not None and c.get("end") is not None),
                key=lambda c: float(c["start"]),
            ) if raw is not None else []

            refined = []
            for word in words:
                if sum(1 for c in word.normalized_token if c in VOWELS) < 2:
                    continue
                label = lexicon_ordinal(trie, word.normalized_token)
                if label is None:
                    continue
                if args.slow:
                    try:
                        result = fine.align(waveform, rate, word, word.normalized_token)
                    except Exception:  # noqa: BLE001 - one word cannot stop a window
                        continue
                    vowels, quality = result.vowels, result.quality
                else:
                    inside = [c for c in chars
                              if word.start_s - 0.02 <= float(c["start"])
                              and float(c["end"]) <= word.end_s + 0.02]
                    vowels = _vowels_from_whisperx_chars(inside, word.normalized_token)
                    expected = sum(1 for c in word.normalized_token if c in VOWELS)
                    quality = len(vowels) / expected if expected else 0.0
                if len(vowels) < 2 or quality < args.min_quality:
                    continue
                result = SimpleNamespace(vowels=vowels, quality=quality)
                durations = [round((v.end_s - v.start_s) * 1000) for v in vowels]
                if len(set(durations)) == 1:
                    continue  # the fallback span, not a measurement
                refined.append((word, result, label, durations))

            vowel_s = time.time() - stage
            stage = time.time()
            # One encoder pass for the window; the frames are then pooled per
            # vowel. Encoding each word separately would repeat the same work
            # the F0 track already taught us not to repeat.
            frames = encoder.encode(waveform, rate) if encoder is not None else None
            every = [v for _, r, _, _ in refined for v in r.vowels]
            # One F0 track for the window, handed to every word in it.
            f0_cache = f0_track(waveform, rate, min_hz=60.0, max_hz=400.0,
                                frame_ms=40, hop_ms=10) if refined else None
            for word, result, label, durations in refined:
                features = extract_prosodic_features(
                    waveform, rate, result.vowels,
                    word_start_s=word.start_s, word_end_s=word.end_s,
                    utterance_vowels=every, f0_cache=f0_cache,
                )
                ssl_index = None
                if frames is not None:
                    pooled = mean_pool_intervals(
                        frames, result.vowels,
                        frame_shift_s=encoder.frame_shift_s)
                    ssl_index = [len(ssl_blocks), len(ssl_blocks) + len(pooled)]
                    ssl_blocks.extend(np.asarray(v, dtype=np.float16) for v in pooled)
                handle.write(json.dumps({
                    "token": word.token,
                    "ssl_index": ssl_index,
                    "form": word.normalized_token,
                    "label": label,
                    "offset_s": offset,
                    "quality": result.quality,
                    "durations_ms": durations,
                    "features": features,
                }, ensure_ascii=False) + "\n")
                kept += 1
            handle.flush()
            prosody_s = time.time() - stage
            done.add(offset)
            done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
            elapsed = time.time() - started
            print(f"  window {index + 1}/{args.windows} at {offset:.0f}s: "
                  f"+{len(refined)} rows, {kept} total, {elapsed / 60:.1f} min | "
                  f"asr {asr_s:.0f}s  align {align_s:.0f}s  "
                  f"vowels {vowel_s:.0f}s  prosody {prosody_s:.0f}s", flush=True)
    finally:
        handle.close()

    if ssl_blocks:
        # A separate matrix, not JSON: 1024 floats per vowel would make the
        # rows file unreadable and enormous.
        matrix = np.vstack(ssl_blocks)
        target = args.out.with_suffix(".ssl.npy")
        np.save(target, matrix)
        print(f"ssl features: {matrix.shape} -> {target}")

    print(f"\nkept {kept:,} labelled words -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
