"""Mine stress labels from Common Voice, one clip at a time, resumably.

The audiobook miner had to guess what was said: it ran ASR over a window and
aligned against whatever came back. Common Voice ships the sentence with the
clip, so the transcript is exact and ASR leaves the loop entirely — which is
both faster and more honest, because a misheard word used to become a wrongly
labelled row.

The other thing it brings is people. Every model trained here so far learned
one narrator's voice, and a stress classifier that only works on him is not a
stress classifier. Validated Ukrainian is 118 hours over 1,107 speakers, and
each row records whose voice it came from, so a benchmark can hold speakers
out rather than holding out sentences the same speaker also read.

Filters carry over from the audiobook miner unchanged: a word needs two
vowels, an unambiguous trie entry, enough characters aligned, and vowels that
are not all the same length (that pattern is WhisperX falling back to the word
span, not a measurement).

SSL vectors go to a raw float16 file rather than growing a list in memory —
the audiobook run held its blocks until the end, which is fine for 4,000 rows
and not fine for a hundred thousand.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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


def load_clip(path: Path) -> tuple[np.ndarray, int]:
    """Read one clip as 16 kHz mono.

    Common Voice is 48 kHz mp3, so this decimates by three. resample_poly
    filters first; np.interp would alias the top of the band straight down
    onto the formants the features are meant to measure.
    """
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = data.mean(axis=1)
    if rate != 16_000:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(rate), 16_000)
        waveform = resample_poly(waveform, 16_000 // divisor, int(rate) // divisor)
        rate = 16_000
    return np.ascontiguousarray(waveform, dtype=np.float32), rate


def part_pool(frames, intervals, *, frame_shift_s: float, parts: int):
    """Whole-vowel mean, then the mean of each of `parts` equal slices.

    Keeping the mean in the first columns makes the comparison exact: an arm
    that reads only those columns sees precisely what the mean-pooled run saw,
    on the same rows from the same audio.
    """
    from ukstress.features.ssl import time_to_frame_bounds

    pooled = []
    for interval in intervals:
        start, end = time_to_frame_bounds(
            interval.start_s, interval.end_s,
            frame_shift_s=frame_shift_s, frame_count=frames.shape[0])
        end = max(end, start + 1)
        block = frames[start:end]
        vector = [np.mean(block, axis=0, dtype=np.float64)]
        edges = np.linspace(0, len(block), parts + 1).astype(int)
        for index in range(parts):
            low, high = edges[index], max(edges[index + 1], edges[index] + 1)
            vector.append(np.mean(block[low:min(high, len(block))] if low < len(block)
                                  else block[-1:], axis=0, dtype=np.float64))
        pooled.append(np.concatenate(vector))
    return pooled


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


WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def previous_of(sentence: str, form: str) -> str:
    """The word before this one, kept so context questions stay answerable.

    The first use is prepositional clitics: the trie labels мене́, the speaker
    says до ме́не, and without the preceding word the row looks like a model
    error rather than a mislabelled one.
    """
    words = [w.lower() for w in WORD.findall(sentence)]
    target = form.lower()
    for index, word in enumerate(words):
        if word == target and index:
            return words[index - 1]
    return ""


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))


def split_map(root: Path) -> dict[str, str]:
    """Which official split each clip belongs to.

    Common Voice builds train/dev/test so that no speaker appears in two of
    them. Recording the split keeps that property available later instead of
    inviting a random shuffle to destroy it.
    """
    mapping: dict[str, str] = {}
    for name in ("train", "dev", "test"):
        path = root / f"{name}.tsv"
        if path.is_file():
            for row in read_rows(path):
                mapping[row["path"]] = name
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/cv-corpus-26.0-2026-06-12/uk"))
    parser.add_argument("--tsv", default="validated.tsv")
    parser.add_argument("--clips", type=int, default=2000)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--max-down-votes", type=int, default=0)
    parser.add_argument("--per-speaker", type=int, default=0,
                        help="cap clips per speaker so one prolific voice "
                             "cannot dominate the training set (0 = no cap). "
                             "The busiest voice here reads 7,346 of the 73,166 "
                             "validated clips, a tenth of the corpus on its own.")
    parser.add_argument("--seed", type=int, default=17,
                        help="shuffles the queue, so a run that stops early "
                             "is still a sample of the whole corpus rather "
                             "than of whoever sorts first")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--ssl-parts", type=int, default=0,
                        help="also pool each vowel in N equal slices and "
                             "concatenate them after the whole-vowel mean. "
                             "A single mean throws away the shape of the "
                             "vowel, and the rise and fall inside it is a "
                             "stress cue; with N the first 1024 columns are "
                             "still exactly the mean, so both can be trained "
                             "on the same rows")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/commonvoice/rows.jsonl"))
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_path = args.out.with_suffix(".clips.json")
    ssl_path = args.out.with_suffix(".ssl.f16")

    done: set[str] = set()
    if done_path.is_file():
        done = set(json.loads(done_path.read_text(encoding="utf-8")))
        print(f"resuming: {len(done):,} clips already processed", flush=True)

    catalogue = read_rows(args.corpus / args.tsv)
    splits = split_map(args.corpus)
    seen_per_speaker: dict[str, int] = {}
    queue: list[dict] = []
    for row in catalogue:
        if int(row.get("up_votes") or 0) < args.min_votes:
            continue
        if int(row.get("down_votes") or 0) > args.max_down_votes:
            continue
        if args.per_speaker:
            count = seen_per_speaker.get(row["client_id"], 0)
            if count >= args.per_speaker:
                continue
            seen_per_speaker[row["client_id"]] = count + 1
        queue.append(row)
    random.Random(args.seed).shuffle(queue)
    queue = [r for r in queue if r["path"] not in done]
    queue = queue[args.skip:args.skip + args.clips]
    speakers = len({r["client_id"] for r in queue})
    print(f"{len(queue):,} clips queued from {speakers:,} speakers", flush=True)
    if not queue:
        return 0

    import marisa_trie

    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.config.models import AlignmentConfig
    from ukstress.features.prosody import extract_prosodic_features, f0_track

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)
    config = AlignmentConfig(device=args.device, return_char_alignments=True)
    word_aligner = WhisperXWordAligner(config)

    encoder = None
    ssl_dim = 0
    ssl_written = 0
    if args.ssl:
        import transformers

        from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals

        encoder = HuggingFaceSpeechEncoder(
            args.ssl_model, device=args.device,
            processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
            model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())
        meta_path = args.out.with_suffix(".ssl.json")
        if meta_path.is_file():
            ssl_dim = json.loads(meta_path.read_text(encoding="utf-8"))["dim"]
            ssl_written = ssl_path.stat().st_size // (ssl_dim * 2)

    kept = 0
    failed = 0
    started = time.time()
    handle = args.out.open("a", encoding="utf-8")
    ssl_handle = ssl_path.open("ab") if args.ssl else None
    try:
        for index, entry in enumerate(queue, 1):
            clip = args.corpus / "clips" / entry["path"]
            sentence = entry["sentence"].strip()
            if not clip.is_file() or not sentence:
                done.add(entry["path"])
                continue
            try:
                waveform, rate = load_clip(clip)
            except Exception as error:  # noqa: BLE001 - one bad clip is not fatal
                print(f"  {entry['path']} unreadable: {str(error)[:60]}", flush=True)
                done.add(entry["path"])
                failed += 1
                continue
            if len(waveform) < rate // 2:
                done.add(entry["path"])
                continue

            try:
                result, raw = word_aligner.align_with_chars(waveform, rate, sentence)
                words = result.words
            except Exception as error:  # noqa: BLE001
                print(f"  {entry['path']} alignment failed: {str(error)[:60]}", flush=True)
                done.add(entry["path"])
                failed += 1
                continue

            chars = sorted(
                (c for c in _raw_chars(raw or {})
                 if c.get("start") is not None and c.get("end") is not None),
                key=lambda c: float(c["start"]),
            )

            refined = []
            for word in words:
                if sum(1 for c in word.normalized_token if c in VOWELS) < 2:
                    continue
                label = lexicon_ordinal(trie, word.normalized_token)
                if label is None:
                    continue
                inside = [c for c in chars
                          if word.start_s - 0.02 <= float(c["start"])
                          and float(c["end"]) <= word.end_s + 0.02]
                vowels = _vowels_from_whisperx_chars(inside, word.normalized_token)
                expected = sum(1 for c in word.normalized_token if c in VOWELS)
                quality = len(vowels) / expected if expected else 0.0
                if len(vowels) < 2 or quality < args.min_quality:
                    continue
                durations = [round((v.end_s - v.start_s) * 1000) for v in vowels]
                if len(set(durations)) == 1:
                    continue  # the fallback span, not a measurement
                refined.append((word, SimpleNamespace(vowels=vowels, quality=quality),
                                label, durations))

            if refined:
                frames = encoder.encode(waveform, rate) if encoder is not None else None
                every = [v for _, r, _, _ in refined for v in r.vowels]
                f0_cache = f0_track(waveform, rate, min_hz=60.0, max_hz=400.0,
                                    frame_ms=40, hop_ms=10)
                for word, measured, label, durations in refined:
                    features = extract_prosodic_features(
                        waveform, rate, measured.vowels,
                        word_start_s=word.start_s, word_end_s=word.end_s,
                        utterance_vowels=every, f0_cache=f0_cache,
                    )
                    ssl_index = None
                    if frames is not None:
                        pooled = (
                            part_pool(frames, measured.vowels,
                                      frame_shift_s=encoder.frame_shift_s,
                                      parts=args.ssl_parts)
                            if args.ssl_parts else
                            mean_pool_intervals(frames, measured.vowels,
                                                frame_shift_s=encoder.frame_shift_s))
                        block = np.asarray(pooled, dtype=np.float16)
                        if ssl_dim == 0:
                            ssl_dim = int(block.shape[1])
                            args.out.with_suffix(".ssl.json").write_text(
                                json.dumps({"dim": ssl_dim}), encoding="utf-8")
                        ssl_handle.write(block.tobytes())
                        ssl_index = [ssl_written, ssl_written + len(pooled)]
                        ssl_written += len(pooled)
                    handle.write(json.dumps({
                        "token": word.token,
                        "ssl_index": ssl_index,
                        "form": word.normalized_token,
                        "label": label,
                        "clip": entry["path"],
                        "speaker": entry["client_id"],
                        "split": splits.get(entry["path"], "other"),
                        "gender": entry.get("gender") or "",
                        "accent": entry.get("accents") or "",
                        "prev_word": previous_of(sentence, word.normalized_token),
                        # Absolute, not just durations: fine-tuning the
                        # encoder has to pool its own frames over these
                        # spans, and a duration cannot say where in the clip
                        # the vowel was.
                        "vowel_spans": [[round(v.start_s, 4), round(v.end_s, 4)]
                                        for v in measured.vowels],
                        "quality": measured.quality,
                        "durations_ms": durations,
                        "features": features,
                    }, ensure_ascii=False) + "\n")
                    kept += 1
                handle.flush()
                if ssl_handle is not None:
                    ssl_handle.flush()

            done.add(entry["path"])
            if index % 25 == 0 or index == len(queue):
                done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
                elapsed = time.time() - started
                print(f"  {index}/{len(queue)} clips: {kept:,} rows, "
                      f"{failed} failed, {elapsed / 60:.1f} min "
                      f"({elapsed / index:.2f} s/clip)", flush=True)
    finally:
        handle.close()
        if ssl_handle is not None:
            ssl_handle.close()
        done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")

    print(f"\nkept {kept:,} labelled words -> {args.out}")
    if args.ssl:
        print(f"ssl features: {ssl_written:,} x {ssl_dim} -> {ssl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
