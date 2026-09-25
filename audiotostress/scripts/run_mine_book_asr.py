"""Mine stress labels from an audiobook, taking the transcript from the audio.

The first miner trusted the anchors to say which words a ten-minute window
holds, cut that window into sixty-second slices by proportion, and handed
forced alignment the book's words for each fraction — padded by 35% so the true
words were certainly inside. Every part of that was wrong. The anchors drift by
hundreds of words (at 1200 s into Костенко the anchor claims book word 3250 and
the narrator is reading word 2838), speech rate is not uniform so the
proportional cut adds its own error, and forced alignment cannot leave a word
out: it must place every one it is given. Measured without any model, the
longest vowel guessed stress 38% of the time on what that miner produced
against 35% chance; on Common Voice the same baseline is 63% against 38%. The
boundaries were not measurements, and neither were the 348,835 labels built on
them.

This turns the dependency around.

    ASR the whole window once   →  what is actually said, with timings
    cut into ~60 s chunks       →  by the ASR's own segment boundaries
    force-align each chunk      →  text and audio come from the same seconds
    vote the chunk into place   →  rare heard words find the book position
    match word by word          →  the book supplies spelling and lexicon

The book is still what labels the rows. It is no longer what times them, and
it is no longer trusted to say where in the recording it is being read.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402

from run_mine_book import slice_audio  # noqa: E402
from run_mine_commonvoice import VOWELS, lexicon_ordinal  # noqa: E402

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def book_words(path: Path) -> tuple[list[str], list[tuple[int, int, int]]]:
    """The book as one lowercase word stream, plus where each word came from.

    Identical to run_anchor_book.book_words, so a position here means the same
    thing it means in the anchors.
    """
    stream, origin = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for index, word in enumerate(WORD.findall(row["text"])):
            stream.append(unicodedata.normalize("NFC", word.lower()))
            origin.append((row["doc"], row["paragraph"], index))
    return stream, origin


def locate(heard: list[str], rare: dict[str, list[int]],
           low: int, high: int, bucket: int = 12) -> tuple[int, int] | None:
    """Where in the book does this chunk start?

    Each heard word that is rare enough to be informative votes for the offset
    that would put it where it was heard. Genuine agreement concentrates in one
    place; coincidence does not. Votes are bucketed because ASR inserts and
    drops words, so the offsets of a correct match scatter by a few positions.
    """
    votes: collections.Counter[int] = collections.Counter()
    for index, word in enumerate(heard):
        for spot in rare.get(word, ()):
            if low <= spot < high:
                votes[(spot - index) // bucket] += 1
    if not votes:
        return None
    best, support = votes.most_common(1)[0]
    return best * bucket, support


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True)
    parser.add_argument("--anchors", type=Path, required=True,
                        help="a coarse prior only: used to start the walk and "
                             "to recover from a chunk that cannot be placed")
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    parser.add_argument("--ssl-model", default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--chunk", type=float, default=60.0)
    parser.add_argument("--chunk-max", type=float, default=75.0,
                        help="hard ceiling on a chunk. Whisper sometimes returns "
                             "one segment covering minutes of audio, and forced "
                             "alignment of that on an 8 GB card stalls without "
                             "erroring — the first full run sat on one window "
                             "for a quarter of an hour")
    parser.add_argument("--rare", type=int, default=40,
                        help="a word occurring more often than this in the book "
                             "says nothing about position and does not vote")
    parser.add_argument("--support", type=int, default=4,
                        help="rare words that must agree before a chunk is "
                             "placed; below this the chunk is skipped rather "
                             "than guessed at")
    parser.add_argument("--lookahead", type=int, default=40,
                        help="how far ahead a heard word may be found before it "
                             "is treated as an ASR error rather than a match")
    parser.add_argument("--min-quality", type=float, default=1.0,
                        help="share of the form's vowels the aligner must place. "
                             "The first miner accepted 0.8, which shifts every "
                             "index after the missing vowel and makes the row a "
                             "lie; only a complete word is a measurement")
    parser.add_argument("--stride", type=int, default=1,
                        help="mine every Nth window. 369 hours of audio is "
                             "three days of compute, so the first pass takes a "
                             "third of it spread evenly over every book and "
                             "every part of every book; a later pass with "
                             "--stride 1 fills in what this skips. Each window "
                             "still places its own chunks, so a gap costs "
                             "nothing but the audio it skips")
    parser.add_argument("--windows", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8,
                        help="Whisper's VAD batch. Sixteen fits until the "
                             "allocator is under pressure, and then costs far "
                             "more than it saves")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit", action="store_true",
                        help="report the model-free baseline instead of writing "
                             "rows: the longest vowel against the lexicon, which "
                             "is what caught the first miner")
    args = parser.parse_args()

    import marisa_trie
    import torch
    import transformers
    import whisperx

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

    stream, origin = book_words(args.text)
    positions: dict[str, list[int]] = {}
    for spot, word in enumerate(stream):
        positions.setdefault(word, []).append(spot)
    rare = {w: p for w, p in positions.items() if len(p) <= args.rare}
    anchors = [json.loads(line) for line in
               args.anchors.read_text(encoding="utf-8").splitlines() if line.strip()]
    anchors = [a for a in anchors if "word_start" in a][args.skip::args.stride]
    if args.windows:
        anchors = anchors[:args.windows]
    print(f"{len(anchors)} windows, {len(stream):,} book words, "
          f"{len(rare):,} rare types", flush=True)

    asr = whisperx.load_model(args.asr_model, args.device,
                              compute_type="float16", language="uk")
    aligner = WhisperXWordAligner(AlignmentConfig(device=args.device,
                                                  return_char_alignments=True))
    encoder = HuggingFaceSpeechEncoder(
        args.ssl_model, device=args.device,
        processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
        model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())

    counts: collections.Counter[str] = collections.Counter()
    spent: collections.Counter[str] = collections.Counter()
    audit = [0, 0, 0.0]
    cursor = anchors[0]["word_start"] if anchors else 0
    drift: list[int] = []
    started = time.time()
    handle = None if args.audit else args.out.open("a", encoding="utf-8")
    try:
        for position, anchor in enumerate(anchors, 1):
            span = anchor.get("seconds") or 600.0
            try:
                clock = time.time()
                window = slice_audio(args.audio, anchor["start_s"], span)
                spent["decode"] += time.time() - clock
                clock = time.time()
                heard = asr.transcribe(window, batch_size=args.batch, language="uk")
                spent["asr"] += time.time() - clock
            except Exception as error:  # noqa: BLE001 - one window is not fatal
                print(f"  window {anchor['window']}: {str(error)[:70]}", flush=True)
                continue

            # Group the ASR's own segments into chunks of about --chunk
            # seconds. Its boundaries fall in silence, so a chunk never cuts a
            # word in half and its text is exactly its audio.
            chunks: list[tuple[float, float, str]] = []
            begin = last = None
            words_of: list[str] = []
            for segment in heard["segments"]:
                text = segment["text"].strip()
                if not text:
                    continue
                head, tail = float(segment["start"]), float(segment["end"])
                # A segment longer than the ceiling cannot be used whole and
                # cannot be cut without cutting words, so it is dropped and
                # counted. Left in, it stalls forced alignment on an 8 GB card
                # with no error at all.
                if tail - head > args.chunk_max:
                    counts["segment_too_long"] += 1
                    if words_of and begin is not None and last is not None:
                        chunks.append((begin, last, " ".join(words_of)))
                    begin, words_of = None, []
                    continue
                # Close before overshooting, not after. Closing after meant a
                # chunk reached ninety seconds whenever the segments were
                # coarse, and forced alignment of ninety seconds costs several
                # times what sixty does on this card — five minutes a window
                # against ninety seconds.
                if begin is not None and words_of and tail - begin > args.chunk:
                    chunks.append((begin, last, " ".join(words_of)))
                    begin, words_of = None, []
                if begin is None:
                    begin = head
                words_of.append(text)
                last = tail
            if begin is not None and last is not None and words_of:
                chunks.append((begin, last, " ".join(words_of)))

            print(f"    window {anchor['window']}: {len(chunks)} chunks, "
                  f"{max((b - a) for a, b, _ in chunks) if chunks else 0:.0f}s longest",
                  flush=True)
            for begin, end, text in chunks:
                counts["chunks"] += 1
                spoken = [unicodedata.normalize("NFC", w.lower())
                          for w in WORD.findall(text)]
                # Near the running cursor first; the anchor is the fallback,
                # and it is only ever a hint.
                placed = locate(spoken, rare, max(0, cursor - 120), cursor + 1500)
                if placed is None or placed[1] < args.support:
                    placed = locate(spoken, rare,
                                    max(0, anchor["word_start"] - 4000),
                                    min(len(stream), anchor["word_end"] + 4000))
                if placed is None or placed[1] < args.support:
                    counts["chunk_not_placed"] += 1
                    continue
                start, support = placed
                drift.append(start - cursor)
                cursor = max(0, start - 30)
                counts["chunk_placed"] += 1

                try:
                    waveform = window[int(begin * 16_000):int(end * 16_000)]
                    if len(waveform) < 16_000:
                        continue
                    clock = time.time()
                    result, raw = aligner.align_with_chars(waveform, 16_000, text)
                    spent["align"] += time.time() - clock
                    clock = time.time()
                    frames = encoder.encode(waveform, 16_000)
                    spent["ssl"] += time.time() - clock
                except Exception as error:  # noqa: BLE001
                    print(f"  chunk {anchor['window']}@{begin:.0f}: {str(error)[:70]}",
                          flush=True)
                    continue
                chars = sorted((c for c in _raw_chars(raw or {})
                                if c.get("start") is not None and c.get("end") is not None),
                               key=lambda c: float(c["start"]))
                base = anchor["start_s"] + begin
                # Align the whole chunk against the book once, rather than
                # hunting each word forward from a cursor: a greedy walk takes
                # the first occurrence of a frequent word and drags the cursor
                # past the true position, which is what the drift showed.
                aligned = [w.normalized_token for w in result.words]
                low = max(0, start - 60)
                high = min(len(stream), start + len(aligned) + 120)
                window_words = stream[low:high]
                where: dict[int, int] = {}
                for a, b, size in difflib.SequenceMatcher(
                        None, aligned, window_words, autojunk=False).get_matching_blocks():
                    for step in range(size):
                        where[a + step] = low + b + step
                if where:
                    cursor = max(where.values()) + 1
                with torch.no_grad():
                    for order, word in enumerate(result.words):
                        form = word.normalized_token
                        counts["heard"] += 1
                        found = where.get(order)
                        if found is None:
                            counts["not_in_book"] += 1
                            continue
                        expected = sum(1 for c in form if c in VOWELS)
                        if expected < 2:
                            counts["one_vowel"] += 1
                            continue
                        inside = [c for c in chars
                                  if word.start_s - 0.02 <= float(c["start"])
                                  and float(c["end"]) <= word.end_s + 0.02]
                        vowels = _vowels_from_whisperx_chars(inside, form)
                        if len(vowels) / expected < args.min_quality:
                            counts["vowels_missing"] += 1
                            continue
                        durations = [round((v.end_s - v.start_s) * 1000) for v in vowels]
                        if len(set(durations)) == 1:
                            counts["flat_durations"] += 1
                            continue
                        kind = readings(form)
                        label = lexicon_ordinal(trie, form) if kind == 1 else None
                        if args.audit:
                            if label is not None and 0 <= label < len(durations):
                                audit[0] += int(max(range(len(durations)),
                                                    key=lambda i: durations[i]) == label)
                                audit[1] += 1
                                audit[2] += 1 / len(durations)
                            counts["kept"] += 1
                            continue
                        pooled = np.asarray(mean_pool_intervals(
                            frames, vowels, frame_shift_s=encoder.frame_shift_s),
                            dtype=np.float32)
                        doc, paragraph, index = origin[found]
                        handle.write(json.dumps({
                            "book": args.book, "window": anchor["window"],
                            "doc": doc, "paragraph": paragraph, "word_index": index,
                            "stream": found, "token": word.token, "form": form,
                            # Absolute, and so are the spans: nothing downstream
                            # reconstructs which chunk this came from.
                            "start_s": round(base + word.start_s, 3),
                            "durations_ms": durations,
                            "readings": kind, "label": label,
                            "vowel_spans": [[round(base + v.start_s, 3),
                                             round(base + v.end_s, 3)]
                                            for v in vowels],
                        }, ensure_ascii=False) + "\n")
                        counts["kept"] += 1
                        counts["lexicon" if kind == 1 else
                               "ambiguous" if kind > 1 else "oov"] += 1
                if handle:
                    handle.flush()
                del frames, result, raw
                torch.cuda.empty_cache()
            print(f"  window {position}/{len(anchors)}  kept {counts['kept']:,}  "
                  f"placed {counts['chunk_placed']}/{counts['chunks']}  "
                  f"matched {1 - counts['not_in_book'] / max(counts['heard'], 1):.0%}  "
                  f"{(time.time() - started) / 60:.1f} min  "
                  # Which stage costs what, per window rather than at the end:
                  # one book ran eight times slower than its neighbour with the
                  # same chunk count, and there was no way to see where it went.
                  + " ".join(f"{k} {v:.0f}s" for k, v in spent.most_common()),
                  flush=True)
    finally:
        if handle:
            handle.close()

    print(f"\ncounts {dict(counts)}")
    print("seconds per stage: " + ", ".join(f"{k} {v:.0f}" for k, v in spent.most_common()))
    if drift:
        drift.sort()
        print(f"chunk start minus running cursor: median {drift[len(drift) // 2]}, "
              f"range {drift[0]}..{drift[-1]} words")
    if args.audit and audit[1]:
        print(f"\nlongest vowel {audit[0] / audit[1]:.1%} over {audit[1]:,} words "
              f"the lexicon names, chance {audit[2] / audit[1]:.1%}")
        print("Common Voice, the same baseline: 63.4% against 37.6%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
