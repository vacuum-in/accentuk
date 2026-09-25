"""Mine an audiobook in one pass: hear it, align it, place it, label it.

The pipeline this replaces touched every second of audio four times — an
anchoring pass with Whisper large-v3 to place ten-minute windows in the book,
a mining pass (Whisper turbo, forced alignment, SSL encoding), a labelling
pass that decoded and SSL-encoded the same windows again to run the ranker,
and audits and calibrations on top — and the mining pass pooled SSL
embeddings it then threw away. Roughly 2.3 units of GPU work per second of
audio, of which one unit produced rows.

This does it once, per ten-minute window of the file, with no anchors:

    decode the window          ffmpeg, seek is sample-accurate
    Whisper turbo              what is said, with segment times
    cut into chunks ≤ 60 s     on the ASR's own boundaries
    force-align each chunk     text and audio from the same seconds
    SSL-encode each chunk      once, and used
    place the chunk            rare heard words vote for a book position;
                               near the running cursor first, whole book after
    match words                difflib against the book stream
    per kept word              vowel spans, durations, pooled SSL, and the
                               ranker's answer — for every word, so the
                               lexicon-named ones calibrate the ranker for free

Every row carries the lexicon's reading where there is exactly one, the
ranker's pick and confidence always, and the book position exactly. The
longest-vowel baseline and the ranker's agreement with the lexicon print per
window: a book whose baseline sits near chance is noise and says so before
the window is over, which is the check that was missing the first time.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


def default_trie() -> str:
    """The lexicon trie from the installed package, wherever uv put it. The
    earlier scripts hard-coded one machine's uv archive hash and broke on the
    next machine."""
    import ukrainian_word_stress

    return str(Path(ukrainian_word_stress.__file__).parent / "data" / "stress.trie")

from run_mine_commonvoice import VOWELS, lexicon_ordinal  # noqa: E402

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def audio_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def decode(path: Path, start: float, seconds: float) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start}", "-t", f"{seconds}",
         "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def book_words(path: Path):
    """The book as one lowercase word stream, plus where each word came from."""
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
           low: int, high: int, bucket: int = 12):
    """Where in the book does this chunk start? Rare heard words vote for the
    offset that puts them where they were heard; agreement concentrates,
    coincidence does not."""
    votes: collections.Counter[int] = collections.Counter()
    for index, word in enumerate(heard):
        for spot in rare.get(word, ()):
            if low <= spot < high:
                votes[(spot - index) // bucket] += 1
    if not votes:
        return None
    best, support = votes.most_common(1)[0]
    return best * bucket, support


def chunk_segments(segments, target: float, ceiling: float, counts):
    """Group the ASR's segments into chunks of about `target` seconds, never
    over `ceiling`, closing before overshoot. A single segment over the
    ceiling is dropped: it cannot be cut without cutting words, and left in it
    stalls forced alignment on a small card without erroring."""
    chunks, begin, last, words = [], None, None, []
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        head, tail = float(segment["start"]), float(segment["end"])
        if tail - head > ceiling:
            counts["segment_too_long"] += 1
            if words and begin is not None:
                chunks.append((begin, last, " ".join(words)))
            begin, words = None, []
            continue
        if begin is not None and words and tail - begin > target:
            chunks.append((begin, last, " ".join(words)))
            begin, words = None, []
        if begin is None:
            begin = head
        words.append(text)
        last = tail
    if begin is not None and words:
        chunks.append((begin, last, " ".join(words)))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book")
    parser.add_argument("--text", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--out", type=Path,
                        help="one book: the rows file. A plan: the directory")
    parser.add_argument("--plan", type=Path,
                        help="mine every book in this plan.json with the models "
                             "loaded once; --only lists slugs, --suffix names "
                             "the output files, existing outputs are skipped")
    parser.add_argument("--only", type=Path,
                        help="a JSON list of slugs to restrict --plan to")
    parser.add_argument("--suffix", default="rows3")
    parser.add_argument("--trie", default=None, help="defaults to the installed package's")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/all_model/ranker.pt"))
    parser.add_argument("--ssl-model", default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--window", type=float, default=600.0)
    parser.add_argument("--chunk", type=float, default=60.0)
    parser.add_argument("--chunk-max", type=float, default=75.0)
    parser.add_argument("--rare", type=int, default=40)
    parser.add_argument("--support", type=int, default=4,
                        help="rare words that must agree near the cursor")
    parser.add_argument("--support-far", type=int, default=8,
                        help="… and when the whole book is searched instead")
    parser.add_argument("--windows", type=int, default=0,
                        help="stop after this many windows: the pilot")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import marisa_trie
    import torch
    import transformers
    import whisperx

    import run_train_speakers as trainer
    from run_train_speakers import prosody_matrix
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value
    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.config.models import AlignmentConfig
    from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals
    from ukstress.ranker.model import load_ranker_checkpoint

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie or default_trie())

    def readings(word: str) -> int:
        value = _trie_value(trie, unicodedata.normalize("NFC", word))
        if value is None:
            return 0
        return len({tuple(a) for _, a in _parse_dictionary_value(value[0])})

    asr = whisperx.load_model(args.asr_model, args.device,
                              compute_type="float16", language="uk")
    aligner = WhisperXWordAligner(AlignmentConfig(device=args.device,
                                                  return_char_alignments=True))
    encoder = HuggingFaceSpeechEncoder(
        args.ssl_model, device=args.device,
        processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
        model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())
    ranker, _ = load_ranker_checkpoint(args.checkpoint, device=args.device)
    ranker.module.eval()
    trainer.PROSODY_MODE = {26: "full", 5: "position", 1: "none"}.get(
        ranker.architecture["prosody_size"], "full")

    def rank(pooled: np.ndarray, form: str):
        block = (pooled - pooled.mean(axis=0, keepdims=True))[None, :, :]
        stub = {"form": form, "features": [{} for _ in range(len(pooled))]}
        prosody = prosody_matrix(stub, trainer.PROSODY_MODE)[None, :, :]
        valid = np.ones((1, len(pooled)), dtype=bool)
        logits = ranker.module(torch.as_tensor(block, device=args.device),
                               torch.as_tensor(prosody, device=args.device),
                               torch.as_tensor(valid, device=args.device))[0]
        pick = int(logits.argmax())
        return pick, round(float(torch.softmax(logits, dim=-1)[pick]), 4)


    def mine(book: str, text: Path, audio: Path, out: Path) -> None:
        stream, origin = book_words(text)
        positions: dict[str, list[int]] = {}
        for spot, word in enumerate(stream):
            positions.setdefault(word, []).append(spot)
        rare = {w: p for w, p in positions.items() if len(p) <= args.rare}
        duration = audio_seconds(audio)
        starts = [s for s in np.arange(0.0, duration, args.window)][args.skip:]
        if args.windows:
            starts = starts[:args.windows]
        print(f"{book}: {duration / 3600:.1f} h, {len(starts)} windows to mine, "
              f"{len(stream):,} book words, {len(rare):,} rare types", flush=True)

        counts: collections.Counter[str] = collections.Counter()
        spent: collections.Counter[str] = collections.Counter()
        baseline = [0, 0, 0.0]      # longest vowel: hits, total, chance
        agreement = [0, 0, 0, 0]    # ranker vs lexicon: hits, total, hits≥.99, total≥.99
        cursor = 0
        started = time.time()
        handle = out.open("a", encoding="utf-8")
        try:
            for position, start in enumerate(starts, 1):
                clock = time.time()
                try:
                    window = decode(audio, float(start), args.window)
                    spent["decode"] += time.time() - clock
                    clock = time.time()
                    heard = asr.transcribe(window, batch_size=args.batch, language="uk")
                    spent["asr"] += time.time() - clock
                except Exception as error:  # noqa: BLE001 - one window is not fatal
                    print(f"  window {position}: {str(error)[:70]}", flush=True)
                    continue

                chunks = chunk_segments(heard["segments"], args.chunk, args.chunk_max, counts)
                for begin, end, text in chunks:
                    counts["chunks"] += 1
                    spoken = [unicodedata.normalize("NFC", w.lower()) for w in WORD.findall(text)]
                    placed = locate(spoken, rare, max(0, cursor - 120), cursor + 1500)
                    if placed is None or placed[1] < args.support:
                        placed = locate(spoken, rare, 0, len(stream))
                        if placed is None or placed[1] < args.support_far:
                            counts["chunk_not_placed"] += 1
                            continue
                        counts["placed_far"] += 1
                    chunk_start = placed[0]
                    counts["chunk_placed"] += 1

                    waveform = window[int(begin * 16_000):int(end * 16_000)]
                    if len(waveform) < 16_000:
                        continue
                    try:
                        clock = time.time()
                        result, raw = aligner.align_with_chars(waveform, 16_000, text)
                        spent["align"] += time.time() - clock
                        clock = time.time()
                        frames = encoder.encode(waveform, 16_000)
                        spent["ssl"] += time.time() - clock
                    except Exception as error:  # noqa: BLE001
                        print(f"  chunk @{start + begin:.0f}s: {str(error)[:70]}", flush=True)
                        continue
                    chars = sorted((c for c in _raw_chars(raw or {})
                                    if c.get("start") is not None and c.get("end") is not None),
                                   key=lambda c: float(c["start"]))

                    aligned = [w.normalized_token for w in result.words]
                    low = max(0, chunk_start - 60)
                    high = min(len(stream), chunk_start + len(aligned) + 120)
                    where: dict[int, int] = {}
                    for a, b, size in difflib.SequenceMatcher(
                            None, aligned, stream[low:high], autojunk=False).get_matching_blocks():
                        for step in range(size):
                            where[a + step] = low + b + step
                    if where:
                        cursor = max(where.values()) + 1

                    base = float(start) + begin
                    clock = time.time()
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
                            if len(vowels) != expected:
                                counts["vowels_missing"] += 1
                                continue
                            durations = [round((v.end_s - v.start_s) * 1000) for v in vowels]
                            if len(set(durations)) == 1:
                                counts["flat_durations"] += 1
                                continue
                            kind = readings(form)
                            label = lexicon_ordinal(trie, form) if kind == 1 else None
                            pooled = np.asarray(mean_pool_intervals(
                                frames, vowels, frame_shift_s=encoder.frame_shift_s),
                                dtype=np.float32)
                            pick, confidence = rank(pooled, form)
                            if label is not None and 0 <= label < len(durations):
                                baseline[0] += int(max(range(len(durations)),
                                                       key=lambda i: durations[i]) == label)
                                baseline[1] += 1
                                baseline[2] += 1 / len(durations)
                                agreement[0] += pick == label
                                agreement[1] += 1
                                if confidence >= 0.99:
                                    agreement[2] += pick == label
                                    agreement[3] += 1
                            doc, paragraph, index = origin[found]
                            handle.write(json.dumps({
                                "book": book, "window": position - 1 + args.skip,
                                "doc": doc, "paragraph": paragraph, "word_index": index,
                                "stream": found, "token": word.token, "form": form,
                                "start_s": round(base + word.start_s, 3),
                                "durations_ms": durations,
                                "readings": kind, "label": label,
                                "audio": pick, "confidence": confidence,
                                "vowel_spans": [[round(base + v.start_s, 3),
                                                 round(base + v.end_s, 3)] for v in vowels],
                            }, ensure_ascii=False) + "\n")
                            counts["kept"] += 1
                            counts["lexicon" if kind == 1 else
                                   "ambiguous" if kind > 1 else "oov"] += 1
                    spent["rank"] += time.time() - clock
                    handle.flush()
                    del frames, result, raw
                    torch.cuda.empty_cache()

                lift = (100 * (baseline[0] / baseline[1] - baseline[2] / baseline[1])
                        if baseline[1] else 0.0)
                print(f"  window {position}/{len(starts)}  kept {counts['kept']:,}  "
                      f"placed {counts['chunk_placed']}/{counts['chunks']}  "
                      f"matched {1 - counts['not_in_book'] / max(counts['heard'], 1):.0%}  "
                      f"vowel lift {lift:+.1f}pp  "
                      f"ranker {agreement[0] / max(agreement[1], 1):.1%}  "
                      f"{(time.time() - started) / 60:.1f} min  "
                      + " ".join(f"{k} {v:.0f}s" for k, v in spent.most_common()),
                      flush=True)
        finally:
            handle.close()

        print(f"\ncounts {dict(counts)}")
        if baseline[1]:
            print(f"longest vowel {baseline[0] / baseline[1]:.1%} over {baseline[1]:,} "
                  f"lexicon-named words, chance {baseline[2] / baseline[1]:.1%} "
                  f"(Common Voice: 63.4% against 37.6%)")
            print(f"ranker agrees with the lexicon {agreement[0] / agreement[1]:.2%}; "
                  f"at ≥0.99: {agreement[2] / max(agreement[3], 1):.2%} "
                  f"on {agreement[3] / agreement[1]:.0%} of words")
        hours = len(starts) * args.window / 3600
        print(f"{hours:.2f} h of audio in {(time.time() - started) / 60:.1f} min "
              f"= {hours * 3600 / (time.time() - started):.1f}x realtime")

    if args.plan:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        only = set(json.loads(args.only.read_text(encoding="utf-8"))) if args.only else None
        out_dir = args.out or args.plan.parent
        for entry in plan:
            if only is not None and entry["slug"] not in only:
                continue
            target = out_dir / f"{entry['slug']}.{args.suffix}.jsonl"
            if target.exists() and target.stat().st_size > 0:
                print(f"=== {entry['slug']}: exists, skipping", flush=True)
                continue
            print(f"=== {entry['slug']} ({entry['hours']} h) {time.strftime('%H:%M')}", flush=True)
            try:
                mine(entry["slug"], Path(entry["text"]), Path(entry["audio"]), target)
            except Exception as error:  # noqa: BLE001 - one book must not end the plan
                print(f"=== {entry['slug']} FAILED: {str(error)[:120]}", flush=True)
        print(f"=== PLAN DONE {time.strftime('%H:%M')}", flush=True)
        return 0

    mine(args.book, args.text, args.audio, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
