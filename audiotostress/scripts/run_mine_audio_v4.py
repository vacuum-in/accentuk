"""Mine any audio: hear it, align it, label it — no matching text needed.

RUAccent's 108,000 hours are podcasts, YouTube and radio with no text at all.
Its pipeline needs none, and neither does this one: Whisper says what was
said, forced alignment places those words in the audio, the SSL encoder and
the ranker read each vowel, and the lexicon says what the word's readings
are. The book used to supply the spelling, the paragraph and a check that the
words were real; here the transcript supplies the sentence, and two gates
stand in for the check:

  * a word is kept only if the lexicon knows it (readings >= 1) and the
    aligner placed every one of its vowels — a misheard word rarely passes
    both;
  * a chunk is kept only if at least --chunk-known of its multi-letter
    words are lexicon words — a stretch of music, mumbling or another
    language fails this the way a wrong-edition book failed the pilot.

Rows carry the chunk's text as the sentence with the word's span in it, so
the corpus builder needs no locator. The longest-vowel lift and the ranker's
agreement with the lexicon print per window as before, and a source whose
lift is below --abort-below after --abort-after windows is stopped.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import resource
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
    parser.add_argument("--source", help="a name for the rows, e.g. youtube:<id>")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--out", type=Path,
                        help="one source: the rows file. A folder of sources: the directory")
    parser.add_argument("--folder", type=Path,
                        help="mine every <id>/audio.* under this folder (as "
                             "run_fetch_youtube.py lays them out) with the models "
                             "loaded once; existing outputs are skipped")
    parser.add_argument("--suffix", default="rows4")
    parser.add_argument("--shard", default="0/1",
                        help="k/n or a-b/n: mine the sources whose index mod n is k "
                             "(or in a..b), so several machines of different speed "
                             "can split one folder — the 5090 takes four eighths, "
                             "a 3070 or an M4 one each")
    parser.add_argument("--reverse", action="store_true",
                        help="walk the folder from the end: a slow machine sweeping "
                             "backwards meets the fast one sweeping forwards with "
                             "almost no video done twice")
    parser.add_argument("--max-sources", type=int, default=0,
                        help="mine at most this many new sources, then exit so the "
                             "driver can push rows and pull new audio")
    parser.add_argument("--chunk-known", type=float, default=0.8,
                        help="share of a chunk's multi-letter words the lexicon "
                             "must know, or the chunk is dropped as misheard")
    parser.add_argument("--abort-after", type=int, default=6,
                        help="windows after which a source is judged")
    parser.add_argument("--abort-below", type=float, default=8.0,
                        help="longest-vowel lift in pp a source must show by then")
    parser.add_argument("--trie", default=None, help="defaults to the installed package's")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/all_model/ranker.pt"))
    parser.add_argument("--ssl-model", default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--asr", choices=("whisperx", "mlx"), default="whisperx",
                        help="whisperx (CUDA, faster-whisper) or mlx (Apple Silicon, "
                             "mlx-whisper). Alignment, SSL and the ranker run in torch "
                             "on --device either way: cuda, mps or cpu")
    parser.add_argument("--asr-model", default=None,
                        help="large-v3-turbo for whisperx, "
                             "mlx-community/whisper-large-v3-turbo for mlx")
    parser.add_argument("--window", type=float, default=600.0)
    parser.add_argument("--chunk", type=float, default=60.0)
    parser.add_argument("--chunk-max", type=float, default=75.0)
    parser.add_argument("--windows", type=int, default=0,
                        help="stop after this many windows: the pilot")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import marisa_trie
    import torch
    import transformers
    if args.asr == "whisperx":
        import whisperx
    else:
        import mlx_whisper

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

    if args.asr == "whisperx":
        asr = whisperx.load_model(args.asr_model or "large-v3-turbo", args.device,
                                  compute_type="float16", language="uk")

        def transcribe(window):
            return asr.transcribe(window, batch_size=args.batch, language="uk")
    else:
        repo = args.asr_model or "mlx-community/whisper-large-v3-turbo"

        def transcribe(window):
            # mlx-whisper returns the same shape: segments with start, end, text.
            return mlx_whisper.transcribe(window, path_or_hf_repo=repo, language="uk",
                                          condition_on_previous_text=False)
    aligner = WhisperXWordAligner(AlignmentConfig(device=args.device,
                                                  return_char_alignments=True))
    encoder = HuggingFaceSpeechEncoder(
        args.ssl_model, device=args.device,
        processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
        model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())
    ranker, _ = load_ranker_checkpoint(args.checkpoint, device=args.device)
    if args.device == "mps":
        # The nested-tensor fast path checks an op MPS lacks; the plain path
        # gives the same numbers and stays on the GPU.
        for module in ranker.module.modules():
            if isinstance(module, torch.nn.TransformerEncoder):
                module.enable_nested_tensor = False

    def release():
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()
            if args.asr == "mlx":
                import mlx.core as mx
                mx.clear_cache()
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


    def known(word: str) -> bool:
        return _trie_value(trie, unicodedata.normalize("NFC", word)) is not None

    def mine(book: str, audio: Path, out: Path) -> None:
        duration = audio_seconds(audio)
        starts = [s for s in np.arange(0.0, duration, args.window)][args.skip:]
        if args.windows:
            starts = starts[:args.windows]
        print(f"{book}: {duration / 3600:.1f} h, {len(starts)} windows to mine", flush=True)

        counts: collections.Counter[str] = collections.Counter()
        spent: collections.Counter[str] = collections.Counter()
        baseline = [0, 0, 0.0]      # longest vowel: hits, total, chance
        agreement = [0, 0, 0, 0]    # ranker vs lexicon: hits, total, hits≥.99, total≥.99
        started = time.time()
        # Rows go to a .part file that becomes `out` only when the whole source
        # is done: a killed run must not leave a short file that the next pass
        # takes for a finished one.
        part = out.with_name(out.name + ".part")
        handle = part.open("w", encoding="utf-8")
        try:
            for position, start in enumerate(starts, 1):
                clock = time.time()
                try:
                    window = decode(audio, float(start), args.window)
                    spent["decode"] += time.time() - clock
                    clock = time.time()
                    heard = transcribe(window)
                    spent["asr"] += time.time() - clock
                except Exception as error:  # noqa: BLE001 - one window is not fatal
                    print(f"  window {position}: {str(error)[:70]}", flush=True)
                    continue

                chunks = chunk_segments(heard["segments"], args.chunk, args.chunk_max, counts)
                for begin, end, text in chunks:
                    counts["chunks"] += 1

                    spoken = [w for w in WORD.findall(text) if len(w) > 2]
                    if not spoken:
                        counts["chunk_empty"] += 1
                        continue
                    share = sum(1 for w in spoken if known(w.lower())) / len(spoken)
                    if share < args.chunk_known:
                        counts["chunk_unknown"] += 1
                        continue
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


                    # Where each aligned word sits in the chunk text, in order.
                    spans, cursor_c = [], 0
                    lowered = text.lower()
                    for word in result.words:
                        found = lowered.find(word.normalized_token, cursor_c)
                        spans.append((found, found + len(word.normalized_token)) if found >= 0 else None)
                        if found >= 0:
                            cursor_c = found + len(word.normalized_token)

                    base = float(start) + begin
                    clock = time.time()
                    with torch.no_grad():
                        for order, word in enumerate(result.words):
                            form = word.normalized_token
                            counts["heard"] += 1
                            if spans[order] is None or not known(form):
                                counts["not_in_lexicon"] += 1
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
                            span = spans[order]
                            handle.write(json.dumps({
                                "book": book, "window": position - 1 + args.skip,
                                "sentence": text, "start": span[0], "end": span[1],
                                "token": word.token, "form": form,
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
                    release()

                lift = (100 * (baseline[0] / baseline[1] - baseline[2] / baseline[1])
                        if baseline[1] else 0.0)
                del heard, window
                release()
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_gb = rss / (2**30 if sys.platform == "darwin" else 2**20)
                print(f"  window {position}/{len(starts)}  kept {counts['kept']:,}  "
                      f"placed {counts['chunk_placed']}/{counts['chunks']}  "
                      f"known {1 - counts['not_in_lexicon'] / max(counts['heard'], 1):.0%}  "
                      f"vowel lift {lift:+.1f}pp  "
                      f"ranker {agreement[0] / max(agreement[1], 1):.1%}  "
                      f"{(time.time() - started) / 60:.1f} min  "
                      + " ".join(f"{k} {v:.0f}s" for k, v in spent.most_common())
                      + f"  rss {rss_gb:.1f}G",
                      flush=True)
        finally:
            handle.close()
        part.replace(out)

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

    if args.folder:
        out_dir = args.out or args.folder
        part, shards = args.shard.split("/")
        low, high = (int(x) for x in part.split("-")) if "-" in part else (int(part), int(part))
        shards = int(shards)
        mined = 0
        for entry in sorted((p for p in args.folder.iterdir() if p.is_dir()), reverse=args.reverse):
            if args.max_sources and mined >= args.max_sources:
                break
            # Shard by a hash of the folder name, not its position: folders keep
            # arriving while the fetch runs, and a position-based split would
            # hand the same video to different machines on different passes.
            digest = int(hashlib.md5(entry.name.encode()).hexdigest(), 16)
            if not low <= digest % shards <= high:
                continue
            if not (entry / "meta.json").exists():
                continue   # still downloading
            target = out_dir / f"{entry.name}.{args.suffix}.jsonl"
            if target.exists() and target.stat().st_size > 0:
                continue   # done here or on another machine
            if target.with_suffix(".failed").exists():
                continue   # a broken download: one try was enough
            try:
                meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
            except ValueError:
                continue   # an rsync caught the fetcher mid-write; next pass
            # The file named by the fetcher: a folder can also hold the
            # pre-conversion download that an rsync caught in passing.
            audio = entry / meta["audio"] if meta.get("audio") else None
            if audio is None or not audio.exists():
                audio = next((p for p in entry.iterdir()
                              if p.name.startswith("audio.") and not p.name.endswith(".part")), None)
            if audio is None:
                continue
            print(f"=== {entry.name} {str(meta.get('title', ''))[:60]!r} {time.strftime('%H:%M')}", flush=True)
            try:
                mine(f"youtube:{entry.name}", audio, target)
            except Exception as error:  # noqa: BLE001 - one source must not end the run
                print(f"=== {entry.name} FAILED: {str(error)[:120]}", flush=True)
                target.with_suffix(".failed").write_text(str(error)[:500], encoding="utf-8")
            mined += 1
        print(f"=== FOLDER DONE {time.strftime('%H:%M')} mined {mined}", flush=True)
        return 0

    mine(args.source or args.audio.stem, args.audio, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
