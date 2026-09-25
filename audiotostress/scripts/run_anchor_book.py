"""Find where in a book each window of its audiobook is being read.

An epub gives the words and an audiobook gives the sound, and nothing connects
them: eighteen hours against three hundred pages, with no timestamps. This
builds that connection once per book, so mining can then align each window
against the words actually spoken in it rather than against ASR output, whose
errors move vowel boundaries.

The method is deliberately cheap. Transcribe a window, take its rare words, and
look for the run of the book's word stream where those words cluster. Anchors
must advance: a window cannot start before the previous one ended, which turns
a fuzzy search into a nearly linear scan and rejects the false matches that a
repeated phrase would otherwise produce.

What this reports matters more than what it stores. A book whose windows anchor
poorly is abridged, differently edited, or read from another translation, and
forcing it produces labels that look fine and are wrong.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def audio_seconds(path: Path) -> float:
    """Duration via ffprobe, not soundfile: libsndfile cannot open m4a, and six
    of these books are m4a."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def read_window(path: Path, start: float, seconds: float) -> np.ndarray:
    """One window as 16 kHz mono. ffmpeg seeks the container; soundfile would
    decode from the beginning of a fifteen-hour file every time."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start}", "-t", f"{seconds}",
         "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def book_words(path: Path) -> tuple[list[str], list[dict]]:
    """The book as one lowercase word stream, plus where each word came from."""
    stream, origin = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for index, word in enumerate(WORD.findall(row["text"])):
            stream.append(unicodedata.normalize("NFC", word.lower()))
            origin.append({"doc": row["doc"], "paragraph": row["paragraph"],
                           "index": index})
    return stream, origin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True,
                        help="the JSONL written by run_extract_books.py")
    parser.add_argument("--window", type=float, default=600.0)
    parser.add_argument("--windows", type=int, default=0, help="0 = the whole book")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-rare", type=int, default=6,
                        help="a window needs this many rare words to anchor")
    parser.add_argument("--abort-after", type=int, default=15,
                        help="check the anchor rate once this many windows have "
                             "been tried, and stop if it is hopeless")
    parser.add_argument("--abort-below", type=float, default=0.25,
                        help="the rate that counts as hopeless. A mismatched "
                             "text does not improve with more audio: Гюго sat "
                             "at 4 anchored windows from the fourth to the "
                             "hundred-and-thirty-third and spent 70 minutes of "
                             "ASR proving it. Books that recover from a slow "
                             "start do so within the first few windows, because "
                             "front matter is short")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from ukstress.asr.whisper_backend import WhisperCompatibleBackend
    from ukstress.config.models import ASRConfig

    stream, origin = book_words(args.text)
    frequency = collections.Counter(stream)
    # Where each rare word occurs. Common words carry no positional information
    # and a stop list built from the book itself beats any fixed one.
    positions: dict[str, list[int]] = collections.defaultdict(list)
    for index, word in enumerate(stream):
        if frequency[word] <= 60 and len(word) >= 5:
            positions[word].append(index)
    print(f"{len(stream):,} words, {len(positions):,} rare types", flush=True)

    duration = audio_seconds(args.audio)
    total = int(duration // args.window) + 1
    if args.windows:
        total = min(total, args.windows)
    print(f"{duration/3600:.1f} h of audio, {total} windows of "
          f"{args.window/60:.0f} min", flush=True)

    asr = WhisperCompatibleBackend(ASRConfig(
        model_id=args.asr_model, device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8"))

    anchors, floor, started = [], 0, time.time()
    for index in range(total):
        offset = index * args.window
        waveform = read_window(args.audio, offset, args.window)
        if len(waveform) < 16_000:
            break
        heard = " ".join(s.text for s in asr.transcribe(waveform, 16_000).segments)
        spoken = [unicodedata.normalize("NFC", w.lower()) for w in WORD.findall(heard)]
        rare = [w for w in spoken if w in positions]

        # Vote for a start position. Each rare word that occurs at book index p
        # votes for the window starting near p minus how far into the window it
        # was heard, so the votes of a correctly-placed window all agree.
        votes: collections.Counter[int] = collections.Counter()
        for order, word in enumerate(rare):
            for place in positions[word]:
                if place < floor:
                    continue
                votes[(place - order) // 50] += 1
        best, support = (votes.most_common(1)[0] if votes else (None, 0))
        anchor = {
            "window": index, "start_s": offset, "heard_words": len(spoken),
            "rare_matched": len(rare), "support": support,
        }
        if best is not None and support >= args.min_rare:
            start = max(best * 50, floor)
            end = min(start + int(len(spoken) * 1.3) + 40, len(stream))
            span = stream[start:end]
            # How much of what was heard actually appears in the span the
            # anchor points at. A high support score with a low overlap means
            # the vote found a repeated phrase somewhere else in the book.
            overlap = (len(set(spoken) & set(span)) / len(set(spoken))
                       if spoken else 0.0)
            anchor.update({"word_start": start, "word_end": end,
                           "text": " ".join(span), "overlap": round(overlap, 3),
                           "doc": origin[start]["doc"],
                           "paragraph": origin[start]["paragraph"]})
            floor = start + int(len(spoken) * 0.6)
        anchors.append(anchor)
        if index + 1 == args.abort_after:
            rate = sum(1 for a in anchors if "word_start" in a) / len(anchors)
            if rate < args.abort_below:
                print(f"\nstopping after {len(anchors)} windows: only "
                      f"{100*rate:.0f}% anchored. The text does not match this "
                      f"recording — a different edition, a different "
                      f"translation, or a different work. Nothing here is "
                      f"worth mining.", flush=True)
                break
        if (index + 1) % 5 == 0 or index + 1 == total:
            good = sum(1 for a in anchors if "word_start" in a)
            print(f"  {index+1}/{total} windows, {good} anchored "
                  f"({100*good/len(anchors):.0f}%), "
                  f"{(time.time()-started)/60:.1f} min", flush=True)

    args.out.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in anchors) + "\n",
                        encoding="utf-8")
    good = [a for a in anchors if "word_start" in a]
    if len(anchors) < total:
        print(f"aborted after {len(anchors)} of {total} windows")
    print(f"\n{len(good)}/{len(anchors)} windows anchored "
          f"({100*len(good)/max(len(anchors),1):.0f}%)")
    if good:
        overlaps = sorted(a.get("overlap", 0.0) for a in good)
        weak = sum(1 for o in overlaps if o < 0.5)
        print(f"heard-word overlap with the anchored span: "
              f"median {overlaps[len(overlaps)//2]:.2f}, "
              f"worst {overlaps[0]:.2f}, below 0.5: {weak}")
    if len(good) > 1:
        steps = [b["word_start"] - a["word_start"]
                 for a, b in zip(good, good[1:], strict=False)]
        print(f"words advanced per window: median {int(np.median(steps)):,}, "
              f"min {min(steps):,}, max {max(steps):,}")
        print(f"book covered: words {good[0]['word_start']:,}–{good[-1]['word_end']:,} "
              f"of {len(stream):,}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
