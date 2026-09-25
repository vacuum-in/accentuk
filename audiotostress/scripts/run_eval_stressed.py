"""Evaluate the acoustic model on a recording whose stress is known.

Mining labels words from the lexicon, so it can only ever learn from forms the
dictionary already names — and those are exactly the words that need no model.
This set is the other kind: a reading of a passage written specifically to pack
in homographs, with the stress marked in the text, so `за́мок` against `замки́`
and `лу́па` against `лупа́` carry a truth no dictionary could supply.

It is also a genuine held-out test. The model trains on one narrator reading a
novel and is tested on a different voice reading different material, which the
by-form split inside a single recording cannot check.

Audio files and paragraphs are matched by transcribing each file and finding
the paragraph it overlaps, rather than by assuming the order lines up: there
are nine files and eleven paragraphs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

ACUTE = "́"
VOWELS = "аеєиіїоуюяй"
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’́-]+")


def stripped(word: str) -> str:
    return unicodedata.normalize("NFC", word.replace(ACUTE, "")).lower()


def stress_ordinal(word: str) -> int | None:
    """Which vowel the text marks, counting as the pipeline counts."""
    decomposed = unicodedata.normalize("NFD", word)
    ordinal = -1
    for character in decomposed:
        if character.lower() in VOWELS:
            ordinal += 1
        if character == ACUTE:
            return ordinal
    return None


def paragraphs(path: Path) -> list[list[str]]:
    text = unicodedata.normalize("NFD", path.read_text(encoding="utf-8"))
    return [WORD.findall(block) for block in re.split(r"\n\s*\n", text) if block.strip()]


def load_16k(path: Path) -> tuple[np.ndarray, int]:
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
    return np.ascontiguousarray(waveform), rate


def best_paragraph(spoken: list[str], blocks: list[list[str]]) -> tuple[int, float]:
    needle = [stripped(w) for w in spoken]
    best, score = -1, 0.0
    for index, block in enumerate(blocks):
        ratio = difflib.SequenceMatcher(
            None, [stripped(w) for w in block], needle, autojunk=False).ratio()
        if ratio > score:
            best, score = index, ratio
    return best, score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path,
                        default=Path("/mnt/c/lmfiles/streesedaudio"))
    parser.add_argument("--text", type=Path,
                        default=Path("/mnt/c/lmfiles/streesedaudio/text.txt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--ssl", action="store_true", default=True,
                        help="pool wav2vec2 frames over each vowel, as the "
                             "training set does; the arms are not comparable "
                             "unless both sides carry the same features")
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--ssl-parts", type=int, default=0,
                        help="match the miner's --ssl-parts, so a checkpoint "
                             "trained on sliced vowels can be evaluated here. "
                             "A model expecting 4096 columns cannot read a "
                             "1024-column evaluation set")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/stressed_eval/rows.jsonl"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.asr.whisper_backend import WhisperCompatibleBackend
    from ukstress.config.models import AlignmentConfig, ASRConfig
    from ukstress.features.prosody import extract_prosodic_features, f0_track

    encoder = None
    ssl_blocks: list[np.ndarray] = []
    if args.ssl:
        import transformers

        from run_mine_commonvoice import part_pool

        from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals

        # A feature extractor rather than AutoProcessor: this checkpoint ships
        # a KenLM decoder meant for transcription, and loading it costs two
        # dependencies that frame embeddings never touch.
        encoder = HuggingFaceSpeechEncoder(
            args.ssl_model, device=args.device,
            processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
            model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())

    blocks = paragraphs(args.text)
    print(f"paragraphs: {len(blocks)}", flush=True)

    asr = WhisperCompatibleBackend(ASRConfig(
        model_id=args.asr_model, device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8"))
    config = AlignmentConfig(device=args.device, return_char_alignments=True)
    aligner = WhisperXWordAligner(config)

    files = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() == ".wav")
    rows: list[dict] = []
    for path in files:
        waveform, rate = load_16k(path)
        heard = WORD.findall(" ".join(
            s.text for s in asr.transcribe(waveform, rate).segments))
        index, score = best_paragraph(heard, blocks)
        block = blocks[index]
        print(f"  {path.name[:38]:<38} -> paragraph {index + 1} ({score:.2f})", flush=True)
        if score < 0.4:
            continue

        transcript = " ".join(stripped(w) for w in block)
        result, raw = aligner.align_with_chars(waveform, rate, transcript)
        chars = sorted(
            (c for c in _raw_chars(raw) if c.get("start") is not None
             and c.get("end") is not None),
            key=lambda c: float(c["start"]))

        # Occurrence order, not just the form. `ві́кна` and `вікна́` are the
        # same spelling in one paragraph, and pairing by form alone cannot say
        # which of them a given aligned token is — which would lose exactly the
        # words this set exists to measure. The aligned words come back in the
        # order they were spoken and the paragraph is read straight through, so
        # the nth occurrence of a form matches the nth marked one.
        occurrences: dict[str, list[int]] = {}
        for word in block:
            ordinal = stress_ordinal(word)
            if ordinal is not None:
                occurrences.setdefault(stripped(word), []).append(ordinal)
        marked = occurrences
        seen: dict[str, int] = {}

        refined = []
        for word in result.words:
            form = stripped(word.normalized_token)
            if sum(1 for c in form if c in VOWELS) < 2 or form not in marked:
                continue
            inside = [c for c in chars
                      if word.start_s - 0.02 <= float(c["start"])
                      and float(c["end"]) <= word.end_s + 0.02]
            vowels = _vowels_from_whisperx_chars(inside, word.normalized_token)
            expected = sum(1 for c in word.normalized_token if c in VOWELS)
            quality = len(vowels) / expected if expected else 0.0
            durations = [round((v.end_s - v.start_s) * 1000) for v in vowels]
            if len(vowels) < 2 or quality < args.min_quality or len(set(durations)) == 1:
                continue
            labels = marked[form]
            index_here = seen.get(form, 0)
            seen[form] = index_here + 1
            label = labels[index_here] if index_here < len(labels) else labels[-1]
            refined.append((word, vowels, labels, quality, durations, label))

        frames = encoder.encode(waveform, rate) if encoder is not None else None
        every = [v for _, vowels, _, _, _, _ in refined for v in vowels]
        cache = f0_track(waveform, rate, min_hz=60.0, max_hz=400.0,
                         frame_ms=40, hop_ms=10) if refined else None
        for word, vowels, labels, quality, durations, label in refined:
            features = extract_prosodic_features(
                waveform, rate, vowels,
                word_start_s=word.start_s, word_end_s=word.end_s,
                utterance_vowels=every, f0_cache=cache)
            ssl_index = None
            if frames is not None:
                pooled = (
                    part_pool(frames, vowels,
                              frame_shift_s=encoder.frame_shift_s,
                              parts=args.ssl_parts)
                    if args.ssl_parts else
                    mean_pool_intervals(frames, vowels,
                                        frame_shift_s=encoder.frame_shift_s))
                ssl_index = [len(ssl_blocks), len(ssl_blocks) + len(pooled)]
                ssl_blocks.extend(np.asarray(v, dtype=np.float16) for v in pooled)
            rows.append({
                "token": word.token, "form": stripped(word.normalized_token),
                "ssl_index": ssl_index,
                # A form written with two different accents in this passage is
                # the case the whole exercise is about; both are recorded and
                # the prediction counts as right if it matches either, because
                # which occurrence this is cannot be recovered here.
                "labels": sorted(set(labels)),
                "label": label,
                "ambiguous_in_text": len(set(labels)) > 1,
                "source": path.name, "quality": quality,
                # Absolute spans, as the miner records: a fine-tuned encoder
                # has to pool its own frames here, and only the passage knows
                # where in its recording each vowel fell.
                "vowel_spans": [[round(v.start_s, 4), round(v.end_s, 4)]
                                for v in vowels],
                "durations_ms": durations, "features": features,
            })

    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if ssl_blocks:
        np.save(args.out.with_suffix(".ssl.npy"), np.vstack(ssl_blocks))
        print(f"ssl features: {len(ssl_blocks)} vowels")

    both = sum(1 for r in rows if r["ambiguous_in_text"])
    print(f"\nrows: {len(rows)}  over {len({r['form'] for r in rows})} forms")
    print(f"forms the passage stresses two ways: {both}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
