"""Read stress off a recording: audio plus its text, in; accented text, out.

This is the chain closed. Everything else in `scripts/` mines rows, trains on
them, or scores what came out; this is the part that answers the actual
question — given a recording and what was said, where did the speaker put the
stress?

The default is a hybrid, and deliberately so. Where the lexicon names a form
unambiguously it wins: a dictionary that is right by construction should not
be overruled by a model that is right 95% of the time. The ranker is asked
about the forms the dictionary offers two readings for, which is where a
recording carries information no text has — and about anything the dictionary
has never heard of. `--model-only` ignores the lexicon entirely, which is how
you find out whether the two agree.

Accents go in as U+0301 after the vowel, and the insertion steps over any
combining marks already there: writing it blindly at the character index puts
the acute inside a decomposed `ї`, which reads as `киі́̈в`. That bug cost 345
rows in the text pipeline and is not repeated here.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402

from run_mine_commonvoice import VOWELS, load_clip  # noqa: E402
import run_train_speakers as trainer  # noqa: E402
from run_train_speakers import prosody_matrix  # noqa: E402

ACUTE = "́"


def lexicon_options(trie, word: str) -> list[int]:
    """Every stress ordinal the dictionary allows for this form.

    One means the dictionary settles it; two or more mean only the recording
    can; none means the dictionary has never seen the word.
    """
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value

    value = _trie_value(trie, unicodedata.normalize("NFC", word))
    if value is None:
        return []
    ordinals = set()
    for _, accents in _parse_dictionary_value(value[0]):
        for position in accents:
            ordinal = -1
            for index, character in enumerate(unicodedata.normalize("NFC", word), 1):
                if character.lower() in VOWELS:
                    ordinal += 1
                if index == position:
                    ordinals.add(ordinal)
                    break
    return sorted(ordinals)


def accented(token: str, ordinal: int) -> str:
    """The token with an acute after its nth vowel, marks stepped over."""
    out: list[str] = []
    seen = -1
    index = 0
    placed = False
    while index < len(token):
        character = token[index]
        out.append(character)
        index += 1
        if character.lower() not in VOWELS:
            continue
        seen += 1
        # Any combining marks belong to this vowel; the acute goes after them,
        # never between a base letter and its diaeresis.
        while index < len(token) and unicodedata.combining(token[index]):
            out.append(token[index])
            index += 1
        if seen == ordinal and not placed:
            out.append(ACUTE)
            placed = True
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--text", required=True,
                        help="what was said, or a path to a file holding it")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/all_model/ranker.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-only", action="store_true",
                        help="ignore the lexicon and let the ranker answer "
                             "every word, to see where the two disagree")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="leave a word unaccented rather than answer below "
                             "this confidence. The model is calibrated — it "
                             "says 97.3%% and is right 95.9%%, says 99.4%% and "
                             "is right 98.8%% — so a threshold buys predictable "
                             "precision: 0.95 keeps 89%% of words at 97.4%%, "
                             "0.99 keeps 34%% at 98.8%%. Lexicon answers are "
                             "never withheld; they are not the model's guess")
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--trie", default="/home/devops/.cache/uv/archive-v0/"
                                          "Fd1M0Xx2Ca_isCM2/ukrainian_word_stress/data/stress.trie")
    parser.add_argument("--json", type=Path, help="also write per-word detail")
    args = parser.parse_args()

    text = Path(args.text).read_text(encoding="utf-8").strip() \
        if Path(args.text).is_file() else args.text

    import marisa_trie
    import torch
    import transformers

    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.config.models import AlignmentConfig
    from ukstress.features.prosody import extract_prosodic_features, f0_track
    from run_mine_commonvoice import part_pool

    from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals
    from ukstress.ranker.model import load_ranker_checkpoint

    trie = marisa_trie.BytesTrie()
    trie.load(args.trie)
    ranker, metadata = load_ranker_checkpoint(args.checkpoint, device=args.device)
    ranker.module.eval()
    ssl_size = ranker.architecture["ssl_size"]
    # The checkpoint's prosody width says what it was trained on: 26 the full
    # set, 5 the free positional features, 1 none. Measured prosody turned out
    # to cost 0.69 points on unfamiliar voices, so the shipped model uses none
    # of it — and this then skips extracting it, F0 track included.
    trainer.PROSODY_MODE = {26: "full", 5: "position", 1: "none"}.get(
        ranker.architecture["prosody_size"], "full")
    measures_prosody = trainer.PROSODY_MODE == "full"

    waveform, rate = load_clip(args.audio)
    aligner = WhisperXWordAligner(AlignmentConfig(device=args.device,
                                                  return_char_alignments=True))
    result, raw = aligner.align_with_chars(waveform, rate, text)
    words = result.words
    chars = sorted((c for c in _raw_chars(raw or {})
                    if c.get("start") is not None and c.get("end") is not None),
                   key=lambda c: float(c["start"]))

    encoder = None
    if ssl_size:
        encoder = HuggingFaceSpeechEncoder(
            args.ssl_model, device=args.device,
            processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
            model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())
    frames = encoder.encode(waveform, rate) if encoder is not None else None

    measured = []
    for word in words:
        expected = sum(1 for c in word.normalized_token if c in VOWELS)
        if expected < 2:
            continue
        inside = [c for c in chars
                  if word.start_s - 0.02 <= float(c["start"])
                  and float(c["end"]) <= word.end_s + 0.02]
        vowels = _vowels_from_whisperx_chars(inside, word.normalized_token)
        if len(vowels) < 2 or len(vowels) / expected < args.min_quality:
            continue
        durations = [round((v.end_s - v.start_s) * 1000) for v in vowels]
        if len(set(durations)) == 1:
            continue
        measured.append((word, vowels))

    f0_cache = f0_track(waveform, rate, min_hz=60.0, max_hz=400.0,
                        frame_ms=40, hop_ms=10) if measured and measures_prosody else None
    every = [v for _, vowels in measured for v in vowels]

    decisions: dict[int, dict] = {}
    with torch.no_grad():
        for word, vowels in measured:
            options = [] if args.model_only else lexicon_options(trie, word.normalized_token)
            features = (
                extract_prosodic_features(
                    waveform, rate, vowels, word_start_s=word.start_s,
                    word_end_s=word.end_s, utterance_vowels=every,
                    f0_cache=f0_cache)
                if measures_prosody else [{} for _ in vowels])
            row = {"form": word.normalized_token, "features": features}
            prosody = prosody_matrix(row, trainer.PROSODY_MODE)[None, :, :]
            if ssl_size and frames is not None:
                # How many slices the checkpoint was trained on is written in
                # its own width: 1024 columns is the whole-vowel mean, 4096 is
                # that mean plus three slices. Reading it from the checkpoint
                # means a model and its pooling can never fall out of step.
                parts = ssl_size // 1024 - 1
                pooled = np.asarray(
                    part_pool(frames, vowels,
                              frame_shift_s=encoder.frame_shift_s, parts=parts)
                    if parts > 0 else
                    mean_pool_intervals(frames, vowels,
                                        frame_shift_s=encoder.frame_shift_s),
                    dtype=np.float32)
                ssl = (pooled - pooled.mean(axis=0, keepdims=True))[None, :, :]
            else:
                ssl = np.zeros((1, len(vowels), ssl_size), dtype=np.float32)
            valid = np.ones((1, len(vowels)), dtype=bool)
            logits = ranker.module(
                torch.as_tensor(ssl, device=args.device),
                torch.as_tensor(prosody, device=args.device),
                torch.as_tensor(valid, device=args.device))[0]
            # The lexicon narrows the field before the model chooses. Where it
            # names one reading the model is not consulted at all; where it
            # names two, the model picks between them rather than anywhere.
            if len(options) == 1:
                pick, source = options[0], "lexicon"
            else:
                allowed = options if len(options) > 1 else list(range(len(vowels)))
                scores = {o: float(logits[o]) for o in allowed if o < len(vowels)}
                if not scores:
                    continue
                pick = max(scores, key=scores.get)
                source = "model" if len(options) > 1 else "model (unknown form)"
            probabilities = torch.softmax(logits, dim=-1)
            if source != "lexicon" and float(probabilities[pick]) < args.min_confidence:
                continue
            decisions[id(word)] = {
                "form": word.normalized_token, "ordinal": pick, "source": source,
                "options": options, "confidence": float(probabilities[pick]),
                "model_pick": int(logits.argmax()),
            }

    pieces = []
    for word in words:
        decision = decisions.get(id(word))
        pieces.append(accented(word.token, decision["ordinal"]) if decision
                      else word.token)
    print(" ".join(pieces))

    detail = [d for d in decisions.values()]
    by_model = [d for d in detail if d["source"].startswith("model")]
    disagreed = [d for d in detail
                 if d["source"] == "lexicon" and d["model_pick"] != d["ordinal"]]
    print(f"\n{len(detail)} words measured of {len(words)} | "
          f"{len(detail) - len(by_model)} settled by the lexicon, "
          f"{len(by_model)} by the model", file=sys.stderr)
    if disagreed:
        print(f"model disagreed with the lexicon on {len(disagreed)}: " +
              ", ".join(d["form"] for d in disagreed[:12]), file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(detail, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
