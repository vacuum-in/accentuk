"""Label the mined ambiguous and OOV words with the audio ranker.

The miner knows where every word is and which readings the lexicon offers, but
not which one was spoken — for a homograph the lexicon has a guess, and for an
out-of-vocabulary word it has nothing. This pass answers that, and it is the
point of the whole book corpus.

It does not realign. The miner stored each word's vowel spans, so this seeks
the audio, encodes once per window, pools the encoder over spans that are
already known, and asks the ranker. That makes it cheaper than mining by the
cost of forced alignment, which was most of it.

Rows without vowel spans are skipped and counted: the first mining run stored
an empty embedding field instead, and those books need re-mining rather than
labelling.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_train_speakers as trainer  # noqa: E402
from run_train_speakers import prosody_matrix  # noqa: E402


def read_window(path: Path, start: float, seconds: float) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start}", "-t", f"{seconds}",
         "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True,
                        help="needed to put the vowel spans back on the clock: "
                             "the miner stored them relative to its 60-second "
                             "slice while the row's start_s is absolute, and "
                             "the slice's own start is anchor + piece * 60")
    parser.add_argument("--slice", type=float, default=60.0)
    parser.add_argument("--calibrate", type=int, default=0,
                        help="instead of labelling the ambiguous words, sample "
                             "this many the lexicon already names and record "
                             "what the ranker says about them. Confidence means "
                             "different things on different material — 0.99 was "
                             "96.8% right on Common Voice, and these are ten "
                             "narrators reading fiction — and this is the only "
                             "way to find out before trusting a gate")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/all_model/ranker.pt"))
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--window", type=float, default=60.0,
                        help="how much audio to decode and encode at once")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers

    from ukstress.datasets import VowelInterval
    from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals
    from ukstress.ranker.model import load_ranker_checkpoint

    ranker, _ = load_ranker_checkpoint(args.checkpoint, device=args.device)
    ranker.module.eval()
    trainer.PROSODY_MODE = {26: "full", 5: "position", 1: "none"}.get(
        ranker.architecture["prosody_size"], "full")
    encoder = HuggingFaceSpeechEncoder(
        args.ssl_model, device=args.device,
        processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
        model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())

    window_start = {}
    for line in args.anchors.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            if "word_start" in a:
                window_start[a["window"]] = a["start_s"]

    wanted, skipped = [], 0
    for line in args.rows.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row["readings"] == 1) != bool(args.calibrate):
            continue   # normally skip what the lexicon answers; invert to calibrate
        if args.calibrate and row.get("label") is None:
            continue
        if not row.get("vowel_spans") or row["window"] not in window_start:
            skipped += 1
            continue
        # Put the spans on the same clock as everything else.
        base = window_start[row["window"]]
        piece = int((row["start_s"] - base) // args.slice)
        origin = base + piece * args.slice
        row["absolute_spans"] = [[origin + a, origin + b]
                                 for a, b in row["vowel_spans"]]
        wanted.append(row)
    if args.calibrate and len(wanted) > args.calibrate:
        import random as _random
        _random.Random(17).shuffle(wanted)
        wanted = wanted[:args.calibrate]
    print(f"{len(wanted):,} rows to label, {skipped:,} skipped for missing spans",
          flush=True)
    if not wanted:
        return 0

    # Group by the window each word falls in, so the encoder runs once per
    # window instead of once per word.
    windows: dict[int, list[dict]] = collections.defaultdict(list)
    for row in wanted:
        windows[int(row["absolute_spans"][0][0] // args.window)].append(row)

    done = 0
    started = time.time()
    handle = args.out.open("a", encoding="utf-8")
    try:
        with torch.no_grad():
            for index in sorted(windows):
                group = windows[index]
                try:
                    audio = read_window(args.audio, index * args.window, args.window)
                    frames = encoder.encode(audio, 16_000)
                except Exception as error:  # noqa: BLE001
                    print(f"  window {index}: {str(error)[:60]}", flush=True)
                    continue
                base = index * args.window
                for row in group:
                    relative = [(a - base, b - base) for a, b in row["absolute_spans"]]
                    if (len(relative) < 2 or relative[0][0] < 0
                            or relative[-1][1] > args.window
                            or any(b <= a for a, b in relative)):
                        continue
                    spans = [VowelInterval(start_s=a, end_s=b, vowel_index=i,
                                           grapheme="а")
                             for i, (a, b) in enumerate(relative)]
                    pooled = np.asarray(mean_pool_intervals(
                        frames, spans, frame_shift_s=encoder.frame_shift_s),
                        dtype=np.float32)
                    block = (pooled - pooled.mean(axis=0, keepdims=True))[None, :, :]
                    stub = {"form": row["form"],
                            "features": [{} for _ in spans]}
                    prosody = prosody_matrix(stub, trainer.PROSODY_MODE)[None, :, :]
                    valid = np.ones((1, len(spans)), dtype=bool)
                    logits = ranker.module(
                        torch.as_tensor(block, device=args.device),
                        torch.as_tensor(prosody, device=args.device),
                        torch.as_tensor(valid, device=args.device))[0]
                    probability = torch.softmax(logits, dim=-1)
                    pick = int(logits.argmax())
                    payload = {
                        **{k: row[k] for k in ("book", "doc", "paragraph", "token",
                                               "form", "start_s", "readings")},
                        "audio": pick,
                        "confidence": round(float(probability[pick]), 4),
                    }
                    if args.calibrate:
                        payload["label"] = row["label"]
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    done += 1
                handle.flush()
                if len(windows) > 20 and index % max(1, len(windows) // 20) == 0:
                    print(f"  {done:,}/{len(wanted):,} labelled, "
                          f"{(time.time()-started)/60:.1f} min", flush=True)
    finally:
        handle.close()
    print(f"\nlabelled {done:,} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
