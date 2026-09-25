"""Ask the audio ranker what was said, for the words the lexicon cannot answer.

The rows from run_mine_book_asr carry absolute vowel spans, so this seeks the
audio, encodes once per window, pools the encoder over spans that are already
known, and asks the ranker. Nothing is realigned and no slice arithmetic is
reconstructed — the previous labeller had to rebuild which sixty-second slice a
row came from, and that class of bookkeeping is what this pipeline has already
lost a week to.

--calibrate inverts the row filter: instead of the words the lexicon cannot
read, it takes the words it can, and scores the ranker against them. On
audiobook material that number had never been measured, and when it finally was
it came out at chance — which is how the first mining run was found to be void.
Run it on every book before trusting a single label.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
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


def wilson(hits: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z, p = 1.96, hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/all_model/ranker.pt"))
    parser.add_argument("--ssl-model", default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--window", type=float, default=60.0)
    parser.add_argument("--calibrate", type=int, default=0,
                        help="score this many words the lexicon already names "
                             "instead of labelling the ones it cannot")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

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

    wanted = []
    for line in args.rows.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row["readings"] == 1) != bool(args.calibrate):
            continue
        if args.calibrate and row.get("label") is None:
            continue
        if not row.get("vowel_spans"):
            continue
        wanted.append(row)
    if args.calibrate and len(wanted) > args.calibrate:
        import random
        random.Random(17).shuffle(wanted)
        wanted = wanted[:args.calibrate]
    print(f"{len(wanted):,} rows", flush=True)
    if not wanted:
        return 0

    windows: dict[int, list[dict]] = collections.defaultdict(list)
    for row in wanted:
        windows[int(row["vowel_spans"][0][0] // args.window)].append(row)

    scored: list[dict] = []
    done = 0
    started = time.time()
    handle = args.out.open("w", encoding="utf-8")
    try:
        with torch.no_grad():
            for index in sorted(windows):
                group = windows[index]
                base = index * args.window
                try:
                    audio = read_window(args.audio, base, args.window)
                    frames = encoder.encode(audio, 16_000)
                except Exception as error:  # noqa: BLE001
                    print(f"  window {index}: {str(error)[:60]}", flush=True)
                    continue
                for row in group:
                    relative = [(a - base, b - base) for a, b in row["vowel_spans"]]
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
                    stub = {"form": row["form"], "features": [{} for _ in spans]}
                    prosody = prosody_matrix(stub, trainer.PROSODY_MODE)[None, :, :]
                    valid = np.ones((1, len(spans)), dtype=bool)
                    logits = ranker.module(
                        torch.as_tensor(block, device=args.device),
                        torch.as_tensor(prosody, device=args.device),
                        torch.as_tensor(valid, device=args.device))[0]
                    pick = int(logits.argmax())
                    payload = {
                        **{k: row[k] for k in ("book", "doc", "paragraph", "word_index",
                                               "stream", "token", "form", "start_s",
                                               "readings")},
                        "audio": pick,
                        "confidence": round(float(torch.softmax(logits, -1)[pick]), 4),
                    }
                    if args.calibrate:
                        payload["label"] = row["label"]
                        scored.append(payload)
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    done += 1
                handle.flush()
                if len(windows) > 20 and index % max(1, len(windows) // 20) == 0:
                    print(f"  {done:,}/{len(wanted):,}, "
                          f"{(time.time() - started) / 60:.1f} min", flush=True)
    finally:
        handle.close()

    if args.calibrate and scored:
        hits = sum(r["audio"] == r["label"] for r in scored)
        low, high = wilson(hits, len(scored))
        print(f"\n{len(scored):,} words the lexicon names")
        print(f"overall {hits / len(scored):.2%}  [{low:.1%}, {high:.1%}]")
        print("\n| gate | kept | share | precision | 95% CI |")
        print("| ---: | ---: | ---: | ---: | --- |")
        for gate in (0.0, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
            kept = [r for r in scored if r["confidence"] >= gate]
            if not kept:
                continue
            good = sum(r["audio"] == r["label"] for r in kept)
            low, high = wilson(good, len(kept))
            print(f"| ≥{gate:g} | {len(kept):,} | {len(kept) / len(scored):.0%} "
                  f"| {good / len(kept):.2%} | [{low:.1%}, {high:.1%}] |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
