"""Decode every clip the rows reference into one 16 kHz float16 archive.

Fine-tuning the encoder reads the same clips once per epoch, and decoding an
mp3 and resampling 48 kHz to 16 kHz costs more than the forward pass saves.
This pays that cost once: one flat file of samples plus an index saying where
each clip starts, so an epoch is a memmap slice rather than thirty thousand
decodes.

Around 48 hours of audio lands in roughly 5.5 GB, which is cheap against the
disk and free against the training loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_mine_commonvoice import load_clip  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/audio_runs/cv_spans/rows.jsonl"))
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/cv-corpus-26.0-2026-06-12/uk"))
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/cv_spans/audio"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    wanted: list[str] = []
    seen: set[str] = set()
    for line in args.rows.read_text(encoding="utf-8").splitlines():
        if line.strip():
            clip = json.loads(line)["clip"]
            if clip not in seen:
                seen.add(clip)
                wanted.append(clip)
    print(f"{len(wanted):,} distinct clips to cache", flush=True)

    index: dict[str, list[int]] = {}
    offset = 0
    started = time.time()
    with args.out.with_suffix(".f16").open("wb") as sink:
        for position, clip in enumerate(wanted, 1):
            try:
                waveform, _ = load_clip(args.corpus / "clips" / clip)
            except Exception as error:  # noqa: BLE001 - one bad clip is not fatal
                print(f"  {clip}: {str(error)[:60]}", flush=True)
                continue
            block = waveform.astype(np.float16)
            sink.write(block.tobytes())
            index[clip] = [offset, offset + len(block)]
            offset += len(block)
            if position % 2000 == 0:
                elapsed = time.time() - started
                print(f"  {position:,}/{len(wanted):,}  "
                      f"{offset * 2 / 1e9:.2f} GB  {elapsed / 60:.1f} min",
                      flush=True)

    args.out.with_suffix(".json").write_text(
        json.dumps({"sample_rate": 16_000, "clips": index}), encoding="utf-8")
    print(f"\ncached {len(index):,} clips, {offset / 16_000 / 3600:.1f} h, "
          f"{offset * 2 / 1e9:.2f} GB -> {args.out.with_suffix('.f16')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
