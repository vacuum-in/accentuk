"""Decode a long recording once, so seeking into it stops costing anything.

`soundfile` cannot jump into an mp3: it decodes from the beginning to reach the
offset, at about 9.7 ms per second of offset. Measured on this book, reading one
120-second window costs 7 s at the 10-minute mark, 117 s at the 3-hour mark and
232 s at 6.5 hours — and would cost 19 minutes near the end. Mining walks
forward through a file, so every window paid for all the audio before it.

A 16-kHz mono WAV of the same recording is about 3.7 GB and reads at any offset
in constant time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("/mnt/c/lmfiles/11.22.63.mp3"))
    parser.add_argument("--out", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/11.22.63.16k.wav"))
    parser.add_argument("--rate", type=int, default=16_000)
    parser.add_argument("--block-s", type=float, default=600.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(args.audio))
    print(f"source: {info.duration / 3600:.2f} h, {info.samplerate} Hz, "
          f"{info.channels} ch", flush=True)

    started = time.time()
    written = 0
    # Sequential blocks: reading forward never seeks, so the whole file decodes
    # once rather than once per window.
    with sf.SoundFile(str(args.out), "w", samplerate=args.rate,
                      channels=1, subtype="PCM_16") as sink:
        for block in sf.blocks(str(args.audio), blocksize=int(args.block_s * info.samplerate),
                               dtype="float32", always_2d=True):
            mono = block.mean(axis=1)
            if info.samplerate != args.rate:
                length = int(round(len(mono) * args.rate / info.samplerate))
                mono = np.interp(
                    np.linspace(0.0, len(mono) - 1, length, dtype=np.float64),
                    np.arange(len(mono), dtype=np.float64),
                    mono,
                ).astype(np.float32)
            sink.write(mono)
            written += len(mono)
            if written % (args.rate * 3600) < len(mono):
                print(f"  {written / args.rate / 3600:.1f} h written, "
                      f"{(time.time() - started) / 60:.1f} min", flush=True)

    print(f"\nwrote {written / args.rate / 3600:.2f} h to {args.out} "
          f"({args.out.stat().st_size / 1e9:.1f} GB) in "
          f"{(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
