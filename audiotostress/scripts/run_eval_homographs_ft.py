"""Score a fine-tuned encoder on the words a dictionary cannot settle.

The unfrozen model was measured on forms the trie names unambiguously — the
forms that need no model — where it beat the frozen one by two or three tenths
of a point. Homographs are a different question: they are the only rows where
the recording carries information no text has, and a representation that has
been adapted to Ukrainian might help there more than it helps on average, or
not at all.

The checkpoint carries its own encoder, so this runs that encoder over the
passage audio rather than reading cached vectors, and pools the same three
slices the training used.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_finetune_encoder import pool_spans  # noqa: E402
from run_mine_commonvoice import load_clip  # noqa: E402
from run_train_speakers import prosody_matrix  # noqa: E402


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/finetuned_v2/finetuned.pt"))
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/audio_runs/stressed_spans/rows.jsonl"))
    parser.add_argument("--audio-dir", type=Path,
                        default=Path("/mnt/c/lmfiles/streesedaudio"))
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/homograph_finetuned.json"))
    args = parser.parse_args()

    import torch
    import transformers

    from ukstress.ranker.model import VowelStressRanker

    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    encoder = transformers.AutoModel.from_pretrained(args.ssl_model)
    encoder.load_state_dict(payload["encoder"])
    encoder.to(args.device).eval()
    ranker = VowelStressRanker(**payload["architecture"])
    ranker.module.load_state_dict(payload["ranker"])
    ranker.module.to(args.device).eval()
    parts = payload.get("parts", 3)
    print(f"checkpoint: {args.checkpoint}  (dev {payload.get('dev', 0):.4f})")

    rows = []
    for line in args.rows.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if 0 <= row["label"] < len(row["features"]) and row.get("vowel_spans"):
                rows.append(row)
    print(f"eval rows:  {len(rows):,}")

    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)

    picks: dict[int, int] = {}
    with torch.no_grad():
        for source, words in by_source.items():
            waveform, rate = load_clip(args.audio_dir / source)
            audio = torch.as_tensor(waveform, device=args.device)
            frames = encoder(audio[None, :]).last_hidden_state[0]
            for row in words:
                spans = row["vowel_spans"]
                if len(spans) != len(row["features"]):
                    continue
                block = pool_spans(frames, spans, frame_shift_s=0.02, parts=parts)
                prosody = torch.as_tensor(prosody_matrix(row),
                                          device=args.device)[None, :, :]
                valid = torch.ones(1, len(spans), dtype=torch.bool, device=args.device)
                logits = ranker.module(block[None, :, :], prosody, valid)[0]
                picks[id(row)] = int(logits.argmax())

    def report(name: str, subset: list[dict]) -> dict:
        scored = [(r, picks[id(r)]) for r in subset if id(r) in picks]
        if not scored:
            return {}
        hits = sum(1 for row, pick in scored if pick == row["label"])
        low, high = wilson(hits, len(scored))
        chance = sum(1 / len(row["features"]) for row, _ in scored) / len(scored)
        longest = sum(1 for row, _ in scored
                      if int(np.argmax([f["duration_s"] for f in row["features"]]))
                      == row["label"])
        print(f"\n{name}  ({len(scored)} rows, "
              f"{len({r['form'] for r, _ in scored})} forms)")
        print(f"  chance          {100 * chance:.1f}%")
        print(f"  longest vowel   {100 * longest / len(scored):.1f}%")
        print(f"  fine-tuned      {100 * hits / len(scored):.1f}%  "
              f"[{100 * low:.1f}, {100 * high:.1f}]")
        return {"rows": len(scored), "hits": hits, "accuracy": hits / len(scored),
                "ci": [low, high], "chance": chance,
                "longest_vowel": longest / len(scored)}

    summary = {"checkpoint": str(args.checkpoint),
               "all": report("all words", rows),
               "homographs": report("homographs only",
                                    [r for r in rows if r.get("ambiguous_in_text")])}
    wrong = collections.Counter(
        r["form"] for r in rows
        if r.get("ambiguous_in_text") and id(r) in picks and picks[id(r)] != r["label"])
    if wrong:
        print("\nmissed homograph forms: " +
              ", ".join(f"{f}×{c}" for f, c in wrong.most_common()))
        summary["missed"] = dict(wrong)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
