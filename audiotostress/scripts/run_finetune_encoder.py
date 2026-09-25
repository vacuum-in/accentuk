"""Fine-tune the top wav2vec2 layers together with the ranker head.

Capacity in the head is settled: three times the parameters scored the same
figure to two decimals. Pooling the vowel in three slices instead of one
bought 0.6 points, which says the ceiling is in the representation. This
tests the representation directly by letting the encoder move.

Only the top layers train. The bottom of an xls-r-300m is generic acoustics
that 48 hours of Ukrainian will not improve and can easily damage, the
convolutional feature extractor stays frozen as it always should, and the
optimiser then carries state for a few tens of millions of parameters rather
than three hundred, which is what makes this fit in 8 GB at all.

The comparison is against `artifacts/audio_runs/production/ranker.pt` on the
same speaker-held-out rows, and the split is computed the same way, so the
two numbers are paired and McNemar applies.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_train_speakers import prosody_matrix  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if 0 <= row["label"] < len(row["features"]) and row.get("vowel_spans"):
            rows.append(row)
    return rows


def speaker_split(rows: list[dict], seed: int) -> dict[str, str]:
    voices = sorted({row["speaker"] for row in rows})
    np.random.default_rng(seed).shuffle(voices)
    cut = int(0.75 * len(voices)), int(0.85 * len(voices))
    return {speaker: ("train" if index < cut[0]
                      else "dev" if index < cut[1] else "test")
            for index, speaker in enumerate(voices)}


def pool_spans(frames, spans, *, frame_shift_s: float, parts: int):
    """Mean over the vowel, then over each of `parts` equal slices.

    Written in torch and kept differentiable — this is the path the encoder's
    gradient comes back through, so the numpy version in the miner cannot be
    reused here.
    """
    import torch

    blocks = []
    total = frames.shape[0]
    for start_s, end_s in spans:
        start = min(int(start_s / frame_shift_s), total - 1)
        end = min(max(int(end_s / frame_shift_s) + 1, start + 1), total)
        window = frames[start:end]
        pieces = [window.mean(dim=0)]
        edges = np.linspace(0, window.shape[0], parts + 1).astype(int)
        for index in range(parts):
            low = int(edges[index])
            high = max(int(edges[index + 1]), low + 1)
            pieces.append(window[low:min(high, window.shape[0])].mean(dim=0))
        blocks.append(torch.cat(pieces))
    stacked = torch.stack(blocks)
    # Against the word's own mean, as every frozen run did: stress is a
    # contrast between a vowel and its neighbours, not an absolute embedding.
    return stacked - stacked.mean(dim=0, keepdim=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/audio_runs/cv_spans/rows.jsonl"))
    parser.add_argument("--audio", type=Path,
                        default=Path("artifacts/audio_runs/cv_spans/audio"))
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--unfreeze", type=int, default=4,
                        help="how many of the encoder's 24 layers may train")
    parser.add_argument("--parts", type=int, default=3)
    parser.add_argument("--accumulate", type=int, default=4,
                        help="clips per optimiser step; one clip at a time\n                             fits the memory, accumulation gives the step\n                             a batch's worth of gradient")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-clips", type=int, default=0,
                        help="cap the clips seen per epoch, for a quick read")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/finetuned"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers

    from ukstress.ranker.model import (
        VowelStressRanker,
        configure_ssl_trainability,
        masked_cross_entropy,
    )

    rows = [r for r in load_rows(args.rows) if not r.get("prepositional_clitic")]
    home = speaker_split(rows, args.seed)
    catalogue = json.loads(args.audio.with_suffix(".json").read_text(encoding="utf-8"))
    samples = np.memmap(args.audio.with_suffix(".f16"), dtype=np.float16, mode="r")
    spans_of = catalogue["clips"]
    rows = [r for r in rows if r["clip"] in spans_of]

    grouped: dict[str, dict[str, list[dict]]] = {"train": {}, "dev": {}, "test": {}}
    for row in rows:
        grouped[home[row["speaker"]]].setdefault(row["clip"], []).append(row)
    for name, part in grouped.items():
        voices = {r["speaker"] for words in part.values() for r in words}
        print(f"{name:<6} {sum(len(w) for w in part.values()):,} words / "
              f"{len(part):,} clips / {len(voices):,} speakers")

    encoder = transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device)
    encoder.freeze_feature_encoder()
    configure_ssl_trainability(encoder, trainable_last_n=args.unfreeze)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"\nencoder: {trainable / 1e6:.0f}M of {total / 1e6:.0f}M parameters train")

    width = len(prosody_matrix(rows[0])[0])
    ranker = VowelStressRanker(ssl_size=1024 * (args.parts + 1), prosody_size=width,
                               hidden_size=args.hidden, transformer_layers=args.layers,
                               attention_heads=4, ffn_size=4 * args.hidden)
    ranker.module.to(args.device)
    optimiser = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": args.lr_encoder},
        {"params": list(ranker.parameters()), "lr": args.lr_head},
    ], weight_decay=0.01)
    torch.manual_seed(args.seed)
    frame_shift = 0.02

    def waveform_of(clip: str):
        low, high = spans_of[clip]
        return torch.as_tensor(np.asarray(samples[low:high], dtype=np.float32),
                               device=args.device)

    def run_clip(clip: str, words: list[dict], *, train: bool):
        audio = waveform_of(clip)
        if audio.numel() < 400:
            return None
        with torch.set_grad_enabled(train):
            frames = encoder(audio[None, :]).last_hidden_state[0]
        blocks, prosody, targets = [], [], []
        for row in words:
            spans = row["vowel_spans"]
            if len(spans) != len(row["features"]):
                continue
            blocks.append(pool_spans(frames, spans, frame_shift_s=frame_shift,
                                     parts=args.parts))
            prosody.append(torch.as_tensor(prosody_matrix(row), device=args.device))
            targets.append(row["label"])
        if not blocks:
            return None
        widest = max(b.shape[0] for b in blocks)
        ssl = torch.zeros(len(blocks), widest, blocks[0].shape[1], device=args.device)
        pros = torch.zeros(len(blocks), widest, prosody[0].shape[1], device=args.device)
        valid = torch.zeros(len(blocks), widest, dtype=torch.bool, device=args.device)
        for index, (block, feature) in enumerate(zip(blocks, prosody, strict=True)):
            ssl[index, :block.shape[0]] = block
            pros[index, :block.shape[0]] = feature
            valid[index, :block.shape[0]] = True
        logits = ranker.module(ssl, pros, valid)
        return logits, valid, torch.as_tensor(targets, device=args.device), words

    def evaluate(part: dict[str, list[dict]]):
        encoder.eval()
        ranker.module.eval()
        per = collections.defaultdict(lambda: [0, 0])
        marks: list[tuple[dict, int]] = []
        with torch.no_grad():
            for clip, words in part.items():
                outcome = run_clip(clip, words, train=False)
                if outcome is None:
                    continue
                logits, valid, targets, kept = outcome
                picks = logits.masked_fill(~valid, float("-inf")).argmax(dim=-1)
                for row, pick, gold in zip(kept, picks.tolist(), targets.tolist(),
                                           strict=False):
                    hit = int(pick == gold)
                    per[row["speaker"]][0] += hit
                    per[row["speaker"]][1] += 1
                    marks.append((row, hit))
        scored = sum(hit for _, hit in marks)
        return scored / max(len(marks), 1), per, marks

    order = list(grouped["train"].items())
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        ranker.module.train()
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(order)
        schedule = order[:args.max_train_clips] if args.max_train_clips else order
        loss_sum = steps = 0.0
        started = time.time()
        optimiser.zero_grad()
        for index, (clip, words) in enumerate(schedule, 1):
            outcome = run_clip(clip, words, train=True)
            if outcome is None:
                continue
            logits, valid, targets, _ = outcome
            loss = masked_cross_entropy(logits, valid, targets) / args.accumulate
            loss.backward()
            loss_sum += float(loss.detach()) * args.accumulate
            steps += 1
            if index % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimiser.param_groups for p in group["params"]], 1.0)
                optimiser.step()
                optimiser.zero_grad()
            if index % 2000 == 0:
                print(f"  epoch {epoch} {index:,}/{len(schedule):,} clips  "
                      f"loss {loss_sum / max(steps, 1):.4f}  "
                      f"{(time.time() - started) / 60:.1f} min", flush=True)
        optimiser.step()
        optimiser.zero_grad()
        dev_score, _, _ = evaluate(grouped["dev"])
        print(f"epoch {epoch}: loss {loss_sum / max(steps, 1):.4f}  "
              f"dev {100 * dev_score:.2f}%", flush=True)
        if dev_score >= best:
            best = dev_score
            torch.save({"encoder": encoder.state_dict(),
                        "ranker": ranker.module.state_dict(),
                        "architecture": ranker.architecture,
                        "parts": args.parts,
                        "dev": dev_score}, args.out / "finetuned.pt")

    test_score, per_speaker, marks = evaluate(grouped["test"])
    scored = sorted(hit / seen for hit, seen in per_speaker.values() if seen >= 20)
    print(f"\nranker + encoder (test): {100 * test_score:.2f}%")
    if scored:
        print(f"per speaker over {len(scored)} voices with 20+ words:")
        print(f"  median          {100 * statistics.median(scored):.1f}%")
        print(f"  worst / best    {100 * scored[0]:.1f}% / {100 * scored[-1]:.1f}%")
        print(f"  10th pct        {100 * scored[max(0, len(scored) // 10)]:.1f}%")

    (args.out / "marks.jsonl").write_text(
        "\n".join(json.dumps({"clip": r["clip"], "form": r["form"], "hit": h})
                  for r, h in marks), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps({
        "unfrozen_layers": args.unfreeze, "test_accuracy": test_score,
        "dev_accuracy": best, "test_rows": len(marks),
        "per_speaker": {"median": statistics.median(scored) if scored else None,
                        "worst": scored[0] if scored else None},
    }, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
