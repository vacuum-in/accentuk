"""Read stress off recordings, then ask the text pipeline what it expects.

Two independent witnesses. The audio model hears which vowel was lengthened
and pitched; the text pipeline reads a dictionary and a morphological parse.
Neither sees what the other saw, so agreement is evidence and disagreement is
information — either a dictionary entry the speakers do not follow, or a word
worth a human's attention.

The audio side never consults the lexicon here. Letting it would make the two
witnesses the same dictionary twice, and agreement would prove nothing.

Where the pipeline reports `dictionary_default` it is guessing among readings
it cannot separate from context. That is exactly where a recording carries
information no text has, and the column to read first.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2")

import numpy as np  # noqa: E402

import run_train_speakers as trainer  # noqa: E402
from run_mine_commonvoice import VOWELS, load_clip, part_pool  # noqa: E402
from run_train_speakers import prosody_matrix  # noqa: E402

ACUTE = "́"

# The audio model's one known blind spot. 1,936 prepositional clitics were
# dropped from its training as suspected-bad labels, so it never learned the
# shift — and on lang-uk's gold the rule it disputes scores 36 of 37. Its
# readings here are confidently wrong, which is worse than uncertain: a whole
# section of a plan and an entry in the dictionary audit were built on them
# before anyone checked. Rows are marked, reported apart, and kept out of the
# disputed-forms list.
PREPOSITIONS = {
    "до", "у", "в", "на", "за", "під", "над", "про", "від", "од", "біля",
    "для", "без", "при", "через", "перед", "між", "із", "з", "зі", "повз",
    "поза", "попід", "коло", "після", "проти", "серед", "крім", "щодо",
}
CLITICS = {
    "мене", "тебе", "себе", "нього", "неї", "них", "нас", "вас",
    "мною", "тобою", "собою", "ньому", "нім", "нею", "ними",
}
WORD_RE = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def is_prepositional_clitic(sentence: str, form: str) -> bool:
    """Does a governing preposition stand immediately before this form?"""
    lowered = form.lower()
    if lowered not in CLITICS:
        return False
    words = [w.lower() for w in WORD_RE.findall(sentence)]
    return any(w == lowered and index and words[index - 1] in PREPOSITIONS
               for index, w in enumerate(words))


def ordinal_of(accented: str) -> int | None:
    """Which vowel the acute sits after, counting from zero.

    Decomposing first matters: the same word arrives composed from one source
    and decomposed from another, and an acute found by character index lands
    somewhere different in each.
    """
    ordinal = -1
    for character in unicodedata.normalize("NFD", accented):
        if character.lower() in VOWELS:
            ordinal += 1
        elif character == ACUTE and ordinal >= 0:
            return ordinal
    return None


def ask_pipeline(text: str, endpoint: str, *, attempts: int = 6) -> list[dict]:
    """Post one sentence, waiting out a service that is briefly gone.

    A sweep of thirty thousand clips runs for hours, and the text service will
    be restarted under it at least once. Without this a single reset connection
    ended a two-hour run and threw away every row it had measured.
    """
    payload = json.dumps({"text": text}).encode("utf-8")
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())["tokens"]
        except Exception as error:  # noqa: BLE001 - any transport failure retries
            last = error
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"pipeline unreachable after {attempts} attempts: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("--text", help="what was said, or a file holding it")
    parser.add_argument("--corpus", type=Path,
                        help="check many Common Voice clips instead of one "
                             "file, which is what turns the agreement rate "
                             "from an anecdote into a statistic")
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--skip", type=int, default=0,
                        help="skip this many of the shuffled clips first, so a "
                             "later sweep covers ground the earlier one did not")
    parser.add_argument("--endpoint", default="http://localhost:8080/v1/stress")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/all_model/ranker.pt"))
    parser.add_argument("--ssl-model",
                        default="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="ignore audio answers below this confidence")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    jobs: list[tuple[Path, str]] = []
    if args.corpus:
        with (args.corpus / "validated.tsv").open(encoding="utf-8", newline="") as handle:
            entries = [r for r in csv.DictReader(handle, delimiter="\t",
                                                 quoting=csv.QUOTE_NONE)
                       if int(r.get("up_votes") or 0) >= 2
                       and int(r.get("down_votes") or 0) == 0]
        random.Random(23).shuffle(entries)
        for entry in entries[args.skip:args.skip + args.clips]:
            clip = args.corpus / "clips" / entry["path"]
            if clip.is_file() and entry["sentence"].strip():
                jobs.append((clip, entry["sentence"].strip()))
    else:
        text = (Path(args.text).read_text(encoding="utf-8").strip()
                if Path(args.text).is_file() else args.text)
        jobs = [(args.audio, text)]
    # Rows land on disk as they are measured. The previous version held
    # everything in memory and wrote once at the end, so a service restart in
    # hour two cost the whole run.
    done: set[str] = set()
    sink_path = args.json.with_suffix(".jsonl") if args.json else None
    if sink_path and sink_path.exists():
        for line in sink_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])
        jobs = [(c, t) for c, t in jobs if c.name not in done]
        print(f"resuming: {len(done):,} clips already measured", flush=True)
    print(f"{len(jobs)} recordings to check", flush=True)

    import torch
    import transformers

    from ukstress.alignment.whisperx_backend import WhisperXWordAligner
    from ukstress.alignment.whisperx_char_backend import (
        _raw_chars,
        _vowels_from_whisperx_chars,
    )
    from ukstress.config.models import AlignmentConfig
    from ukstress.features.ssl import HuggingFaceSpeechEncoder, mean_pool_intervals
    from ukstress.ranker.model import load_ranker_checkpoint

    ranker, _ = load_ranker_checkpoint(args.checkpoint, device=args.device)
    ranker.module.eval()
    ssl_size = ranker.architecture["ssl_size"]
    trainer.PROSODY_MODE = {26: "full", 5: "position", 1: "none"}.get(
        ranker.architecture["prosody_size"], "full")
    parts = ssl_size // 1024 - 1
    aligner = WhisperXWordAligner(AlignmentConfig(device=args.device,
                                                  return_char_alignments=True))
    encoder = HuggingFaceSpeechEncoder(
        args.ssl_model, device=args.device,
        processor=transformers.AutoFeatureExtractor.from_pretrained(args.ssl_model),
        model=transformers.AutoModel.from_pretrained(args.ssl_model).to(args.device).eval())

    rows: list[dict] = []
    for position, (clip, text) in enumerate(jobs, 1):
        try:
            waveform, rate = load_clip(clip)
            result, raw = aligner.align_with_chars(waveform, rate, text)
            frames = encoder.encode(waveform, rate)
        except Exception as error:  # noqa: BLE001 - one bad clip is not fatal
            print(f"  {clip.name}: {str(error)[:60]}", flush=True)
            continue
        chars = sorted((c for c in _raw_chars(raw or {})
                        if c.get("start") is not None and c.get("end") is not None),
                       key=lambda c: float(c["start"]))

        heard: dict[tuple[str, int], tuple[int, float]] = {}
        counted: Counter[str] = Counter()
        with torch.no_grad():
            for word in result.words:
                form = word.normalized_token
                expected = sum(1 for c in form if c in VOWELS)
                if expected < 2:
                    continue
                inside = [c for c in chars
                          if word.start_s - 0.02 <= float(c["start"])
                          and float(c["end"]) <= word.end_s + 0.02]
                vowels = _vowels_from_whisperx_chars(inside, form)
                if len(vowels) < 2 or len(vowels) / expected < args.min_quality:
                    continue
                pooled = np.asarray(
                    part_pool(frames, vowels, frame_shift_s=encoder.frame_shift_s,
                              parts=parts) if parts > 0 else
                    mean_pool_intervals(frames, vowels,
                                        frame_shift_s=encoder.frame_shift_s),
                    dtype=np.float32)
                block = (pooled - pooled.mean(axis=0, keepdims=True))[None, :, :]
                stub = {"form": form, "features": [{} for _ in vowels]}
                prosody = prosody_matrix(stub, trainer.PROSODY_MODE)[None, :, :]
                valid = np.ones((1, len(vowels)), dtype=bool)
                logits = ranker.module(
                    torch.as_tensor(block, device=args.device),
                    torch.as_tensor(prosody, device=args.device),
                    torch.as_tensor(valid, device=args.device))[0]
                probability = torch.softmax(logits, dim=-1)
                pick = int(logits.argmax())
                index = counted[form]
                counted[form] += 1
                if float(probability[pick]) >= args.min_confidence:
                    heard[(form, index)] = (pick, float(probability[pick]))

        try:
            tokens = ask_pipeline(text, args.endpoint)
        except Exception as error:  # noqa: BLE001 - only a sustained outage stops us
            print(f"  pipeline unavailable: {str(error)[:70]}", flush=True)
            return 1
        fresh: list[dict] = []

        seen: Counter[str] = Counter()
        for token in tokens:
            form = unicodedata.normalize("NFC", token["text"]).lower()
            index = seen[form]
            seen[form] += 1
            if (form, index) not in heard:
                continue
            pick, confidence = heard[(form, index)]
            fresh.append({
                "clip": clip.name, "form": token["text"],
                # Kept for the corpus builder: a token classifier trains on
                # (sentence, span, gold), and the span must be the API's own
                # tokenisation, not a re-tokenisation that may differ by one.
                "sentence": text, "start": token.get("start"), "end": token.get("end"),
                "blind_spot": is_prepositional_clitic(text, token["text"]),
                "audio": pick, "text": ordinal_of(token.get("output_text") or ""),
                "confidence": confidence, "status": token.get("status", ""),
                "candidates": token.get("candidates") or [],
            })
        rows.extend(fresh)
        if sink_path:
            with sink_path.open("a", encoding="utf-8") as sink:
                for row in fresh or [{"clip": clip.name}]:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.corpus and position % 25 == 0:
            print(f"  {position}/{len(jobs)} clips, {len(rows):,} words",
                  flush=True)

    for row in rows:
        row["agree"] = row["text"] is not None and row["text"] == row["audio"]
    blind = [r for r in rows if r.get("blind_spot")]
    rows = [r for r in rows if not r.get("blind_spot")]
    both = [r for r in rows if r["text"] is not None]
    if not both:
        print("nothing to compare")
        return 1

    if not args.corpus:
        print(f"\n{'word':<18}{'heard':<12}{'pipeline':<12}{'conf':>6}  status")
        for row in rows:
            expected = "—" if row["text"] is None else f"vowel {row['text']}"
            print(f"{' ' if row['agree'] else '!'} {row['form']:<16}"
                  f"vowel {row['audio']:<6}{expected:<12}"
                  f"{row['confidence']:>6.2f}  {row['status']}")

    agreed = sum(1 for r in both if r["agree"])
    if blind:
        scored = [r for r in blind if r["text"] is not None]
        hit = sum(1 for r in scored if r["text"] == r["audio"])
        print(f"\nexcluded: {len(blind):,} prepositional clitics "
              f"({100 * hit / max(len(scored), 1):.0f}% agreement) — the audio "
              f"model never trained on these and is confidently wrong on them; "
              f"the rule scores 36/37 on lang-uk's gold. Not evidence.")
    print(f"\n{len(rows):,} words heard, {len(both):,} also stressed by the pipeline")
    print(f"agreement: {agreed:,}/{len(both):,} ({100*agreed/len(both):.1f}%)\n")

    print(f"{'pipeline status':<24}{'words':>8}{'agree':>9}")
    by_status: dict[str, list[dict]] = defaultdict(list)
    for row in both:
        by_status[row["status"]].append(row)
    for status, group in sorted(by_status.items(), key=lambda x: -len(x[1])):
        hit = sum(1 for r in group if r["agree"])
        print(f"  {status:<22}{len(group):>8,}{100*hit/len(group):>8.1f}%")

    print(f"\n{'audio confidence':<24}{'words':>8}{'agree':>9}")
    for low, high in ((0.0, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)):
        group = [r for r in both if low <= r["confidence"] < high]
        if group:
            hit = sum(1 for r in group if r["agree"])
            print(f"  {low:.2f} – {high:<16.2f}{len(group):>8,}{100*hit/len(group):>8.1f}%")

    disputed = Counter(r["form"].lower() for r in both if not r["agree"])
    seen_form = Counter(r["form"].lower() for r in both)
    stubborn = [(f, c, seen_form[f]) for f, c in disputed.items()
                if seen_form[f] >= 3 and c / seen_form[f] >= 0.7]
    if stubborn:
        print("\nforms the recordings consistently dispute "
              "(seen 3+ times, disagreeing 70%+):")
        for form, wrong, total in sorted(stubborn, key=lambda x: -x[2])[:40]:
            print(f"  {form:<20}{wrong}/{total}")
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
