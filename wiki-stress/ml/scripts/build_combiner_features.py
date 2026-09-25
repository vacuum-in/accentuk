"""Every tier's opinion about every candidate, independently.

The live API returns one answer per token, from whichever tier came first in
a fixed order; the others are never asked. A combiner needs all of them at
once, so this runs each signal on its own over the labelled tokens
(`build_combiner_tokens.py`) and writes, per token and per candidate:

  lexicon      served rank; whether every stored spelling is capitalised
  surface      token capitalised mid-sentence; a 2/3/4 numeral just before
  morphology   the morphology tier's pick (trie readings × the tagger's tags),
               and how the tagger's Case/Number/Gender/upos match the
               reading's tags from the readings table
  crossenc     the cross-encoder's softmax over the candidates, when the
               form is in its inventory with the same candidate set
  classifier   tok-v5's probability for the candidate, the form's training
               count and how one-sided its training rows were
  audio        the share of narrators (books + YouTube, ranker >= 0.99) who
               said this reading of this form, and how many said anything
  readings     whether the readings are told apart by sense / feats / proper

Missing opinions are features too (an `*_available` flag and a zero), so a
tier's silence is something the combiner can learn to read.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "etl" / "src"))

from ukstress_ml.combiner import Assets, Opinions, features, modifier_agreement, softmax  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=Path("output/ml/combiner/tokens.jsonl"))
    parser.add_argument("--readings", type=Path, default=Path("output/ml/readings.jsonl"))
    parser.add_argument("--spacy", default="models/uk_core_news_trf_380")
    parser.add_argument("--crossenc", default="output/ml/models/v19-v10")
    parser.add_argument("--classifier", type=Path, default=Path("models/tok-v5"))
    parser.add_argument("--classifier-corpus", type=Path, default=Path("output/ml/corpus/tok-v5/train.jsonl"))
    parser.add_argument("--audio-rows", nargs="*", default=[
        "/home/devops/audiotostress/artifacts/books/*.rows3.jsonl",
        "/home/devops/audiotostress/youtube/*.rows4.jsonl"])
    parser.add_argument("--audio-prior-cache", type=Path, default=Path("output/ml/combiner/audio_prior_all.json"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/combiner/features.jsonl"))
    parser.add_argument("--pack", type=Path,
                        help="also write the serving assets (readings, narrators, classifier profile) here")
    args = parser.parse_args()

    tokens = [json.loads(line) for line in args.tokens.read_text(encoding="utf-8").splitlines() if line.strip()]
    forms = {t["form"] for t in tokens}
    print(f"{len(tokens):,} tokens, {len(forms):,} forms", flush=True)

    # Readings table, narrators and classifier profile for every ambiguous
    # form: the same tables the service loads, so training sees what serving sees.
    readings_lines = args.readings.read_text(encoding="utf-8").splitlines()
    all_forms = {json.loads(line)["form"] for line in readings_lines}
    # forms made ambiguous only by the cross-encoder's inventory widening the
    # candidate list, and every labelled form: all need their narrators too
    manifest = json.loads((Path(args.crossenc) / "serving_manifest.json").read_text(encoding="utf-8"))
    all_forms |= {unicodedata.normalize("NFC", f) for f in manifest.get("forms", {})} | forms
    if args.audio_prior_cache.exists():
        prior = {k: collections.Counter(v) for k, v in
                 json.loads(args.audio_prior_cache.read_text(encoding="utf-8")).items()}
    else:
        prior = collections.defaultdict(collections.Counter)
        files = [p for pattern in args.audio_rows for p in sorted(glob.glob(pattern))]
        for i, path in enumerate(files):
            for line in open(path, encoding="utf-8"):
                start = line.find('"form": "')
                if start < 0:
                    continue
                end = line.find('"', start + 9)
                if line[start + 9:end].lower() not in all_forms:
                    continue
                r = json.loads(line)
                if r.get("confidence", 0) >= 0.99 and r.get("readings", 1) > 1:
                    prior[r["form"].lower()][str(r["audio"])] += 1
            if i % 200 == 0:
                print(f"  audio prior: {i}/{len(files)} files", flush=True)
        args.audio_prior_cache.parent.mkdir(parents=True, exist_ok=True)
        args.audio_prior_cache.write_text(json.dumps(prior, ensure_ascii=False), encoding="utf-8")
    print(f"audio prior for {len(prior):,} forms", flush=True)
    trained: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in args.classifier_corpus.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if len(r["candidates"]) > 1:
            trained[r["form"].lower()][str(r["gold"])] += 1
    if args.pack:
        args.pack.mkdir(parents=True, exist_ok=True)
        with (args.pack / "readings.jsonl").open("w", encoding="utf-8") as handle:
            for line in readings_lines:
                r = json.loads(line)
                handle.write(json.dumps({"form": r["form"], "told_apart_by": r["told_apart_by"], "readings": [
                    {"signature": x["signature"], "feats": x["feats"], "upos": x["upos"],
                     "proper": bool(x.get("proper")), "senses": bool(x["senses"])} for x in r["readings"]]},
                    ensure_ascii=False) + "\n")
        (args.pack / "audio_prior.json").write_text(json.dumps(prior, ensure_ascii=False), encoding="utf-8")
        (args.pack / "classifier_profile.json").write_text(json.dumps(trained, ensure_ascii=False), encoding="utf-8")
        print(f"serving assets -> {args.pack}", flush=True)
        assets = Assets.load(args.pack)
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "readings.jsonl").write_text("\n".join(readings_lines), encoding="utf-8")
            (tmp / "audio_prior.json").write_text(json.dumps(prior, ensure_ascii=False), encoding="utf-8")
            (tmp / "classifier_profile.json").write_text(json.dumps(trained, ensure_ascii=False), encoding="utf-8")
            assets = Assets.load(tmp)

    # The tagger, and the morphology tier's own decision over trie readings.
    import spacy
    from ukstress_ml.morphology import resolve as morphology_resolve
    from ukstress_ml.morphology import agreement_repair
    from ukrainian_word_stress.stressify_ import _load_dictionary, _parse_dictionary_value, _trie_value
    from ukstress.normalizer import stress_signature
    nlp = spacy.load(args.spacy, exclude=["lemmatizer", "parser", "ner", "senter"])
    trie = _load_dictionary()
    sentences = sorted({t["sentence"] for t in tokens})
    parsed: dict[str, dict[int, tuple[str, str, int]]] = {}
    for doc, sentence in zip(nlp.pipe(sentences, batch_size=64), sentences):
        parsed[sentence] = {tok.idx: (tok.pos_, str(tok.morph), i) for i, tok in enumerate(doc)}
        parsed[sentence]["_words"] = [(tok.idx, tok.text.lower(), tok.pos_) for tok in doc]
        parsed[sentence]["_spans"] = [(tok.idx, tok.text, tok.pos_, str(tok.morph)) for tok in doc]
    print(f"tagged {len(parsed):,} sentences", flush=True)

    def morph_pick(t) -> str | None:
        tag = parsed[t["sentence"]].get(t["start"])
        if tag is None:
            return None
        value = _trie_value(trie, t["form"])
        if not value:
            return None
        readings = [(list(tags), list(acc)) for tags, acc in _parse_dictionary_value(value[0])]
        spans = {(idx, idx + len(text)): (pos, morph_tags)
                 for idx, text, pos, morph_tags in parsed[t["sentence"]]["_spans"]}
        repaired = agreement_repair(readings, t["sentence"], spans, t["start"])
        if repaired is not None:
            accents = list(repaired)
        else:
            resolution = morphology_resolve(readings, tag[0], tag[1])
            accents = list(resolution.accents) if resolution is not None else []
        if not accents:
            return None
        position = accents[0]
        word = t["form"]
        try:
            return stress_signature(word[:position] + "́" + word[position:])
        except Exception:  # noqa: BLE001
            return None

    # The cross-encoder, exactly as served.
    from ukstress_ml.serving import ServingError, Target, load_model
    crossenc = load_model(args.crossenc)
    xenc: dict[int, dict[str, float]] = {}
    for i, t in enumerate(tokens):
        try:
            decision = crossenc.resolve([Target(sentence=t["sentence"], start=t["start"], end=t["end"],
                                                form=t["form"], candidates=tuple(t["candidates"]))])[0]
        except ServingError:
            continue
        scores = decision.get("scores") or {}
        if scores:
            xenc[i] = softmax(scores)
        if i % 2000 == 0:
            print(f"  cross-encoder {i:,}/{len(tokens):,}", flush=True)
    print(f"cross-encoder answered {len(xenc):,} tokens", flush=True)

    # The classifier, with its probability for every candidate.
    from ukstress_ml.token_resolver import TokenResolver
    resolver = TokenResolver(args.classifier, min_seen=0)
    tok: dict[int, dict[str, float]] = {}
    batch = []
    for i, t in enumerate(tokens):
        batch.append({**t, "index": i})
        if len(batch) == 64 or i == len(tokens) - 1:
            for d in resolver.resolve(batch, with_probabilities=True):
                tok[d["index"]] = d["probabilities"]
            batch = []
    print(f"classifier answered {len(tok):,} tokens", flush=True)

    import pymorphy3
    morph = pymorphy3.MorphAnalyzer(lang="uk")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for i, t in enumerate(tokens):
            words = parsed[t["sentence"]]["_words"]
            position = next((k for k, (idx, _, _) in enumerate(words) if idx == t["start"]), None)
            tag = parsed[t["sentence"]].get(t["start"])
            previous = words[position - 1][1] if position else ""
            opinions = Opinions(tag=(tag[0], tag[1]) if tag else None, previous=previous,
                                first=position in (None, 0), morph_pick=morph_pick(t),
                                xenc=xenc.get(i, {}), tok=tok.get(i),
                                modifier=modifier_agreement(previous, morph))
            per_candidate = features(t["form"], t["text"], t["candidates"], opinions, assets)
            handle.write(json.dumps({"i": i, "split": t["split"], "source": t["source"], "form": t["form"],
                                     "gold": t["gold"], "pipeline": t["pipeline"], "status": t["status"],
                                     "candidates": t["candidates"], "features": per_candidate},
                                    ensure_ascii=False) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
