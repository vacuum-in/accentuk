"""Learn how much to trust each tier, per candidate, instead of a fixed order.

A conditional logit (and a one-hidden-layer variant): every candidate of an
ambiguous token gets a score from its features (`build_combiner_features.py`),
a softmax over the token's candidates gives the decision. Trained on the
Common Voice train split; the model is chosen on cv:dev + lang-uk:dev and
reported once on cv:test and lang-uk:test, against the live pipeline and
against each tier alone. `--with-languk-dev` also trains on lang-uk's dev
half (then lang-uk:test is the only honest lang-uk number).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ukstress_ml.combiner import add_pipeline  # noqa: E402


def load(path: Path, residual: bool):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if residual:
        # The pipeline's own answer and the tier that gave it: the combiner
        # then learns when to depart from the pipeline rather than rebuilding
        # it — the reviewed decisions, the prepositional and counted-form
        # rules have no feature of their own and live only in this answer.
        for r in rows:
            add_pipeline(r["features"], r["pipeline"], r["status"])
    names = sorted(next(iter(rows[0]["features"].values())))
    return rows, names


def tensors(rows, names, mean, std):
    out = []
    for r in rows:
        x = torch.tensor([[r["features"][c][n] for n in names] for c in r["candidates"]], dtype=torch.float32)
        out.append(((x - mean) / std, r["candidates"].index(r["gold"])))
    return out


class Scorer(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.net = (nn.Linear(width, 1) if hidden == 0 else
                    nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1)))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def accuracy(model, data) -> float:
    with torch.no_grad():
        return sum(int(model(x).argmax()) == y for x, y in data) / max(len(data), 1)


def train(data, width, hidden, epochs, lr, weight_decay, seed):
    torch.manual_seed(seed)
    model = Scorer(width, hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(data)
        for x, y in data:
            loss = nn.functional.cross_entropy(model(x).unsqueeze(0), torch.tensor([y]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/ml/combiner/features.jsonl"))
    parser.add_argument("--with-languk-dev", action="store_true")
    parser.add_argument("--languk-weight", type=int, default=4,
                        help="copies of each lang-uk dev token in training (it is 20x smaller than cv)")
    parser.add_argument("--no-residual", action="store_true",
                        help="leave the pipeline's own answer out of the features")
    parser.add_argument("--out", type=Path, default=Path("output/ml/combiner/model.pt"))
    parser.add_argument("--top200", type=Path, default=Path("output/ml/top200_full.jsonl"),
                        help="a top-200 run; its forms mark the frequent-form subset of cv:test")
    args = parser.parse_args()

    rows, names = load(args.features, residual=not args.no_residual)
    by = collections.defaultdict(list)
    for r in rows:
        by[f"{r['source']}:{r['split']}"].append(r)
    train_rows = list(by["cv:train"]) + (by["languk:dev"] * args.languk_weight if args.with_languk_dev else [])
    matrix = torch.tensor([[r["features"][c][n] for n in names] for r in train_rows for c in r["candidates"]])
    mean, std = matrix.mean(0), matrix.std(0).clamp(min=1e-6)
    data = {k: tensors(v, names, mean, std) for k, v in by.items()}
    train_data = tensors(train_rows, names, mean, std)

    # Model selection on the dev halves of both sources.
    best = None
    for hidden in (0, 16, 32):
        for weight_decay in (0.0, 1e-4, 1e-3):
            model = train(list(train_data), len(names), hidden, 4, 3e-3, weight_decay, 17)
            dev = (accuracy(model, data["cv:dev"]) + accuracy(model, data["languk:dev"])) / 2
            print(f"  hidden {hidden:>2} wd {weight_decay:<6} dev cv {accuracy(model, data['cv:dev']):.2%} "
                  f"languk {accuracy(model, data['languk:dev']):.2%}", flush=True)
            if best is None or dev > best[0]:
                best = (dev, hidden, weight_decay, model)
    _, hidden, weight_decay, model = best
    print(f"chosen: hidden {hidden}, weight decay {weight_decay}")

    # A departure gate: keep the pipeline's answer unless the combiner prefers
    # another candidate by at least `tau` in probability. Chosen on the dev
    # halves: the largest cv:dev gain whose lang-uk:dev cost stays within 0.3.
    def decide(x, r, tau):
        with torch.no_grad():
            probs = torch.softmax(model(x), 0)
        pick = int(probs.argmax())
        if r["pipeline"] in r["candidates"]:
            keep = r["candidates"].index(r["pipeline"])
            if float(probs[pick] - probs[keep]) < tau:
                return keep
        return pick

    def gated(split_key, tau):
        rows_ = by[split_key]
        return sum(decide(data[split_key][i][0], r, tau) == data[split_key][i][1]
                   for i, r in enumerate(rows_)) / max(len(rows_), 1)

    base_cv = sum(r["pipeline"] == r["gold"] for r in by["cv:dev"]) / len(by["cv:dev"])
    base_lu = sum(r["pipeline"] == r["gold"] for r in by["languk:dev"]) / len(by["languk:dev"])
    tau = None
    for candidate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        cv, lu = gated("cv:dev", candidate), gated("languk:dev", candidate)
        print(f"  tau {candidate:.1f}: dev cv {cv - base_cv:+.2%}  languk {lu - base_lu:+.2%}")
        if lu - base_lu >= -0.003 and (tau is None or cv > tau[1]):
            tau = (candidate, cv)
    tau = tau[0] if tau else 1.0
    print(f"gate chosen: tau {tau}")

    top_forms = set()
    if args.top200.exists():
        top_forms = {json.loads(line)["form"] for line in args.top200.read_text(encoding="utf-8").splitlines() if line}

    def single(rows_, pick):
        return sum(pick(r) == r["gold"] for r in rows_) / max(len(rows_), 1)

    def argmax_of(feature, fallback):
        def pick(r):
            scores = {c: r["features"][c][feature] for c in r["candidates"]}
            if not any(scores.values()):
                return fallback(r)
            return max(scores, key=scores.get)
        return pick

    def pipeline(r):
        return r["pipeline"]

    def lexicon(r):
        return r["candidates"][0]

    report = {}
    for split in ("cv:test", "languk:test", "cv:test:top200"):
        base = split if split != "cv:test:top200" else "cv:test"
        rows_ = by[base] if split != "cv:test:top200" else [r for r in by["cv:test"] if r["form"] in top_forms]
        index = {id(r): i for i, r in enumerate(by[base])}
        combined = sum(int(model(data[base][index[id(r)]][0]).argmax()) == data[base][index[id(r)]][1]
                       for r in rows_) / max(len(rows_), 1)
        combined_gated = sum(decide(data[base][index[id(r)]][0], r, tau) == data[base][index[id(r)]][1]
                             for r in rows_) / max(len(rows_), 1)
        report[split] = {"tokens": len(rows_), "combiner": combined, f"combiner, gate {tau}": combined_gated,
                         "pipeline": single(rows_, pipeline),
                         "lexicon order": single(rows_, lexicon),
                         "classifier alone": single(rows_, argmax_of("tok_p", lexicon)),
                         "cross-encoder, else lexicon": single(rows_, argmax_of("xenc_p", lexicon)),
                         "morphology, else lexicon": single(rows_, argmax_of("morph_says", lexicon)),
                         "audio prior, else lexicon": single(rows_, argmax_of("audio_share", lexicon))}
    print(f"\n{'':<28}" + "".join(f"{k:>18}" for k in report))
    for key in next(iter(report.values())):
        print(f"{key:<28}" + "".join(
            f"{v[key]:>18,}" if key == "tokens" else f"{100 * v[key]:>17.2f}%" for v in report.values()))

    if hidden == 0:
        weights = model.net.weight.detach()[0]
        print("\nweights (standardised):")
        for name, w in sorted(zip(names, weights.tolist()), key=lambda kv: -abs(kv[1])):
            print(f"  {name:<18}{w:+.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "names": names, "mean": mean, "std": std, "hidden": hidden,
                "tau": tau}, args.out)
    (args.out.with_suffix(".json")).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
