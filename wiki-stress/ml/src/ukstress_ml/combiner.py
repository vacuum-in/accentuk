"""The learned combiner over the tiers, shared by training and serving.

One definition of the features, so what the model is served is what it was
trained on. The callers compute the tiers' raw opinions (the tagger's tags,
the morphology tier's pick, the cross-encoder's scores, the classifier's
probabilities) in whatever way suits them; this module turns them into the
per-candidate feature vectors and applies the scorer and its departure gate.
See RESULTS.md, "A learned combiner over the tiers".
"""
from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COUNTED = frozenset({"два", "дві", "три", "чотири", "обидва", "обидві", "півтора", "півтори"})
FEATS = ("Case", "Number", "Gender")
STATUSES = ("morphology", "agreement", "token_model", "dictionary_default", "stressed", "counted_form",
            "prepositional", "suffix", "compound", "ambiguous")


def softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    exp = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


@dataclass
class Opinions:
    """What each tier said about one token, before any combining."""
    tag: tuple[str, str] | None          # (upos, feats) from the tagger
    previous: str                        # the preceding tagger token, lowercased
    first: bool                          # the token opens the sentence
    morph_pick: str | None               # the morphology tier's signature, or None
    xenc: dict[str, float]               # cross-encoder softmax, {} when it did not answer
    tok: dict[str, float] | None         # classifier probabilities, None when it did not answer
    # (Case, Number) pairs the adjacent modifier's surface form can carry, from
    # the dictionary rather than the tagger; None when there is no modifier
    modifier: frozenset[tuple[str, str]] | None = None


class Assets:
    """The per-form tables the features read: readings, narrators, training profile."""

    def __init__(self, readings: dict[tuple[str, str], dict], told: dict[str, set],
                 prior: dict[str, collections.Counter], trained: dict[str, collections.Counter]):
        self.readings, self.told, self.prior, self.trained = readings, told, prior, trained

    @classmethod
    def load(cls, directory: Path) -> "Assets":
        readings: dict[tuple[str, str], dict] = {}
        told: dict[str, set] = {}
        for line in (directory / "readings.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            told[r["form"]] = set(r["told_apart_by"])
            for x in r["readings"]:
                readings[(r["form"], x["signature"])] = {
                    "feats": x.get("feats", []), "upos": x.get("upos", []),
                    "proper": bool(x.get("proper")), "senses": bool(x.get("senses"))}
        prior = {k: collections.Counter(v) for k, v in
                 json.loads((directory / "audio_prior.json").read_text(encoding="utf-8")).items()}
        trained = {k: collections.Counter(v) for k, v in
                   json.loads((directory / "classifier_profile.json").read_text(encoding="utf-8")).items()}
        return cls(readings, told, prior, trained)


MODIFIER_POS = frozenset({"ADJF", "NPRO", "PRTF"})
#: pronouns that stand for a noun rather than modify one
PERSONAL = frozenset({"я", "ти", "він", "вона", "воно", "ми", "ви", "вони", "себе"})
PYMORPHY_CASE = {"nomn": "Nom", "gent": "Gen", "datv": "Dat", "accs": "Acc", "ablt": "Ins",
                 "loct": "Loc", "voct": "Voc"}


def modifier_agreement(previous: str, morph: Any) -> frozenset[tuple[str, str]] | None:
    """The (Case, Number) pairs the word before the target can agree in.

    Only an adjective, a pronoun or a participle counts, and only as the
    dictionary lists it: the tagger reads «Мої сестри.» as a genitive
    singular throughout, «мої» included, though only «моєї» can be one.
    Numerals are left out on purpose: after 2, 3 and 4 the noun takes the
    nominative plural with the genitive singular's stress (три сестри́), and
    the counted-form feature says so. None when the word is not a modifier.
    """
    if morph is None or not previous or not previous[:1].isalpha():
        return None
    parses = morph.parse(previous)
    # every reading of the word must be a modifier: ханів, козаків, Артемові
    # parse as possessive adjectives too, but are nouns in running text
    if not parses or any(p.tag.POS not in MODIFIER_POS or p.normal_form in PERSONAL for p in parses):
        return None
    pairs = set()
    for parse in parses:
        case = PYMORPHY_CASE.get(parse.tag.case or "")
        # VESUM marks a singular modifier by its gender and leaves the number
        # empty (моєї, старий); only the plural says so
        number = ("Plur" if parse.tag.number == "plur" else
                  "Sing" if parse.tag.number == "sing" or parse.tag.gender else None)
        if case and number:
            pairs.add((case, number))
    return frozenset(pairs) or None


def features(form: str, text: str, candidates: list[str], opinions: Opinions,
             assets: Assets) -> dict[str, dict[str, float]]:
    """The feature vector of every candidate of one token (without the pipeline's answer)."""
    tag = opinions.tag
    tag_feats = dict(f.split("=", 1) for f in (tag[1].split("|") if tag and tag[1] else []) if "=" in f)
    cap_mid = text[:1].isupper() and not opinions.first
    heard = assets.prior.get(form, collections.Counter())
    heard_total = sum(heard.values())
    seen = assets.trained.get(form, collections.Counter())
    seen_total = sum(seen.values())
    minority = (min(seen.values()) / seen_total) if len(seen) > 1 else 0.0
    answered = opinions.tok is not None
    out = {}
    for rank, c in enumerate(candidates):
        r = assets.readings.get((form, c), {})
        r_feats = dict(f.split("=", 1) for f in r.get("feats", []) if "=" in f)
        matches = sum(1 for k in FEATS if k in tag_feats and r_feats.get(k) == tag_feats[k])
        conflicts = sum(1 for k in FEATS if k in tag_feats and k in r_feats
                        and not any(f == f"{k}={tag_feats[k]}" for f in r.get("feats", [])))
        own = {(case.split("=", 1)[1], number.split("=", 1)[1])
               for case in r.get("feats", []) if case.startswith("Case=")
               for number in r.get("feats", []) if number.startswith("Number=")}
        agrees = bool(opinions.modifier and own and own & opinions.modifier)
        disagrees = bool(opinions.modifier and own and not own & opinions.modifier)
        out[c] = {
            "mod_agree": float(agrees), "mod_disagree": float(disagrees),
            "lex_rank": rank, "lex_first": float(rank == 0),
            "proper": float(bool(r.get("proper"))),
            "proper_x_capmid": float(bool(r.get("proper")) and cap_mid),
            "proper_x_lower": float(bool(r.get("proper")) and not text[:1].isupper()),
            "counted_x_gensg": float(opinions.previous in COUNTED and r_feats.get("Case") == "Gen"
                                     and r_feats.get("Number") == "Sing"),
            "morph_says": float(opinions.morph_pick == c),
            "morph_other": float(opinions.morph_pick is not None and opinions.morph_pick != c),
            "tag_match": matches, "tag_conflict": conflicts,
            "upos_match": float(bool(tag) and tag[0] in r.get("upos", [])),
            "xenc_p": opinions.xenc.get(c, 0.0), "xenc_avail": float(bool(opinions.xenc)),
            "tok_p": (opinions.tok or {}).get(c, 0.0), "tok_avail": float(answered),
            "tok_seen_log": math.log1p(seen_total) if answered else 0.0,
            "tok_minority": minority if answered else 0.0,
            "audio_share": heard[c] / heard_total if heard_total else 0.0,
            "audio_log_n": math.log1p(heard_total),
            "has_sense": float(bool(r.get("senses"))),
            "told_sense": float("sense" in assets.told.get(form, ())),
            "told_feats": float("feats" in assets.told.get(form, ())),
        }
    return out


def add_pipeline(per_candidate: dict[str, dict[str, float]], pipeline: str | None, status: str | None) -> None:
    """The pipeline's own answer and tier: the combiner learns when to depart from it."""
    for c, f in per_candidate.items():
        says = float(pipeline == c)
        f["pipe_says"] = says
        for s in STATUSES:
            f[f"pipe_says_x_{s}"] = says * float(status == s)


class Combiner:
    """The trained scorer and its gate."""

    def __init__(self, path: Path, tau: float | None = None):
        import torch
        from torch import nn
        self._torch = torch
        saved = torch.load(path, map_location="cpu")
        self.names: list[str] = saved["names"]
        self.mean, self.std = saved["mean"], saved["std"]
        width, hidden = len(self.names), saved["hidden"]
        net = (nn.Linear(width, 1) if hidden == 0 else
               nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1)))
        self.net = net
        self.net.load_state_dict({k.removeprefix("net."): v for k, v in saved["state"].items()})
        self.net.eval()
        self.tau = float(tau if tau is not None else saved.get("tau", 1.0))
        # The tiers the model saw deciding in training. A rule added later
        # (the agreement repair) is outside its experience: the model would
        # overrule it from the classifier and the prior alone, which is how
        # combiner-v1 put «Мої сестри́» back.
        self.known_statuses = {n.removeprefix("pipe_says_x_") for n in self.names
                               if n.startswith("pipe_says_x_")}

    def decide(self, candidates: list[str], per_candidate: dict[str, dict[str, float]],
               pipeline: str | None, status: str | None = None) -> tuple[str, float, bool]:
        """(signature, probability, departed from the pipeline)."""
        if status and self.known_statuses and status not in self.known_statuses and pipeline in candidates:
            return pipeline, 1.0, False
        torch = self._torch
        x = torch.tensor([[per_candidate[c][n] for n in self.names] for c in candidates], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.net((x - self.mean) / self.std).squeeze(-1), 0)
        pick = int(probs.argmax())
        if pipeline in candidates:
            keep = candidates.index(pipeline)
            if float(probs[pick] - probs[keep]) < self.tau:
                return pipeline, float(probs[keep]), False
        return candidates[pick], float(probs[pick]), candidates[pick] != pipeline
