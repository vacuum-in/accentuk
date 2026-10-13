"""Learned combiner over inference tiers."""

from __future__ import annotations

import collections
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

COUNTED = frozenset({"два", "дві", "три", "чотири", "обидва", "обидві", "півтора", "півтори"})
FEATS = ("Case", "Number", "Gender")
STATUSES = (
    "morphology",
    "agreement",
    "token_model",
    "dictionary_default",
    "stressed",
    "counted_form",
    "prepositional",
    "suffix",
    "compound",
    "ambiguous",
)

MODIFIER_POS = frozenset({"ADJF", "NPRO", "PRTF"})
PERSONAL = frozenset({"я", "ти", "він", "вона", "воно", "ми", "ви", "вони", "себе"})
PYMORPHY_CASE = {
    "nomn": "Nom",
    "gent": "Gen",
    "datv": "Dat",
    "accs": "Acc",
    "ablt": "Ins",
    "loct": "Loc",
    "voct": "Voc",
}


def softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    exp = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


@dataclass
class Opinions:
    """What each tier said about one token, before combining."""

    tag: tuple[str, str] | None
    previous: str
    first: bool
    morph_pick: str | None
    xenc: dict[str, float]
    tok: dict[str, float] | None
    modifier: frozenset[tuple[str, str]] | None = None


class Assets:
    """Per-form combiner tables: readings, narrators, training profile."""

    def __init__(
        self,
        readings: dict[tuple[str, str], dict],
        told: dict[str, set],
        prior: dict[str, collections.Counter],
        trained: dict[str, collections.Counter],
    ) -> None:
        self.readings = readings
        self.told = told
        self.prior = prior
        self.trained = trained

    @classmethod
    def load(cls, directory: Path) -> Assets:
        readings: dict[tuple[str, str], dict] = {}
        told: dict[str, set] = {}
        readings_file = directory / "readings.jsonl"
        if readings_file.is_file():
            for line in readings_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                told[r["form"]] = set(r.get("told_apart_by", []))
                for x in r.get("readings", []):
                    readings[(r["form"], x["signature"])] = {
                        "feats": x.get("feats", []),
                        "upos": x.get("upos", []),
                        "proper": bool(x.get("proper")),
                        "senses": bool(x.get("senses")),
                    }
        prior_file = directory / "audio_prior.json"
        prior = (
            {k: collections.Counter(v) for k, v in json.loads(prior_file.read_text(encoding="utf-8")).items()}
            if prior_file.is_file()
            else {}
        )
        trained_file = directory / "classifier_profile.json"
        trained = (
            {k: collections.Counter(v) for k, v in json.loads(trained_file.read_text(encoding="utf-8")).items()}
            if trained_file.is_file()
            else {}
        )
        return cls(readings, told, prior, trained)


def modifier_agreement(previous: str, morph: Any) -> frozenset[tuple[str, str]] | None:
    """The (Case, Number) pairs the word before the target can agree in."""
    if morph is None or not previous or not previous[:1].isalpha():
        return None
    parses = morph.parse(previous)
    if not parses or any(
        p.tag.POS not in MODIFIER_POS or p.normal_form in PERSONAL for p in parses
    ):
        return None
    pairs = set()
    for parse in parses:
        case = PYMORPHY_CASE.get(parse.tag.case or "")
        number = (
            "Plur"
            if parse.tag.number == "plur"
            else "Sing"
            if parse.tag.number == "sing" or parse.tag.gender
            else None
        )
        if case and number:
            pairs.add((case, number))
    return frozenset(pairs) or None


def features(
    form: str,
    text: str,
    candidates: list[str],
    opinions: Opinions,
    assets: Assets,
) -> dict[str, dict[str, float]]:
    """Compute feature vector for every candidate of one token."""
    tag = opinions.tag
    tag_feats = (
        dict(f.split("=", 1) for f in (tag[1].split("|") if tag and tag[1] else []) if "=" in f)
    )
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
        conflicts = sum(
            1
            for k in FEATS
            if k in tag_feats
            and k in r_feats
            and not any(f == f"{k}={tag_feats[k]}" for f in r.get("feats", []))
        )
        own = {
            (case.split("=", 1)[1], number.split("=", 1)[1])
            for case in r.get("feats", [])
            if case.startswith("Case=")
            for number in r.get("feats", [])
            if number.startswith("Number=")
        }
        agrees = bool(opinions.modifier and own and own & opinions.modifier)
        disagrees = bool(opinions.modifier and own and not own & opinions.modifier)
        out[c] = {
            "mod_agree": float(agrees),
            "mod_disagree": float(disagrees),
            "lex_rank": rank,
            "lex_first": float(rank == 0),
            "proper": float(bool(r.get("proper"))),
            "proper_x_capmid": float(bool(r.get("proper")) and cap_mid),
            "proper_x_lower": float(bool(r.get("proper")) and not text[:1].isupper()),
            "counted_x_gensg": float(
                opinions.previous in COUNTED
                and r_feats.get("Case") == "Gen"
                and r_feats.get("Number") == "Sing"
            ),
            "morph_says": float(opinions.morph_pick == c),
            "morph_other": float(opinions.morph_pick is not None and opinions.morph_pick != c),
            "tag_match": matches,
            "tag_conflict": conflicts,
            "upos_match": float(bool(tag) and tag[0] in r.get("upos", [])),
            "xenc_p": opinions.xenc.get(c, 0.0),
            "xenc_avail": float(bool(opinions.xenc)),
            "tok_p": (opinions.tok or {}).get(c, 0.0),
            "tok_avail": float(answered),
            "tok_seen_log": math.log1p(seen_total) if answered else 0.0,
            "tok_minority": minority if answered else 0.0,
            "audio_share": heard[c] / heard_total if heard_total else 0.0,
            "audio_log_n": math.log1p(heard_total),
            "has_sense": float(bool(r.get("senses"))),
            "told_sense": float("sense" in assets.told.get(form, ())),
            "told_feats": float("feats" in assets.told.get(form, ())),
        }
    return out


def add_pipeline(
    per_candidate: dict[str, dict[str, float]],
    pipeline: str | None,
    status: str | None,
) -> None:
    """Add pipeline decision and status features to candidate vectors."""
    for c, f in per_candidate.items():
        says = float(pipeline == c)
        f["pipe_says"] = says
        for s in STATUSES:
            f[f"pipe_says_x_{s}"] = says * float(status == s)


class Combiner:
    """Trained combiner scorer and departure gate."""

    def __init__(self, path: Path | str, tau: float | None = None) -> None:
        import torch
        from torch import nn

        self._torch = torch
        saved = torch.load(path, map_location="cpu")
        self.names: list[str] = saved["names"]
        self.mean, self.std = saved["mean"], saved["std"]
        width, hidden = len(self.names), saved["hidden"]
        net = (
            nn.Linear(width, 1)
            if hidden == 0
            else nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        )
        self.net = net
        self.net.load_state_dict({k.removeprefix("net."): v for k, v in saved["state"].items()})
        self.net.eval()
        self.tau = float(tau if tau is not None else saved.get("tau", 1.0))
        self.known_statuses = {
            n.removeprefix("pipe_says_x_") for n in self.names if n.startswith("pipe_says_x_")
        }

    def decide(
        self,
        candidates: list[str],
        per_candidate: dict[str, dict[str, float]],
        pipeline: str | None,
        status: str | None = None,
    ) -> tuple[str, float, bool]:
        """Return (signature, probability, departed from pipeline)."""
        if (
            status
            and self.known_statuses
            and status not in self.known_statuses
            and pipeline in candidates
        ):
            return pipeline, 1.0, False
        torch = self._torch
        x = torch.tensor(
            [[per_candidate[c][n] for n in self.names] for c in candidates],
            dtype=torch.float32,
        )
        with torch.no_grad():
            probs = torch.softmax(self.net((x - self.mean) / self.std).squeeze(-1), 0)
        pick = int(probs.argmax())
        if pipeline in candidates:
            keep = candidates.index(pipeline)
            if float(probs[pick] - probs[keep]) < self.tau:
                return pipeline, float(probs[keep]), False
        return candidates[pick], float(probs[pick]), candidates[pick] != pipeline


def load_combiner(directory: Path | str | None) -> tuple[Combiner | None, Assets | None]:
    """Load Combiner and its Assets from directory if present."""
    if not directory:
        return None, None
    path = Path(directory)
    if not (path / "model.pt").is_file():
        return None, None
    try:
        combiner = Combiner(path / "model.pt")
        assets = Assets.load(path)
        return combiner, assets
    except Exception as error:  # noqa: BLE001
        log.warning("Combiner disabled: %s", error)
        return None, None
