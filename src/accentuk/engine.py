"""Accentor orchestration and lifecycle."""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .combiner import (
    Assets as CombinerAssets,
    Combiner,
    Opinions,
    add_pipeline,
    features as combiner_features,
    load_combiner,
    modifier_agreement,
    softmax,
)
from .errors import AssetError
from .inference import (
    ContextualStressModel,
    CT2ContextualStressModel,
    Target,
    load_contextual_model,
    validate_target,
)
from .lexicon import Lexicon
from .morphology import (
    SpacyMorphologyTier,
    agreement_repair,
    apply_accents,
    imperfective_infinitive,
    resolve as morphology_resolve,
)
from .normalizer import ACUTE, lookup_key
from .text import (
    BREAKS,
    StressToken,
    apply_signature,
    by_capitalisation,
    counts_as_vowel,
    sentence_initial,
    signature_of_output,
    vowel_count,
    word_tokens,
)
from .token_resolver import CT2TokenResolver, TokenResolver, load_token_resolver

PREPOSITIONAL_PRONOUNS = {
    "мене": "0",
    "тебе": "0",
    "себе": "0",
    "нього": "0",
    "неї": "0",
    "нею": "0",
    "ними": "0",
}

GOVERNING_PREPOSITIONS = frozenset({
    "без", "біля", "в", "від", "для", "до",
    "за", "з", "зі", "із", "коло", "крізь",
    "круг", "між", "на", "над", "о", "об",
    "округ", "перед", "під", "по", "повз",
    "поза", "поміж", "попри", "при", "про",
    "серед", "у", "через", "од",
})


@dataclass(frozen=True)
class StressResult:
    """The result of stress marking on an input text."""

    text: str
    tokens: list[StressToken]
    dataset_id: int = 2
    model_version: str = ""
    coverage_manifest_hash: str = ""
    warnings: list[str] = field(default_factory=list)
    mode: str = ""


class SuffixTable:
    """Analogy-based suffix stress table for out-of-vocabulary words."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.from_end: dict[str, int] = {}
        self.max_suffix = 13
        if path:
            p = Path(path)
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for suffix, val in data.items():
                        self.from_end[suffix] = int(val)
                except Exception:
                    pass

    def stress(self, surface: str) -> str:
        if not self.from_end:
            return ""
        decomposed = list(unicodedata.normalize("NFD", surface.lower()))
        vowels = sum(1 for r in decomposed if counts_as_vowel(r))
        if vowels < 2:
            return ""
        limit = min(self.max_suffix, len(decomposed))
        for length in range(limit, 0, -1):
            sub = "".join(decomposed[len(decomposed) - length :])
            if sub in self.from_end:
                pos = self.from_end[sub]
                if pos < vowels:
                    return str(vowels - 1 - pos)
        return ""


def apply_prepositional_shift(tokens: list[StressToken], text: str) -> None:
    for i in range(1, len(tokens)):
        sig = PREPOSITIONAL_PRONOUNS.get(tokens[i].text.lower())
        if not sig:
            continue
        if tokens[i - 1].text.lower() not in GOVERNING_PREPOSITIONS:
            continue
        between = text[tokens[i - 1].end : tokens[i].start]
        if not between or not between.isspace():
            continue
        try:
            stressed = apply_signature(tokens[i].text, sig)
            tokens[i].status = "prepositional"
            tokens[i].output_text = stressed
        except Exception:
            continue


def apply_compound_fallback(tokens: list[StressToken], lexicon: Lexicon) -> None:
    for t in tokens:
        if t.status != "not_found" or "-" not in t.text:
            continue
        parts = t.text.split("-")
        out: list[str] = []
        marked = False
        for part in parts:
            entry = lexicon.lookup(part)
            if not entry or not entry.get("candidates"):
                out.append(part)
                continue
            try:
                stressed = apply_signature(part, entry["candidates"][0])
                out.append(stressed)
                marked = True
            except Exception:
                out.append(part)
        if marked:
            t.status = "compound"
            t.output_text = "-".join(out)


def apply_suffix_fallback(tokens: list[StressToken], table: SuffixTable | None) -> None:
    if table is None:
        return
    for t in tokens:
        if t.status != "not_found":
            continue
        sig = table.stress(t.text)
        if not sig:
            continue
        try:
            stressed = apply_signature(t.text, sig)
            t.status = "suffix"
            t.output_text = stressed
        except Exception:
            continue


def apply_positional_default(tokens: list[StressToken]) -> None:
    for t in tokens:
        if t.status != "not_found" or vowel_count(t.text) < 2:
            continue
        segment = 0
        per_segment = [0]
        for c in t.text:
            if c in BREAKS or c.isspace():
                segment += 1
                per_segment.append(0)
            elif counts_as_vowel(c):
                per_segment[segment] += 1
        last = len(per_segment) - 1
        while last > 0 and per_segment[last] == 0:
            last -= 1
        ordinal = max(0, per_segment[last] - 2)
        try:
            stressed = apply_signature(t.text, f"{last}:{ordinal}")
            t.status = "default_position"
            t.output_text = stressed
        except Exception:
            continue


def apply_dictionary_default(tokens: list[StressToken]) -> None:
    for t in tokens:
        if t.status not in ("ambiguous", "model_ineligible"):
            continue
        if not t.candidates:
            continue
        try:
            stressed = apply_signature(t.text, t.candidates[0])
            t.status = "dictionary_default"
            t.output_text = stressed
        except Exception:
            t.status = "invalid_candidate"


class Accentor:
    """Primary local Ukrainian stress marking engine."""

    def __init__(
        self,
        asset_dir: Path | str | None = None,
        *,
        mode: str = "auto",
        device: str = "cpu",
        enable_combiner: bool = False,
    ) -> None:
        if mode not in ("auto", "light", "full"):
            raise ValueError(f"mode must be 'auto', 'light', or 'full', got '{mode}'")
        self.mode = mode
        self.device = device
        self.default_combiner = enable_combiner
        self._closed = False

        if asset_dir is not None:
            self.asset_dir = Path(asset_dir)
            trie_file = self.asset_dir / "lexicon.marisa"
            if not trie_file.is_file():
                trie_file = self.asset_dir / "lexicon" / "lexicon.marisa"
            if not trie_file.is_file():
                raise AssetError(
                    f"Accentor requires installed assets at {self.asset_dir}.",
                    "Run 'accentuk download' to fetch the required models and lexicon.",
                )
        elif os.environ.get("ACCENTUK_ASSETS"):
            self.asset_dir = Path(os.environ["ACCENTUK_ASSETS"])
            trie_file = self.asset_dir / "lexicon.marisa"
            if not trie_file.is_file():
                trie_file = self.asset_dir / "lexicon" / "lexicon.marisa"
            if not trie_file.is_file():
                raise AssetError(f"ACCENTUK_ASSETS at {self.asset_dir} has no lexicon.marisa.")
        else:
            candidates = [Path.home() / ".cache" / "accentuk"]
            trie_file = None
            for cand in candidates:
                tf = cand / "lexicon.marisa" if (cand / "lexicon.marisa").is_file() else cand / "lexicon" / "lexicon.marisa"
                if tf.is_file():
                    self.asset_dir = cand
                    trie_file = tf
                    break
            if trie_file is None:
                self.asset_dir = Path.home() / ".cache" / "accentuk"
                raise AssetError(
                    f"Accentor requires installed assets at {self.asset_dir}.",
                    "Run 'accentuk download' to fetch the required models and lexicon.",
                )

        self.lexicon = Lexicon(trie_file)

        # Small asset tables
        suffix_path = self.asset_dir / "suffix_table.json"
        if not suffix_path.is_file():
            suffix_path = self.asset_dir / "lexicon" / "suffix_table.json"
        self.suffix_table = SuffixTable(suffix_path if suffix_path.is_file() else None)

        counted_forms_path = self.asset_dir / "counted_forms.json"
        if not counted_forms_path.is_file():
            counted_forms_path = self.asset_dir / "lexicon" / "counted_forms.json"
        counted_set: set[str] = set()
        if counted_forms_path.is_file():
            try:
                for entry in json.loads(counted_forms_path.read_text(encoding="utf-8")):
                    if entry.get("verdict") == "counted":
                        counted_set.add(unicodedata.normalize("NFC", entry["form"]).lower())
            except Exception:
                pass
        self.counted_forms = frozenset(counted_set)

        # Manifest metadata
        manifest_path = self.asset_dir / "manifest.json"
        self.dataset_id = 2
        self.model_version = ""
        self.coverage_manifest_hash = ""
        if manifest_path.is_file():
            try:
                mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.dataset_id = int(mdata.get("dataset_id", 2))
                self.model_version = str(mdata.get("model_version", ""))
                self.coverage_manifest_hash = str(mdata.get("inventory_hash", ""))
            except Exception:
                pass

        # Local model tiers (lazy loaded on first use or initialized)
        self._context_model: ContextualStressModel | CT2ContextualStressModel | None = None
        self._spacy_tier: SpacyMorphologyTier | None = None
        self._token_resolver: TokenResolver | CT2TokenResolver | None = None
        self._combiner: Combiner | None = None
        self._combiner_assets: CombinerAssets | None = None
        self._combiner_morph: Any = None

    def _ensure_context_model(self) -> ContextualStressModel | CT2ContextualStressModel | None:
        if self._context_model is not None:
            return self._context_model

        candidates = [
            self.asset_dir,
            self.asset_dir / "stress-model",
            self.asset_dir / "ct2_models" / "stress-model-ct2-int8",
            self.asset_dir / "stress-model-ct2-int8",
        ]
        # One serving manifest for both backends: the CT2 conversion directory does not carry its own,
        # and without it the light model failed to load and light mode ran without a cross-encoder.
        manifest = next((m for m in (self.asset_dir / "stress-model" / "serving_manifest.json",
                                     self.asset_dir / "serving_manifest.json") if m.is_file()), None)
        for cand in candidates:
            if cand.is_dir():
                self._context_model = load_contextual_model(
                    cand, manifest, mode=self.mode, device=self.device
                )
                if self._context_model is not None:
                    ver, inv = self._context_model.metadata()
                    self.model_version = self.model_version or ver
                    self.coverage_manifest_hash = self.coverage_manifest_hash or inv
                    break
        return self._context_model

    def _ensure_spacy_tier(self) -> SpacyMorphologyTier | None:
        if self._spacy_tier is not None:
            return self._spacy_tier
        # Full mode parses with the transformer pipeline; light mode with a small CPU one
        # (ACCENTUK_LIGHT_SPACY=sm|lg|off picks it), since without any tagger the morphology
        # tier, which decides most heteronyms, would be skipped altogether.
        if self.mode == "light":
            choice = os.environ.get("ACCENTUK_LIGHT_SPACY", "sm")
            if choice == "off":
                return None
            names: tuple[str, ...] = (f"spacy-uk_core_news_{choice}",)
        else:
            names = ("spacy-uk_core_news_trf_380", "spacy-uk_core_news_trf", "spacy")
        for candidate_name in names:
            p = self.asset_dir / candidate_name
            if p.is_dir():
                try:
                    self._spacy_tier = SpacyMorphologyTier(
                        p, counted_forms=self.counted_forms, counted_form_enabled=True
                    )
                    break
                except Exception:
                    pass
        return self._spacy_tier

    def _ensure_token_resolver(self) -> TokenResolver | CT2TokenResolver | None:
        if self._token_resolver is not None:
            return self._token_resolver
        candidates = [
            self.asset_dir,
            self.asset_dir / "tok-v5",
            self.asset_dir / "ct2_models" / "tok-v5-ct2-int8",
            self.asset_dir / "tok-v5-ct2-int8",
        ]
        for cand in candidates:
            if cand.is_dir():
                self._token_resolver = load_token_resolver(
                    cand, mode=self.mode, device=self.device
                )
                if self._token_resolver is not None:
                    break
        return self._token_resolver

    def _ensure_combiner(self) -> tuple[Combiner | None, CombinerAssets | None]:
        if self.mode == "light":
            return None, None
        if self._combiner is not None:
            return self._combiner, self._combiner_assets
        for candidate_name in ("combiner-v1", "combiner"):
            p = self.asset_dir / candidate_name
            if p.is_dir() and (p / "model.pt").is_file():
                self._combiner, self._combiner_assets = load_combiner(p)
                if self._combiner:
                    try:
                        import pymorphy3

                        self._combiner_morph = pymorphy3.MorphAnalyzer(lang="uk")
                    except Exception:
                        pass
                    break
        return self._combiner, self._combiner_assets

    def stress(
        self,
        text: str,
        *,
        on_ambiguity: str = "default",
        combiner: bool | None = None,
    ) -> StressResult:
        """Mark stress in text."""
        if self._closed:
            raise RuntimeError("Accentor is closed.")
        if on_ambiguity not in ("default", "preserve"):
            raise ValueError("on_ambiguity must be 'default' or 'preserve'")

        use_combiner = self.default_combiner if combiner is None else combiner

        if not text:
            return StressResult(
                text="",
                tokens=[],
                dataset_id=self.dataset_id,
                model_version=self.model_version,
                coverage_manifest_hash=self.coverage_manifest_hash,
                warnings=[],
                mode=self.mode,
            )

        tokens = word_tokens(text)
        warnings: list[str] = []

        # Step 1: Lexicon lookup and capitalization filtering
        targets: list[Target] = []
        target_indices: list[int] = []

        for i, token in enumerate(tokens):
            if ACUTE in token.text:
                token.status = "already_stressed"
                continue
            if vowel_count(token.text) == 0:
                token.status = "not_required"
                continue

            entry = self.lexicon.lookup(token.text)
            if entry is None or not entry.get("candidates"):
                token.status = "not_found"
                continue

            is_initial = sentence_initial(text, token.start)
            cands = by_capitalisation(
                entry["candidates"],
                entry.get("proper", {}),
                token.text,
                is_initial,
                form=lookup_key(token.text),
            )
            token.candidates = cands

            if len(cands) == 1:
                try:
                    stressed = apply_signature(token.text, cands[0])
                    token.status = "stressed"
                    token.output_text = stressed
                except Exception:
                    token.status = "invalid_candidate"
            else:
                token.status = "model_ineligible"
                target_indices.append(i)
                targets.append(
                    Target(
                        sentence=text,
                        start=token.start,
                        end=token.end,
                        form=lookup_key(token.text),
                        candidates=tuple(cands),
                    )
                )

        # Step 2: Contextual cross-encoder inference
        context_model = self._ensure_context_model()
        if context_model is not None and targets:
            try:
                decisions = context_model.resolve(targets)
                for d in decisions:
                    idx = target_indices[d["index"]]
                    tok = tokens[idx]
                    tok.scores = d.get("scores")
                    tok.margin = d.get("margin")
                    if d.get("status") == "selected":
                        chosen_sig = d.get("signature", "")
                        if chosen_sig in tok.candidates:
                            try:
                                tok.output_text = apply_signature(tok.text, chosen_sig)
                                tok.status = "stressed"
                            except Exception:
                                tok.status = "invalid_candidate"
                        else:
                            tok.status = "invalid_candidate"
                    elif d.get("status") == "candidate_mismatch":
                        tok.status = "invalid_candidate"
                    elif d.get("status") in ("low_margin", "ambiguous"):
                        tok.status = "ambiguous"
                    elif d.get("status") == "model_ineligible":
                        tok.status = "model_ineligible"

            except Exception as err:
                warnings.append(f"Contextual model resolution error: {err}")

        # Step 3: Morphology tier
        spacy_tier = self._ensure_spacy_tier()
        if spacy_tier is not None:
            # Candidates for morphology: ambiguous tokens with len(candidates) >= 2
            morph_tokens = [
                (i, tok)
                for i, tok in enumerate(tokens)
                if len(tok.candidates) >= 2 and tok.status in ("model_ineligible", "ambiguous", "stressed")
            ]
            if morph_tokens:
                try:
                    spans = spacy_tier.parse(text)
                    for i, tok in morph_tokens:
                        entry = self.lexicon.lookup(tok.text)
                        readings = entry.get("readings", []) if entry else []
                        if not readings:
                            continue
                        parse = spans.get((tok.start, tok.end))
                        if not parse:
                            continue

                        rule = ""
                        repaired = agreement_repair(readings, text, spans, tok.start, self._combiner_morph)
                        resolution = (
                            None
                            if repaired is not None
                            else morphology_resolve(readings, parse[0], parse[1])
                        )

                        if repaired is not None:
                            accents, rule = list(repaired), "agreement"
                        elif resolution is not None:
                            accents = list(resolution.accents)
                            if len(parse) > 2:
                                rule = "counted_form"

                        else:
                            preceding = [
                                (spans[(a, b)][0], spans[(a, b)][1], text[a:b])
                                for (a, b) in sorted(spans)
                                if a < tok.start
                            ]
                            chosen = imperfective_infinitive(lookup_key(tok.text), readings, preceding)
                            if chosen is not None:
                                accents, rule = list(chosen), "aspect"
                            else:
                                continue

                        if "-" not in tok.text and len(accents) > 1:
                            accents = accents[:1]

                        if tok.status == "stressed" and not rule:
                            # Tagger match without a rule does NOT override a confident model decision
                            continue

                        tok.status = rule or ("counted_form" if len(parse) > 2 else "morphology")
                        tok.output_text = apply_accents(tok.text, tuple(accents))
                except Exception as err:
                    warnings.append(f"Morphology tier error: {err}")

        # Step 4: Prepositional shift
        apply_prepositional_shift(tokens, text)

        # Step 5: Compound fallback
        apply_compound_fallback(tokens, self.lexicon)

        # Step 6: Suffix fallback
        apply_suffix_fallback(tokens, self.suffix_table)

        # Step 7: Positional default
        apply_positional_default(tokens)

        # Step 8: Token classifier
        token_resolver = self._ensure_token_resolver()
        if token_resolver is not None:
            tok_targets = [
                {
                    "index": i,
                    "sentence": text,
                    "start": tok.start,
                    "end": tok.end,
                    "form": lookup_key(tok.text),
                    "candidates": tok.candidates,
                }
                for i, tok in enumerate(tokens)
                if len(tok.candidates) >= 2 and tok.status in ("ambiguous", "model_ineligible", "stressed")
            ]
            if tok_targets:
                try:
                    decisions = token_resolver.resolve(tok_targets)
                    for d in decisions:
                        tok = tokens[d["index"]]
                        sig = d.get("signature")
                        if sig and sig in tok.candidates:
                            try:
                                tok.output_text = apply_signature(tok.text, sig)
                                tok.status = "token_model"
                            except Exception:
                                pass
                except Exception as err:
                    warnings.append(f"Token resolver error: {err}")

        # Step 9: Dictionary default
        if on_ambiguity == "default":
            apply_dictionary_default(tokens)

        # Step 10: Learned combiner
        combiner_model, combiner_assets = self._ensure_combiner()
        if use_combiner and combiner_model and combiner_assets and on_ambiguity == "default":
            # Combiner targets: tokens with >= 2 single-digit candidates
            comb_targets = [
                (i, tok)
                for i, tok in enumerate(tokens)
                if len([c for c in tok.candidates if c.isdigit()]) >= 2
            ]
            if comb_targets and spacy_tier:
                try:
                    spans = spacy_tier.parse(text)
                    sorted_spans = sorted((a, b, v[0], v[1]) for (a, b), v in spans.items())
                    spans_dict = {(a, b): (u, f) for a, b, u, f in sorted_spans}
                    for i, tok in comb_targets:
                        pos = next((k for k, span in enumerate(sorted_spans) if span[0] == tok.start), None)
                        tag = (sorted_spans[pos][2], sorted_spans[pos][3]) if pos is not None else None
                        prev = text[sorted_spans[pos - 1][0]:sorted_spans[pos - 1][1]].lower() if pos else ""
                        norm_form = lookup_key(tok.text)
                        entry = self.lexicon.lookup(tok.text)
                        readings = entry.get("readings", []) if entry else []

                        morph_pick = None
                        if tag and readings:
                            rep = agreement_repair(readings, text, spans_dict, tok.start, self._combiner_morph)
                            res = None if rep is not None else morphology_resolve(readings, tag[0], tag[1])
                            acc = list(rep) if rep is not None else (list(res.accents) if res is not None else [])
                            if acc:
                                try:
                                    morph_pick = signature_of_output(apply_accents(tok.text, tuple(acc)))
                                except Exception:
                                    pass

                        xenc_scores = tok.scores if tok.scores else {}
                        tok_probs = None
                        if token_resolver and token_resolver.covers(norm_form):
                            try:
                                d_list = token_resolver.resolve(
                                    [{"index": 0, "sentence": text, "start": tok.start, "end": tok.end, "form": norm_form, "candidates": tok.candidates}],
                                    with_probabilities=True,
                                    ignore_coverage=True,
                                )
                                if d_list and "probabilities" in d_list[0]:
                                    tok_probs = d_list[0]["probabilities"]
                            except Exception:
                                pass

                        opinions = Opinions(
                            tag=tag,
                            previous=prev,
                            first=(pos in (None, 0)),
                            morph_pick=morph_pick,
                            xenc=softmax(xenc_scores),
                            tok=tok_probs,
                            modifier=modifier_agreement(prev, self._combiner_morph),
                        )
                        per_cand = combiner_features(norm_form, tok.text, tok.candidates, opinions, combiner_assets)
                        pipe_sig = signature_of_output(tok.output_text)
                        add_pipeline(per_cand, pipe_sig or None, tok.status or None)
                        chosen_sig, _, departed = combiner_model.decide(
                            tok.candidates, per_cand, pipe_sig or None, tok.status or None
                        )
                        if departed and chosen_sig in tok.candidates:
                            try:
                                tok.output_text = apply_signature(tok.text, chosen_sig)
                                tok.status = "combiner"
                            except Exception:
                                pass
                except Exception as err:
                    warnings.append(f"Combiner error: {err}")

        # Step 11: Rebuild output text
        rebuilt: list[str] = []
        pos = 0
        for tok in tokens:
            rebuilt.append(text[pos : tok.start])
            rebuilt.append(tok.output_text)
            pos = tok.end
        rebuilt.append(text[pos:])
        result_text = "".join(rebuilt)

        return StressResult(
            text=result_text,
            tokens=tokens,
            dataset_id=self.dataset_id,
            model_version=self.model_version,
            coverage_manifest_hash=self.coverage_manifest_hash,
            warnings=warnings,
            mode=self.mode,
        )

    def stress_many(
        self,
        texts: list[str],
        *,
        on_ambiguity: str = "default",
        combiner: bool | None = None,
    ) -> list[StressResult]:
        """Mark stress in multiple texts."""
        return [self.stress(t, on_ambiguity=on_ambiguity, combiner=combiner) for t in texts]

    def close(self) -> None:
        """Release lexicon and model resources."""
        self._closed = True
        self.lexicon.close()

    def __enter__(self) -> Accentor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
