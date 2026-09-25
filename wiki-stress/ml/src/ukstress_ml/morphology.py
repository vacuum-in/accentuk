"""Tier 2: resolve grammatical stress ambiguity from a morphological parse.

The pipeline has three tiers, not two. Tier 1 is the PostgreSQL lookup for
forms with a single stress. Tier 3 is the cross-encoder for semantic
homographs. Between them sits a class the triage identified and nothing served:
forms whose readings differ in accent but are separated by their **tags** —
`ко́леса` (nominative plural) versus `коле́са` (genitive singular).

Measured on 3M words of OpenSubtitles, `grammatical` forms are **47% of all
ambiguous-but-unserved tokens**, the single largest gap, and they need no model
and no annotation spend — only the parse.

Two limits are deliberate:

* Only `Case`, `Number`, `Gender` and `Animacy` are trusted to route
  (`RELIABLE_SEPARATING_FEATURES`). A split that rests on `upos` alone is
  usually patronymic-versus-surname, which Stanza does not decide reliably.
* Forms with a noun/verb accent split are refused outright
  (`has_unreliable_pos_split`). Stanza tags a Ukrainian imperative as a
  nominative plural noun — `Коси траву` parses identically to `довгі коси` —
  so tag matching would return the noun reading with false confidence. Those
  belong to the model tier.

When the matched readings disagree, or nothing matches, this returns None and
the caller falls through rather than guessing.
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ukstress_ml.triage import (
    RELIABLE_SEPARATING_FEATURES,
    has_unreliable_pos_split,
    separating_features,
)

Reading = tuple[list[str], list[int]]


@dataclass(frozen=True)
class Resolution:
    accents: tuple[int, ...]
    matched_readings: int
    reason: str


#: Part-of-speech labels the two tag vocabularies disagree about while meaning
#: the same thing. The trie writes `upos=PRON` for determiners like `цьому`,
#: `всього`, `усі`; UD taggers write `DET`. Case, gender and number agree
#: exactly in these cases, so refusing to match on the label alone threw away
#: readings the parser had in fact resolved.
_EQUIVALENT_UPOS = {
    "PRON": ("PRON", "DET"),
    "DET": ("DET", "PRON"),
    "NOUN": ("NOUN", "PROPN"),
    "PROPN": ("PROPN", "NOUN"),
}


#: Numerals that put a following noun in the *counted form* — historically a
#: dual, and the one construction where a UD parse cannot help. `дві сестри́`
#: and `мої се́стри` are both tagged `Case=Nom|Number=Plur`, identically, but
#: only the second is the nominative plural. The counted form carries the
#: genitive singular's accent, so a noun whose two readings differ there is
#: silently given the wrong one.
COUNTED_NUMERALS = frozenset({"два", "дві", "три", "чотири", "обидва", "обидві"})

#: Tags a numeral may carry and still govern a counted form. In an oblique
#: case (`двома сестрами`, `двох сестер`) the noun agrees normally and nothing
#: needs rewriting.
_COUNTED_NUMERAL_CASES = frozenset({"Nom", "Acc"})

#: Word classes that may stand between the numeral and its noun while still
#: agreeing with it: `дві великі площини́`, `три смугасті полотна́`.
_AGREEING_BETWEEN = frozenset({"ADJ", "DET", "NUM"})


def _feats_to_dict(feats: str | None) -> dict[str, str]:
    return dict(
        part.split("=", 1) for part in (feats or "").split("|") if "=" in part
    )


def counted_form_feats(upos: str | None, feats: str | None) -> str | None:
    """Rewrite a nominative plural as a genitive singular, or return None.

    Returns None whenever the rewrite does not apply, so a caller can leave the
    parse untouched. Only `Case` and `Number` change; `Gender` and `Animacy`
    are what `resolve()` matches on besides, and they are unaffected.
    """
    if upos not in {"NOUN", "PROPN"}:
        return None
    parts = _feats_to_dict(feats)
    # Accusative as well as nominative: `він має дві руки́` is parsed
    # `Case=Acc` throughout, and restricting the rewrite to `Nom` left every
    # object of a verb unstressed by the tier.
    if parts.get("Case") not in ("Nom", "Acc") or parts.get("Number") != "Plur":
        return None
    parts["Case"], parts["Number"] = "Gen", "Sing"
    return "|".join(f"{key}={value}" for key, value in sorted(parts.items()))


def governed_by_counted_numeral(
    preceding: list[tuple[str, str, str]],
) -> bool:
    """Does a 2/3/4 numeral govern the token these tokens precede?

    `preceding` is `(upos, feats, text)` for the tokens before the target, in
    order. Only the immediately preceding word counts: anything else, an
    attributive adjective included, means the numeral does not govern this
    noun directly.
    """
    for upos, feats, text in reversed(preceding):
        if upos not in _AGREEING_BETWEEN:
            return False
        if upos != "NUM":
            # An attributive adjective between the numeral and the noun takes
            # the phrase into the plural: gold reads «дві вели́кі площи́ни» and
            # «три смуга́сті поло́тна», against «дві площини́» with nothing
            # between. Walking past the adjective produced exactly those two
            # errors and no correct answers.
            return False
        if text.lower() not in COUNTED_NUMERALS:
            return False  # `п'ять сестер` governs a genitive plural instead
        case = _feats_to_dict(feats).get("Case")
        return case is None or case in _COUNTED_NUMERAL_CASES
    return False


#: Forms the Orthoepic Dictionary lists with a 2/3/4 numeral, loaded from
#: `COUNTED_FORM_LIST`. Empty means "rewrite every noun", which is the
#: behaviour the benchmark refuted — see `counted_form_enabled`.
_COUNTED_FORMS: frozenset[str] | None = None


def counted_forms() -> frozenset[str]:
    """Surface forms known to take the counted form, or an empty set.

    Membership is lexical and not inferable: `три сестри́` but `дві вели́кі
    площи́ни`. The list is built by `ml/scripts/run_counted_forms.py`, which
    reads it off the Orthoepic Dictionary rather than deciding anything.
    """
    global _COUNTED_FORMS
    if _COUNTED_FORMS is None:
        path = os.environ.get("COUNTED_FORM_LIST", "").strip()
        forms: set[str] = set()
        if path and Path(path).is_file():
            for entry in json.loads(Path(path).read_text(encoding="utf-8")):
                if entry.get("verdict") == "counted":
                    forms.add(unicodedata.normalize("NFC", entry["form"]).lower())
        _COUNTED_FORMS = frozenset(forms)
    return _COUNTED_FORMS


def counted_form_enabled() -> bool:
    """Off by default, and the measurement is the reason.

    Enabling it fixes `три сестри́`, `чотири стіни́` and `три вікна́`, and breaks
    `дві вели́кі площи́ни` and `три смуга́сті поло́тна`, which lang-uk's gold keeps
    in the nominative plural. On that benchmark the trade is negative: heteronym
    82.22% -> 82.12%, macro-F1 62.47% -> 62.05%, sentence 65.01% -> 64.81%.

    The counted form is a lexical property of a noun, not a consequence of its
    paradigm, so a rule over the whole class cannot be right. Doing this
    properly needs a list of the nouns that take it; until that list exists,
    the code stays here, tested, and off.
    """
    return os.environ.get("COUNTED_FORM", "").strip().lower() in {"1", "true", "yes", "on"}


#: Verbs that take an infinitive and require it to be imperfective. «почала
#: засипа́ти», «став засипа́ти», «продовжували засипа́ти» — the phase verb names
#: a stage of an ongoing action, and a perfective infinitive cannot express one.
#:
#: Modals are deliberately absent. «не можна виносити» and «не можна винести»
#: are both grammatical, so `могти`, `можна`, `треба` and `слід` decide nothing
#: and adding them would trade a rule for a guess.
PHASE_VERBS = frozenset({
    "почати", "почав", "почала", "почало", "почали", "починати", "починає",
    "починають", "починав", "починала", "починали",
    "стати", "став", "стала", "стало", "стали",
    "продовжити", "продовжив", "продовжила", "продовжили", "продовжувати",
    "продовжував", "продовжувала", "продовжували", "продовжує", "продовжують",
    "перестати", "перестав", "перестала", "перестали", "переставати",
    "припинити", "припинив", "припинила", "припинили", "припиняти",
    "закінчити", "закінчив", "закінчила", "закінчили",
    "взятися", "взявся", "взялася", "взялися", "братися", "береться",
})


def later_accent(accents: tuple[int, ...]) -> int:
    """Rightmost accent position, the one that orders an aspect pair.

    Within one of these pairs the perfective always carries the earlier accent
    and the imperfective the later one, and it holds across both shapes the
    class takes: a stressed prefix (`ви́носити` against `вино́сити`) and a
    stressed root against a stressed suffix (`заси́пати` against `засипа́ти`).
    Keying on the `ви-` prefix alone missed every verb of the second shape.

    The trie records no aspect — both readings come back
    `VerbForm=Inf|upos=VERB` — so accent order is the only thing that
    distinguishes them, and it is enough for this decision.
    """
    return max(accents) if accents else -1


def imperfective_infinitive(
    form: str,
    readings: list[Reading],
    preceding: list[tuple[str, str, str]],
) -> tuple[int, ...] | None:
    """The imperfective reading, when a phase verb governs this infinitive.

    Returns None whenever the question does not arise: not an infinitive, not a
    `ви-` pair, no phase verb governing it, or the readings do not actually
    disagree. The tier declines these forms otherwise — identical tags leave
    `separating_features` empty — so this is the difference between an answer
    and a dictionary default.
    """
    infinitives = [(tuple(accents), frozenset(tags)) for tags, accents in readings
                   if "VerbForm=Inf" in tags and "upos=VERB" in tags]
    if len(infinitives) < 2:
        return None
    if len({tags for _, tags in infinitives}) != 1:
        return None  # the tags separate them; ordinary resolution applies
    ordered = sorted({a for a, _ in infinitives}, key=later_accent)
    if len(ordered) != 2:
        return None
    imperfective = [ordered[-1]]
    if later_accent(ordered[0]) == later_accent(ordered[-1]):
        return None
    for _, _, text in reversed(preceding):
        lowered = unicodedata.normalize("NFC", text).lower()
        if lowered in PHASE_VERBS:
            return imperfective[0]
        if lowered in ("не", "б", "би", "ж", "же"):
            continue  # «не почала засипа́ти» is still governed by `почала`
        return None
    return None


def apply_counted_form(
    spans: dict[tuple[int, int], tuple[str, str]],
    tokens: list[tuple[tuple[int, int], str, str, str]],
) -> set[tuple[int, int]]:
    """Rewrite in place the parses a counted-form numeral governs.

    Returns the spans it rewrote. A caller needs to tell these apart from an
    ordinary tag match: the answer here comes from a syntactic rule the model
    cannot see, and the corpus the model was trained on labels these tokens
    the other way, so it must not be allowed to overrule them.

    This is the one place the tier looks past the token it is resolving. It
    stays here, next to the feature vocabulary it edits, rather than inside
    `resolve()`, which must keep working from a parse alone.
    """
    rewritten_spans: set[tuple[int, int]] = set()
    if not counted_form_enabled():
        return rewritten_spans
    known = counted_forms()
    for index, (span, upos, feats, text) in enumerate(tokens):
        # An empty list means no list was configured. Rewriting every noun
        # after a numeral is what cost 0.42 macro-F1, so the list is required.
        if known and unicodedata.normalize("NFC", text).lower() not in known:
            continue
        if not governed_by_counted_numeral(
            [(u, f, x) for _, u, f, x in tokens[:index]]
        ):
            continue
        rewritten = counted_form_feats(upos, feats)
        if rewritten is not None:
            spans[span] = (upos, rewritten)
            rewritten_spans.add(span)
    return rewritten_spans


def _feature_set(upos: str | None, feats: str | None) -> set[str]:
    """Flatten a parse into the tag vocabulary the trie's readings use."""
    observed = set((feats or "").split("|")) - {""}
    for label in _EQUIVALENT_UPOS.get(upos or "", (upos,)):
        if label:
            observed.add(f"upos={label}")
    return observed


def resolve(readings: list[Reading], upos: str | None, feats: str | None) -> Resolution | None:
    """Pick the accent whose tags the parse satisfies, or None."""
    if len(readings) < 2:
        return None
    if has_unreliable_pos_split(readings):
        return None
    features = separating_features(readings)
    if not (features & RELIABLE_SEPARATING_FEATURES):
        return None

    observed = _feature_set(upos, feats)
    matched = [tuple(accents) for tags, accents in readings if set(tags) <= observed]
    if not matched:
        return None
    distinct = set(matched)
    if len(distinct) > 1:
        # The parse satisfies readings that disagree; morphology has not
        # decided anything and pretending otherwise would be a coin flip.
        return None
    return Resolution(next(iter(distinct)), len(matched), "tags matched")


def apply_accents(surface: str, accents: tuple[int, ...], acute: str = "́") -> str:
    """Insert the acute at the trie's accent positions.

    These are **character offsets**, not vowel ordinals — `коси` yields `[2]`
    and `[4]`, which cannot be ordinals for a word with two vowels. Note this
    is the opposite convention to `ukstress.normalizer.stress_signature`, which
    counts vowels; conflating the two silently marks the wrong syllable.
    """
    text = surface
    for position in sorted(accents, reverse=True):
        if 0 <= position <= len(text):
            text = text[:position] + acute + text[position:]
    return unicodedata.normalize("NFC", text)


class SpacyMorphologyTier:
    """spaCy-backed resolver, interface-compatible with `MorphologyTier`.

    `resolve()` needs exactly one thing from a parser: `(upos, feats)` per token
    in context, where feats carry Case, Number, Gender and Animacy. Stanza
    supplies that and costs ~500 MB, ~30 s to construct, and is not safe to
    share across threads — measured, one instance serialised the whole API and
    dropped word accuracy from 87.89% to 72.16% under eight concurrent callers.

    `uk_core_news_sm` supplies the same features from 15 MB and loads in under a
    second. Everything but the morphologizer is excluded: the parser, lemmatizer
    and NER cost time and nothing here reads them.
    """

    #: Not `stanza`'s tag names — spaCy writes feats as `A=b|C=d`, which is the
    #: same shape `_feature_set` already splits on.
    def __init__(self, model_path: str | None = None, nlp: Any = None) -> None:
        self._nlp = nlp
        # Default to the vendored copy: the published wheel cannot be
        # installed on this Python, so `spacy.load("uk_core_news_sm")` by name
        # fails. `ml/scripts/fetch_spacy_model.sh` puts it here.
        self._path = (model_path or os.environ.get("SPACY_UK_MODEL")
                      or str(Path(__file__).resolve().parents[3] / "models" / "uk_core_news_sm"))
        self._trie: Any = None

    @property
    def pipeline(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self._path,
                                   exclude=["ner", "lemmatizer", "parser", "senter"])
        return self._nlp

    @property
    def trie(self) -> Any:
        if self._trie is None:
            from ukrainian_word_stress.stressify_ import _load_dictionary

            self._trie = _load_dictionary()
        return self._trie

    def readings(self, form: str) -> list[Reading]:
        from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value

        value = _trie_value(self.trie, unicodedata.normalize("NFC", form))
        if value is None:
            return []
        return [(list(t), list(a)) for t, a in _parse_dictionary_value(value[0])]

    def parse(self, sentence: str) -> dict[tuple[int, int], tuple[str, str]]:
        return self.parse_batch([sentence])[0]

    def parse_batch(self, sentences: list[str]) -> list[dict[tuple[int, int], tuple[str, str]]]:
        if not sentences:
            return []
        results: list[dict[tuple[int, int], tuple[str, str]]] = []
        for doc in self.pipeline.pipe(sentences):
            spans: dict[tuple[int, int], tuple[str, str]] = {}
            tokens: list[tuple[tuple[int, int], str, str, str]] = []
            for token in doc:
                span = (token.idx, token.idx + len(token.text))
                spans[span] = (token.pos_, str(token.morph))
                tokens.append((span, token.pos_, str(token.morph), token.text))
            for span in apply_counted_form(spans, tokens):
                # A third element rather than a parallel structure: callers
                # index `parse[0]` and `parse[1]`, so a longer tuple passes
                # through them untouched and the flag reaches whoever wants it.
                spans[span] = (*spans[span], True)
            results.append(spans)
        return results


class MorphologyTier:
    """Stanza-backed resolver, loaded lazily so importing costs nothing."""

    def __init__(self, pipeline: Any = None, trie: Any = None) -> None:
        self._pipeline = pipeline
        self._trie = trie

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            import stanza

            self._pipeline = stanza.Pipeline(
                "uk", processors="tokenize,pos,mwt",
                download_method=None, logging_level="ERROR",
            )
        return self._pipeline

    @property
    def trie(self) -> Any:
        if self._trie is None:
            from ukrainian_word_stress.stressify_ import _load_dictionary

            self._trie = _load_dictionary()
        return self._trie

    def readings(self, form: str) -> list[Reading]:
        from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value

        value = _trie_value(self.trie, unicodedata.normalize("NFC", form))
        if value is None:
            return []
        return [(list(t), list(a)) for t, a in _parse_dictionary_value(value[0])]

    def parse(self, sentence: str) -> dict[tuple[int, int], tuple[str, str]]:
        """Map each token's character span to its `(upos, feats)`."""
        return self.parse_batch([sentence])[0]

    def parse_batch(self, sentences: list[str]) -> list[dict[tuple[int, int], tuple[str, str]]]:
        """Parse many sentences in one pipeline call.

        Stanza's per-call overhead dominates at this granularity — calling it
        once per sentence made a 200k-word pass take over ten minutes against
        sixteen seconds without it. Batching amortises the overhead; offsets
        are rebased per sentence so callers still index by their own spans.
        """
        if not sentences:
            return []
        import stanza

        docs = self.pipeline.bulk_process([stanza.Document([], text=s) for s in sentences])
        results: list[dict[tuple[int, int], tuple[str, str]]] = []
        for doc in docs:
            spans: dict[tuple[int, int], tuple[str, str]] = {}
            tokens: list[tuple[tuple[int, int], str, str, str]] = []
            for token in doc.iter_tokens():
                word = token.to_dict()[0]
                span = (token.start_char, token.end_char)
                upos = word.get("upos", "")
                feats = word.get("feats", "") or ""
                spans[span] = (upos, feats)
                tokens.append((span, upos, feats, token.text))
            for span in apply_counted_form(spans, tokens):
                # A third element rather than a parallel structure: callers
                # index `parse[0]` and `parse[1]`, so a longer tuple passes
                # through them untouched and the flag reaches whoever wants it.
                spans[span] = (*spans[span], True)
            results.append(spans)
        return results


_DICTIONARY_MORPH: Any = None


def dictionary_morph() -> Any:
    """pymorphy3 over VESUM, built once; None when the package is missing."""
    global _DICTIONARY_MORPH
    if _DICTIONARY_MORPH is None:
        try:
            import pymorphy3

            _DICTIONARY_MORPH = pymorphy3.MorphAnalyzer(lang="uk")
        except Exception:  # noqa: BLE001 - the repair is optional; the tier works without it
            _DICTIONARY_MORPH = False
    return _DICTIONARY_MORPH or None


def agreement_repair(readings: list[Reading], sentence: str,
                     spans: dict[tuple[int, int], tuple], start: int) -> tuple[int, ...] | None:
    """The accent the modifier before the target demands, when the tagger misread it.

    The tagger reads «Мої сестри.» as a genitive singular throughout —
    «мої» included, though only «моєї» can be one — and the tag match then
    serves сестри́. When the word before the target is a modifier whose
    dictionary cases and numbers do not include what the tagger gave it, the
    tagger has contradicted the dictionary on this phrase, and the reading of
    the target that agrees with what the modifier can be is taken instead —
    if exactly one reading does. Numerals are not modifiers here: 2/3/4
    govern the counted form (три сестри́).
    """
    from ukstress_ml.combiner import modifier_agreement

    morph = dictionary_morph()
    if morph is None:
        return None
    ordered = sorted(spans)
    position = next((k for k, (a, _) in enumerate(ordered) if a == start), None)
    if not position:
        return None
    a, b = ordered[position - 1]
    # the target must be a noun and the word before it tagged as its modifier:
    # measured on the top-200 set, firing on «вони», «була», «того» after a
    # noun that pymorphy can also read as a possessive adjective was wrong
    # seven times in nine
    if spans[(a, b)][0] not in ("DET", "ADJ"):
        return None
    if not readings or not all("upos=NOUN" in tags for tags, _ in readings):
        return None
    allowed = modifier_agreement(sentence[a:b].lower(), morph)
    if not allowed:
        return None
    tagged = _feats_to_dict(spans[(a, b)][1])
    if "Case" not in tagged or (tagged.get("Case"), tagged.get("Number")) in allowed:
        return None   # the tagger agrees with the dictionary: nothing to repair
    matching = set()
    for tags, accents in readings:
        own = _feats_to_dict("|".join(tags))
        if (own.get("Case"), own.get("Number")) in allowed:
            matching.add(tuple(accents))
    return next(iter(matching)) if len(matching) == 1 else None
