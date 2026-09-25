"""Gradio demo of the full two-tier stress pipeline.

Tier 1 — PostgreSQL `stress_lookup`: forms with a single stress signature are
answered outright. Tier 2 — the contextual cross-encoder: forms the database
reports as ambiguous are scored against their candidate glosses, and the answer
is kept only when the decision margin clears the abstention threshold.

Below the threshold the pipeline deliberately keeps the dictionary's default
rather than the model's pick: a confident wrong stress is worse than none.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import gradio as gr
import psycopg
from ukstress.normalizer import lookup_key

from ukstress_ml.morphology import MorphologyTier, SpacyMorphologyTier, apply_accents
from ukstress_ml.morphology import resolve as morph_resolve
from ukstress_ml.serving import ContextualStressModel, Target

WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
VOWELS = frozenset("аеєиіїоуюя")


def is_monosyllabic(word: str) -> bool:
    """Ukrainian does not mark stress on single-syllable words.

    The lexicon still records one (`у` -> `у́`), because a one-syllable entry is
    trivially "stressed on its only vowel", but applying it produces `У́ впра́ві`
    where the correct output is `У впра́ві`. Filtering here rather than in the
    data keeps the lexicon faithful to its source while the rendering stays
    orthographically correct."""
    return sum(1 for c in word.lower() if c in VOWELS) <= 1


class Pipeline:
    def __init__(self, dsn: str, model_dir: Path, manifest: Path, device: str,
                 extra_datasets: tuple[int, ...] = (),
                 spacy_model: str | None | bool = None) -> None:
        self.dsn = dsn
        self.extra_datasets = extra_datasets
        self.model = ContextualStressModel(model_dir, manifest, backend="torch", device=device)
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        with psycopg.connect(dsn) as conn:
            row = conn.execute("SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()
            self.dataset_id = int(row[0]) if row else 0
        # spaCy by default: same accuracy as Stanza on this tier from 15 MB
        # instead of ~500 MB and 0.8s instead of ~30s to load, which matters a
        # great deal for a demo someone starts by hand.
        self._morph = SpacyMorphologyTier() if spacy_model is not False else MorphologyTier()
        self._readings: dict[str, list] = {}
        self._parses: dict[int, dict] = {}

    def lookup(self, keys: list[str]) -> dict[str, list[tuple[str, str]]]:
        if not keys:
            return {}
        with psycopg.connect(self.dsn) as conn:
            rows = conn.execute(
                """
                SELECT form_normalized, stress_signature, min(stressed_form)
                FROM stress_lookup
                WHERE dataset_id = ANY(%s) AND form_normalized = ANY(%s)
                GROUP BY form_normalized, stress_signature
                """,
                ([self.dataset_id, *self.extra_datasets], keys),
            ).fetchall()
        out: dict[str, list[tuple[str, str]]] = {}
        for form, signature, stressed in rows:
            out.setdefault(str(form), []).append((str(signature), str(stressed)))
        return out

    def run(self, text: str) -> tuple[str, str, str]:
        self._parses.clear()
        spans = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]
        keys = {surface: lookup_key(surface) for surface, _, _ in spans}
        found = self.lookup(sorted(set(keys.values())))

        targets: list[Target] = []
        target_index: list[int] = []
        resolved: list[str | None] = [None] * len(spans)
        tiers: list[str] = []

        for index, (surface, start, end) in enumerate(spans):
            if is_monosyllabic(surface):
                tiers.append("monosyllabic")
                continue
            variants = found.get(keys[surface], [])
            if not variants:
                tiers.append("not in lexicon")
                continue
            if len({s for s, _ in variants}) == 1:
                resolved[index] = self._match_case(surface, variants[0][1])
                tiers.append("dictionary")
                continue
            entry = self.manifest["forms"].get(keys[surface])
            db_signatures = sorted({s for s, _ in variants})
            if entry is None or sorted(entry["signatures"]) != db_signatures:
                # Tier 2: when the readings are separated by their tags, the
                # parse decides and no model is needed. Resolves 2% of running
                # text that would otherwise go unstressed.
                got = self._morphology(text, surface, start, end)
                if got is not None:
                    resolved[index] = got
                    tiers.append("morphology")
                    continue
                # The model cannot be asked about a form outside its manifest,
                # but leaving the word bare is the worst of the three options:
                # it is neither the dictionary's answer nor the model's. Serve
                # the lexicon's best-ranked reading instead — stress_lookup
                # already orders by confidence and source rank.
                resolved[index] = self._match_case(surface, dict(variants)[db_signatures[0]])
                tiers.append("ambiguous — outside model coverage (dictionary default)")
                continue
            targets.append(Target(text, start, end, keys[surface], tuple(db_signatures)))
            target_index.append(index)
            tiers.append("model")

        decisions: list[dict[str, Any]] = self.model.resolve(targets) if targets else []
        notes: list[str] = []
        for decision, index in zip(decisions, target_index, strict=True):
            surface, start, end = spans[index]
            variants = dict(found[keys[surface]])
            picked = variants.get(decision["signature"])
            if decision["status"] == "selected" and picked:
                resolved[index] = self._match_case(surface, picked)
                notes.append(
                    f"**{surface}** → {picked}  (margin {decision['margin']:.1f} ≥ "
                    f"{self.model.threshold}, model)"
                )
            else:
                # Below threshold the design falls back to the dictionary's
                # default, not to nothing: leaving the word bare is a worse
                # answer than the more frequent reading. The default here is the
                # first signature, which stress_lookup orders by confidence and
                # source rank.
                options = " / ".join(sorted(variants.values()))
                # The observed dominant sense, not the lowest signature.
                # ANALYSIS_TRIAGE.md §4: a frequency-derived default beats an
                # arbitrary pick, which had been choosing "за́мки" (castles) for
                # a sentence about locks on doors.
                entry = self.manifest["forms"].get(keys[surface], {})
                default = variants.get(entry.get("default_signature", "")) or variants[min(variants)]
                resolved[index] = self._match_case(surface, default)
                notes.append(
                    f"**{surface}** → {default} *(dictionary default; model "
                    f"abstained, margin {decision['margin']:.1f} < "
                    f"{self.model.threshold}; {options})*"
                )

        pieces: list[str] = []
        cursor = 0
        for index, (surface, start, end) in enumerate(spans):
            pieces.append(text[cursor:start])
            pieces.append(resolved[index] or surface)
            cursor = end
        pieces.append(text[cursor:])

        summary = {
            "monosyllabic (no mark)": tiers.count("monosyllabic"),
            "dictionary": tiers.count("dictionary"),
            "morphology": tiers.count("morphology"),
            "model": tiers.count("model"),
            "outside model coverage": tiers.count("ambiguous — outside model coverage"),
            "not in lexicon": tiers.count("not in lexicon"),
        }
        table = "\n".join(f"| {k} | {v} |" for k, v in summary.items())
        stats = (
            f"| tier | tokens |\n| --- | --- |\n{table}\n\n"
            f"active dataset `{self.dataset_id}` · manifest forms "
            f"{len(self.manifest['forms']):,} · threshold {self.model.threshold}"
        )
        return "".join(pieces), stats, "\n\n".join(notes) or "_no ambiguous tokens_"

    def _morphology(self, text: str, surface: str, start: int, end: int) -> str | None:
        if self._morph is None:
            return None
        key = lookup_key(surface)
        if key not in self._readings:
            self._readings[key] = self._morph.readings(key)
        readings = self._readings[key]
        if not readings:
            return None
        if id(text) not in self._parses:
            try:
                self._parses[id(text)] = self._morph.parse(text)
            except Exception:  # noqa: BLE001 - never fail a request on a parse
                self._parses[id(text)] = {}
        parse = self._parses[id(text)].get((start, end))
        if not parse:
            return None
        got = morph_resolve(readings, parse[0], parse[1])
        if got is None:
            return None
        return self._match_case(surface, apply_accents(surface, got.accents))

    @staticmethod
    def _match_case(surface: str, stressed: str) -> str:
        """Return `stressed` wearing the caller's own casing and spelling.

        Two things are restored, not just case.

        A stored form may carry several acutes on one token — `пе́ре́д` — which
        is the wordlist's "either stress is acceptable" notation, not two
        stresses to pronounce. Only the first is kept.

        And the lexicon stores apostrophes as U+02BC while the caller may well
        have typed U+0027. Returning `здоро́вʼя` for `здоров'я` rewrites the
        user's text, which is not this function's business.
        """
        acute = "\u0301"
        # NFC first. The lexicon stores NFD, where `й` is `и` plus a breve and
        # `ї` is `і` plus a diaeresis; the caller's text is composed. Walking
        # the two in step without this desynchronises them and duplicates
        # characters — `старови́нний̆`, `Ма́йстерр`.
        stressed = unicodedata.normalize("NFC", stressed)
        # Collapse free variation on a single token.
        if "-" not in surface and stressed.count(acute) > 1:
            kept, seen = [], False
            for character in stressed:
                if character == acute:
                    if seen:
                        continue
                    seen = True
                kept.append(character)
            stressed = "".join(kept)
        # Put back the caller's own characters, keeping only the acute.
        source = iter(surface)
        out = []
        for character in stressed:
            if character == acute:
                out.append(character)
                continue
            original = next(source, character)
            out.append(original.upper() if original.isupper() else original)
        return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-url",
        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress",
    )
    parser.add_argument("--model", type=Path, default=Path("output/ml/models/v5-balanced"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("output/ml/serving_manifest_expanded.json")
    )
    parser.add_argument("--device", default="cpu",
                        help="cpu is the default: the demo must never contend "
                             "with training for the 8 GB card")
    parser.add_argument("--stanza", action="store_true",
                        help="use Stanza for morphology instead of spaCy")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--extra-dataset", type=int, action="append", default=[])
    args = parser.parse_args()

    pipeline = Pipeline(args.database_url, args.model, args.manifest, args.device,
                        tuple(args.extra_dataset), spacy_model=False if args.stanza else None)

    with gr.Blocks(title="Ukrainian stress — two-tier pipeline") as demo:
        gr.Markdown(
            "## Ukrainian word stress\n"
            "**Tier 1** PostgreSQL lookup for forms with one stress · "
            "**Tier 2** contextual cross-encoder for homographs, which abstains "
            "below the decision margin rather than guessing."
        )
        with gr.Row():
            box = gr.Textbox(
                label="Text",
                lines=4,
                value="Він відчинив замок ключем і зайшов у старий замок над річкою.",
            )
        run = gr.Button("Stress", variant="primary")
        out = gr.Textbox(label="Stressed", lines=4)
        with gr.Row():
            stats = gr.Markdown()
            notes = gr.Markdown()
        run.click(pipeline.run, inputs=box, outputs=[out, stats, notes])
        box.submit(pipeline.run, inputs=box, outputs=[out, stats, notes])
        gr.Examples(
            [
                "Він відчинив замок ключем і зайшов у старий замок над річкою.",
                "Замками фортеці милувалися туристи, а двері зачинили замками.",
                "Старший син і старший за званням офіцер прибули разом.",
                "Об'єднання зусиль дало нове об'єднання підприємств.",
            ],
            inputs=box,
        )

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
