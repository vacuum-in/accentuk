"""Check that everything the live API serves agrees with itself.

The contextual tier is fed from four places that are built and deployed
separately: the serving manifest the model loads, the copy the API loads, the
candidate rows in PostgreSQL, and the lexicon the API reads candidates from.
On 2026-09-23 three disagreements between them were live at once, none of
which raised an error anywhere:

* the database held 570 forms under an old inventory hash while the API asked
  for another, so 14,686 forms never reached the model;
* 331 manifest forms carried each other's signatures (`Баска́ми` labelled 0),
  so the model's pick was rendered as the other reading;
* 1,701 forms (24 once the supplementary datasets are counted) had a lexicon
  candidate set wider than the manifest's, so the model refused them and the
  word went out with no stress at all (`Валові`).

This runs the API's own candidate computation over every manifest form and
compares. `--fix` writes a repaired manifest (signatures re-derived from the
spellings, unservable forms dropped, the inventory hash kept), backs up the
original, and reloads the database rows for that hash.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg

from ukstress.normalizer import NormalizationError, lookup_key, stress_signature
from ukstress.serving_inventory import import_serving_inventory


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    swapped: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    unservable: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _drop_subsumed_free_variation(signatures: list[str]) -> list[str]:
    """The API's rule: `0|1` is dropped when `0` or `1` is also present."""
    specific = {s for s in signatures if "|" not in s}
    if not specific:
        return signatures
    return [s for s in signatures
            if "|" not in s or not any(m in specific for m in s.split("|"))]


def api_candidates(connection: psycopg.Connection, forms: list[str], datasets: list[int],
                   inventory_hash: str, exclusive: list[int]) -> dict[str, list[str]]:
    """The candidate set the API sends to the model, per form.

    Mirrors `repository.batchSignatures`: lexicon rows from the active and
    supplementary datasets plus the contextual rows for the inventory hash,
    grouped by signature; an exclusive reviewed reading collapses the form to
    one candidate; free variation subsumed by a specific reading is dropped.
    Order is irrelevant here — the model compares sets.
    """
    rows = connection.execute(
        """
        WITH input AS (SELECT word FROM unnest(%s::text[]) AS value(word)),
        candidates AS (
            SELECT input.word, lookup.stress_signature,
                   lookup.dataset_id = ANY(%s) AS exclusive
            FROM input JOIN stress_lookup lookup
              ON lookup.dataset_id = ANY(%s) AND lookup.form_normalized = input.word
            UNION ALL
            SELECT input.word, contextual.stress_signature, false
            FROM input JOIN contextual_stress_candidate contextual
              ON contextual.inventory_hash = %s AND contextual.form_normalized = input.word
        )
        SELECT word, stress_signature, bool_or(exclusive)
        FROM candidates GROUP BY word, stress_signature
        """,
        (forms, exclusive, datasets, inventory_hash),
    ).fetchall()
    signatures: dict[str, list[str]] = defaultdict(list)
    only: dict[str, str] = {}
    for form, signature, is_exclusive in rows:
        signatures[form].append(signature)
        if is_exclusive:
            only.setdefault(form, signature)
    return {form: [only[form]] if form in only else _drop_subsumed_free_variation(sigs)
            for form, sigs in signatures.items()}


def check(database_url: str, *, manifest_path: Path, api_manifest_path: Path | None,
          active_hash: str, supplementary: list[int], exclusive: list[int],
          token_model_dir: Path | None) -> tuple[Report, dict[str, Any]]:
    report = Report()
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    forms: dict[str, Any] = manifest.get("forms") or {}
    manifest_hash = str(manifest.get("inventory_hash"))
    report.facts.update(manifest=str(manifest_path), manifest_forms=len(forms),
                        manifest_hash=manifest_hash, model_version=manifest.get("model_version"))

    # 1. One inventory hash everywhere.
    if manifest_hash != active_hash:
        report.errors.append(f"ACTIVE_INVENTORY_HASH {active_hash[:12]} != model manifest {manifest_hash[:12]}")
    if api_manifest_path is not None:
        if api_manifest_path.resolve() != manifest_path.resolve():
            api_hash = str(json.loads(api_manifest_path.read_text(encoding="utf-8")).get("inventory_hash"))
            if api_hash != manifest_hash:
                report.errors.append(f"API manifest {api_manifest_path} has hash {api_hash[:12]}, "
                                     f"model manifest {manifest_hash[:12]}")
            else:
                report.warnings.append("API and model read different manifest files with the same hash; "
                                       "their form lists can still drift")

    # 2. Every signature is the one its spelling carries.
    for form, entry in forms.items():
        bad = []
        for candidate in entry.get("candidates", []):
            spelling = str(candidate.get("stressed", ""))
            try:
                real = stress_signature(spelling)
            except NormalizationError:
                bad.append({"stressed": spelling, "labelled": candidate.get("signature"), "real": "invalid"})
                continue
            if lookup_key(spelling) != lookup_key(form):
                bad.append({"stressed": spelling, "labelled": candidate.get("signature"), "real": "other form"})
            elif real != str(candidate.get("signature")):
                bad.append({"stressed": spelling, "labelled": candidate.get("signature"), "real": real})
        if bad:
            report.swapped[form] = bad
    if report.swapped:
        report.errors.append(f"{len(report.swapped):,} manifest forms carry a signature their "
                             f"spelling does not (e.g. {next(iter(report.swapped))})")

    with psycopg.connect(database_url) as connection:
        active = connection.execute("SELECT dataset_id FROM active_dataset").fetchone()
        if active is None:
            report.errors.append("no active dataset")
            return report, manifest
        datasets = [int(active[0]), *supplementary]
        report.facts["datasets"] = datasets

        # 3. The database holds this inventory, and only it is what the API asks for.
        by_hash = dict(connection.execute(
            "SELECT inventory_hash, count(DISTINCT form_normalized) FROM contextual_stress_candidate "
            "GROUP BY 1").fetchall())
        served = by_hash.get(active_hash, 0)
        report.facts["database_forms_for_active_hash"] = served
        stale = {h[:12]: n for h, n in by_hash.items() if h != active_hash}
        if stale:
            report.warnings.append(f"database rows under other hashes (never served): {stale}")
        if served == 0:
            report.errors.append("the database has no candidate rows for the active inventory hash: "
                                 "the contextual tier reaches no form")
        elif served < 0.95 * len(forms):
            report.errors.append(f"the database has {served:,} forms for the active hash, "
                                 f"the manifest {len(forms):,}")

        # 4. What the API would send is what the model accepts.
        names = list(forms)
        sent: dict[str, list[str]] = {}
        for start in range(0, len(names), 5000):
            sent.update(api_candidates(connection, names[start:start + 5000], datasets,
                                       active_hash, exclusive))
    reached = 0
    for form, entry in forms.items():
        api = sorted(set(sent.get(form, [])))
        if len(api) < 2:
            continue   # one candidate (or an exclusive decision): never sent to the model
        reached += 1
        model = sorted(set(str(s) for s in entry.get("signatures", [])))
        if api != model:
            report.unservable[form] = {"api": api, "manifest": model}
    report.facts["forms_reaching_model"] = reached
    if report.unservable:
        example = next(iter(report.unservable))
        report.errors.append(
            f"{len(report.unservable):,} forms reach the model with a candidate set it refuses, "
            f"and go out unstressed (e.g. {example}: API {report.unservable[example]['api']}, "
            f"manifest {report.unservable[example]['manifest']})")

    # 5. The token classifier's coverage is where the API expects it.
    if token_model_dir is not None:
        coverage = token_model_dir / "coverage.json"
        if not coverage.exists():
            report.errors.append(f"TOKEN_MODEL_DIR {token_model_dir} has no coverage.json")
        else:
            data = json.loads(coverage.read_text(encoding="utf-8"))
            gate = int(data.get("min_seen", 10))
            eligible = sum(1 for n in data.get("forms", {}).values() if n >= gate)
            report.facts["token_model"] = {"version": data.get("model_version"), "min_seen": gate,
                                           "eligible": eligible}
    return report, manifest


def repair(manifest: dict[str, Any], report: Report) -> tuple[dict[str, Any], dict[str, int]]:
    """Re-derive signatures from spellings; drop forms the model cannot serve."""
    counts = {"signatures_rederived": 0, "forms_dropped": 0}
    repaired = dict(manifest)
    forms: dict[str, Any] = {}
    for form, entry in manifest["forms"].items():
        if form in report.unservable:
            counts["forms_dropped"] += 1
            continue
        candidates = []
        for candidate in entry.get("candidates", []):
            spelling = str(candidate.get("stressed", ""))
            try:
                real = stress_signature(spelling)
            except NormalizationError:
                continue
            if lookup_key(spelling) != lookup_key(form):
                continue
            if real != str(candidate.get("signature")):
                candidate = {**candidate, "signature": real}
                counts["signatures_rederived"] += 1
            candidates.append(candidate)
        signatures = sorted({c["signature"] for c in candidates})
        if len(signatures) < 2:
            counts["forms_dropped"] += 1
            continue
        forms[form] = {**entry, "candidates": candidates, "signatures": signatures,
                       "sense_ids": [c["sense_id"] for c in candidates]}
    repaired["forms"] = forms
    return repaired, counts


def check_and_fix(database_url: str, *, manifest_path: Path, fix: bool, **kwargs: Any) -> dict[str, Any]:
    report, manifest = check(database_url, manifest_path=manifest_path, **kwargs)
    result: dict[str, Any] = {"ok": report.ok, "errors": report.errors, "warnings": report.warnings,
                              **report.facts}
    if not fix or report.ok:
        return result
    repaired, counts = repair(manifest, report)
    backup = manifest_path.with_suffix(".json.before-check")
    if not backup.exists():
        shutil.copyfile(manifest_path, backup)
    manifest_path.write_text(json.dumps(repaired, ensure_ascii=False), encoding="utf-8")
    imported = import_serving_inventory(database_url, manifest_path)
    after, _ = check(database_url, manifest_path=manifest_path, **kwargs)
    result.update(fixed=counts, backup=str(backup), reimported=imported,
                  ok_after_fix=after.ok, errors_after_fix=after.errors)
    return result
