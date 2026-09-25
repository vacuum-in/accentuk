"""Derive a gloss-conditioned corpus from an existing one.

Same rows, same splits, same labels — only the source encoding changes, so the
comparison against the ungloss(ed) model is clean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ukstress_ml import ambiguity, corpus

SOURCE_VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
TARGET_VERSION = sys.argv[2] if len(sys.argv) > 2 else "v2g"


def main() -> None:
    src = Path(f"output/ml/corpus/{SOURCE_VERSION}")
    dst = Path(f"output/ml/corpus/{TARGET_VERSION}")
    dst.mkdir(parents=True, exist_ok=True)

    forms = {f.form: f for f in ambiguity.load(Path("output/ml/ambiguous_forms.jsonl"))}
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))

    counts: dict[str, int] = {}
    missing_gloss = 0
    for split in ("train", "dev", "test", "test_unseen"):
        path = src / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = corpus.load_split(path)
        written = 0
        with (dst / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                form = forms.get(row["form"])
                if form is None:
                    continue
                candidates = [
                    {
                        "sense_id": c.sense_id,
                        "signature": c.signature,
                        "stressed": c.stressed,
                        "definition": c.definition,
                    }
                    for c in form.candidates
                ]
                if any(not c["definition"] for c in candidates):
                    missing_gloss += 1
                    continue
                row = dict(row)
                row["source_marked_only"] = row["source"]
                row["source"] = corpus.encode_source_with_glosses(
                    row["sentence"],
                    row["start"],
                    row["end"],
                    candidates,
                    seed_key=row["sentence_id"],
                )
                row["candidates"] = candidates
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        counts[split] = written
        print(f"{split:<12} {written:,}", flush=True)

    manifest["corpus_version"] = TARGET_VERSION
    manifest["derived_from"] = SOURCE_VERSION
    manifest["encoding"] = "marked span + per-candidate stressed form and gloss, order shuffled"
    manifest["rows_dropped_missing_gloss"] = missing_gloss
    manifest["split_counts"] = counts
    (dst / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"dropped for missing gloss: {missing_gloss}", flush=True)

    example = corpus.load_split(dst / "train.jsonl")[0]
    print("\nexample source:\n ", example["source"][:320], flush=True)
    print("target:", example["target"], flush=True)


if __name__ == "__main__":
    main()
