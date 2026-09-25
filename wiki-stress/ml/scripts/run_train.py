"""Train the Marian span model on the assembled corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ukstress_ml import corpus, train

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"

CORPUS = Path(f"output/ml/corpus/{VERSION}")
OUT = Path(f"output/ml/models/{VERSION}")


def main() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    train_rows = corpus.load_split(CORPUS / "train.jsonl")
    dev_rows = corpus.load_split(CORPUS / "dev.jsonl")

    # Gloss-conditioned sources carry the sentence plus one gloss per candidate,
    # so they need roughly twice the encoder window.
    config = train.TrainConfig(max_source_length=256 if VERSION.endswith("g") else 128)
    run = train.train(
        train_rows,
        dev_rows,
        OUT,
        config=config,
        corpus_version=manifest["corpus_version"],
        inventory_hash=manifest["inventory_hash"],
    )
    print(
        f"best epoch {run.best_epoch}  dev group-macro {run.best_dev_group_macro:.4f}  "
        f"{run.seconds / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
