"""Full mining pass over the Ukrainian Wikipedia dump."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ukstress_ml import ambiguity, mine

DUMP = Path("data/ukwiki-latest-pages-articles.xml.bz2")
OUT = Path("output/ml/mined/ukwiki")
CAP = 80
SEED = 20260815


def main() -> None:
    forms = ambiguity.load(Path("output/ml/ambiguous_forms.jsonl"))
    manifest = json.loads(Path(str(DUMP) + ".manifest.json").read_text(encoding="utf-8"))
    print(f"forms: {len(forms)}  dump: {manifest['byte_size'] / 1e9:.2f} GB", flush=True)

    started = time.time()
    reservoirs, stats = mine.mine(DUMP, forms, cap_per_form=CAP, seed=SEED, progress_every=100_000)
    elapsed = time.time() - started

    coverage = mine.write(
        reservoirs,
        stats,
        forms,
        OUT,
        corpus="ukwiki",
        licence="CC BY-SA 4.0",
        dump_sha256=manifest["sha256"],
        config={"cap_per_form": CAP, "seed": SEED, "min_tokens": 6, "max_tokens": 40},
    )
    print(json.dumps(coverage["stats"], indent=2), flush=True)
    print(
        f"rows={coverage['rows_written']:,} "
        f"forms_with_candidates={coverage['forms_with_candidates']}/{coverage['forms_total']} "
        f"elapsed={elapsed / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
