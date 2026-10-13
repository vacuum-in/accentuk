"""Command-line interface for accentuk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assets import download_assets
from .engine import Accentor
from .errors import AccentukError
from .export import build_bundle


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="accentuk",
        description="Offline Ukrainian stress marking and optional TTS preparation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = subparsers.add_parser("download", help="Download and verify required stress models/lexicon")
    p_dl.add_argument("--light", action="store_true", help="Download light bundle (CTranslate2 INT8 + Lexicon, ~450 MB)")
    p_dl.add_argument("--full", action="store_true", help="Download full bundle (PyTorch weights + spaCy TRF, ~2.2 GB)")
    p_dl.add_argument("--tts", action="store_true", help="Also download optional TTS verbalizer model")
    p_dl.add_argument("--directory", type=Path, default=None, help="Target cache directory")

    # export
    p_exp = subparsers.add_parser("export", help="Export serving archive into MARISA bundle")
    p_exp.add_argument("--archive", type=Path, required=True, help="Path to serving archive or directory")
    p_exp.add_argument("--manifest", type=Path, required=True, help="Path to serving_manifest.json")
    p_exp.add_argument("--output", type=Path, required=True, help="Output bundle directory")
    p_exp.add_argument("--assets", type=Path, default=None, help="Path to lexicon-assets.tar.gz or dir")

    # stress
    p_stress = subparsers.add_parser("stress", help="Mark stress on Ukrainian text")
    p_stress.add_argument("text", help="Ukrainian input text")
    p_stress.add_argument("--mode", choices=["auto", "light", "full"], default="auto", help="Inference mode: light (CT2 INT8) or full (PyTorch/spaCy)")
    p_stress.add_argument("--ambiguity", choices=["default", "preserve"], default="default", help="Handling of ambiguous words")
    p_stress.add_argument("--combiner", action="store_true", help="Enable learned combiner")
    p_stress.add_argument("--assets", type=Path, default=None, help="Path to asset bundle directory")

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Prepare text for TTS (verbalize and stress mark)")
    p_prep.add_argument("text", help="Input text with numbers/dates/symbols")
    p_prep.add_argument("--mode", choices=["auto", "light", "full"], default="auto", help="Inference mode: light (CT2 INT8) or full (PyTorch/spaCy)")
    p_prep.add_argument("--ambiguity", choices=["default", "preserve"], default="default", help="Handling of ambiguous words")
    p_prep.add_argument("--combiner", action="store_true", help="Enable learned combiner")
    p_prep.add_argument("--assets", type=Path, default=None, help="Path to asset bundle directory")

    parsed = parser.parse_args(args)

    try:
        if parsed.command == "download":
            download_assets(
                target_dir=parsed.directory,
                include_tts=parsed.tts,
                light=parsed.light,
                full=parsed.full,
            )
            print("Download completed successfully.")
            return 0

        elif parsed.command == "export":
            build_bundle(
                archive=parsed.archive,
                model_manifest=parsed.manifest,
                output=parsed.output,
                assets_dir=parsed.assets,
            )
            print(f"Export completed successfully to {parsed.output}.")
            return 0

        elif parsed.command == "stress":
            with Accentor(asset_dir=parsed.assets, mode=parsed.mode, enable_combiner=parsed.combiner) as engine:
                res = engine.stress(parsed.text, on_ambiguity=parsed.ambiguity, combiner=parsed.combiner)
                print(res.text)
            return 0

        elif parsed.command == "prepare":
            try:
                from .frontend import Frontend

                with Frontend(asset_dir=parsed.assets, mode=parsed.mode) as frontend:
                    res = frontend.prepare(parsed.text, on_ambiguity=parsed.ambiguity, combiner=parsed.combiner)
                    print(res.text)
                return 0
            except ImportError as err:
                print(f"TTS extra not available: {err}. Install with 'accentuk[tts]'.", file=sys.stderr)
                return 1

    except AccentukError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected error: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
