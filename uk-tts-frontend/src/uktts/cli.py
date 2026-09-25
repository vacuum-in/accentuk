"""Command line entry point: `uktts prepare`, `uktts ready`, `uktts serve`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import Config
from .pipeline import Pipeline
from .stress import StressUnavailable


def _config(args: argparse.Namespace) -> Config:
    config = Config.from_env()
    verbalizer = config.verbalizer
    stress = config.stress
    if args.checkpoint:
        verbalizer = replace(verbalizer, checkpoint=Path(args.checkpoint))
    if args.device:
        verbalizer = replace(verbalizer, device=args.device)
    if args.batch_size:
        verbalizer = replace(verbalizer, batch_size=args.batch_size)
    if args.no_verbalize:
        verbalizer = replace(verbalizer, enabled=False)
    if args.stress_url:
        stress = replace(stress, base_url=args.stress_url.rstrip("/"))
    if args.on_ambiguity:
        stress = replace(stress, on_ambiguity=args.on_ambiguity)
    if args.no_stress:
        stress = replace(stress, enabled=False)
    return Config(verbalizer=verbalizer, stress=stress)


def _inputs(args: argparse.Namespace) -> list[str]:
    if args.text:
        return list(args.text)
    raw = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    if args.lines:
        return [line for line in raw.splitlines() if line.strip()]
    return [raw]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", help="verbalizer checkpoint directory")
    parser.add_argument("--device", help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stress-url", help="base URL of the Go stress API")
    parser.add_argument("--on-ambiguity", choices=["default", "preserve"])
    parser.add_argument("--no-verbalize", action="store_true", help="skip stage 1")
    parser.add_argument("--no-stress", action="store_true", help="skip stage 2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uktts", description="Prepare Ukrainian text for speech synthesis."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="verbalize and stress text")
    prepare.add_argument("text", nargs="*", help="text to prepare; omit to read stdin")
    prepare.add_argument("--file", help="read the input from this file instead of stdin")
    prepare.add_argument("--lines", action="store_true", help="treat each line as one input")
    prepare.add_argument("--json", action="store_true", help="emit one JSON object per input")
    prepare.add_argument("--mode", choices=["both", "verbalize", "stress"], default="both",
                         help="which stages to run")
    _common(prepare)

    ready = commands.add_parser("ready", help="report component readiness as JSON")
    _common(ready)

    serve = commands.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    _common(serve)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from .service import create_app

        uvicorn.run(create_app(_config(args)), host=args.host, port=args.port)
        return 0

    pipeline = Pipeline(_config(args))

    if args.command == "ready":
        status = pipeline.ready()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["ready"] else 1

    try:
        results = pipeline.prepare_many(_inputs(args), mode=args.mode)
    except StressUnavailable as error:
        print(f"error: {error}", file=sys.stderr)
        print("hint: start the stress stack, or pass --no-stress", file=sys.stderr)
        return 2

    for result in results:
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        else:
            print(result.text)
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
