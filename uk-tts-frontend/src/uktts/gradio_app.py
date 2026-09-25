"""A small Gradio page for exercising the running `/v1/prepare` service.

It is a client, not a second entry point: it calls the HTTP API over the
network exactly as any other consumer would, so what you see here is what the
service actually returns.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import gradio as gr
import httpx

DEFAULT_API = "http://127.0.0.1:8000"

EXAMPLES = [
    "Замок на горі. Замок у дверях.",
    "О 7:45 ранку 3 квітня 2025 р. потяг № 12 прибув до Львова.",
    "Компанія Apple випустила iPhone 15 Pro за $999.",
    "Температура від -5 до +12 °C, вологість 47 %.",
    "У 1991 році Україна проголосила незалежність.",
]

# Everything the stress API can say about a token, in the order a reader wants
# to scan them: decided first, undecided next, unreachable last.
STATUS_HELP = {
    "stressed": "a tier decided it",
    "morphology": "the grammatical tier decided it",
    "compound": "stressed from its halves",
    "suffix": "guessed by analogy with word endings",
    "dictionary_default": "several readings, none chosen on evidence",
    "ambiguous": "several readings, served unstressed on request",
    "already_stressed": "the input carried an accent",
    "not_required": "no vowel to mark",
    "not_found": "no reading in the lexicon",
    "model_ineligible": "outside the model's coverage",
    "invalid_candidate": "the stored signature would not apply",
}


MODES = {
    "Verbalize + stress": "both",
    "Verbalize only": "verbalize",
    "Stress only": "stress",
}


def prepare(
    api: str, text: str, mode_label: str, on_ambiguity: str, timeout: float,
    combiner: bool = False,
) -> tuple[str, str, Any, str]:
    """Prepare each line of the box as its own input.

    A pasted block is usually a list of separate sentences to try, not one
    passage, and sending them together makes the per-word table one long run
    with no way to tell which line a row came from. Blank lines are kept in
    place rather than sent, so the output has the shape of the input.
    """

    if not text.strip():
        return "", "", [], "Enter some text."
    lines = text.splitlines() or [text]
    numbered = [(index, line) for index, line in enumerate(lines) if line.strip()]

    mode = MODES.get(mode_label, "both")
    body: dict[str, Any] = {
        "texts": [line for _, line in numbered],
        "mode": mode,
        "include_tokens": True,
    }
    if on_ambiguity and mode != "verbalize":
        body["on_ambiguity"] = on_ambiguity
    if mode != "verbalize":
        body["combiner"] = bool(combiner)
    try:
        response = httpx.post(f"{api.rstrip('/')}/v1/prepare", json=body, timeout=timeout)
    except Exception as error:  # noqa: BLE001
        # The endpoint is deployment detail. It goes to the operator's log, not
        # onto a page that may be publicly shared.
        print(f"prepare failed against {api}: {error}", file=sys.stderr)
        return "", "", [], "The service is unreachable. Try again shortly."
    if response.status_code != 200:
        print(
            f"prepare returned HTTP {response.status_code}: {response.text[:400]}",
            file=sys.stderr,
        )
        if response.status_code == 503:
            return "", "", [], "The service is starting up or a stage is down."
        return "", "", [], f"The request failed (HTTP {response.status_code})."

    results = response.json()["results"]
    stressed = list(lines)
    verbalized = list(lines)
    rows: list[list[str]] = []
    chunks = 0
    timings: dict[str, float] = {}
    warnings: list[str] = []
    multiline = len(numbered) > 1

    for (index, _), result in zip(numbered, results, strict=True):
        stressed[index] = result["text"]
        verbalized[index] = result["verbalized"]
        chunks += result.get("chunks", 0)
        for stage, value in result.get("timings_ms", {}).items():
            timings[stage] = timings.get(stage, 0.0) + value
        warnings.extend(result.get("warnings", []))
        for token in result.get("tokens", []):
            row = [
                token["text"],
                token["output_text"],
                token["status"],
                STATUS_HELP.get(token["status"], ""),
            ]
            rows.append([str(index + 1), *row] if multiline else row)

    notes = [
        f"{mode} · {len(numbered)} line{'s' if multiline else ''}, {chunks} chunks",
        "  ".join(f"{stage} {value:.0f} ms" for stage, value in timings.items()),
    ]
    notes.extend(f"warning: {warning}" for warning in warnings)
    return (
        "\n".join(stressed),
        "\n".join(verbalized),
        rows,
        "\n".join(part for part in notes if part),
    )


def health(api: str, timeout: float) -> str:
    """Report readiness as a sentence, not as the raw document.

    The health payload carries the stress endpoint and the checkpoint path;
    neither belongs on a page anyone can open.
    """

    try:
        response = httpx.get(f"{api.rstrip('/')}/health/ready", timeout=timeout)
        payload = response.json()
    except Exception as error:  # noqa: BLE001
        print(f"health check failed against {api}: {error}", file=sys.stderr)
        return "Unreachable."
    if response.status_code != 200:
        return "Not ready."
    stress = payload.get("stress", {})
    return (
        f"Ready. Verbalizer: {payload.get('verbalizer', 'unknown')}. "
        f"Stress: {'ready' if stress.get('ready') else 'not ready'}."
    )


def build(api: str, timeout: float) -> gr.Blocks:
    with gr.Blocks(title="Ukrainian TTS text frontend") as demo:
        gr.Markdown(
            "# Ukrainian TTS text frontend\n"
            "Raw text in, TTS-ready text out: the verbalizer spells out numbers, "
            "dates and Latin words, then the stress API marks the stressed vowel "
            "of each one."
        )
        with gr.Row():
            mode = gr.Radio(
                list(MODES),
                # stress alone by default: the verbalizer is a separate model
                # with its own error rate, and it is not what is under test
                value="Stress only",
                label="Pipeline",
                info="run both stages, or one of them alone",
            )
            ambiguity = gr.Radio(
                ["default", "preserve"],
                value="default",
                label="Undecided word",
                info="serve the lexicon's best reading, or leave it unstressed",
            )
            combiner = gr.Checkbox(
                value=False,
                label="Learned combiner",
                info="weigh every tier per word: better on spoken, everyday "
                     "sentences, worse on literary prose",
            )
        source = gr.Textbox(
            label="Text",
            lines=5,
            placeholder="Введіть текст…\nКожен рядок обробляється окремо.",
            info="one line = one input",
        )
        run = gr.Button("Prepare", variant="primary")

        stressed = gr.Textbox(label="Result", lines=5)
        verbalized = gr.Textbox(
            label="After verbalization, before stress",
            lines=5,
            info="equals the input when verbalization is off",
        )
        notes = gr.Textbox(label="Notes", lines=2)
        tokens = gr.Dataframe(
            headers=["line", "word", "output", "status", "meaning"],
            label="Per-word decisions",
            wrap=True,
        )

        with gr.Accordion("Service status", open=False):
            check = gr.Button("Check")
            status = gr.Textbox(label="Status", lines=1)
            check.click(lambda: health(api, timeout), None, status)

        gr.Examples(EXAMPLES, source)

        inputs = [source, mode, ambiguity, combiner]
        outputs = [stressed, verbalized, tokens, notes]

        def handler(text: str, mode_label: str, policy: str,
                    use_combiner: bool) -> tuple[str, str, Any, str]:
            return prepare(api, text, mode_label, policy, timeout, use_combiner)

        run.click(handler, inputs, outputs)
        source.submit(handler, inputs, outputs)
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gradio page for /v1/prepare.")
    parser.add_argument("--api", default=DEFAULT_API, help="base URL of the prepare service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--share",
        action="store_true",
        help="publish a public *.gradio.live tunnel; needs a writable filesystem "
             "and outbound network, so it does not work in the read-only container",
    )
    parser.add_argument(
        "--auth",
        metavar="USER:PASSWORD",
        help="require a login; strongly advised with --share, which is otherwise "
             "an open door to the pipeline for anyone holding the URL",
    )
    args = parser.parse_args(argv)

    credentials = None
    if args.auth:
        user, separator, password = args.auth.partition(":")
        if not separator or not user or not password:
            parser.error("--auth expects USER:PASSWORD")
        credentials = (user, password)

    build(args.api, args.timeout).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=credentials,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
