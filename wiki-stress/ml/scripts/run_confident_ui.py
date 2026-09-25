"""A page for checking what the confident policy marks, and what a model adds.

The point of the confident mode is that a marked word needs no review, so the
thing to look at is the split: which words it kept, which it left bare, and —
when the last tier is on — what a model proposed for the blanks and whether
that proposal is worth trusting. Measured on lang-uk it was not: Gemma 4 E4B
answered the blanks at 50.3% and gpt-5.6-luna at 57.5%, against 65.0% for the
dictionary default they would be replacing.

So the model's answers are shown separately rather than folded into the text
by default. Turning them on is a deliberate act.
"""

from __future__ import annotations

import argparse
import importlib.util as importlib_util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import gradio as gr

_spec = importlib_util.spec_from_file_location(
    "run_confident", Path(__file__).with_name("run_confident.py"))
_confident = importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_confident)


def build(api: str, model: str) -> gr.Blocks:
    with gr.Blocks(title="Stress — confident mode", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## Наголос: лише впевнене\n"
            "Позначається те, що виміряно як надійне — однозначні словникові "
            "влучання та правила, **99.0%** точності на 78% слів. Решта "
            "лишається без наголосу.\n\n"
            "Останній тир питає модель про порожнечі. На бенчмарку він програв "
            "словниковому дефолту (50–58% проти 65%), тож його відповіді "
            "показані окремо — вмикайте свідомо."
        )
        with gr.Row():
            text = gr.Textbox(label="Текст", lines=8,
                              placeholder="Кожен рядок обробляється окремо…")
        with gr.Row():
            use_llm = gr.Checkbox(label=f"Останній тир: {model}", value=False)
            effort = gr.Radio(["low", "medium", "high"], value="high",
                              label="Зусилля міркування")
            run = gr.Button("Розставити наголоси", variant="primary")
        out = gr.Textbox(label="Результат", lines=8)
        report = gr.Textbox(label="Що лишилось порожнім", lines=12)

        def handle(source: str, llm: bool, level: str):
            if not source.strip():
                return "", ""
            argv = ["--api", api, "--effort", level, "--model", model]
            if llm:
                argv.append("--llm")

            def run_once(extra: list[str]) -> str:
                buffer = io.StringIO()
                saved = sys.argv
                sys.argv = ["run_confident.py", source, *argv, *extra]
                try:
                    with redirect_stdout(buffer):
                        _confident.main()
                finally:
                    sys.argv = saved
                return buffer.getvalue()

            # Two passes: the text as it should be used, and the accounting.
            # The report re-queries rather than sharing state because the LLM
            # pass rewrites offsets, and a stale report is worse than a slow one.
            return run_once([]).rstrip("\n"), run_once(["--report"]).rstrip("\n")

        run.click(handle, [text, use_llm, effort], [out, report])
        gr.Examples(
            examples=[
                ["Колись це був замок на горі, а тепер замок на дверях."],
                ["Кам'яні сходи вели до вежі. Сходи обережно, каміння слизьке."],
                ["Він має дві руки. У неї три вікна. Заходи до мене в гості."],
                ["Дитина почала засипати. Робітники продовжували засипати яму."],
            ],
            inputs=[text],
        )
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8080/v1/stress")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    build(args.api, args.model).launch(server_name=args.host, server_port=args.port,
                                       share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
