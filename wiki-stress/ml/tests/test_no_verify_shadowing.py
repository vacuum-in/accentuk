"""Guard against scripts re-declaring the blind-verify deployment.

Both `run_label.py` and `run_generate.py` shipped a local

    VERIFY_DEPLOYMENT = annotate.Deployment(
        name="DeepSeek-V3.2",
        endpoint_var="AZURE_OPENAI_ENDPOINT",   # the *labelling* variables
        key_var="AZURE_OPENAI_API_KEY",
    )

which borrows the labelling resource's credentials. The failure is quiet in the
worst way: credentials resolve, the run proceeds, and blind verification is
performed by the resource that produced the labels — so the strongest filter in
the corpus build (it rejected 12.9% of labels) silently becomes a model
agreeing with itself.

`annotate.VERIFY_DEPLOYMENT` carries its own AZURE_OPENAI_*_VERIFY credentials
and is the only definition that should exist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))
LABEL_VARS = {"AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"}


def _verify_assignments(tree: ast.AST) -> list[ast.Assign]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VERIFY_DEPLOYMENT" for t in node.targets
        ):
            found.append(node)
    return found


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_do_not_construct_their_own_verify_deployment(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in _verify_assignments(tree):
        assert not isinstance(node.value, ast.Call), (
            f"{script.name} constructs its own VERIFY_DEPLOYMENT; use "
            "annotate.VERIFY_DEPLOYMENT so verification keeps its own credentials"
        )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_do_not_point_verification_at_labelling_credentials(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    if "VERIFY_DEPLOYMENT" not in source:
        return
    tree = ast.parse(source)
    for node in _verify_assignments(tree):
        used = {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        assert not (used & LABEL_VARS), (
            f"{script.name} resolves VERIFY_DEPLOYMENT from {used & LABEL_VARS}, "
            "which are the labelling credentials"
        )


def test_the_module_level_verify_deployment_is_distinct_from_the_labeller() -> None:
    from ukstress_ml.annotate import LABEL_DEPLOYMENT, VERIFY_DEPLOYMENT

    assert VERIFY_DEPLOYMENT.endpoint_var != LABEL_DEPLOYMENT.endpoint_var
    assert VERIFY_DEPLOYMENT.key_var != LABEL_DEPLOYMENT.key_var
    # And it must not inherit the labeller's fallbacks either, or a
    # single-deployment environment collapses them onto one model.
    assert not set(VERIFY_DEPLOYMENT.fallback_endpoint_vars) & set(
        LABEL_DEPLOYMENT.fallback_endpoint_vars
    )
