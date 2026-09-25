import pytest

from ukstress_ml.annotate import (
    LABEL_DEPLOYMENT,
    VERIFY_DEPLOYMENT,
    Deployment,
    run_job,
)


def test_label_deployment_accepts_the_foundry_portal_variable_names() -> None:
    """The live .env defines AZURE_FOUNDRY_OPENAI_*, not AZURE_OPENAI_*."""
    config = {
        "AZURE_FOUNDRY_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
        "AZURE_FOUNDRY_OPENAI_API_KEY": "secret",
    }
    client = LABEL_DEPLOYMENT.client(config)
    assert str(client.base_url).startswith("https://example.invalid")


def test_original_variable_names_still_win_when_both_are_present() -> None:
    config = {
        "AZURE_OPENAI_ENDPOINT": "https://primary.invalid/openai/v1",
        "AZURE_OPENAI_API_KEY": "primary",
        "AZURE_FOUNDRY_OPENAI_BASE_URL": "https://fallback.invalid/openai/v1",
        "AZURE_FOUNDRY_OPENAI_API_KEY": "fallback",
    }
    assert str(LABEL_DEPLOYMENT.client(config).base_url).startswith("https://primary.invalid")


def test_missing_credentials_name_every_accepted_variable() -> None:
    with pytest.raises(RuntimeError) as error:
        LABEL_DEPLOYMENT.client({})
    message = str(error.value)
    assert "AZURE_OPENAI_ENDPOINT" in message
    assert "AZURE_FOUNDRY_OPENAI_BASE_URL" in message


def test_model_name_comes_from_the_environment_when_set() -> None:
    deployment = Deployment(
        name="fallback-literal",
        endpoint_var="E",
        key_var="K",
        model_var="MODEL",
    )
    assert deployment.resolved_model({}) == "fallback-literal"
    assert deployment.resolved_model({"MODEL": "real-deployment"}) == "real-deployment"


def test_verify_deployment_does_not_inherit_the_label_fallback() -> None:
    """A single configured deployment must not silently serve both jobs.

    If VERIFY fell back to the same AZURE_FOUNDRY_* variables as LABEL, blind
    verification would become self-verification with nothing to signal it.
    """
    assert VERIFY_DEPLOYMENT.fallback_endpoint_vars == ()
    assert VERIFY_DEPLOYMENT.fallback_key_vars == ()
    with pytest.raises(RuntimeError, match="no credentials"):
        VERIFY_DEPLOYMENT.client(
            {
                "AZURE_FOUNDRY_OPENAI_BASE_URL": "https://example.invalid",
                "AZURE_FOUNDRY_OPENAI_API_KEY": "secret",
            }
        )


def test_verification_refuses_when_both_roles_resolve_to_one_model(tmp_path) -> None:
    """The guard must compare resolved names, not the dataclass literals."""
    config = {
        "AZURE_OPENAI_DEPLOYMENT": "same-model",
        "AZURE_OPENAI_DEPLOYMENT_VERIFY": "same-model",
        "AZURE_OPENAI_ENDPOINT_VERIFY": "https://example.invalid",
        "AZURE_OPENAI_API_KEY_VERIFY": "secret",
    }
    assert VERIFY_DEPLOYMENT.resolved_model(config) == LABEL_DEPLOYMENT.resolved_model(config)

    with pytest.raises(RuntimeError, match="same model that produced the labels"):
        run_job(
            job="verify",
            rows=[],
            forms={},
            deployment=VERIFY_DEPLOYMENT,
            config=config,
            raw_path=tmp_path / "raw.jsonl",
        )


def test_verification_proceeds_when_the_two_roles_differ(tmp_path) -> None:
    config = {
        "AZURE_OPENAI_DEPLOYMENT": "labeller",
        "AZURE_OPENAI_DEPLOYMENT_VERIFY": "verifier",
        "AZURE_OPENAI_ENDPOINT_VERIFY": "https://example.invalid",
        "AZURE_OPENAI_API_KEY_VERIFY": "secret",
    }
    # No rows, so nothing is sent; reaching the empty-work path means the
    # guard allowed the run.
    usage = run_job(
        job="verify",
        rows=[],
        forms={},
        deployment=VERIFY_DEPLOYMENT,
        config=config,
        raw_path=tmp_path / "raw.jsonl",
    )
    assert usage.calls == 0
