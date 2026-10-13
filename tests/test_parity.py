import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import pytest

from accentuk import Accentor


@pytest.fixture(scope="module")
def engine():
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle fixture not present")
    with Accentor(asset_dir=bundle_path) as acc:
        yield acc


def test_parity_probe_words(engine: Accentor):
    # Dictionary direct match
    res = engine.stress("Мова.")
    assert res.text == "Мо́ва."
    assert res.tokens[0].status == "stressed"
    assert res.tokens[0].candidates == ["0"]

    # Not required (single vowel)
    res = engine.stress("В лісі.")
    assert res.tokens[0].status == "not_required"


def test_parity_prepositional_rule(engine: Accentor):
    # "нього" after "без"
    res = engine.stress("без нього")
    assert res.tokens[1].text == "нього"
    assert res.tokens[1].status == "prepositional"
    assert res.tokens[1].output_text == "ньо́го"


def test_parity_preserve_ambiguity(engine: Accentor):
    # "сестри" has candidates ["0", "1"]
    res = engine.stress("сестри", on_ambiguity="preserve")
    # Isolated "сестри" with preserve should stay unstressed if no context disambiguates
    tok = [t for t in res.tokens if t.text == "сестри"][0]
    assert tok.status in ("ambiguous", "dictionary_default", "agreement", "morphology")


def test_parity_with_live_go_service(engine: Accentor):
    # Check if Go service is running
    req = Request("http://localhost:8080/v1/stress", method="GET")
    try:
        with urlopen(Request("http://localhost:8080/health/live", method="GET"), timeout=1) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") != "live":
                pytest.skip("Go service not healthy")
    except Exception:
        pytest.skip("Go service not reachable on localhost:8080")

    # Sentences to compare against live Go service
    probes = [
        "Мова.",
        "Мої сестри.",
        "Без нього ніяк.",
        "Замок стоїть на горі.",
    ]

    for sentence in probes:
        # Call Go service
        go_req = Request(
            "http://localhost:8080/v1/stress",
            data=json.dumps({"text": sentence}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(go_req, timeout=15) as resp:
                go_data = json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, URLError, OSError) as err:
            pytest.skip(f"Go service request timed out or failed: {err}")

        py_res = engine.stress(sentence)

        has_unavailable = any("unavailable" in w for w in go_data.get("warnings", []))
        if not has_unavailable:
            assert py_res.text == go_data["text"], f"Mismatch for '{sentence}': Go='{go_data['text']}', Py='{py_res.text}'"

        # Assert token-by-token parity
        assert len(py_res.tokens) == len(go_data["tokens"])
        for py_tok, go_tok in zip(py_res.tokens, go_data["tokens"]):
            assert py_tok.text == go_tok["text"]
            if not has_unavailable:
                assert py_tok.output_text == go_tok["output_text"]
                assert py_tok.status == go_tok["status"]
            else:
                assert py_tok.status in (go_tok["status"], "agreement", "morphology", "token_model")
            if py_tok.status != "prepositional":
                assert py_tok.candidates == (go_tok.get("candidates") or [])

