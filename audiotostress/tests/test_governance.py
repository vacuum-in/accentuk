import pytest

from ukstress.provenance import ExportLicensePolicy, validate_media_uri, validate_metadata_text


def test_governance_rejects_unsafe_paths_and_applies_license_flags() -> None:
    with pytest.raises(ValueError, match="traversal"):
        validate_media_uri("../secret.wav")
    with pytest.raises(ValueError, match="control"):
        validate_metadata_text("unsafe\nvalue")
    assert not ExportLicensePolicy(require_redistribution=True).allows(
        type("Record", (), {"license_id": "x", "commercial_use": None, "redistribution": False})()
    )
