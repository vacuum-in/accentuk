from ukstress.serving_check import Report, _drop_subsumed_free_variation, repair


def test_free_variation_is_dropped_only_when_a_member_is_specific():
    assert _drop_subsumed_free_variation(["0|1", "1"]) == ["1"]
    assert _drop_subsumed_free_variation(["0|1", "2"]) == ["0|1", "2"]
    assert _drop_subsumed_free_variation(["0|1"]) == ["0|1"]


def test_repair_rederives_swapped_signatures_and_drops_unservable_forms():
    manifest = {"inventory_hash": "h", "forms": {
        "басками": {"signatures": ["0", "1"], "sense_ids": ["a", "b"], "candidates": [
            {"sense_id": "a", "signature": "0", "stressed": "Баска́ми"},
            {"sense_id": "b", "signature": "1", "stressed": "ба́сками"}]},
        "валові": {"signatures": ["1", "2"], "sense_ids": ["a", "b"], "candidates": [
            {"sense_id": "a", "signature": "2", "stressed": "валові́"},
            {"sense_id": "b", "signature": "1", "stressed": "вало́ві"}]},
    }}
    report = Report(unservable={"валові": {"api": ["0", "1", "2"], "manifest": ["1", "2"]}})
    repaired, counts = repair(manifest, report)
    assert counts == {"signatures_rederived": 2, "forms_dropped": 1}
    assert set(repaired["forms"]) == {"басками"}
    assert {c["stressed"]: c["signature"] for c in repaired["forms"]["басками"]["candidates"]} == {
        "Баска́ми": "1", "ба́сками": "0"}
    assert repaired["inventory_hash"] == "h"
