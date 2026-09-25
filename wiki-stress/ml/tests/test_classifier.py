"""The classification reformulation: predict the stressed vowel directly."""

import torch

from ukstress_ml.classifier import (
    SIGNATURE_INDEX,
    SIGNATURES,
    candidate_mask,
    make_collate,
    masked_logits,
    prepare_rows,
)

MANIFEST = {
    "замок": {"candidates": [
        {"sense_id": "593.a", "signature": "0", "stressed": "за́мок"},
        {"sense_id": "593.b", "signature": "1", "stressed": "замо́к"},
    ]},
    "коси": {"candidates": [
        {"sense_id": "12.a", "signature": "0", "stressed": "ко́си"},
        {"sense_id": "12.b", "signature": "1", "stressed": "коси́"},
    ]},
}


def row(sentence: str, form: str, sense: str, group: int = 1) -> dict:
    start = sentence.index(form)
    return {"sentence": sentence, "form": form, "start": start,
            "end": start + len(form), "gold_sense": sense, "group_id": group}


def test_prepare_attaches_the_gold_signature() -> None:
    got = prepare_rows([row("Старовинний замок на горі.", "замок", "593.a")], MANIFEST)
    assert len(got) == 1
    assert got[0]["gold_signature"] == "0"
    assert [c["signature"] for c in got[0]["candidates"]] == ["0", "1"]


def test_a_row_whose_gold_is_not_a_candidate_is_dropped() -> None:
    # The masked softmax could never select it, so training on it is training
    # against an unreachable target.
    assert prepare_rows(
        [row("Старовинний замок на горі.", "замок", "999.z")], MANIFEST) == []


def test_a_form_absent_from_the_manifest_is_dropped() -> None:
    assert prepare_rows([row("Тут макариха була.", "макариха", "9.a")], MANIFEST) == []


def test_the_mask_keeps_only_the_signatures_this_form_offers() -> None:
    rows = prepare_rows([row("Старовинний замок на горі.", "замок", "593.a")], MANIFEST)
    mask = candidate_mask(rows)
    assert mask.shape == (1, len(SIGNATURES))
    assert mask[0, SIGNATURE_INDEX["0"]] and mask[0, SIGNATURE_INDEX["1"]]
    assert not mask[0, SIGNATURE_INDEX["2"]]


def test_masking_removes_non_candidates_from_the_softmax() -> None:
    logits = torch.zeros(1, len(SIGNATURES))
    logits[0, SIGNATURE_INDEX["3"]] = 99.0        # a signature this form lacks
    mask = torch.zeros(1, len(SIGNATURES), dtype=torch.bool)
    mask[0, SIGNATURE_INDEX["0"]] = True
    mask[0, SIGNATURE_INDEX["1"]] = True
    assert masked_logits(logits, mask).argmax(dim=-1).item() != SIGNATURE_INDEX["3"]
    assert torch.softmax(masked_logits(logits, mask), dim=-1)[0, SIGNATURE_INDEX["3"]] == 0


class FakeTokenizer:
    """Character-level stand-in with the offset mapping the collate relies on."""

    def __call__(self, texts, **kwargs):
        width = max(len(t) for t in texts) + 2
        ids, attn, offs = [], [], []
        for t in texts:
            row_ids = [0] + [ord(c) % 100 + 5 for c in t] + [2]
            row_off = [(0, 0)] + [(i, i + 1) for i in range(len(t))] + [(0, 0)]
            pad = width - len(row_ids)
            ids.append(row_ids + [1] * pad)
            attn.append([1] * len(row_ids) + [0] * pad)
            offs.append(row_off + [(0, 0)] * pad)
        import transformers
        return transformers.BatchEncoding({
            "input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn),
            "offset_mapping": torch.tensor(offs)})


def test_the_span_mask_covers_exactly_the_target_word() -> None:
    sentence = "Старовинний замок на горі."
    rows = prepare_rows([row(sentence, "замок", "593.a")], MANIFEST)
    batch = make_collate(FakeTokenizer(), 192)(rows)
    covered = [j for j, on in enumerate(batch["span"][0].tolist()) if on]
    # One leading special token, so character i sits at position i+1.
    assert covered == [i + 1 for i in range(rows[0]["start"], rows[0]["end"])]


def test_two_occurrences_of_one_form_get_different_spans() -> None:
    """The failure RUAccent lists as its own limitation."""
    sentence = "Замок на горі, а замок на дверях."
    first = {"sentence": sentence, "form": "замок", "start": 0, "end": 5,
             "gold_sense": "593.a", "group_id": 1}
    second = {"sentence": sentence, "form": "замок", "start": 17, "end": 22,
              "gold_sense": "593.b", "group_id": 1}
    rows = prepare_rows([first, second], MANIFEST)
    batch = make_collate(FakeTokenizer(), 192)(rows)
    assert batch["span"][0].tolist() != batch["span"][1].tolist()
    assert batch["golds"].tolist() == [SIGNATURE_INDEX["0"], SIGNATURE_INDEX["1"]]


def test_a_truncated_target_falls_back_rather_than_dividing_by_zero() -> None:
    sentence = "Старовинний замок на горі."
    rows = prepare_rows([row(sentence, "замок", "593.a")], MANIFEST)
    rows[0]["start"], rows[0]["end"] = 900, 905     # beyond every token
    batch = make_collate(FakeTokenizer(), 192)(rows)
    assert batch["span"][0].any()


def test_a_row_carrying_its_signature_directly_is_accepted() -> None:
    """Forms mined from the trie have no senses, only signatures."""
    r = row("Старовинний замок на горі.", "замок", "unused")
    r.pop("gold_sense")
    r["gold_signature"] = "1"
    got = prepare_rows([r], MANIFEST)
    assert len(got) == 1 and got[0]["gold_signature"] == "1"


def test_a_direct_signature_outside_the_candidates_is_still_dropped() -> None:
    r = row("Старовинний замок на горі.", "замок", "unused")
    r.pop("gold_sense")
    r["gold_signature"] = "4"
    assert prepare_rows([r], MANIFEST) == []
