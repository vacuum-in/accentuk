---
language: uk
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: FacebookAI/xlm-roberta-base
tags: [ukrainian, stress, homograph, word-sense-disambiguation, cross-encoder, tts]
---

# Ukrainian homograph cross-encoder (`v19-v10`)

Scores how well a gloss fits one occurrence of an ambiguous Ukrainian word.
The pipeline scores every candidate reading of the word, takes the best one,
and so gets the word's stress: за́мок "castle" or замо́к "lock", бо́ку or боку́.

## Input and output

- **Input:** a pair of strings, truncated to 192 subword tokens.
  - **First:** the sentence with the target word wrapped in `⟦ ⟧`.
  - **Second:** the candidate gloss, `"<stressed form>: <definition>"`.
- **Output:** one logit. The pipeline:
  - scores all senses of the form;
  - takes the maximum per stress signature;
  - answers when the margin between the top two signatures is at least **0.5**, and abstains otherwise.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("aloudreader/uk-stress-crossencoder")
model = AutoModelForSequenceClassification.from_pretrained("aloudreader/uk-stress-crossencoder").eval()
pairs = [("Старий ⟦замок⟧ стоїть на горі.", "за́мок: Укріплене житло феодала доби середньовіччя з оборонними, господарськими, культовими і т. ін. будівлями, зазвичай оточене високим кам'яним муром із кількома вежами. Великий поміщицький будинок; палац."),
         ("Старий ⟦замок⟧ стоїть на горі.", "замо́к: Пристрій для замикання дверей у приміщеннях, дверцят шафи, скринь, шухляд і т. ін. На замку бути — бути замкненим; бути недоступним для ворогів, добре охоронятися.")]
enc = tok([a for a, _ in pairs], [b for _, b in pairs], return_tensors="pt", padding=True)
print(model(**enc).logits.squeeze(-1))   # higher = better fit
```

The candidate senses, their glosses and stress signatures come from the serving manifest. It covers 15,203 forms and ships with the model as `serving_manifest.json`. Forms outside the manifest are not scored.

## Training

- **Architecture:** XLM-RoBERTa base, 278M parameters, with a sequence-classification head. Trained in two stages from `xlm-roberta-base`:
  - `v3-xenc`, on 12,351 mined, sense-labelled sentences;
  - `v19-v10`, on 463,867 rows.
- **Data of `v19-v10`:**
  - 411,049 sentences mined from the Malyuk corpus;
  - 23,287 from Ukrainian Wikipedia (CC BY-SA 4.0);
  - 29,531 LLM-generated, used for training only.
  - The sense labels come from an LLM (DeepSeek-V4-Pro), blind-verified by a second LLM (DeepSeek-V3.2). Senses and glosses come from the project's homograph inventory.
- **Split:** by form, 25% of forms held out. Forms whose training rows show only one sense were dropped (222,217 rows).
- **Settings:** 2 epochs, lr 1e-5, batch 16, max length 192, seed 20260824, listwise cross-entropy over each row's candidates.
- **Result:** held-out dev macro accuracy per sense group 0.8443, micro accuracy 0.8939.
- **Hardware:** one RTX 3070, 3.4 h.
- **Reproducibility:** retrained from the same data on 2026-09-25, it scored 0.8429. Every pipeline test landed within 0.3 points.

## Evaluation

As a single tier, taking the lexicon's first reading where it abstains:
- 72.3% on ambiguous Common Voice tokens (audio gold);
- 66.6% on lang-uk;
- 71.0% on the top-200 forms.

It is strongest on *semantic* homographs, where the two readings differ in meaning (а́тлас / атла́с). In the pipeline, morphology decides grammatical alternations (се́ла / села́) and the token classifier decides frequent forms. The cross-encoder mainly answers forms that neither of those covers.

## Limitations

- **Labels:** the training labels are LLM judgements, not human ones. Re-adjudicating the model's high-confidence "errors" overturned the silver label in 57% of cases, so the reported accuracy is bounded by label noise.
- **Rare readings:** it favours the common reading in contexts built to force the rare one (м'який *атла́с*).
- **Coverage:** it only covers forms that have glosses. About 1,900 senses of the inventory lack a definition.
- **Weak spot:** on modern text it is weakest on ambiguous forms the token classifier does not cover, at 77.9% there.
