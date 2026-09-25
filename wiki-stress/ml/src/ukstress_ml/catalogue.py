"""What each dataset is, so a file name is not the only thing to go on.

Descriptions are written from what the files are actually used for in this
project. Where a name follows a pattern rather than being one specific artifact,
the pattern is described and the entry says so, rather than inventing detail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CORPUS = "Training corpus"
INVENTORY = "Sense inventory"
MANIFEST = "Serving manifest"
TABLE = "Lookup table"
EVAL = "Evaluation"
ANALYSIS = "Analysis"
OTHER = "Other"

#: Order the groups appear in the sidebar: what you train on, what defines the
#: label space, what is served, then what measures it.
GROUP_ORDER = (CORPUS, MANIFEST, INVENTORY, TABLE, EVAL, ANALYSIS, OTHER)


@dataclass(frozen=True)
class Entry:
    group: str
    title: str
    detail: str


EXACT: dict[str, Entry] = {
    "silver_mined_v10.json": Entry(
        CORPUS, "The corpus behind the shipped model",
        "463,867 rows over 7,079 ambiguous forms; what `v19-v10` was trained on. "
        "One row is a sentence, the target span, and the sense that won. "
        "88.6% malyuk, 5% ukwiki, 6.4% synthetic. Labels are silver: two models "
        "agreeing, residual error near 2%."),
    "silver_mined_v11.json": Entry(
        CORPUS, "Corpus v11", "The next mining round after v10. Trained no shipped model."),
    "classifier_rows_v25.json": Entry(
        CORPUS, "Classifier rows, corpus plus the frequent-form gap",
        "467,623 rows: the v10 corpus joined to 19,897 rows mined for the most "
        "frequent ambiguous forms, which have no dictionary senses and so were "
        "outside model coverage entirely. Carries `gold_signature` directly."),
    "top_forms_labelled.json": Entry(
        CORPUS, "The frequent-form gap, labelled by the parser",
        "19,897 rows across 106 forms — `вони`, `тому`, `яка`, `була` — mined from "
        "real text and labelled morphologically, because the tags separate their "
        "readings. The majority label agrees with lang-uk gold on 84.4% of 244 "
        "benchmark tokens, so these inherit the tagger's errors."),
    "silver_natural.json": Entry(
        CORPUS, "Natural sentences only",
        "No generated data. Trained `v14-natural`, which scored 79.72% against "
        "v12's 81.24% — the measurement that refuted 'generated data is worthless'."),
    "silver_inflected.json": Entry(
        CORPUS, "Inflected-forms pilot",
        "23,287 rows. The narrow set that asked whether fine-tuning on inflected "
        "surfaces restores calibration."),
    "gold_set_1000.jsonl": Entry(
        EVAL, "External gold set, 1,000 rows",
        "30 heteronyms, human-labelled, never used in training. Reconstructed "
        "from an evaluation run and verified identical to the two source CSVs on "
        "all 1,000 sentences. Note some rows are one sentence in three wrappers, "
        "so it is fewer than 1,000 independent contexts."),
    "ukrainian_homographs_1000.csv": Entry(
        EVAL, "Gold set sentences", "The `--sentences` half of the gold set."),
    "ukrainian_homographs_1000_gold.csv": Entry(
        EVAL, "Gold set answers", "The `--gold` half: stressed target and sense."),
    "serving_manifest_v22.json": Entry(
        MANIFEST, "Shipped model coverage",
        "15,332 forms with their candidate senses, glosses and signatures, plus "
        "`threshold` 0.5 and the inventory hash the API checks. It is an "
        "inventory of forms that have glosses, not a claim about what the model "
        "can do: 55.7% of it never appears in the training corpus."),
    "serving_manifest_v25.json": Entry(
        MANIFEST, "Coverage extended to the frequent forms",
        "15,460 forms: v22 plus 128 built from the stress trie for high-frequency "
        "words with no dictionary senses. Takes top-200 coverage from 61 to 189."),
    "serving_manifest_v23.json": Entry(
        MANIFEST, "Coverage minus the grammatical forms",
        "14,159 forms — v22 with the 1,162 grammatical splits removed, to route "
        "them to the parser. Measured 5 points worse and kept only as the starting "
        "point for a narrower version of that idea."),
    "top_forms_manifest.json": Entry(
        MANIFEST, "Frequent forms, built from the trie",
        "128 entries with signatures and stressed spellings but no definitions, "
        "which is all the signature classifier needs and all the trie can give."),
    "ambiguous_forms_glossed.jsonl": Entry(
        INVENTORY, "Glossed sense inventory",
        "13,565 forms with candidate senses and definitions. Covers only 63% of "
        "the v10 corpus, so training on it silently drops 171,245 rows."),
    "ambiguous_forms_merged.jsonl": Entry(
        INVENTORY, "Glossed inventory plus the manifest",
        "Closes that gap to 97% by folding in the 2,111 forms the serving manifest "
        "already had. Merged entries are marked `paradigm_source: serving_manifest`."),
    "ambiguous_forms_inflected.jsonl": Entry(
        INVENTORY, "Inflected inventory", "12,651 forms, the inflected pilot's inventory."),
    "ambiguous_forms.jsonl": Entry(
        INVENTORY, "First inventory", "2,035 forms. Superseded."),
    "counted_forms.json": Entry(
        TABLE, "Counted form after 2/3/4",
        "27 counted, 76 nominative, 5 unresolved. Read off the Orthoepic "
        "Dictionary, which prints the numeral phrase for exactly the nouns that "
        "take it — the criterion no language model reproduced. Behind "
        "`COUNTED_FORM=1`."),
    "counted_form_worklist.json": Entry(
        ANALYSIS, "Counted-form candidates with contexts",
        "108 forms attested after a 2/3/4 numeral in 454,708 sentences, each with "
        "up to eight real contexts."),
    "counted_form_candidates.json": Entry(
        ANALYSIS, "Counted-form candidates",
        "1,318 of 2,892,732 trie forms whose genitive-singular accent differs "
        "from the nominative plural, which is the precondition."),
    "suffix_table.json": Entry(
        TABLE, "Suffix analogy fallback",
        "149,085 endings for words no tier reaches. 74.5% precise on held-out "
        "lexicon forms; took `uncovered` errors from 188 to 35. Served as "
        "`SUFFIX_TABLE`."),
    "trie_defaults_map.json": Entry(
        TABLE, "Candidate ordering from the trie",
        "53,995 forms where the source trie names one reading and the lexicon "
        "leaves several. Right 26 times against the lexicon order's 12. Served as "
        "`TRIE_DEFAULTS`."),
    "ambiguous_frequency.json": Entry(
        ANALYSIS, "Ambiguous forms by real-text frequency",
        "The 5,000 most frequent, counted over 4.5M words of corpus sentences. "
        "Ambiguous words are 14.3% of running text; the top 2,000 forms cover "
        "only 62.7% of their occurrences, so the tail is fat."),
    "heteronym_frequency.json": Entry(
        ANALYSIS, "lang-uk heteronyms by frequency",
        "All 492 forms of the benchmark's list. Its top 200 cover 83% of "
        "heteronym occurrences, which is why RUAccent's top-200 evaluation is "
        "defensible rather than cherry-picked."),
    "exemplar_ablation.json": Entry(
        ANALYSIS, "Exemplars against definitions",
        "The matched pair that found no benefit in scoring senses by their "
        "labelled usages instead of their dictionary definitions."),
}

PATTERNS: tuple[tuple[re.Pattern[str], Entry], ...] = (
    (re.compile(r"^counted_candidates"), Entry(
        INVENTORY, "Counted-form review queue",
        "795 nouns whose plural and genitive singular are spelt alike and "
        "stressed apart — the homography that makes «три сестри́» possible at "
        "all. Derived from the stress trie's own case and number tags, not "
        "scraped, and paired within one gender so that two lexemes sharing a "
        "spelling — коли, masculine plural at one accent and feminine genitive "
        "singular at another — do not invent a counted form between them. "
        "Membership is not the verdict: whether a word actually takes "
        "the counted form is lexical, and the earlier attempt to settle it by "
        "asking whether a dictionary printed a numeral phrase read silence as a "
        "no and filed вікна́ wrong. Click the reading «три …» takes; the click "
        "records the verdict.")),
    (re.compile(r"^counted_forms\.json$"), Entry(
        TABLE, "Counted forms, in service",
        "What the morphology tier reads when COUNTED_FORM=1. Only rows with "
        "verdict `counted` are used — 27 of 108. Superseded as a review "
        "surface by counted_candidates.json, which covers the whole class.")),
    (re.compile(r"^propernames"), Entry(
        INVENTORY, "Proper names, held out of training",
        "1,142 forms split out of the inventory because at least one of their "
        "readings is a surname, a toponym or a given name — 602, 436 and 142 "
        "readings respectively, plus 92 the glosses call proper without saying "
        "which kind. A name's stress belongs to the name, not to the sentence, "
        "so context cannot decide it and the rows only teach the model to "
        "guess. Every candidate carries a `category`, so re-admitting one kind "
        "is a filter rather than another run. Material for a gazetteer tier.")),
    (re.compile(r"\.nonames\.json$"), Entry(
        CORPUS, "Corpus with proper names removed",
        "The same mining round minus every row whose form group left for the "
        "proper-name inventory: 4.5% of rows. Use this to train, not the "
        "unfiltered file beside it.")),
    (re.compile(r"^silver_mined_v(\d+)\.json$"), Entry(
        CORPUS, "Corpus, earlier round",
        "A superseded mining round. The ladder ran v4 through v11; seven models "
        "trained across 85k-478k rows all landed between 79.7% and 82.4%.")),
    (re.compile(r"^silver_plus_generated"), Entry(
        CORPUS, "Corpus with generated sentences",
        "Natural rows plus synthetic ones built for under-represented senses.")),
    (re.compile(r"^mined_.*rows?\.json$"), Entry(
        CORPUS, "Raw mining output",
        "Contexts pulled from a corpus before labelling or filtering.")),
    (re.compile(r"^ambiguous_surface_v\d+\.jsonl$"), Entry(
        ANALYSIS, "Surface triage",
        "42,395 forms classified by *why* they are ambiguous — free variation, "
        "grammatical, homograph, artifact. Not a sense inventory: it has no "
        "candidates, and passing it to training raises KeyError.")),
    (re.compile(r"^ambiguous_forms_"), Entry(
        INVENTORY, "Sense inventory variant",
        "Forms with candidate senses at some stage of the inventory pipeline.")),
    (re.compile(r"^serving_manifest_v26\.json$"), Entry(
        MANIFEST, "Serving manifest, names removed",
        "v25 with 1,140 proper-name form groups dropped and 2 trimmed: 14,320 "
        "forms. The two trimmed ones carry three readings each — Чо́пові the "
        "town beside чопо́ві and чопові́, both real — so only the name left.")),
    (re.compile(r"^serving_manifest_v\d+\.json$"), Entry(
        MANIFEST, "Serving manifest, earlier version",
        "Model coverage at an earlier point. The API checks its inventory hash "
        "against the database before trusting it.")),
    (re.compile(r"^gold_eval"), Entry(
        EVAL, "Gold evaluation output",
        "Per-row decisions from a run against the external gold set, with every "
        "candidate, confidence, parse and score vector kept so a threshold sweep "
        "is a file re-read rather than another GPU pass.")),
    (re.compile(r"^(bench|langukbench)"), Entry(
        EVAL, "Benchmark output", "Metrics from a lang-uk benchmark run.")),
    (re.compile(r"^suffix_table"), Entry(
        TABLE, "Suffix analogy variant", "An alternative build of the ending table.")),
    (re.compile(r"\.manifest\.json$"), Entry(
        OTHER, "Build manifest", "Provenance for the file it sits beside.")),
)

FALLBACK = Entry(OTHER, "", "No description recorded for this file.")


def describe(name: str) -> Entry:
    if name in EXACT:
        return EXACT[name]
    for pattern, entry in PATTERNS:
        if pattern.search(name):
            return entry
    return FALLBACK
