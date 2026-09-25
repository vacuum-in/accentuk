# State of the stress pipeline

Rewritten 2026-09-20. This page is what is running, what it scores, what is
in flight, and where the work is going. `RESULTS.md` has every number's
history; `PLAN_PARITY.md` and `PLAN_QUALITY.md` have the plans this page
tracks; `docs/open-issues.md` has what is still wrong; `docs/lessons.md` has
why the pipeline looks the way it does.

## Deployed (both machines, identical)

Two hosts run the same stack from `~/uk-tts-frontend/deploy`: the desktop
(RTX 3070 Laptop 8 GB) and the laptop `gpu-host` (RTX 5090 32 GB). Same
user, same paths, same `.env` except `POSTGRES_VOLUME`. The laptop is the
primary for anything GPU-heavy; both serve.

| component | value |
| --- | --- |
| lexicon | Postgres, 2,996,570 forms; `SUPPLEMENTARY_DATASETS=5,7,9,10,12,13` (dataset 3, the LLM gap-fill, dropped 09-18); `REVIEWED_DATASETS=12,13`, `REVIEWED_EXCLUSIVE_DATASETS=13` |
| manual corrections | dataset 12: 231 reviewed readings *lead* (20 from rounds 1–2, 211 dictionary-default reorders from round 3); dataset 13: 90 readings *exclusive* |
| cross-encoder | `output/ml/models/v19-v10`, threshold 0.5; its manifest pruned of every reviewed form |
| morphology | spaCy `uk_core_news_trf_380`, counted-form and aspect rules |
| token classifier | **`tok-v5`** (`models/tok-v5`): XLM-R-uk over the sentence, softmax over the served candidates; answers where the pipeline would take the dictionary default or the cross-encoder's pick, for forms seen 80+ times in training (461 forms); status `token_model` |
| learned combiner | **`combiner-v1`** (`models/combiner-v1`, `COMBINER_ENABLED=1`): weighs every tier's opinion of each ambiguous token plus the tier order's answer, departs only by a margin of 0.2; status `combiner`. Held out live: top-200 95.93% → 97.51%, lang-uk 83.53% → 83.27% |
| fallbacks | compound → suffix table → positional default (penultimate vowel of the last segment; nothing with two vowels leaves unmarked) |
| audio side | `all_model/ranker.pt`, 97.0% on held-out Common Voice speakers, 96.2–96.6% against the lexicon on book and YouTube audio, 98.8–99.0% at ≥0.99 |

**Precedence** for an ambiguous token: reviewed-exclusive → reviewed-lead →
lexicon → morphology → cross-encoder → token classifier (on `dictionary_default`
and `stressed` only) → dictionary default → compound → suffix → positional.

**Measured** (2026-09-20, `live_review21`):

| test | value | note |
| --- | ---: | --- |
| top-200 ambiguous forms, audio gold ≥0.99 (RUAccent's yardstick) | **95.88%** | RUAccent 96.37% on Russian |
| modern text, all ambiguous tokens (6,000 live) | 92.9% | pipeline alone 85.9% |
| lang-uk words | 95.98% | human gold, adversarial sentences |
| lang-uk heteronyms | 84.62% | |
| lang-uk sentences | 70.37% | |

**After any manifest, model or dataset change:** `docker compose restart
stress-model api` — compose does not recreate a container for a changed
bind-mounted file (cost 0.22 points for an hour once). **Rollback** of the
classifier: point `TOKEN_MODEL_DIR` at `models/tok-v3d`; of the reorders:
drop dataset 12's round-3 rows; of everything else: `git log` on both repos.

## Data

| corpus | size | where |
| --- | ---: | --- |
| audiobooks with matching text | 460 folders, 5,686 h, 196 GB; **375 usable** (pilot), 344 mined, 30 dropped for mismatched editions, 3 Russian texts | laptop `~/books`, `~/audiotostress/books-input`; inventory in `artifacts/books/inventory.{json,md}` |
| mined rows (v3 miner, one pass, ranker inline) | 318 `rows3` files + 24 first-batch books in `rows2`/`labels2`; ~9.4M rows after batch 5; batch 6 (60 books, 614 h) mining now | laptop `~/audiotostress/artifacts/books` |
| YouTube, no text (v4 miner) | 1,089 videos from 6 sources being fetched; 4 + 17 pilot sources passed | laptop `~/audiotostress/youtube`, desktop mirror |
| classifier corpus `tok-v5` | 336,189 ambiguous rows, 11,054 forms, 100 modern books, ambiguity by the served lexicon | laptop `output/ml/corpus/tok-v5` |
| placer corpus | 125,372 OOV rows, 50,870 forms; +600k lexicon forms in `placer-v3` | laptop |
| Common Voice 26 uk | 118 h, 1,099 speakers: the ranker's training and the modern-text test | both |
| lang-uk benchmark | 1,026 sentences, 493 heteronym forms | desktop scratchpad, laptop `~/ukrainian-tts-preprocessing` |

The book exclusion list (`ml/data/book_exclusions.json`, 37 entries: originals
before ~1930, verse, duplicates, wrong-audio folders) is read by the corpus
builder by default; measured, training on them costs ~1 point on modern text.

## In flight (2026-09-20)

* **Batch 6** of books mining on the laptop, two workers, 60 books, done ~12:40.
* **YouTube**: fetch of 1,089 videos running on the laptop; mining shard 0/2
  queued on the laptop after batch 6, shard 1/2 on the desktop's 3070 pulling
  folders by rsync. Both write `youtube/*.rows4.jsonl`; the laptop is the
  home of the results.
* **Waiting on the reviewer**: `DICTIONARY_AUDIT_ROUND3.md` table A (72
  lexicon readings the narrators contradict) and the gold draft
  `ml/data/gold_modern_draft_1.txt` (306 sentences, 50 forms).

## Where the work is going

The yardstick is RUAccent's: accuracy on the frequent homographs against
audio gold, which is where TTS on modern text lives. The pipeline is 0.5
points from it. lang-uk is carried alongside as the independent, human-gold,
adversarial check; no change ships that costs it more than 0.3.

1. **Audio without text** (PLAN_PARITY step 2, running). The v4 miner needs
   no matching book: the Whisper transcript is the sentence, the lexicon vets
   each word, the share of lexicon words vets each chunk. The pilot on nine
   channels scored like the books (lift +17…+28, ranker 90–98%). Spontaneous
   speech gives the ranker 90–94% against audiobooks' 96–97% — the report's
   90% floor is what keeps podcast rows in. This removes the supply ceiling:
   RUAccent used 108,000 hours, this project has 5,700 with text.
2. **`tok-v6`** from books + YouTube rows, the `tok-v5` recipe. The question
   it answers is whether spontaneous-speech rows help or hurt the classifier
   on the two tests.
3. **The ranker's precision on homographs** (PLAN_PARITY step 3): synthesise
   the two readings of each heteronym with the project's own TTS, mine the
   audio, and measure the labeller on the words that actually train the
   classifier — the number that has never been measurable.
4. **Human gold** (PLAN_QUALITY step 4) and then a **learned combiner** over
   the tiers (step 5). Every rule change since 09-18 has been a compromise
   between two tests that are each wrong in a known way; the draft that
   unblocks this is written and waits for review.

## What was tried and closed

| idea | result | where recorded |
| --- | --- | --- |
| confidence gate on the classifier | no help on lang-uk; costs modern text | `PLAN_QUALITY` step 0 |
| more books (`tok-v4`, 225 books) | equal on modern text, −1.1 to −1.5 on lang-uk | `RESULTS.md` |
| balancing minority readings | no change | `RESULTS.md` |
| a stress placer for OOV words (6 models incl. mmBERT) | +7 on modern text, parity on lang-uk, ~+0.1 overall; not deployed | `RESULTS.md` |
| the classifier overriding morphology | +0.35 modern, −0.5 to −1.1 lang-uk; not deployed | this session's log |
| dictionary-default reorders (211) | +1.2 modern, 0 on lang-uk; **live** | `RESULTS.md` |
| ambiguity by the served lexicon (`tok-v5`) | +1.3 on top-200, +0.9 modern, −0.3 lang-uk; **live** | `RESULTS.md` |

## How to verify a machine

```
curl -s http://localhost:8080/health/ready
docker exec uk-tts-stress-model-1 python -c "import urllib.request,json; print(json.load(urllib.request.urlopen('http://127.0.0.1:8090/health/ready'))['token_model'])"
ml/.venv/bin/python ml/scripts/run_live_bench.py --benchmark <lang-uk clone> --save output/ml/live_review_N.json
ml/.venv/bin/python ml/scripts/run_tier_attribution.py output/ml/live_review_N.json
```

Expect `tok-v5`, 461 eligible forms; lang-uk 95.97% / 84.62%; a `token_model`
row in the tier table.
