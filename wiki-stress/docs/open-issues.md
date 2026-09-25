# Open issues — 2026-09-18

What is still wrong or unfinished, after `tok-v3d` went live on both machines.
Numbers are from `docs/state.md` and `RESULTS.md`.

## 1. Output quality

| # | issue | evidence | where it lives |
| --- | --- | --- | --- |
| 1 | **Rare readings in a forcing context are still lost.** The cross-encoder picks the common sense (м'який *атла́с* → а́тлас; замки́ з воріт → за́мки; лупа́ після шампуню → лу́па); morphology cannot tell senses with identical tags (обра́зи від критиків → о́брази; його вина́ → ви́на); the classifier misses some too (підземні хо́ди → ходи́; почервоніла по́ра → пора́). | lang-uk heteronyms 84.95% — one in seven wrong; the test paragraph of 2026-09-18 | cross-encoder, morphology, classifier |
| 2 | **The cross-encoder is weak on ambiguous forms the classifier does not cover.** | 77.9% on modern text (`stressed` tier, forms seen <100); classifier covers only 299 forms at ≥100 | cross-encoder v19-v10 |
| 3 | **Dictionary default is a coin with a bias** on uncovered forms. | 89–90% on modern text where it decides | lexicon ordering |
| 4 | **Suffix fallback carries a third of the benchmark loss.** | 461 tokens at 72.0%, 32% of lost words | `suffix_table.json` |
| 5 | **Lexicon defects surface one at a time.** воду́ and хо́ча were found by reading one paragraph; two audit rounds fixed 23 + 85 before that. More exist. | | datasets 12/13 via `run_manual_corrections.py` |
| 6 | **Single-vowel words come out marked** («У́», «а́»). Harmless for TTS, noise on screen; lang-uk ignores them. | | lexicon entries for monosyllables |
| 7 | **The positional default is a guess** (penultimate vowel of the last segment): 11 of 15 right on the benchmark. | | `applyPositionalDefault` |
| 8 | **Classifier generalisation to unseen forms is modest**: 76.0% inside its corpus (v3c 80.3%), and it is never used on them in production — coverage gating is what keeps it safe. | | `tok-v3d` |

## 2. Measurement

| # | issue |
| --- | --- |
| 9 | **The two tests disagree and neither is the target.** lang-uk is human gold on sentences built to force rare readings; the modern-text test is Common Voice with gold from the audio ranker — the same ranker that labelled the training books, so a systematic ranker error on a form counts as agreement. There is no human-gold modern-text set. The deployment rule (dictionary default + cross-encoder picks, forms seen 100+) is a compromise between them. |
| 10 | **The "before" column of the modern-text test is stale**: the sweeps' `text` field is the pipeline's answer at sweep time, before 100+ lexicon corrections. Gains attributed to a tier partly belong to the corrections. |
| 11 | **Ranker precision on ambiguous words is unmeasured.** 98.2% at ≥0.95 was measured on lexicon-named words; on the homographs that actually train the classifier it cannot be measured without human gold. |
| 12 | **The lang-uk clone lives in a session scratchpad** on the desktop (`/tmp/claude-…/ukrainian-tts-preprocessing`); on the laptop it is `~/ukrainian-tts-preprocessing`. |

## 3. Data

| # | issue |
| --- | --- |
| 13 | **Batch 3 is mined but not in any corpus.** 36 books passed the pilot and were mined on 2026-09-17/18 after `tok-v3d` was trained. A `tok-v4` on 24 + 95 + 36 books is untrained. |
| 14 | **Wrong audio under a title** in the input folders: «Жінки, які кохають…» carries «Книга змін»'s recording; «Янссон» carries Жолдак's; «Петров» carries Франкл's; Логвин and Андрухович share one recording that matches neither text. The inventory flags them; the folders are still there. |
| 15 | **Texts that do not match their recordings** (other edition or translation): Толкін «Хранителі», Капоте, Манн, Ролінг «Напівкровний принц», Барка, Ремарк «Життя у позику», Гуцало «Голодомор», Прохасько, Літопис, Яновський «Вершники», Панас Мирний «Хіба ревуть…», Вишня. Dropped by the pilot, not re-sourced. |
| 16 | **Partial matches mined as-is**: Шевчук «Дім на горі» at 41% word match, Гюго at 76%; a handful of books yield <1,000 rows/hour. Their rows are good, their coverage is not. |
| 17 | **Archaic and verse books are excluded by a hand-written list** (21 slugs in `/tmp/archaic.txt` on the laptop, not in the repo). The rule — no originals before ~1930, no verse — is in `RESULTS.md`, the list is not. |
| 18 | **The per-reading cap still drops rows** (91,848 at cap 2000). Uncapped was not tried. |
| 19 | **`--min-quality 1.0` discards ~5% of aligned words** (a vowel the aligner missed). Correct, but yield. |
| 20 | **Two duplicate books** exist under two slugs (Багряний «Людина біжить…», Гоукінз «Дівчина у потягу»); one copy of each is excluded by hand, not by the planner. |

## 4. Engineering and operations

| # | issue |
| --- | --- |
| 21 | **Compose does not recreate `api` for a changed bind-mounted manifest.** Documented in `docs/state.md`; not automated. Cost 0.22 points for an hour on 2026-09-18. |
| 22 | **`run_manual_corrections.py --manifest` keeps only the first backup** of the manifest; successive prunes are not versioned. |
| 23 | **The token classifier runs in torch on CPU** in the container, one request at a time under a lock: ~100–400 ms per request. No ONNX export for `tok-v3d`; the cross-encoder has one. |
| 24 | **`stressed` means two things** — a single-reading lexicon hit and a cross-encoder decision — so tier attribution has to look at the candidate count to tell them apart. |
| 25 | **The GPU pipeline is half idle**: alignment and SSL are unbatched (52% of time), two workers reach ~65% utilisation. Batching alignment is a change in `ukstress.alignment.whisperx_backend`. |
| 26 | **Old scripts still hard-code one machine**: `/home/devops` and the uv archive hash in `run_mine_commonvoice.py`, `run_verify_stress.py`, the `run_*.sh` drivers. Only `run_mine_book_v3.py` and `build_book_corpus.py` resolve the trie from the package. |
| 27 | **No Python tests for the token tier** (`token_resolver.py`, the `/internal/v1/token` endpoint); Go tests exist for the API side. No CI on either repository. |
| 28 | **Database roles use default passwords** (`ukstress_api`, `ukstress_etl`) on both machines. |
| 29 | **Leftovers**: `artifacts/books/void/` (641 MB of the void mining run) and `~/move-staging` on the desktop; `~/move-staging` (with `env.txt`, secrets) on the laptop; `.codex/` untracked in `audiotostress`. |
| 30 | **The desktop's Gradio is stopped**, the laptop's runs; the two stacks are otherwise identical and nothing keeps them so. |
| 31 | **`aloudreader` integration is a brief, not code** (`docs/aloudreader-integration.md`). |
