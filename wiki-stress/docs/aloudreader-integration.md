# Brief for the agent: the stress pipeline as an accentor backend in `aloudreader`

Written 2026-09-15 for an agent working in `~/aloudreader` on the laptop
(`gpu-host`). Read `aloudreader/AGENTS.md` first; its rules (scoped
edits, tests on touched code, no secrets, thin handlers) apply to everything
below.

## What you are adding, in one sentence

A third Ukrainian accentor backend, `ukstress`, for the StyleTTS2 adapter —
alongside the existing `word_stress` and `transformer` — that gets stressed
text from the `ukstress` HTTP service instead of from an in-process library,
and falls back to `word_stress` when the service is unreachable.

## Why

StyleTTS2 Ukrainian needs stress marks before `ipa_uk` phonemises the text.
Today `backend/app/processing/tts.py::_build_styletts2_stressifier` gives it
`ukrainian_word_stress` (a trie) or `ukrainian_accentor_transformer`. The
`ukstress` service is a four-tier pipeline — reviewed lexicon, morphology, a
gloss cross-encoder, a token classifier trained on 369 hours of audiobooks,
then dictionary defaults — measured on lang-uk's public benchmark at **96.03%
word / 84.84% heteronym accuracy**, and at 96.0% on the ambiguous words of
modern text where the dictionary default was 78.7%. Full numbers:
`~/wiki-stress/RESULTS.md`; what is deployed and how: `~/wiki-stress/docs/state.md`.

## The service you are calling

Runs on this laptop from `~/uk-tts-frontend/deploy` (`docker compose up -d`;
five containers: postgres, stress-model, api, frontend, ui). It is already up.

| | |
| --- | --- |
| endpoint | `POST http://<host>:8080/v1/stress` |
| request | `{"text": "<up to 2 MB>", "on_ambiguity": "default"}` (`on_ambiguity` optional; `"preserve"` leaves undecidable words unmarked) |
| response | `{"text": "<same text with U+0301 after stressed vowels>", "tokens": [{"text","start","end","status","output_text","candidates"}...], "warnings": [], "dataset_id", "model_version", "coverage_manifest_hash"}` |
| health | `GET /health/ready` → `{"status":"ready"}` |
| latency | ~50–300 ms per request on CPU, dominated by the model tiers; safe to call in parallel |

Facts about the contract that decide the implementation:

* **Output is NFC with combining acute U+0301** after the stressed vowel —
  the same symbol the adapter already uses (`_STYLETTS2_COMBINING_ACUTE_ACCENT`).
  Use the returned `text` as-is; do not rebuild it from tokens.
* **Words that already carry an acute are left alone** (`status:
  already_stressed`). So manual marks survive — *if they are acutes*. The
  service does **not** understand the `+` convention: `Ко+леса` is split into
  two tokens and mangled. The adapter converts `+` → acute before calling the
  stressifier today (`_prepare_styletts2_part`); keep that order.
* **Single-vowel words come back marked** («а́», «на» does not — `not_found`).
  `word_stress` does not mark them. Verify `ipa_uk` treats a marked
  monosyllable identically; if not, strip the acute from single-vowel tokens
  before phonemising (the response's `tokens` give you spans).
* Offsets in `tokens` are Unicode code points, not bytes.
* Any non-200, a timeout, or `warnings` mentioning "unavailable" means a tier
  was down; the text is still valid (defaults were used). Treat only
  transport failure as failure.

## Design

**Settings** (`backend/app/core/config.py`, next to `styletts2_ukrainian_accentor`):

```python
styletts2_ukrainian_accentor: Literal['word_stress', 'transformer', 'ukstress'] = 'word_stress'
ukstress_api_url: str | None = None            # e.g. http://gpu-host:8080
ukstress_timeout_seconds: float = 5.0
ukstress_fallback_accentor: Literal['word_stress', 'transformer'] = 'word_stress'
```

**Client** — `backend/app/processing/ukstress_client.py`, no framework
imports, a small class with one method `stress(text: str) -> str` over
`httpx` (already a backend dependency; check `pyproject.toml`) with the
timeout from settings, a persistent client, and a typed error
`UkStressUnavailable`. It never raises anything else to the caller. Keep it
here rather than under `services/` because it is a processing-stage
dependency of an engine adapter, like `ipa_uk`.

**Stressifier** — in `tts.py`, a callable `UkStressStressifier(client,
fallback)` whose `__call__(text)` returns `client.stress(text)` and, on
`UkStressUnavailable`, logs once per runtime (not per part) and returns
`fallback(text)`. `_build_styletts2_stressifier` returns it for
`'ukstress'`, building the fallback with the existing code path for
`settings.ukstress_fallback_accentor`, and returns backend name `'ukstress'`.
Nothing else in `_prepare_styletts2_part` changes.

**Cache correctness** — `_STYLETTS2_MODEL_CACHE` is keyed on the accentor
setting already; confirm `AudioRenderService.build_synthesis_fingerprint`
also incorporates it (or the engine's backend name), so segments synthesised
under `word_stress` are not served from cache after switching to `ukstress`.
If it does not, add it — that is a behaviour change worth a test.

**Fail-open** — a worker must start and render with the service down. Do not
probe `/health/ready` at import or model-load time; the first failed call
switches to the fallback for that call, every call retries the service
(cheap: a connection refusal is milliseconds).

**Out of scope** — other engines (Supertonic, Kokoro) do not take stressed
input today; the browser TTS path is separate; showing stress in the reader
UI is a product question. Do not vendor any `ukstress` code; it is a service.

## Deployment

The StyleTTS2 worker (`backend/Dockerfile.tts-styletts2`, its compose
service) needs `UKSTRESS_API_URL` and `STYLETTS2_UKRAINIAN_ACCENTOR=ukstress`.
The service listens on the host; from a container use the LAN address
(`http://gpu-host:8080`) or `host.docker.internal` with the
`extra_hosts` mapping. Do not merge the two compose files; they have
different lifecycles. Document the variable in `backend/README.md` and the
env examples (`_env.example`), never in `.env.local`.

## Tests (write these; report which ran)

Unit, `backend/tests/` in the existing layout, with `respx` or a monkeypatched
transport — no network:

1. `stress()` returns the response `text` unchanged (NFC, acutes present).
2. Timeout, connection error, HTTP 500, malformed JSON → `UkStressUnavailable`;
   nothing else escapes.
3. `UkStressStressifier` falls back and logs once; the fallback's output is
   what the caller gets.
4. `_build_styletts2_stressifier` with `'ukstress'` returns backend name
   `'ukstress'`; with the setting unset behaviour is byte-identical to today
   (regression guard for `word_stress`).
5. `+` in input reaches the service as an acute, not as `+`.
6. The synthesis fingerprint differs between `word_stress` and `ukstress`.

Contract test, skipped unless `UKSTRESS_API_URL` is set: one request with a
heteronym sentence («Ми довго блукали по заводу.» → contains «заво́ду»), and
`/health/ready` returns `ready`.

Coverage on the new module and the touched functions ≥ 80% (AGENTS.md).

## Verification, end to end

1. `STYLETTS2_UKRAINIAN_ACCENTOR=ukstress`, quick-synthesise (the existing
   `tts_quick_synthesize` path) three sentences and confirm the audio renders
   and the log shows no fallback:
   «Замок стоїть на горі, а замок висить на дверях.»,
   «Ми довго блукали по заводу, вилазили по всіх закутках.»,
   «Через видання заміж доньки у мене зараз багато клопоту.»
2. Stop the `api` container (`docker compose stop api` in
   `~/uk-tts-frontend/deploy`), synthesise again, confirm the fallback
   engaged and audio still renders; start it again.
3. Same three sentences under `word_stress`; note where the marks differ.
   The `ukstress` reading is the reference (85% heteronym accuracy against
   the trie's published 7x%), but if a sentence sounds wrong, say so — the
   pipeline has a known weakness on rare readings in constructed sentences,
   documented in `~/wiki-stress/docs/lessons.md` §6.4.

## Report back with

Which tests ran and which could not; the three sentences under both
backends; observed latency per part; anything in the contract above that
turned out to be false.
