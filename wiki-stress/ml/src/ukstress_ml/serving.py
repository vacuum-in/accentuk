"""Internal, candidate-constrained XLM-R inference service.

The service has no database access and never constructs a stressed spelling.
It scores only manifest senses and returns one of the caller's signatures.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from ukstress.normalizer import lookup_key

from ukstress_ml.corpus import CLOSE_MARK, OPEN_MARK
from ukstress_ml.morphology import apply_accents as morphology_apply
from ukstress_ml.morphology import imperfective_infinitive
from ukstress_ml.morphology import agreement_repair
from ukstress_ml.morphology import resolve as morphology_resolve
from ukstress_ml.token_resolver import load_token_resolver

log = logging.getLogger(__name__)


class ServingError(ValueError):
    """Raised when the immutable serving contract is violated."""


@dataclass(frozen=True)
class Target:
    sentence: str
    start: int
    end: int
    form: str
    candidates: tuple[str, ...]


class TargetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence: str
    start: int
    end: int
    form: str
    candidates: list[str]


class MorphologyTargetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    sentence: str
    start: int
    end: int
    form: str


class MorphologyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[MorphologyTargetBody]


class ResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: str
    inventory_hash: str
    targets: list[TargetBody]


class TokenTargetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    sentence: str
    start: int
    end: int
    form: str
    candidates: list[str]


class TokenBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[TokenTargetBody]


class CombineTargetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    sentence: str
    start: int
    end: int
    form: str
    text: str
    candidates: list[str]
    pipeline: str = ""
    status: str = ""


class CombineBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[CombineTargetBody]


#: Serving tokenization cap. Left at the training value deliberately: measured
#: on `test_natural`, real pairs are p50 81 / p95 136 tokens, so 192 truncates
#: 1.0% of them, and `padding=True` pads to the longest sequence in the batch
#: rather than to this number. Lowering it does not shorten typical inputs; it
#: only starts truncating (64 would cut 74.6% of pairs).
MAX_LENGTH = 192

#: ONNX graph, relative to the model directory, preferred when present.
ONNX_RELATIVE_PATH = Path("onnx") / "model.onnx"


class _TorchBackend:
    """Eager PyTorch scoring — the reference implementation."""

    name = "torch"

    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        self._torch = torch
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        # Honour an explicit device. Left to itself this picks CUDA whenever a
        # GPU is visible, which silently turns a "CPU serving" measurement into
        # a GPU one -- pass "cpu" to compare against the ONNX backend, which is
        # CPUExecutionProvider only.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def logits(self, encoded: dict[str, Any]) -> list[float]:
        moved = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            values = self.model(**moved).logits.squeeze(-1).detach().cpu().tolist()
        return values if isinstance(values, list) else [float(values)]

    @property
    def return_tensors(self) -> str:
        return "pt"


class _OnnxBackend:
    """ONNX Runtime scoring.

    Measured against eager torch on all 759 `test_natural` spans: identical
    micro accuracy (0.9236), argmax agreement 1.0000, max logit deviation
    9e-05 -- so decisions are unchanged and the fp32-calibrated abstention
    threshold stays valid. It is ~2.1x faster on CPU and needs no torch at
    runtime, which is why it is preferred when the graph is present.

    Only use an fp32 graph here. Dynamic int8 measured max|Dlogit| = 13.0,
    which destroys the margin calibration the threshold depends on.
    """

    name = "onnxruntime"

    def __init__(self, graph: Path, threads: int = 8) -> None:
        import onnxruntime  # type: ignore[import-untyped]  # ships no stubs

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            str(graph), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {node.name for node in self.session.get_inputs()}

    def logits(self, encoded: dict[str, Any]) -> list[float]:
        feeds = {key: value for key, value in encoded.items() if key in self._inputs}
        output = self.session.run(None, feeds)[0]
        return [float(value) for value in output.reshape(-1)]

    @property
    def return_tensors(self) -> str:
        return "np"


class ContextualStressModel:
    def __init__(
        self, model_dir: Path, manifest_path: Path, *, backend: str = "auto",
        device: str | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        content = manifest_path.read_bytes()
        self.manifest: dict[str, Any] = json.loads(content)
        self.manifest_hash = hashlib.sha256(content).hexdigest()
        self.model_version = str(self.manifest["model_version"])
        self.inventory_hash = str(self.manifest["inventory_hash"])
        self.threshold = float(self.manifest["threshold"])
        checkpoint = model_dir / "checkpoint"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        graph = model_dir / ONNX_RELATIVE_PATH
        if backend == "onnx" or (backend == "auto" and graph.exists()):
            if not graph.exists():
                raise ServingError(f"onnx backend requested but {graph} is missing")
            self.backend: _TorchBackend | _OnnxBackend = _OnnxBackend(graph)
        elif backend in {"auto", "torch"}:
            self.backend = _TorchBackend(checkpoint, device)
        else:
            raise ServingError(f"unknown backend: {backend}")

    def resolve(self, targets: list[Target]) -> list[dict[str, Any]]:
        left: list[str] = []
        right: list[str] = []
        widths: list[int] = []
        signatures: list[list[str]] = []
        for target in targets:
            entry, allowed = validate_target(target, self.manifest)
            marked = (
                target.sentence[: target.start]
                + OPEN_MARK
                + target.sentence[target.start : target.end]
                + CLOSE_MARK
                + target.sentence[target.end :]
            )
            row_signatures: list[str] = []
            for candidate in entry["candidates"]:
                signature = candidate["signature"]
                if signature not in target.candidates:
                    continue
                gloss = f"{candidate['stressed']}: {candidate['definition']}".strip(": ")
                left.append(marked)
                right.append(gloss)
                row_signatures.append(signature)
            if set(row_signatures) != set(allowed):
                raise ServingError("manifest lacks candidate metadata")
            widths.append(len(row_signatures))
            signatures.append(row_signatures)

        if not left:
            return []
        encoded = self.tokenizer(
            left,
            right,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors=self.backend.return_tensors,
        )
        logits = self.backend.logits(dict(encoded))

        decisions: list[dict[str, Any]] = []
        offset = 0
        for index, (width, row_signatures) in enumerate(zip(widths, signatures, strict=True)):
            # Multiple senses may map to one signature. Max pooling preserves
            # the hard constraint while allowing the best matching gloss to win.
            scores: dict[str, float] = {}
            for signature, score in zip(
                row_signatures, logits[offset : offset + width], strict=True
            ):
                scores[signature] = max(scores.get(signature, float("-inf")), float(score))
            ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
            margin = scores[ranked[0]] - scores[ranked[1]]
            decisions.append(
                {
                    "index": index,
                    "signature": ranked[0],
                    "scores": scores,
                    "margin": margin,
                    "status": "selected" if margin >= self.threshold else "low_margin",
                }
            )
            offset += width
        return decisions


def validate_target(target: Target, manifest: dict[str, Any]) -> tuple[dict, list[str]]:
    """Enforce the preconditions every backend shares.

    Kept in one place so a second model cannot quietly relax them: offsets must
    land on the requested form, the form must be inside frozen coverage, and the
    database's candidates must match the manifest's exactly. A mismatch is an
    inventory-versioning bug and must fail rather than be resolved against a
    stale candidate set.
    """
    if not (0 <= target.start < target.end <= len(target.sentence)):
        raise ServingError("target offsets are outside the sentence")
    if lookup_key(target.sentence[target.start : target.end]) != target.form:
        raise ServingError("target offsets do not identify the requested form")
    entry = manifest["forms"].get(target.form)
    if entry is None:
        raise ServingError("form is outside frozen training coverage")
    allowed = sorted(entry["signatures"])
    if sorted(target.candidates) != allowed:
        raise ServingError("database candidates differ from model inventory")
    return entry, allowed


class _ClassifierBackendName:
    name = "classifier"


class ClassifierStressModel:
    """Tier 3 as classification, serving the same contract as the pair model.

    The cross-encoder scores `(sentence, gloss)` for every candidate sense. This
    scores the sentence once and reads a signature off a linear head, masked to
    the form's candidates.

    Measured on the 383 benchmark heteronym tokens the manifest covers, the two
    are complementary rather than interchangeable: the classifier is better on
    grammatical splits (85.5% against 83.6%) and worse on semantic homographs
    (63.5% against 68.5%), which follows from the pair model seeing a gloss that
    carries sense information the classifier never gets.

    `margin` is a logit difference in both models, so the manifest's abstention
    threshold keeps its meaning and a deployment can swap one for the other
    without recalibrating.
    """

    def __init__(self, model_dir: Path, manifest_path: Path, *,
                 device: str | None = None) -> None:
        import torch
        from transformers import AutoTokenizer

        from ukstress_ml.classifier import SIGNATURES, SignatureClassifier

        content = manifest_path.read_bytes()
        self.manifest = json.loads(content)
        self.manifest_hash = hashlib.sha256(content).hexdigest()
        self.model_version = str(self.manifest["model_version"])
        self.inventory_hash = str(self.manifest["inventory_hash"])
        self.threshold = float(self.manifest["threshold"])
        self.backend = _ClassifierBackendName()
        self._torch = torch
        self._signatures = SIGNATURES

        checkpoint = model_dir / "checkpoint"
        head = checkpoint / "head.pt"
        if not head.is_file():
            raise ServingError(f"classifier checkpoint has no head at {head}")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = SignatureClassifier(str(checkpoint)).to(self.device).eval()
        self.model.head.load_state_dict(torch.load(head, map_location=self.device))

    def _span(self, offsets: list[tuple[int, int]], attention: Any,
              start: int, end: int) -> list[bool]:
        """Which tokens cover the target word.

        Located through the tokenizer's offsets rather than by inserting
        markers, so nothing is added to the text and two occurrences of one form
        in a sentence stay distinguishable.
        """
        mask = [a != b and a < end and b > start for a, b in offsets]
        if not any(mask):
            # Truncation removed the target. Fall back to the whole sequence
            # rather than dividing by zero; the low margin that follows is the
            # honest signal that the input was unusable.
            return [bool(x) for x in attention]
        return mask

    def resolve(self, targets: list[Target]) -> list[dict[str, Any]]:
        torch = self._torch
        allowed_by_row: list[list[str]] = []
        for target in targets:
            _, allowed = validate_target(target, self.manifest)
            allowed_by_row.append(allowed)
        if not targets:
            return []

        encoded = self.tokenizer(
            [t.sentence for t in targets], truncation=True, max_length=MAX_LENGTH,
            padding=True, return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        attention = encoded["attention_mask"].tolist()
        span = torch.tensor(
            [self._span([tuple(o) for o in offsets[i]], attention[i],
                        target.start, target.end)
             for i, target in enumerate(targets)],
            dtype=torch.bool,
        )
        moved = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = self.model(moved["input_ids"], moved["attention_mask"],
                                span.to(self.device)).float().cpu()

        index_of = {s: i for i, s in enumerate(self._signatures)}
        decisions: list[dict[str, Any]] = []
        for index, allowed in enumerate(allowed_by_row):
            scores = {s: float(logits[index, index_of[s]])
                      for s in allowed if s in index_of}
            if len(scores) < 2:
                # A single representable candidate leaves nothing to choose.
                decisions.append({"index": index, "signature": next(iter(scores), ""),
                                  "scores": scores, "margin": 0.0,
                                  "status": "ambiguous"})
                continue
            ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
            margin = scores[ranked[0]] - scores[ranked[1]]
            decisions.append({
                "index": index, "signature": ranked[0], "scores": scores,
                "margin": margin,
                "status": "selected" if margin >= self.threshold else "low_margin",
            })
        return decisions


def load_model(model_dir: str, manifest_path: str | None = None) -> Any:
    """Pick the backend from what the checkpoint contains.

    A classifier checkpoint carries `head.pt` beside the encoder; a pair
    checkpoint does not. Detecting it here means a deployment switches models by
    changing a path, with no second configuration flag to keep in sync.
    """
    directory = Path(model_dir)
    manifest = Path(manifest_path) if manifest_path else directory / "serving_manifest.json"
    if (directory / "checkpoint" / "head.pt").is_file():
        return ClassifierStressModel(directory, manifest)
    return ContextualStressModel(directory, manifest)


class _MorphologyPool:
    """A pool of morphological parsers plus a shared reading cache.

    Two properties this needs that a single lazily-built parser does not have.

    *Warm.* Building the parser loads models and takes tens of seconds. Left
    until the first request, that request blocks past any sane timeout and the
    tier reports itself unavailable — measured, every morphology call in the
    live API failed this way against a 2s budget while the endpoint answered a
    direct probe in 0.5s.

    *Concurrent.* A parser is not safe to share across threads, and the service
    runs requests in a threadpool. With one instance, concurrent requests
    serialise behind it and time out: at eight concurrent callers the same build
    scored 72.16% word accuracy against 87.89% served serially, because the tier
    silently degraded to dictionary defaults. `size` instances are checked out
    and returned around each parse.

    `readings` is pure trie lookup, thread-safe, and worth caching outright.
    """

    def __init__(self, size: int = 2) -> None:
        self._size = max(1, size)
        self._pool: queue.Queue[Any] = queue.Queue()
        self._readings: dict[str, Any] = {}
        self._built = 0
        self._failed = False
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Build every parser now, so no request pays construction cost."""
        for _ in range(self._size):
            parser = self._build()
            if parser is None:
                return
            self._pool.put(parser)

    def _build(self) -> Any:
        if self._failed:
            return None
        try:
            # spaCy where a model is configured, Stanza otherwise. Measured on
            # lang-uk's benchmark the small spaCy model is not a compromise: it
            # scores 73.50% heteronym accuracy against Stanza's 72.96% and
            # 60.05 macro-F1 against 56.91, from 15 MB instead of ~500 MB and a
            # 0.8s load instead of ~30s. The size is what makes a pool viable,
            # and the pool is what stops one parser serialising the API.
            spacy_model = os.environ.get("SPACY_UK_MODEL")
            if spacy_model:
                from ukstress_ml.morphology import SpacyMorphologyTier

                parser = SpacyMorphologyTier(spacy_model)
            else:
                from ukstress_ml.morphology import MorphologyTier

                parser = MorphologyTier()
        except Exception as error:  # noqa: BLE001
            log.warning("morphology tier unavailable: %s", error)
            self._failed = True
            return None
        with self._lock:
            self._built += 1
        return parser

    @contextlib.contextmanager
    def acquire(self, timeout: float = 30.0) -> Any:
        """Check out a parser, building one on demand if the pool is cold."""
        if self._failed:
            yield None
            return
        try:
            parser = self._pool.get(timeout=timeout)
        except queue.Empty:
            parser = self._build() if self._built < self._size else None
            if parser is None:
                yield None
                return
        try:
            yield parser
        finally:
            self._pool.put(parser)

    @property
    def available(self) -> bool:
        return not self._failed

    def readings(self, form: str, parser: Any) -> Any:
        if form not in self._readings:
            self._readings[form] = parser.readings(form) if parser is not None else []
        return self._readings[form]


_morphology = _MorphologyPool(size=int(os.environ.get("MORPHOLOGY_WORKERS", "2")))


class _Memo:
    """A small thread-safe LRU of what the models said about a sentence.

    One API request asks the service about the same sentence up to four times:
    the cross-encoder, the tagger, the classifier, then the combiner, which
    re-runs all three to build its features. Each of those is a pure function
    of the sentence, the span and the candidate set, so the second asking is
    answered from here: the combiner costs its own small scorer instead of a
    second pass through three transformers.
    """

    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)


_MEMO_SIZE = int(os.environ.get("MODEL_MEMO_SIZE", "4096"))
_parses = _Memo(_MEMO_SIZE)


def _parse(parser: Any, sentence: str) -> dict:
    """The tagger's span map for a sentence, parsed once. Callers only read it."""
    parsed = _parses.get(sentence)
    if parsed is None:
        parsed = parser.parse(sentence)
        _parses.put(sentence, parsed)
    return parsed


def _target_key(t: Any) -> tuple:
    return (t.sentence, t.start, t.end, t.form, tuple(t.candidates))


def _memoised(memo: _Memo, keys: list[tuple], compute: Any) -> list[Any]:
    """Look every key up; compute the misses in one batch, in their order."""
    found = [memo.get(k) for k in keys]
    missing = [i for i, v in enumerate(found) if v is None]
    if missing:
        for i, value in zip(missing, compute(missing), strict=True):
            memo.put(keys[i], value)
            found[i] = value
    return found


#: Sentences whose morphological resolution is known and stable. Each is a
#: grammatical stress split — the tags separate the readings — so a working
#: tagger resolves them and a broken one resolves none. Kept small: this runs
#: on every readiness probe.
MORPHOLOGY_CANARY: tuple[tuple[str, str, int, int], ...] = (
    ("Довгі коси спадали на плечі.", "коси", 6, 10),
    ("Ці рідини потрібно змішати.", "рідини", 3, 9),
    ("Він не має руки.", "руки", 11, 15),
    ("Ці села лежать за річкою.", "села", 3, 7),
)



def preceding_tokens(sentence: str, parsed: dict, start: int) -> list:
    """`(upos, feats, text)` for the parsed tokens before this one, in order.

    The parse is a span map, so order is recovered by sorting; the text comes
    from the sentence rather than from the parse, which does not carry it.
    """
    out = []
    for (begin, end) in sorted(parsed):
        if begin >= start:
            break
        upos, feats = parsed[(begin, end)][0], parsed[(begin, end)][1]
        out.append((upos, feats, sentence[begin:end]))
    return out

def morphology_canary() -> dict[str, Any]:
    """Does the morphology tier still resolve anything at all?

    Reports the fraction of a fixed probe set the tier answers. Zero means the
    tier is dead — a tagger that loads without tagging, a pool that failed to
    build — which is a condition no request-level error will surface.
    """
    if not _morphology.available:
        return {"ok": False, "resolved": 0, "probed": len(MORPHOLOGY_CANARY),
                "detail": "morphology tier unavailable"}
    resolved = 0
    try:
        with _morphology.acquire() as parser:
            if parser is None:
                return {"ok": False, "resolved": 0, "probed": len(MORPHOLOGY_CANARY),
                        "detail": "no parser available"}
            for sentence, form, start, end in MORPHOLOGY_CANARY:
                parsed = parser.parse(sentence)
                readings = _morphology.readings(form, parser)
                parse = parsed.get((start, end))
                if not readings or not parse:
                    continue
                # A parse that carries no features is the silent-failure case.
                if not parse[1]:
                    continue
                if morphology_resolve(readings, parse[0], parse[1]) is not None:
                    resolved += 1
    except Exception as error:  # noqa: BLE001 - reported, never raised past readiness
        return {"ok": False, "resolved": resolved, "probed": len(MORPHOLOGY_CANARY),
                "detail": f"probe failed: {error}"}
    return {"ok": resolved > 0, "resolved": resolved,
            "probed": len(MORPHOLOGY_CANARY),
            "detail": "tier resolves nothing" if resolved == 0 else "ok"}


def _load_combiner(directory: str | None) -> tuple[Any, Any]:
    """The learned combiner and its tables, or (None, None) when unconfigured."""
    if not directory:
        return None, None
    path = Path(directory)
    if not (path / "model.pt").is_file():
        log.warning("combiner disabled: %s has no model.pt", path)
        return None, None
    try:
        from ukstress_ml.combiner import Assets, Combiner

        combiner = Combiner(path / "model.pt")
        assets = Assets.load(path)
    except Exception as error:  # noqa: BLE001 - a broken combiner must not take the service down
        log.warning("combiner disabled: %s", error)
        return None, None
    log.info("combiner loaded from %s (tau %.2f, %d readings)", path, combiner.tau, len(assets.readings))
    return combiner, assets


def create_app(
    model_dir: str = "models/v3-xenc", manifest_path: str | None = None
) -> Any:
    """Create the internal FastAPI app; model loading occurs once at startup."""
    from fastapi import FastAPI, HTTPException

    runtime = load_model(model_dir, manifest_path)
    # The token classifier is a separate, optional tier: unset, the service
    # is exactly what it was. Its coverage list travels with the model.
    token_resolver = load_token_resolver(os.environ.get("TOKEN_MODEL_DIR"))
    combiner, combiner_assets = _load_combiner(os.environ.get("COMBINER_DIR"))
    combiner_morph = None
    if combiner is not None:
        try:
            import pymorphy3

            combiner_morph = pymorphy3.MorphAnalyzer(lang="uk")
        except Exception as error:  # noqa: BLE001 - without it the modifier feature is silent
            log.warning("combiner: no pymorphy3, the modifier feature stays empty: %s", error)

    app = FastAPI(title="Internal contextual stress resolver", docs_url=None, redoc_url=None)

    xenc_memo, token_memo = _Memo(_MEMO_SIZE), _Memo(_MEMO_SIZE)

    def xenc_decisions(targets: list[Target]) -> list[dict[str, Any]]:
        """runtime.resolve, each target scored once; indexes are this call's."""
        found = _memoised(xenc_memo, [_target_key(t) for t in targets],
                          lambda missing: runtime.resolve([targets[i] for i in missing]))
        return [{**decision, "index": index} for index, decision in enumerate(found)]

    def token_decisions(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The classifier on every target, coverage aside, with probabilities.

        A target the classifier declines is remembered as False and left out.
        """
        def compute(missing: list[int]) -> list[Any]:
            batch = [{**targets[i], "index": k} for k, i in enumerate(missing)]
            answered = {d["index"]: d for d in token_resolver.resolve(
                batch, with_probabilities=True, ignore_coverage=True)}
            return [answered.get(k, False) for k in range(len(missing))]

        keys = [(t["sentence"], t["start"], t["end"], t["form"], tuple(t["candidates"])) for t in targets]
        return [{**d, "index": t["index"]} for t, d in zip(targets, _memoised(token_memo, keys, compute)) if d]

    @app.on_event("startup")
    def _warm_morphology() -> None:
        # Built here rather than on first use: construction loads models and
        # takes tens of seconds, which no request timeout should have to absorb.
        _morphology.warm()

    @app.post("/internal/v1/resolve")
    def resolve(body: ResolveBody) -> dict[str, Any]:
        if body.model_version != runtime.model_version or body.inventory_hash != runtime.inventory_hash:
            raise HTTPException(status_code=409, detail="model/inventory version mismatch")
        try:
            decisions = xenc_decisions(
                [
                    Target(
                        sentence=item.sentence,
                        start=item.start,
                        end=item.end,
                        form=item.form,
                        candidates=tuple(item.candidates),
                    )
                    for item in body.targets
                ]
            )
        except ServingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "model_version": runtime.model_version,
            "inventory_hash": runtime.inventory_hash,
            "decisions": decisions,
        }

    @app.post("/internal/v1/morphology")
    def morphology(body: MorphologyBody) -> dict[str, Any]:
        """Resolve ambiguities whose readings differ by grammar, not by sense.

        The cross-encoder is the wrong instrument for `се́ла` (nominative
        plural) against `села́` (genitive singular): it scores
        `(sentence, gloss)` similarity, and two glosses that differ only in a
        case label are nearly the same input. Measured on lang-uk's benchmark,
        routing 181 such forms to the model *lowered* heteronym accuracy by
        1.5 points against letting the parser decide them.

        This tier answers them from a morphological parse instead. Measured on
        the same benchmark it resolved 79% of the grammatical heteronyms it saw
        and was right on 74% of those, against a dictionary default that is a
        coin flip.
        """
        if not _morphology.available:
            raise HTTPException(status_code=503, detail="morphology tier unavailable")
        answers: list[dict[str, Any]] = []
        by_sentence: dict[str, list[Any]] = {}
        for item in body.targets:
            by_sentence.setdefault(item.sentence, []).append(item)

        with _morphology.acquire() as parser:
            if parser is None:
                raise HTTPException(status_code=503, detail="no parser available")
            for sentence, items in by_sentence.items():
                try:
                    parsed = _parse(parser, sentence)
                except Exception as error:  # noqa: BLE001
                    log.warning("morphology parse failed: %s", error)
                    continue
                for item in items:
                    readings = _morphology.readings(item.form, parser)
                    if not readings:
                        continue
                    parse = parsed.get((item.start, item.end))
                    if not parse:
                        continue
                    surface = sentence[item.start:item.end]
                    rule = ""
                    repaired = agreement_repair(readings, sentence, parsed, item.start)
                    resolution = (None if repaired is not None
                                  else morphology_resolve(readings, parse[0], parse[1]))
                    if repaired is not None:
                        accents, rule = list(repaired), "agreement"
                    elif resolution is not None:
                        accents = list(resolution.accents)
                    else:
                        # The tier declines a form whose readings carry
                        # identical tags, which is every `ви-` aspect pair:
                        # the trie records no aspect and the accent is the
                        # only thing telling `ви́ходити` from `вихо́дити`. A
                        # phase verb settles it without either.
                        chosen = imperfective_infinitive(
                            item.form, readings,
                            preceding_tokens(sentence, parsed, item.start))
                        if chosen is None:
                            continue
                        accents = list(chosen)
                        rule = "aspect"
                    if "-" not in surface and len(accents) > 1:
                        # Several accents on one token is the wordlist's "either
                        # stress is acceptable" notation, not two to emit.
                        accents = accents[:1]
                    answers.append({
                        "index": item.index,
                        "stressed": morphology_apply(surface, tuple(accents)),
                        # A counted form is a syntactic rule, not a tag match.
                        # The model was trained on a corpus whose silver labels
                        # call these tokens the other way — mining it for
                        # numeral phrases returns «три се́стри» — so a confident
                        # model answer here is confidently wrong, and the caller
                        # is told it may override one.
                        "rule": rule or ("counted_form" if len(parse) > 2 else ""),
                    })
        return {"decisions": answers}

    @app.post("/internal/v1/combine")
    def combine(body: CombineBody) -> dict[str, Any]:
        """Weigh every tier's opinion of each ambiguous token and decide.

        The caller sends the answer its tier order gave; this runs the tagger,
        the morphology tier's rule, the cross-encoder and the classifier on
        every target, builds the features the combiner was trained on (the
        shared `ukstress_ml.combiner`) and returns the combiner's reading and
        whether it departs from the caller's. Trained and measured in
        RESULTS.md, "A learned combiner over the tiers".
        """
        if combiner is None or token_resolver is None:
            raise HTTPException(status_code=503, detail="combiner unavailable")
        from ukstress.normalizer import stress_signature
        from ukstress_ml.combiner import Opinions, add_pipeline, features, modifier_agreement, softmax

        targets = body.targets
        # The tagger, once per sentence.
        tags: dict[str, list[tuple[int, int, str, str]]] = {}
        with _morphology.acquire() as parser:
            if parser is None:
                raise HTTPException(status_code=503, detail="no parser available")
            for sentence in {t.sentence for t in targets}:
                try:
                    parsed = _parse(parser, sentence)
                except Exception as error:  # noqa: BLE001
                    log.warning("combiner parse failed: %s", error)
                    parsed = {}
                tags[sentence] = sorted((a, b, v[0], v[1]) for (a, b), v in parsed.items())
            readings = {t.form: _morphology.readings(t.form, parser) for t in targets}
        # The cross-encoder, where the form is in its inventory with this candidate set.
        xenc: dict[int, dict[str, float]] = {}
        valid = []
        for t in targets:
            target = Target(sentence=t.sentence, start=t.start, end=t.end, form=t.form,
                            candidates=tuple(t.candidates))
            try:
                validate_target(target, runtime.manifest)
            except ServingError:
                continue
            valid.append((t.index, target))
        if valid:
            try:
                for (index, _), decision in zip(valid, xenc_decisions([v for _, v in valid])):
                    if decision.get("scores"):
                        xenc[index] = softmax(decision["scores"])
            except ServingError as error:
                log.warning("combiner cross-encoder failed: %s", error)
        # The classifier, on every form, with every candidate's probability.
        tok: dict[int, dict[str, float]] = {}
        for d in token_decisions([t.model_dump() for t in targets]):
            tok[d["index"]] = d["probabilities"]

        decisions = []
        for t in targets:
            spans = tags.get(t.sentence, [])
            position = next((k for k, span in enumerate(spans) if span[0] == t.start), None)
            tag = (spans[position][2], spans[position][3]) if position is not None else None
            previous = (t.sentence[spans[position - 1][0]:spans[position - 1][1]].lower()
                        if position else "")
            pick = None
            if tag is not None and readings.get(t.form):
                spans_dict = {(a, b): (u, f) for a, b, u, f in spans}
                repaired = agreement_repair(readings[t.form], t.sentence, spans_dict, t.start)
                resolution = None if repaired is not None else morphology_resolve(readings[t.form], tag[0], tag[1])
                accents = list(repaired) if repaired is not None else (
                    list(resolution.accents) if resolution is not None else [])
                if accents:
                    at = accents[0]
                    try:
                        pick = stress_signature(t.form[:at] + "\u0301" + t.form[at:])
                    except Exception:  # noqa: BLE001
                        pick = None
            opinions = Opinions(tag=tag, previous=previous, first=position in (None, 0),
                                morph_pick=pick, xenc=xenc.get(t.index, {}), tok=tok.get(t.index),
                                modifier=modifier_agreement(previous, combiner_morph))
            per_candidate = features(t.form, t.text, t.candidates, opinions, combiner_assets)
            add_pipeline(per_candidate, t.pipeline or None, t.status or None)
            signature, probability, departed = combiner.decide(t.candidates, per_candidate, t.pipeline or None,
                                                               t.status or None)
            decisions.append({"index": t.index, "signature": signature,
                              "probability": round(probability, 4), "departed": departed})
        return {"decisions": decisions}

    @app.post("/internal/v1/token")
    def token(body: TokenBody) -> dict[str, Any]:
        """Decide an ambiguous span from its sentence, with no gloss.

        Answers only for forms on the classifier's coverage list; the caller
        offers every span that would otherwise take the dictionary default and
        gets back decisions for the covered ones. Measured on Common Voice
        against audio gold, that slice moves from 78.5% to 93.8%; on lang-uk,
        whose sentences are built to force the rare reading, the same rule is
        −0.5 on heteronyms. Both numbers are in RESULTS.md.
        """
        if token_resolver is None:
            raise HTTPException(status_code=503, detail="token tier unavailable")
        covered = [item.model_dump() for item in body.targets if token_resolver.covers(item.form)]
        decisions = [{k: v for k, v in d.items() if k != "probabilities"} for d in token_decisions(covered)]
        return {"model_version": token_resolver.model_version, "decisions": decisions}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        # The backend is part of the served contract: two deployments of the
        # same model_version can differ in latency and, for a quantized graph,
        # in decisions. Report it rather than making callers guess.
        #
        # The morphology probe is here because a tagger that loads and tags
        # nothing raises no error anywhere. `uk_core_news_trf` 3.7.2 does
        # exactly that without its curated-transformers factory: the pipeline
        # constructs, every parse comes back featureless, the tier resolves
        # zero tokens, and heteronym accuracy falls 17 points in silence. A
        # deployment that cannot answer the canary must fail readiness rather
        # than serve degraded.
        probe = morphology_canary()
        payload: dict[str, Any] = {
            "status": "ready" if probe["ok"] else "degraded",
            "model_version": runtime.model_version,
            "backend": runtime.backend.name,
            "morphology": probe,
            "token_model": token_resolver.describe() if token_resolver else None,
            "combiner": ({"tau": combiner.tau, "features": len(combiner.names)}
                         if combiner is not None else None),
        }
        if not probe["ok"]:
            raise HTTPException(status_code=503, detail=payload)
        return payload

    return app
