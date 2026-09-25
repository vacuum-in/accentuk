"""Tokenizers for the marked-span model.

Source: SentencePiece unigram over Ukrainian sentences with the span markers as
user-defined symbols. Target: character level, because targets are single words
and the combining acute must stay a first-class unit the decoder emits
explicitly.

Both are trained on the training split only and versioned by checksum with the
model.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

from ukstress_ml.corpus import CLOSE_MARK, OPEN_MARK

PAD, EOS, UNK, BOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "</s>", "<unk>", "<s>"]


@dataclass
class SourceTokenizer:
    """SentencePiece unigram model over encoded source strings."""

    processor: spm.SentencePieceProcessor
    model_path: Path

    @property
    def vocab_size(self) -> int:
        return int(self.processor.get_piece_size())

    @classmethod
    def train(cls, texts: list[str], output_dir: Path, vocab_size: int = 8000) -> SourceTokenizer:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / "source"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            for text in texts:
                fh.write(unicodedata.normalize("NFD", text).replace("\n", " ") + "\n")
            corpus_path = fh.name
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            pad_id=PAD,
            eos_id=EOS,
            unk_id=UNK,
            bos_id=BOS,
            user_defined_symbols=[OPEN_MARK, CLOSE_MARK],
            normalization_rule_name="identity",
            train_extremely_large_corpus=False,
            # Treat vocab_size as a ceiling: a small corpus simply yields a
            # smaller vocabulary instead of failing the run.
            hard_vocab_limit=False,
            minloglevel=2,
        )
        Path(corpus_path).unlink(missing_ok=True)
        return cls.load(output_dir)

    @classmethod
    def load(cls, output_dir: Path) -> SourceTokenizer:
        model_path = output_dir / "source.model"
        processor = spm.SentencePieceProcessor()
        processor.load(str(model_path))
        return cls(processor=processor, model_path=model_path)

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = self.processor.encode(unicodedata.normalize("NFD", text), out_type=int)
        ids = ids[: max_length - 1]
        return [*ids, EOS]

    def checksum(self) -> str:
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()


@dataclass
class TargetVocab:
    """Character vocabulary over stressed target words."""

    chars: list[str]

    @property
    def index(self) -> dict[str, int]:
        return {char: position for position, char in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def build(cls, targets: list[str]) -> TargetVocab:
        seen: set[str] = set()
        for target in targets:
            seen.update(unicodedata.normalize("NFD", target))
        return cls(chars=[*SPECIALS, *sorted(seen)])

    def encode(self, text: str, max_length: int) -> list[int]:
        index = self.index
        ids = [index.get(char, UNK) for char in unicodedata.normalize("NFD", text)]
        ids = ids[: max_length - 1]
        return [*ids, EOS]

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for token in ids:
            if token in (EOS, PAD, BOS):
                continue
            if 0 <= token < len(self.chars):
                out.append(self.chars[token])
        return "".join(out)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> TargetVocab:
        return cls(chars=json.loads(path.read_text(encoding="utf-8"))["chars"])

    def checksum(self) -> str:
        payload = json.dumps(self.chars, ensure_ascii=False, sort_keys=False).encode()
        return hashlib.sha256(payload).hexdigest()
