"""Tokenizers: a byte-level one for random test models and a Hugging Face tokenizer.json loader."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer as HFTokenizerImpl


class Tokenizer(Protocol):
    """What the engine and server need from a tokenizer."""

    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """Text to token ids."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Token ids to text."""
        ...


class ByteTokenizer:
    """One token per UTF-8 byte (ids 0-255); id 256 is end of sequence."""

    vocab_size = 257
    eos_token_id: int | None = 256

    def encode(self, text: str) -> list[int]:
        """Encode text as its UTF-8 bytes."""
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """Decode bytes, dropping ids that are not bytes."""
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class HFTokenizer:
    """A tokenizer.json file loaded with the `tokenizers` library."""

    def __init__(self, path: Path, eos_token: str = "<|endoftext|>") -> None:  # noqa: S107
        """Load the tokenizer and look up its end-of-sequence id."""
        self._tok = HFTokenizerImpl.from_file(str(path))
        self.eos_token_id: int | None = self._tok.token_to_id(eos_token)

    def encode(self, text: str) -> list[int]:
        """Encode without adding special tokens."""
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        """Decode, skipping special tokens."""
        return str(self._tok.decode(ids, skip_special_tokens=True))
