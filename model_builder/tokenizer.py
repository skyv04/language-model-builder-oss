"""Tokenizers: a fast byte-level BPE trained on the project's corpus, or GPT-2's BPE.

Both are byte-level, so any text round-trips.  Every project tokenizer also knows the
special tokens used to separate documents and to mark the roles of a chat transcript.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from . import paths

END_OF_TEXT = "<|endoftext|>"
SYSTEM, USER, ASSISTANT, END = "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"
CHAT_TOKENS = [SYSTEM, USER, ASSISTANT, END]
SPECIAL_TOKENS = [END_OF_TEXT, *CHAT_TOKENS]
BPE_SIZES = (4096, 10000, 16000)
GPT2_VOCAB = 50257 + len(CHAT_TOKENS)
GPT2_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"
_GPT2_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's printable byte-to-character table used by byte-level BPE."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    codes = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


def label(kind: str, vocab_size: int) -> str:
    if kind == "gpt2":
        return "GPT-2 BPE"
    return f"Fast {round(vocab_size / 1000)}K BPE" if vocab_size >= 1000 else f"Fast {vocab_size} BPE"


class TextTokenizer:
    """Thin wrapper over a Hugging Face `tokenizers.Tokenizer` with chat-aware helpers."""

    def __init__(self, tokenizer: Tokenizer, kind: str) -> None:
        self.tokenizer = tokenizer
        self.kind = kind
        missing = [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
        if missing:
            tokenizer.add_special_tokens(missing)
        self.eot_id = tokenizer.token_to_id(END_OF_TEXT)
        self.system_id = tokenizer.token_to_id(SYSTEM)
        self.user_id = tokenizer.token_to_id(USER)
        self.assistant_id = tokenizer.token_to_id(ASSISTANT)
        self.end_id = tokenizer.token_to_id(END)
        self.special_ids = {self.eot_id, self.system_id, self.user_id, self.assistant_id, self.end_id}
        self._pieces: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def label(self) -> str:
        return label(self.kind, self.vocab_size)

    @staticmethod
    def sanitize(text: str) -> str:
        """Stop literal special-token strings inside user text from becoming control tokens."""
        return text.replace("<|", "<\u200b|") if "<|" in text else text

    def encode(self, text: str, special: bool = False) -> list[int]:
        """Encode text; special-token strings are only honoured when `special` is true."""
        if not special:
            text = self.sanitize(text)
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [item.ids for item in self.tokenizer.encode_batch([self.sanitize(text) for text in texts], add_special_tokens=False)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def piece_bytes(self, token_id: int) -> bytes:
        token = self.tokenizer.id_to_token(int(token_id))
        if token is None:
            return b""
        if token_id in self.special_ids or token in SPECIAL_TOKENS:
            return token.encode()
        table = _byte_decoder()
        try:
            return bytes(table[char] for char in token)
        except KeyError:
            return token.encode()

    def piece(self, token_id: int) -> str:
        """Display text of one token (partial UTF-8 sequences show as U+FFFD)."""
        token_id = int(token_id)
        cached = self._pieces.get(token_id)
        if cached is None:
            cached = self.piece_bytes(token_id).decode("utf-8", errors="replace")
            if len(self._pieces) < 200_000:
                self._pieces[token_id] = cached
        return cached

    def tokens(self, text: str, special: bool = True) -> list[dict]:
        return [{"id": token_id, "text": self.piece(token_id)} for token_id in self.encode(text, special=special)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        self.tokenizer.save(str(temporary))
        payload = json.loads(temporary.read_text(encoding="utf-8"))
        payload["lmb_kind"] = self.kind
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TextTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = payload.pop("lmb_kind", "bpe")
        return cls(Tokenizer.from_str(json.dumps(payload)), kind)

    def describe(self) -> dict:
        return {"kind": self.kind, "vocab_size": self.vocab_size, "label": self.label}


def train_bpe(texts: Iterable[str], vocab_size: int, length: int | None = None) -> TextTokenizer:
    """Train a byte-level BPE whose first ids are the special tokens."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)
    return TextTokenizer(tokenizer, "bpe")


def limited_texts(documents: Iterable[str], max_chars: int) -> Iterator[str]:
    """Yield documents until roughly `max_chars` characters have been produced."""
    used = 0
    for document in documents:
        if used >= max_chars:
            return
        piece = document[: max_chars - used]
        used += len(piece)
        yield piece


def gpt2_path() -> Path:
    return paths.cache_dir() / "gpt2" / "tokenizer.json"


def ensure_gpt2(progress=None) -> Path:
    """Download GPT-2's tokenizer.json once (≈1.4 MB, Modified MIT licence) into the cache."""
    path = gpt2_path()
    with _GPT2_LOCK:
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        request = urllib.request.Request(GPT2_URL, headers={"User-Agent": "language-model-builder-oss"})
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0) or None
            received = 0
            while chunk := response.read(1 << 16):
                handle.write(chunk)
                received += len(chunk)
                if progress:
                    progress(received, total)
        Tokenizer.from_file(str(temporary))
        temporary.replace(path)
    return path


def load_gpt2() -> TextTokenizer:
    return TextTokenizer(Tokenizer.from_file(str(ensure_gpt2())), "gpt2")


@lru_cache(maxsize=1)
def shared_gpt2() -> TextTokenizer:
    return load_gpt2()


def byte_tokens(text: str) -> list[dict]:
    """Plain UTF-8 bytes, for teaching comparisons."""
    result = []
    for byte in text.encode("utf-8"):
        char = chr(byte)
        shown = char if 32 <= byte < 127 else ("\n" if byte == 10 else f"\\x{byte:02x}")
        result.append({"id": byte, "text": shown})
    return result
