"""Turn datasets into token streams and training batches.

Pre-training text becomes one long token stream per split (documents separated by
<|endoftext|>), stored as a flat binary file and read through numpy memmaps.  Chat data
becomes padded rows whose targets are -100 everywhere except the assistant replies.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenizer import TextTokenizer

IGNORE = -100
VAL_EVERY = 50  # one document in fifty goes to the validation split
DEFAULT_SYSTEM = "You are a friendly assistant. Give clear, short answers."
MAX_CORPUS_TOKENS = 400_000_000


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 65536 else np.dtype(np.uint32)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _batched(items: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_corpus(
    documents: Iterable[str],
    tokenizer: TextTokenizer,
    folder: Path,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    meta: dict | None = None,
) -> dict:
    """Tokenise documents into folder/train.bin and folder/val.bin; returns the metadata."""
    folder.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(tokenizer.vocab_size)
    files = {split: open(folder / f"{split}.bin.tmp", "wb") for split in ("train", "val")}
    counts = {"train": 0, "val": 0}
    document_count = 0
    characters = 0
    try:
        for batch in _batched(documents, 256):
            if should_stop and should_stop():
                raise InterruptedError("stopped while tokenising")
            for text, ids in zip(batch, tokenizer.encode_batch(batch)):
                if not ids:
                    continue
                split = "val" if document_count % VAL_EVERY == VAL_EVERY - 1 else "train"
                row = np.asarray([tokenizer.eot_id, *ids], dtype=dtype)
                row.tofile(files[split])
                counts[split] += len(row)
                document_count += 1
                characters += len(text)
            if progress:
                progress(document_count, counts["train"] + counts["val"])
            if counts["train"] + counts["val"] >= MAX_CORPUS_TOKENS:
                break
    except BaseException:
        for handle in files.values():
            handle.close()
        for split in files:
            (folder / f"{split}.bin.tmp").unlink(missing_ok=True)
        raise
    for handle in files.values():
        handle.close()
    if counts["val"] == 0 and counts["train"] > 0:
        # Tiny corpora: hold out the tail of the training stream instead.
        train = np.fromfile(folder / "train.bin.tmp", dtype=dtype)
        cut = max(1, len(train) // 20)
        train[-cut:].tofile(folder / "val.bin.tmp")
        train[:-cut].tofile(folder / "train.bin.tmp")
        counts = {"train": len(train) - cut, "val": cut}
    for split in files:
        os.replace(folder / f"{split}.bin.tmp", folder / f"{split}.bin")
    info = {
        "dtype": dtype.name,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "documents": document_count,
        "characters": characters,
        **(meta or {}),
    }
    _atomic_json(folder / "corpus.json", info)
    return info


def corpus_info(folder: Path) -> dict | None:
    path = folder / "corpus.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class TokenStream:
    """Random (tokens, targets) windows from a tokenised split (nanoGPT-style sampling)."""

    def __init__(self, path: Path, dtype: str, context: int, seed: int = 0) -> None:
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.context = context
        if len(self.data) < 2:
            raise ValueError(f"{path.name} holds too few tokens to train on")
        self.window = min(context, len(self.data) - 1)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(0, len(self.data) - self.window, size=batch_size)
        rows = np.stack([np.asarray(self.data[start : start + self.window + 1], dtype=np.int64) for start in starts])
        return rows[:, :-1], rows[:, 1:]

    def fixed_batches(self, batch_size: int, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Deterministic evaluation batches spread evenly over the split."""
        span = len(self.data) - self.window
        total = batch_size * count
        starts = np.linspace(0, max(0, span - 1), num=total).astype(np.int64)
        batches = []
        for index in range(count):
            chunk = starts[index * batch_size : (index + 1) * batch_size]
            rows = np.stack([np.asarray(self.data[start : start + self.window + 1], dtype=np.int64) for start in chunk])
            batches.append((rows[:, :-1], rows[:, 1:]))
        return batches


# ---------- chat formatting ----------


def normalise_messages(messages: list[dict], system: str | None = None) -> list[dict]:
    cleaned = [
        {"role": str(message.get("role", "")).strip().lower(), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict)
    ]
    cleaned = [message for message in cleaned if message["role"] in {"system", "user", "assistant"}]
    if not cleaned or cleaned[0]["role"] != "system":
        cleaned.insert(0, {"role": "system", "content": system if system is not None else DEFAULT_SYSTEM})
    elif system is not None:
        cleaned[0] = {"role": "system", "content": system}
    return cleaned


def encode_chat(
    tokenizer: TextTokenizer,
    messages: list[dict],
    add_generation_prompt: bool = False,
    system: str | None = None,
) -> tuple[list[int], list[int]]:
    """Token ids for a chat transcript plus targets that only cover assistant replies.

    Format: <|role|>\\n content <|end|>\\n for every message.  Targets hold the next token id
    where the model should learn (assistant content and its <|end|>) and -100 elsewhere.
    """
    role_ids = {"system": tokenizer.system_id, "user": tokenizer.user_id, "assistant": tokenizer.assistant_id}
    newline = tokenizer.encode("\n")
    ids: list[int] = []
    learn: list[bool] = []
    for message in normalise_messages(messages, system):
        header = [role_ids[message["role"]], *newline]
        body = tokenizer.encode(message["content"].strip()) + [tokenizer.end_id]
        ids += header
        learn += [False] * len(header)
        ids += body
        learn += [message["role"] == "assistant"] * len(body)
        ids += newline
        learn += [False] * len(newline)
    if add_generation_prompt:
        header = [tokenizer.assistant_id, *newline]
        ids += header
        learn += [False] * len(header)
    targets = [IGNORE] * len(ids)
    for position in range(len(ids) - 1):
        if learn[position + 1]:
            targets[position] = ids[position + 1]
    return ids, targets


def chat_prompt_text(tokenizer: TextTokenizer, messages: list[dict], system: str | None = None) -> str:
    ids, _ = encode_chat(tokenizer, messages, add_generation_prompt=True, system=system)
    return tokenizer.decode(ids)


def build_conversations(
    conversations: Iterable[dict],
    tokenizer: TextTokenizer,
    folder: Path,
    context: int,
    progress: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    meta: dict | None = None,
) -> dict:
    """Encode chat rows (truncated to the context) into flat arrays with offsets."""
    folder.mkdir(parents=True, exist_ok=True)
    tokens: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lengths: list[int] = []
    learnable = 0
    skipped = 0
    for index, row in enumerate(conversations):
        if should_stop and should_stop() and index % 64 == 0:
            raise InterruptedError("stopped while preparing conversations")
        messages = row.get("messages") if isinstance(row, dict) else None
        if not messages:
            skipped += 1
            continue
        ids, target = encode_chat(tokenizer, messages)
        # targets[t] labels position t (the next token), so both lists share the model's context.
        ids, target = ids[:context], target[:context]
        usable = sum(1 for value in target if value != IGNORE)
        if usable == 0:
            skipped += 1
            continue
        tokens.append(np.asarray(ids, dtype=np.int32))
        targets.append(np.asarray(target, dtype=np.int32))
        lengths.append(len(ids))
        learnable += usable
        if progress and len(lengths) % 500 == 0:
            progress(len(lengths))
    if not lengths:
        raise ValueError("the fine-tuning dataset has no usable conversations")
    order = list(range(len(lengths)))
    random.Random(1234).shuffle(order)
    val_count = max(1, min(len(order) // 20, 500)) if len(order) > 1 else 0
    splits = {"val": order[:val_count], "train": order[val_count:] or order}
    for split, indices in splits.items():
        if not indices:
            continue
        offsets = np.zeros(len(indices) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([lengths[index] for index in indices])
        np.save(folder / f"{split}_tokens.npy", np.concatenate([tokens[index] for index in indices]))
        np.save(folder / f"{split}_targets.npy", np.concatenate([targets[index] for index in indices]))
        np.save(folder / f"{split}_offsets.npy", offsets)
    info = {
        "conversations": len(lengths),
        "train_rows": len(splits["train"]),
        "val_rows": len(splits["val"]),
        "tokens": int(sum(lengths)),
        "learnable_tokens": int(learnable),
        "skipped": skipped,
        "context": context,
        **(meta or {}),
    }
    _atomic_json(folder / "conversations.json", info)
    return info


def conversations_info(folder: Path) -> dict | None:
    try:
        return json.loads((folder / "conversations.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class ChatRows:
    """Random padded batches of encoded conversations."""

    def __init__(self, folder: Path, split: str, context: int, pad_id: int, seed: int = 0) -> None:
        self.tokens = np.load(folder / f"{split}_tokens.npy", mmap_mode="r")
        self.targets = np.load(folder / f"{split}_targets.npy", mmap_mode="r")
        self.offsets = np.load(folder / f"{split}_offsets.npy")
        self.context = context
        self.pad_id = pad_id
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def rows(self, indices, fixed_length: bool = False) -> tuple[np.ndarray, np.ndarray]:
        pieces = []
        for index in indices:
            start, end = int(self.offsets[index]), int(self.offsets[index + 1])
            pieces.append((np.asarray(self.tokens[start:end]), np.asarray(self.targets[start:end])))
        width = self.context if fixed_length else min(self.context, max(len(ids) for ids, _ in pieces))
        batch_tokens = np.full((len(pieces), width), self.pad_id, dtype=np.int64)
        batch_targets = np.full((len(pieces), width), IGNORE, dtype=np.int64)
        for row, (ids, target) in enumerate(pieces):
            length = min(width, len(ids))
            batch_tokens[row, :length] = ids[:length]
            batch_targets[row, :length] = target[:length]
        return batch_tokens, batch_targets

    def batch(self, batch_size: int, fixed_length: bool = False) -> tuple[np.ndarray, np.ndarray]:
        return self.rows(self.rng.integers(0, len(self), size=batch_size), fixed_length)

    def fixed_batches(self, batch_size: int, count: int, fixed_length: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
        total = len(self)
        indices = [index % total for index in range(batch_size * count)]
        return [self.rows(indices[i * batch_size : (i + 1) * batch_size], fixed_length) for i in range(count)]


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


def encode_preference(tokenizer: TextTokenizer, pair: PreferencePair, context: int) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
    """Encode the chosen and rejected answers as chat rows sharing the same prompt."""
    encoded = []
    for answer in (pair.chosen, pair.rejected):
        ids, targets = encode_chat(tokenizer, [{"role": "user", "content": pair.prompt}, {"role": "assistant", "content": answer}])
        encoded.append((ids[:context], targets[:context]))
    return encoded[0], encoded[1]


def pad_rows(rows: list[tuple[list[int], list[int]]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(ids) for ids, _ in rows)
    tokens = np.full((len(rows), width), pad_id, dtype=np.int64)
    targets = np.full((len(rows), width), IGNORE, dtype=np.int64)
    for index, (ids, target) in enumerate(rows):
        tokens[index, : len(ids)] = ids
        targets[index, : len(target)] = target
    return tokens, targets
