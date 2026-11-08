"""Dataset catalog, on-demand downloads, previews and custom imports.

Every dataset lives in ``paths.datasets_dir()/<id>/`` as one normalised JSON-lines file and a
``meta.json`` file:

* pre-training corpora use ``documents.jsonl`` with one ``{"text": ...}`` object per document;
* fine-tuning sets use ``conversations.jsonl`` with ``{"messages": [{"role", "content"}, ...]}``
  objects whose roles are ``system``, ``user`` or ``assistant``.

Built-in datasets are fetched from pinned upstream revisions and trimmed to a phone-friendly
size. A download is staged in a hidden sibling directory and renamed into place only when it is
complete, so a cancelled or failed download never leaves a partial dataset behind.
"""

from __future__ import annotations

import codecs
import csv
import fcntl
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import ssl
import struct
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator

from . import __version__, paths

PRETRAIN = "pretrain"
FINETUNE = "finetune"
KINDS = (PRETRAIN, FINETUNE)

DOCUMENTS_FILE = "documents.jsonl"
CONVERSATIONS_FILE = "conversations.jsonl"
META_FILE = "meta.json"
CUSTOM_PREFIX = "custom-"
FORMAT_VERSION = 1

MB = 1_000_000
MAX_IMPORT_BYTES = 200 * 1024 * 1024
PREVIEW_CHARS = 8_000
PREVIEW_DOCUMENTS = 6
PREVIEW_EXAMPLES = 3

USER_AGENT = (
    f"LanguageModelBuilder-OSS/{__version__} "
    "(unofficial open-source edition; on-demand dataset download) Python-urllib"
)
HTTP_TIMEOUT = 20.0
HTTP_RETRIES = 4
RETRY_DELAY = 2.0  # seconds before the first retry; doubles each time
CHUNK_SIZE = 64 * 1024
REPORT_INTERVAL = 0.25
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

ProgressCallback = Callable[[int, "int | None", str], None]

_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")
_ROLE_ALIASES = {
    "system": "system", "developer": "system",
    "user": "user", "human": "user", "prompter": "user", "question": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
    "chatgpt": "assistant", "ai": "assistant", "answer": "assistant",
}
_SKIPPED_ROLES = frozenset({"tool", "function", "ipython"})


class DownloadCancelled(Exception):
    """Raised by :func:`download` when its cancel event is set; partial files are removed."""


class DownloadInProgress(RuntimeError):
    """Raised when the same dataset is already being downloaded (by this or another process)."""


class DownloadError(RuntimeError):
    """A download failed: network trouble, an unexpected response or a corrupt file."""


class DatasetNotDownloaded(FileNotFoundError):
    """The dataset exists in the catalog but has not been downloaded yet."""


@dataclass(frozen=True)
class DatasetInfo:
    """Catalog entry; JSON-serialisable via :func:`dataclasses.asdict`.

    ``size_label`` describes the normalised data the model reads; ``approx_bytes`` is roughly
    how much :func:`download` fetches over the network (for custom imports: the stored size).
    """

    id: str
    kind: str
    name: str
    tagline: str
    description: str
    license: str
    license_url: str
    source_label: str
    source_url: str
    paper_label: str | None
    paper_url: str | None
    size_label: str
    approx_bytes: int
    icon: str
    color: str
    recommended: bool
    custom: bool = False


@dataclass(frozen=True)
class _Recipe:
    build: Callable[["_Job"], Iterable[dict]]
    max_bytes: int = 48 * MB
    max_records: int | None = None


# --------------------------------------------------------------------------- text helpers

_ODD_SPACES = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _ODD_SPACES.sub(" ", text)
    text = _TRAILING_SPACE.sub("\n", text)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _content_text(content: Any) -> str:
    """Message content as plain text; accepts strings and OpenAI-style lists of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _clean_messages(messages: Iterable[Any]) -> list[dict] | None:
    """Normalise roles and content; None unless a user turn and an assistant reply remain."""
    cleaned: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        raw_role = str(message.get("role", message.get("from", ""))).strip().lower()
        if raw_role in _SKIPPED_ROLES:
            continue
        role = _ROLE_ALIASES.get(raw_role)
        if role is None:
            return None
        content = _clean_text(_content_text(message.get("content", message.get("value"))))
        if content:
            cleaned.append({"role": role, "content": content})
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()
    roles = {message["role"] for message in cleaned}
    return cleaned if {"user", "assistant"} <= roles else None


def _shorten(text: str, limit: int) -> str:
    """``text`` cut at a word boundary to at most ``limit`` characters, ending in " …"."""
    if len(text) <= limit:
        return text
    cut = text[: max(1, limit - 2)]
    space = cut.rfind(" ", int(limit * 0.8))
    return (cut[:space] if space > 0 else cut).rstrip() + " …"


def _human_bytes(size: int) -> str:
    if size >= 10 * MB:
        return f"{size / MB:.0f} MB"
    if size >= MB:
        return f"{size / MB:.1f} MB"
    return f"{max(1, round(size / 1000))} KB"


def _human_count(count: int) -> str:
    if count >= 10_000:
        return f"{count / 1000:.0f}k"
    if count >= 1_000:
        return f"{count / 1000:.1f}k"
    return str(count)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- downloading

def _urlopen(request: urllib.request.Request, timeout: float):
    """Single point of network access (tests replace it)."""
    return urllib.request.urlopen(request, timeout=timeout)


class _Truncated(Exception):
    """The connection ended before the requested bytes arrived."""


def _content_range(header: str | None) -> tuple[int, int | None] | None:
    match = re.match(r"bytes (\d+)-\d+/(\d+|\*)", header or "")
    if not match:
        return None
    return int(match.group(1)), None if match.group(2) == "*" else int(match.group(2))


class _Job:
    """One download in progress: HTTP helpers, progress accounting and cancellation."""

    def __init__(self, info: DatasetInfo, workdir: Path, progress: ProgressCallback | None,
                 cancel: threading.Event | None):
        self.info = info
        self.workdir = workdir
        self.done = 0
        self.estimate = info.approx_bytes or None
        self.budget: int | None = None  # normalised bytes wanted (partial downloads)
        self.sources: list[str] = []
        self._progress = progress
        self._cancel = cancel or threading.Event()
        self._last_report = 0.0

    def check(self) -> None:
        if self._cancel.is_set():
            raise DownloadCancelled(f"The download of {self.info.name} was cancelled.")

    def pause(self, seconds: float) -> None:
        if self._cancel.wait(seconds):
            self.check()

    def report(self, message: str, force: bool = False) -> None:
        if self._progress is None:
            return
        now = time.monotonic()
        if not force and now - self._last_report < REPORT_INTERVAL:
            return
        self._last_report = now
        total = max(self.estimate, self.done) if self.estimate else None
        self._progress(self.done, total, message)

    def fetch_file(self, url: str, name: str, *, start: int = 0, end: int | None = None,
                   size: int | None = None, sha256: str | None = None, label: str = "") -> Path:
        """Download bytes ``[start, end)`` of ``url`` (to EOF if ``end`` is None) into the job."""
        path = self.workdir / name
        digest = hashlib.sha256() if sha256 else None
        with open(path, "wb") as handle:
            written = self._stream(url, handle, start, end, digest, label or self.info.name)
        if size is not None and start == 0 and end is None and written != size:
            raise DownloadError(f"{url} returned {written:,} bytes instead of {size:,}.")
        if digest is not None and digest.hexdigest() != sha256:
            raise DownloadError(f"{url} failed its checksum; the file may be corrupt or changed.")
        return path

    def fetch_bytes(self, url: str, start: int, end: int, label: str = "") -> bytes:
        buffer = io.BytesIO()
        self._stream(url, buffer, start, end, None, label or self.info.name)
        return buffer.getvalue()

    def fetch_first(self, urls: Iterable[str], name: str, label: str = "") -> Path:
        """Try mirrors in order until one succeeds."""
        error: Exception | None = None
        for url in urls:
            known = url in self.sources
            try:
                return self.fetch_file(url, name, label=label)
            except DownloadError as problem:
                error = problem
                if not known and url in self.sources:
                    self.sources.remove(url)  # only list mirrors that actually served data
        raise DownloadError(f"No mirror could provide {name}: {error}")

    def _stream(self, url: str, out: BinaryIO, start: int, end: int | None,
                digest: Any, label: str) -> int:
        if url not in self.sources:
            self.sources.append(url)
        written = failures = 0
        progress_at_failure = -1
        while True:
            self.check()
            offset = start + written
            if end is not None and offset >= end:
                return written
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            ranged = offset > 0 or end is not None
            if ranged:
                headers["Range"] = f"bytes={offset}-{'' if end is None else end - 1}"
            try:
                with _urlopen(urllib.request.Request(url, headers=headers), HTTP_TIMEOUT) as response:
                    status = getattr(response, "status", None) or 200
                    discard = 0
                    if ranged and status == 206:
                        answered = _content_range(response.headers.get("Content-Range"))
                        if answered is not None and answered[0] != offset:
                            raise DownloadError(f"{url} answered with an unexpected byte range.")
                        if answered is not None and answered[1] is not None and end is not None:
                            end = min(end, answered[1])
                    else:
                        discard = offset  # the server ignored Range: skip what we already have
                        length = response.headers.get("Content-Length")
                        if end is not None and length and length.isdigit():
                            end = min(end, int(length))
                    while True:
                        self.check()
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        if discard:
                            skipped = min(discard, len(chunk))
                            chunk, discard = chunk[skipped:], discard - skipped
                        if end is not None:
                            chunk = chunk[: end - start - written]
                        if not chunk:
                            if end is not None and start + written >= end:
                                return written
                            continue
                        out.write(chunk)
                        if digest is not None:
                            digest.update(chunk)
                        written += len(chunk)
                        self.done += len(chunk)
                        self.report(f"Downloading {label}")
                        if end is not None and start + written >= end:
                            return written
                if end is None:
                    return written
                raise _Truncated("the connection closed early")
            except urllib.error.HTTPError as error:
                if error.code == 416 and end is None and written:
                    return written
                if error.code not in RETRY_STATUSES:
                    raise DownloadError(f"Could not download {label}: the server answered "
                                        f"HTTP {error.code} for {url}.") from error
                problem: Exception = error
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError, ssl.SSLError, _Truncated) as error:
                problem = error
            if written != progress_at_failure:
                failures, progress_at_failure = 0, written
            failures += 1
            if failures > HTTP_RETRIES:
                reason = getattr(problem, "reason", None) or problem
                raise DownloadError(f"Could not download {label}: {reason}") from problem
            self.report(f"Connection problem; retrying ({failures}/{HTTP_RETRIES})", force=True)
            self.pause(min(RETRY_DELAY * 2.0 ** (failures - 1), 20.0))


class _SplicedFile(io.RawIOBase):
    """Seekable read-only view of a remote file built from a downloaded prefix and its tail."""

    def __init__(self, size: int, tail: bytes, prefix: Path | None = None, prefix_length: int = 0):
        super().__init__()
        self._size = size
        self._tail = tail
        self._tail_start = size - len(tail)
        self._prefix = open(prefix, "rb") if prefix is not None else None
        self._prefix_length = prefix_length if prefix is not None else 0
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        self._position = max(0, base + offset)
        return self._position

    def read(self, size: int = -1) -> bytes:
        remaining = self._size - self._position if size is None or size < 0 else size
        remaining = max(0, min(remaining, self._size - self._position))
        pieces = []
        while remaining > 0:
            position = self._position
            if position < self._prefix_length:
                assert self._prefix is not None
                self._prefix.seek(position)
                piece = self._prefix.read(min(remaining, self._prefix_length - position))
            elif position >= self._tail_start:
                offset = position - self._tail_start
                piece = self._tail[offset: offset + remaining]
            else:
                raise OSError(f"Byte {position:,} of the remote file was not downloaded.")
            if not piece:
                raise OSError("Unexpected end of the downloaded data.")
            pieces.append(piece)
            self._position += len(piece)
            remaining -= len(piece)
        return b"".join(pieces)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if self._prefix is not None:
            self._prefix.close()
            self._prefix = None
        super().close()


def _column_chunk_end(column: Any) -> int:
    offsets = [value for value in (column.dictionary_page_offset, column.data_page_offset) if value]
    return min(offsets) + column.total_compressed_size


def _parquet_rows(job: _Job, url: str, size: int, columns: list[str], *,
                  budget: int | None = None, sha256: str | None = None) -> Iterator[dict]:
    """Rows of a remote Parquet file.

    With a ``budget`` (normalised bytes wanted) only the footer and the leading row groups
    needed to cover it are fetched, using HTTP range requests; otherwise the whole file is.
    """
    import pyarrow.parquet as pq

    tail = b""
    groups: int | None = None
    prefix_end = size
    if budget is not None:
        tail = job.fetch_bytes(url, max(0, size - 65_536), size, "file index")
        if len(tail) < 12 or tail[-4:] != b"PAR1":
            raise DownloadError(f"{url} is not a Parquet file.")
        footer_length = struct.unpack("<I", tail[-8:-4])[0] + 8
        if footer_length > len(tail):
            tail = job.fetch_bytes(url, size - footer_length, size - len(tail), "file index") + tail
        with _SplicedFile(size, tail) as footer:
            metadata = pq.ParquetFile(footer).metadata
        wanted = 0.0
        for index in range(metadata.num_row_groups):
            group = metadata.row_group(index)
            chunks = [group.column(j) for j in range(group.num_columns)]
            wanted += sum(c.total_uncompressed_size for c in chunks
                          if c.path_in_schema.split(".")[0] in columns)
            prefix_end = max(_column_chunk_end(c) for c in chunks)
            groups = index + 1
            if wanted >= budget * 1.1:
                break
        if prefix_end >= size - len(tail):
            groups, prefix_end = None, size
    job.estimate = job.done + prefix_end
    if prefix_end == size:
        local = job.fetch_file(url, "source.parquet", size=size, sha256=sha256)
        handle: Any = open(local, "rb")
    else:
        local = job.fetch_file(url, "source.parquet", end=prefix_end)
        handle = _SplicedFile(size, tail, local, prefix_end)
    job.report("Reading the downloaded data", force=True)
    try:
        reader = pq.ParquetFile(handle)
        for index in range(groups if groups is not None else reader.num_row_groups):
            job.check()
            yield from reader.read_row_group(index, columns=columns).to_pylist()
    finally:
        handle.close()


def _gzip_lines(path: Path) -> Iterator[str]:
    """Complete lines of a (possibly truncated) gzip file."""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            while chunk:
                data = pending + decompressor.decompress(chunk)
                chunk = b""
                if decompressor.eof:
                    chunk = decompressor.unused_data
                    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
                *lines, pending = data.split(b"\n")
                for line in lines:
                    yield line.decode("utf-8", "replace")


def _json_array_items(path: Path) -> Iterator[Any]:
    """Items of a JSON array, stopping quietly at a truncated final item."""
    decoder = json.JSONDecoder()
    text_decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer, position, started = "", 0, False
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            buffer = buffer[position:] + text_decoder.decode(chunk)
            position = 0
            if not started:
                buffer = buffer.lstrip("\ufeff \t\r\n")
                if not buffer:
                    continue
                if buffer[0] != "[":
                    raise DownloadError("Expected a JSON array.")
                position, started = 1, True
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    return
                try:
                    item, position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield item


def _split_on_marker(path: Path, marker: str) -> Iterator[str]:
    """Documents separated by lines consisting of ``marker``."""
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.strip() == marker:
                if text := _clean_text("".join(lines)):
                    yield text
                lines = []
            else:
                lines.append(line)
    if text := _clean_text("".join(lines)):
        yield text


# --------------------------------------------------------------------------- storage

def _data_file(kind: str) -> str:
    return DOCUMENTS_FILE if kind == PRETRAIN else CONVERSATIONS_FILE


def _count_key(kind: str) -> str:
    return "documents" if kind == PRETRAIN else "rows"


def _check_id(dataset_id: str) -> str:
    if not isinstance(dataset_id, str) or not _ID_PATTERN.fullmatch(dataset_id):
        raise KeyError(f"Unknown dataset: {dataset_id!r}")
    return dataset_id


def _read_meta(directory: Path) -> dict | None:
    try:
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _complete_meta(dataset_id: str, kind: str) -> dict | None:
    directory = paths.datasets_dir() / dataset_id
    meta = _read_meta(directory)
    if meta is None or not (directory / _data_file(kind)).is_file():
        return None
    return meta


def _discard(directory: Path) -> None:
    """Remove a directory so that it disappears atomically for readers."""
    if not directory.exists():
        return
    trash = directory.with_name(f".{directory.name}.trash-{os.getpid()}-{time.monotonic_ns()}")
    os.rename(directory, trash)
    shutil.rmtree(trash, ignore_errors=True)


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _thread_lock(dataset_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(dataset_id, threading.Lock())


@contextmanager
def _download_lock(info: DatasetInfo) -> Iterator[None]:
    """Per-dataset lock across threads (threading.Lock) and processes (flock)."""
    lock = _thread_lock(info.id)
    if not lock.acquire(blocking=False):
        raise DownloadInProgress(f"{info.name} is already being downloaded.")
    try:
        descriptor = os.open(paths.datasets_dir() / f".{info.id}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DownloadInProgress(
                    f"{info.name} is already being downloaded by another process.") from None
            yield
        finally:
            os.close(descriptor)
    finally:
        lock.release()


class _Writer:
    """Writes normalised JSON lines while tracking size, count and SHA-256."""

    def __init__(self, path: Path, max_bytes: int | None = None, max_records: int | None = None):
        self.path = path
        self.max_bytes = max_bytes
        self.max_records = max_records
        self.bytes = 0
        self.records = 0
        self._digest = hashlib.sha256()
        self._handle = open(path, "wb")

    def add(self, record: dict) -> bool:
        """Append a record; False (nothing written) once the size or count budget is used up."""
        if self.max_records is not None and self.records >= self.max_records:
            return False
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if self.max_bytes is not None and self.records and self.bytes + len(line) > self.max_bytes:
            return False
        self._handle.write(line)
        self._digest.update(line)
        self.bytes += len(line)
        self.records += 1
        return True

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


# --------------------------------------------------------------------------- public API

def catalog(kind: str | None = None) -> list[DatasetInfo]:
    """Built-in datasets (recommended first) followed by custom imports, oldest first."""
    if kind is not None and kind not in KINDS:
        raise ValueError(f"Unknown dataset kind {kind!r}; expected 'pretrain' or 'finetune'.")
    entries = [info for info in _BUILTINS.values() if kind in (None, info.kind)]
    entries.sort(key=lambda info: (KINDS.index(info.kind), not info.recommended))
    customs = [info for info, _ in sorted(_custom_entries(), key=lambda pair: pair[1])
               if kind in (None, info.kind)]
    return entries + customs


def get(dataset_id: str) -> DatasetInfo:
    """Catalog entry for ``dataset_id``; KeyError if unknown."""
    _check_id(dataset_id)
    if dataset_id in _BUILTINS:
        return _BUILTINS[dataset_id]
    if dataset_id.startswith(CUSTOM_PREFIX):
        info = _custom_info(paths.datasets_dir() / dataset_id)
        if info is not None:
            return info
    raise KeyError(f"Unknown dataset: {dataset_id!r}")


def status(dataset_id: str) -> dict:
    """Local state, e.g. ``{"downloaded": True, "bytes": 123, "documents": 45, ...}``."""
    info = get(dataset_id)
    key = _count_key(info.kind)
    meta = _complete_meta(info.id, info.kind)
    result: dict[str, Any] = {
        "id": info.id, "kind": info.kind, "downloaded": meta is not None,
        "downloading": _thread_lock(info.id).locked(), "bytes": 0, key: 0,
    }
    if meta is not None:
        result.update({
            "bytes": int(meta.get("bytes", 0)), key: int(meta.get(key, 0)),
            "download_bytes": int(meta.get("download_bytes", 0)),
            "sha256": meta.get("sha256"), "created": meta.get("created"),
        })
    return result


def download(dataset_id: str, progress: ProgressCallback | None = None,
             cancel: threading.Event | None = None) -> dict:
    """Fetch and normalise a built-in dataset; returns :func:`status` afterwards.

    ``progress(done_bytes, total_bytes_or_None, message)`` is called regularly. Setting
    ``cancel`` aborts with :class:`DownloadCancelled` and leaves nothing behind. Raises
    :class:`DownloadInProgress` if the dataset is already being downloaded. Already
    downloaded (and custom) datasets return immediately.
    """
    info = get(dataset_id)
    if info.custom:
        return status(info.id)
    recipe = _RECIPES[info.id]
    root = paths.datasets_dir()
    with _download_lock(info):
        if _complete_meta(info.id, info.kind) is not None:
            return status(info.id)
        for stale in root.glob(f".{info.id}.download-*"):
            shutil.rmtree(stale, ignore_errors=True)
        free = shutil.disk_usage(root).free
        if free < 3 * info.approx_bytes + 50 * MB:
            raise DownloadError(f"Not enough free disk space to download {info.name} "
                                f"({_human_bytes(free)} free).")
        workdir = Path(tempfile.mkdtemp(prefix=f".{info.id}.download-", dir=root))
        try:
            job = _Job(info, workdir, progress, cancel)
            job.budget = recipe.max_bytes
            job.report(f"Starting {info.name}", force=True)
            writer = _write_records(job, recipe, workdir / _data_file(info.kind))
            for leftover in workdir.iterdir():
                if leftover.name != writer.path.name:
                    leftover.unlink()
            _write_json(workdir / META_FILE, {
                "format": FORMAT_VERSION, "id": info.id, "kind": info.kind, "name": info.name,
                "license": info.license, "license_url": info.license_url,
                "source_url": info.source_url, "sources": job.sources,
                "download_bytes": job.done, "file": writer.path.name, "bytes": writer.bytes,
                _count_key(info.kind): writer.records, "sha256": writer.sha256,
                "created": _now(),
            })
            job.check()
            target = root / info.id
            _discard(target)
            os.rename(workdir, target)
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        job.estimate = job.done
        job.report(f"{info.name} is ready", force=True)
    return status(info.id)


def _write_records(job: _Job, recipe: _Recipe, path: Path) -> _Writer:
    noun = "documents" if job.info.kind == PRETRAIN else "conversations"
    writer = _Writer(path, recipe.max_bytes, recipe.max_records)
    records = iter(recipe.build(job))
    try:
        for record in records:
            job.check()
            if not writer.add(record):
                break
            if writer.records % 200 == 0:
                job.report(f"Preparing {job.info.name}: {writer.records:,} {noun}")
    finally:
        close = getattr(records, "close", None)
        if close is not None:
            close()
        writer.close()
    if writer.records == 0:
        raise DownloadError(f"{job.info.name} contained no usable {noun}.")
    return writer


def delete_download(dataset_id: str) -> None:
    """Remove the local copy of a dataset (built-in or custom) if present."""
    info = get(dataset_id)
    with _download_lock(info):
        _discard(paths.datasets_dir() / info.id)


def delete_custom(dataset_id: str) -> None:
    """Delete an imported dataset."""
    info = get(dataset_id)
    if not info.custom:
        raise ValueError(f"{info.name} is a built-in dataset and cannot be deleted.")
    delete_download(info.id)


def _require(dataset_id: str, kind: str) -> tuple[DatasetInfo, Path]:
    info = get(dataset_id)
    if info.kind != kind:
        other = "iter_conversations()" if kind == PRETRAIN else "iter_documents()"
        raise ValueError(f"{info.name} is a {info.kind} dataset; use {other}.")
    if _complete_meta(info.id, info.kind) is None:
        raise DatasetNotDownloaded(f"{info.name} has not been downloaded yet.")
    return info, paths.datasets_dir() / info.id / _data_file(info.kind)


def _jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_documents(dataset_id: str) -> Iterator[str]:
    """Texts of a downloaded pre-training dataset, in order."""
    _, path = _require(dataset_id, PRETRAIN)
    return (record["text"] for record in _jsonl(path))


def iter_conversations(dataset_id: str) -> Iterator[dict]:
    """``{"messages": [{"role", "content"}, ...]}`` dicts of a downloaded fine-tuning dataset."""
    _, path = _require(dataset_id, FINETUNE)
    return ({"messages": record["messages"]} for record in _jsonl(path))


def preview(dataset_id: str) -> dict:
    """A few documents (at most ~8,000 characters in total) or the first chat examples."""
    info = get(dataset_id)
    downloaded = _complete_meta(info.id, info.kind) is not None
    if info.kind == FINETUNE:
        examples = []
        if downloaded:
            for conversation in iter_conversations(info.id):
                examples.append(conversation)
                if len(examples) >= PREVIEW_EXAMPLES:
                    break
        return {"kind": "chat", "examples": examples, "downloaded": downloaded}
    documents: list[str] = []
    truncated = False
    if downloaded:
        budget = PREVIEW_CHARS
        for text in iter_documents(info.id):
            if len(text) > budget:
                if not documents or budget >= 400:
                    documents.append(_shorten(text, budget))
                    truncated = True
                break
            documents.append(text)
            budget -= len(text)
            if len(documents) >= PREVIEW_DOCUMENTS or budget <= 0:
                break
    return {"kind": "text", "documents": documents, "truncated": truncated, "downloaded": downloaded}


# --------------------------------------------------------------------------- custom imports

_CUSTOM_COLORS = ("#64748b", "#7c6f64", "#5f7a8a", "#6d6a86", "#7a6375", "#5e7d6e")
_TEXT_EXTENSIONS = (".txt", ".md")
_CHAT_EXTENSIONS = (".jsonl", ".json", ".csv")
_DOCUMENT_MARKER = "<|endoftext|>"
_LIST_KEYS = ("data", "rows", "examples", "train", "items", "records", "conversations", "dataset")
CHAT_FORMATS_HELP = (
    "Each example needs one of these shapes: "
    '{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}, '
    '{"conversations": [{"from": "human", "value": "…"}, {"from": "gpt", "value": "…"}]}, '
    '{"prompt": "…", "response": "…"} (or "completion"), '
    '{"instruction": "…", "input": "…", "output": "…"}, '
    '{"question": "…", "answer": "…"} or {"user": "…", "assistant": "…"}. '
    "CSV files need a header row with those column names."
)
_BAD_ROW = object()


def _load_custom(directory: Path) -> tuple[DatasetInfo, dict] | None:
    meta = _read_meta(directory)
    data = meta.get("info") if meta else None
    if not isinstance(data, dict):
        return None
    names = {field.name for field in fields(DatasetInfo)}
    try:
        info = DatasetInfo(**{key: value for key, value in data.items() if key in names})
    except TypeError:
        return None
    if info.id != directory.name or not info.custom or info.kind not in KINDS:
        return None
    if not (directory / _data_file(info.kind)).is_file():
        return None
    return info, meta


def _custom_info(directory: Path) -> DatasetInfo | None:
    loaded = _load_custom(directory)
    return loaded[0] if loaded else None


def _custom_entries() -> list[tuple[DatasetInfo, str]]:
    entries = []
    for directory in sorted(paths.datasets_dir().glob(CUSTOM_PREFIX + "*")):
        if directory.is_dir() and _ID_PATTERN.fullmatch(directory.name):
            loaded = _load_custom(directory)
            if loaded is not None:
                entries.append((loaded[0], str(loaded[1].get("created", ""))))
    return entries


def _slug(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
    return slug[:40].strip("-") or "dataset"


def _not_utf8() -> ValueError:
    return ValueError("The file is not valid UTF-8 text. Save it with UTF-8 encoding and try again.")


def _text_documents(data: bytes) -> Iterator[str]:
    """Documents of a plain-text import, split on <|endoftext|> lines or 2+ blank lines."""
    if b"\x00" in data:
        raise ValueError("This looks like a binary file (it contains NUL bytes). Pre-training data "
                         "must be plain UTF-8 text; UTF-16 files need to be re-saved as UTF-8.")
    head = data[:65_536].decode("utf-8", "ignore")
    if len(_CONTROL_CHARS.findall(head)) > max(8, len(head) // 100):
        raise ValueError("This looks like a binary file, not plain text.")
    stream = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", errors="strict", newline="")
    lines: list[str] = []
    blank_run = 0
    try:
        for line in stream:
            stripped = line.strip()
            if stripped == _DOCUMENT_MARKER or (stripped and blank_run >= 2):
                if text := _clean_text("".join(lines)):
                    yield text
                lines, blank_run = [], 0
                if stripped == _DOCUMENT_MARKER:
                    continue
            blank_run = 0 if stripped else blank_run + 1
            lines.append(line)
    except UnicodeDecodeError:
        raise _not_utf8() from None
    if text := _clean_text("".join(lines)):
        yield text


def _jsonl_rows(data: bytes) -> Iterator[Any]:
    for line in io.BytesIO(data):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except UnicodeDecodeError:
            raise _not_utf8() from None
        except ValueError:
            yield _BAD_ROW


def _json_rows(data: bytes) -> Iterator[Any]:
    try:
        document = json.loads(data)
    except UnicodeDecodeError:
        raise _not_utf8() from None
    except json.JSONDecodeError as error:
        if error.msg == "Extra data":  # JSON lines saved with a .json extension
            yield from _jsonl_rows(data)
            return
        raise ValueError(f"The file is not valid JSON (line {error.lineno}, column {error.colno}: "
                         f"{error.msg}).") from None
    if isinstance(document, list):
        yield from document
        return
    if isinstance(document, dict):
        if _row_messages(document)[0] is not None:
            yield document
            return
        lists = [value for value in document.values()
                 if isinstance(value, list) and value and isinstance(value[0], dict)]
        if len(lists) != 1:
            lists = [document[key] for key in _LIST_KEYS if isinstance(document.get(key), list)][:1]
        if lists:
            yield from lists[0]
            return
    raise ValueError("Could not find a list of examples in the JSON file. " + CHAT_FORMATS_HELP)


def _csv_rows(data: bytes) -> Iterator[Any]:
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
    stream = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", errors="strict", newline="")
    try:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError("The CSV file is empty; it needs a header row. " + CHAT_FORMATS_HELP)
        names = [name.strip().lower() for name in header]
        for values in reader:
            if any(value.strip() for value in values):
                yield dict(zip(names, values))
    except UnicodeDecodeError:
        raise _not_utf8() from None
    except csv.Error as error:
        raise ValueError(f"The CSV file could not be read: {error}.") from None


def _field_text(row: dict, *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if isinstance(value, str) and value.strip():
            return value
    return None


_PAIR_FORMATS = (
    ("prompt/response", ("prompt",), ("response", "completion")),
    ("instruction", ("instruction",), ("output", "response")),
    ("question/answer", ("question",), ("answer",)),
    ("user/assistant", ("user",), ("assistant",)),
)


def _row_messages(row: Any) -> tuple[list[dict] | None, str | None]:
    """Conversation for one imported example plus the name of the detected format."""
    if not isinstance(row, dict):
        return None, None
    row = {str(key).strip().lower(): value for key, value in row.items()}
    system = _field_text(row, "system", "system_prompt")
    for key, label in (("messages", "messages"), ("conversations", "sharegpt"),
                       ("conversation", "sharegpt")):
        turns = row.get(key)
        if isinstance(turns, str) and turns.lstrip().startswith("["):
            try:
                turns = json.loads(turns)
            except ValueError:
                return None, label
        if isinstance(turns, list):
            if system and not any(isinstance(turn, dict) and str(turn.get(
                    "role", turn.get("from", ""))).strip().lower() in ("system", "developer")
                    for turn in turns):
                turns = [{"role": "system", "content": system}, *turns]
            return _clean_messages(turns), label
    for label, prompt_keys, reply_keys in _PAIR_FORMATS:
        if not (any(key in row for key in prompt_keys) and any(key in row for key in reply_keys)):
            continue
        prompt = _field_text(row, *prompt_keys)
        reply = _field_text(row, *reply_keys)
        if label == "instruction" and prompt and (extra := _field_text(row, "input", "context")):
            prompt = f"{prompt}\n\n{extra}"
        if not prompt or not reply:
            return None, label
        messages = [{"role": "system", "content": system}] if system else []
        messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        return _clean_messages(messages), label
    return None, None


def import_custom(name: str, kind: str, filename: str, data: bytes) -> DatasetInfo:
    """Import a user file as a new dataset (``custom-…`` id); ValueError explains any problem.

    Pre-training: .txt/.md UTF-8 text; documents are separated by lines containing only
    ``<|endoftext|>`` or by two or more blank lines. Fine-tuning: .jsonl/.json/.csv in any of the
    formats listed in :data:`CHAT_FORMATS_HELP`; unusable rows are skipped and counted.
    """
    if kind not in KINDS:
        raise ValueError("The dataset kind must be 'pretrain' or 'finetune'.")
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("The file contents must be bytes.")
    data = bytes(data)
    filename = Path(str(filename or "")).name
    extension = Path(filename).suffix.lower()
    name = _one_line(name or "")[:80] or _one_line(Path(filename).stem)[:80] or "My dataset"
    if len(data) > MAX_IMPORT_BYTES:
        raise ValueError(f"The file is too large ({_human_bytes(len(data))}); "
                         "imports are limited to 200 MB.")
    if kind == PRETRAIN and extension not in _TEXT_EXTENSIONS:
        raise ValueError("Pre-training data must be a plain text file ending in .txt or .md.")
    if kind == FINETUNE and extension not in _CHAT_EXTENSIONS:
        raise ValueError("Fine-tuning data must be a .jsonl, .json or .csv file.")
    if re.search(rb"\S", data) is None:
        raise ValueError("The file is empty.")

    root = paths.datasets_dir()
    base = CUSTOM_PREFIX + _slug(name)
    for number in range(1, 1000):
        dataset_id = base if number == 1 else f"{base}-{number}"
        try:
            os.mkdir(root / dataset_id)
            break
        except FileExistsError:
            continue
    else:
        raise ValueError("Too many datasets with this name; please choose another name.")
    reserved = root / dataset_id
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.import-", dir=root))
    try:
        writer = _Writer(staging / _data_file(kind))
        skipped = 0
        formats: dict[str, int] = defaultdict(int)
        try:
            if kind == PRETRAIN:
                for text in _text_documents(data):
                    writer.add({"text": text})
                formats["text"] = writer.records
            else:
                as_json = extension == ".json" or re.match(rb"(\xef\xbb\xbf)?\s*\[", data[:4096])
                rows = (_csv_rows(data) if extension == ".csv"
                        else _json_rows(data) if as_json else _jsonl_rows(data))
                for row in rows:
                    messages, label = _row_messages(row) if row is not _BAD_ROW else (None, None)
                    if messages is None:
                        skipped += 1
                        continue
                    formats[label or "messages"] += 1
                    writer.add({"messages": messages})
        finally:
            writer.close()
        if writer.records == 0:
            if kind == PRETRAIN:
                raise ValueError("The file contains no text.")
            raise ValueError(f"No usable conversations were found in {filename or 'the file'}. "
                             + CHAT_FORMATS_HELP)
        noun = "documents" if kind == PRETRAIN else "conversations"
        details = f"{writer.records:,} {noun}"
        if skipped:
            details += f"; {skipped:,} unusable {'row was' if skipped == 1 else 'rows were'} skipped"
        info = DatasetInfo(
            id=dataset_id, kind=kind, name=name,
            tagline="Your own text" if kind == PRETRAIN else "Your own conversations",
            description=f"Imported from {filename or 'a file'} ({details}).",
            license="Your own data", license_url="",
            source_label=filename or "Imported file", source_url="",
            paper_label=None, paper_url=None,
            size_label=f"{_human_bytes(writer.bytes)} · {_human_count(writer.records)} {noun}",
            approx_bytes=writer.bytes, icon="file",
            color=_CUSTOM_COLORS[int(hashlib.sha256(dataset_id.encode()).hexdigest(), 16)
                                % len(_CUSTOM_COLORS)],
            recommended=False, custom=True,
        )
        _write_json(staging / META_FILE, {
            "format": FORMAT_VERSION, "id": dataset_id, "kind": kind, "name": name,
            "license": info.license, "license_url": "", "sources": [],
            "original_filename": filename, "original_bytes": len(data),
            "source_format": max(formats, key=formats.get) if formats else None,
            "skipped": skipped, "download_bytes": 0, "file": writer.path.name,
            "bytes": writer.bytes, _count_key(kind): writer.records, "sha256": writer.sha256,
            "created": _now(), "info": asdict(info),
        })
        os.rename(staging, reserved)  # replaces the empty placeholder directory atomically
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            reserved.rmdir()
        except OSError:
            pass
        raise
    return info


# --------------------------------------------------------------------------- built-in recipes

@dataclass(frozen=True)
class _Remote:
    url: str
    size: int
    sha256: str | None = None


def _hf(repo: str, revision: str, file: str, size: int, sha256: str | None = None) -> _Remote:
    return _Remote(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}", size, sha256)


def _fetch(job: _Job, remote: _Remote, name: str) -> Path:
    job.estimate = job.done + remote.size
    return job.fetch_file(remote.url, name, size=remote.size, sha256=remote.sha256)


def _rows(job: _Job, remote: _Remote, columns: list[str], partial: bool = False) -> Iterator[dict]:
    return _parquet_rows(job, remote.url, remote.size, columns,
                         budget=job.budget if partial else None, sha256=remote.sha256)


def _prefix(job: _Job, remote: _Remote, name: str, length: int) -> Path:
    job.estimate = job.done + length
    return job.fetch_file(remote.url, name, end=length)


def _jsonl_file(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def _chat(prompt: Any, reply: Any) -> dict | None:
    messages = _clean_messages([{"role": "user", "content": prompt},
                                {"role": "assistant", "content": reply}])
    return {"messages": messages} if messages else None


_TINYSTORIES = _hf("roneneldan/TinyStories", "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                   "TinyStoriesV2-GPT4-valid.txt", 22_502_601,
                   "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d")
_SHAKESPEARE = _Remote("https://raw.githubusercontent.com/karpathy/char-rnn/"
                       "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
                       1_115_394, "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed")
_WIKITEXT2 = _hf("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
                 "wikitext-2-raw-v1/train-00000-of-00001.parquet", 6_357_543,
                 "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7")
_SIMPLE_WIKI = _hf("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                   "20231101.simple/train-00000-of-00001.parquet", 156_885_218,
                   "31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0")
_GOODWIKI = _hf("euirim/goodwiki", "7491f6da6439cf25e4d783f96fd09c4256d314db",
                "09_04_2023_v1.parquet", 482_741_657,
                "2b8b3edebedec44e8a49994fe1f2a72322fdb8983e0a3bfd914b5014daef57a7")
_ARXIV = _hf("gfissore/arxiv-abstracts-2021", "e4c5fbd4dec8e46a5dc869216fe1c94cc585757a",
             "arxiv-abstracts.jsonl.gz", 939_914_256)
_CHILDREN = _hf("ajibawa-2023/Children-Stories-Collection", "fcd0d59c4970106f980ac9185928e0cfad82e4db",
                "Children-Stories-0-Final.json", 344_470_994)
_COSMOPEDIA = _hf("HuggingFaceTB/cosmopedia-100k", "8fdb6dada238c1d46dd8cf61da2fb4f9c36f654d",
                  "data/train-00000-of-00002.parquet", 153_421_593)
_EVERYDAY = _hf("HuggingFaceTB/everyday-conversations-llama3.1-2k",
                "14f543216b9ba42b6b951dc5bd199460d193b162", "data/train_sft-00000-of-00001.parquet",
                2_050_609, "4750214d12b4b92d07fca953e0b5cc59e313c575282d08b605c639aa87531daa")
_NO_ROBOTS = _hf("HuggingFaceH4/no_robots", "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
                 "data/train_sft-00000-of-00001.parquet", 10_474_682,
                 "76d64120a2ab20eb85d6b22bdced5b1ec638a6213a97004741006745c92489ef")
_DOLLY = _hf("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
             "databricks-dolly-15k.jsonl", 13_085_339,
             "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec")
_OASST1 = _hf("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b",
              "data/train-00000-of-00001-b42a775f407cee45.parquet", 39_516_251,
              "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e")
_SMOLTALK = _hf("HuggingFaceTB/smol-smoltalk", "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
                "data/train-00000-of-00004.parquet", 229_710_350)
_GSM8K = _hf("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
             "main/train-00000-of-00001.parquet", 2_306_545,
             "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d")
_COEDIT = _hf("grammarly/coedit", "e9a255c33ef910bc33a9d2b522653fa87521583e", "train.jsonl",
              19_695_735, "2913249158d6a178dc638e870212ff8a432d128eb6b4bdbe969ee805e6063ce3")
_CODE_ALPACA = _hf("sahil2801/CodeAlpaca-20k", "152bb5e9a29651266b018106053980070a0521a1",
                   "code_alpaca_20k.json", 8_057_896,
                   "58d136bb7ea8e4b7c5b6ef08831da905d0abfafeb56f8fa519083089dd6a210e")
_SQUAD = _hf("rajpurkar/squad", "7b6d24c440a36b6815f21b70d25016731768db1f",
             "plain_text/train-00000-of-00001.parquet", 14_458_314,
             "ea7f52bac024f6b1bdc7aaa2a4ee302cba8c2fdc8d4a235cf18a9a5196b6175b")
_ULTRACHAT = _hf("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
                 "data/train_sft-00000-of-00003-a3ecf92756993583.parquet", 243_999_189)

_ARXIV_PREFIX = 16 * MB
_CHILDREN_PREFIX = 40 * MB


def _build_tinystories(job: _Job) -> Iterator[dict]:
    path = _fetch(job, _TINYSTORIES, "source.txt")
    for index, text in enumerate(_split_on_marker(path, _DOCUMENT_MARKER)):
        if index == 0 and text[:1].islower():
            continue  # the file opens with the tail end of a story
        yield {"text": text}


def _build_shakespeare(job: _Job) -> Iterator[dict]:
    path = _fetch(job, _SHAKESPEARE, "source.txt")
    yield {"text": _clean_text(path.read_text(encoding="utf-8"))}


_WIKITEXT_HEADING = re.compile(r"(=(?: =)*) (.+?) \1")


def _wikitext_line(text: str) -> str:
    """Undo WikiText's word-level tokenisation (``1 @,@ 000`` → ``1,000``, ``it 's`` → ``it's``)."""
    text = text.strip().replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r'" ([^"]*?) "', r'"\1"', text)
    text = re.sub(r" ([.,;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    text = re.sub(r" '(s|t|re|m|ve|ll|d)\b", r"'\1", text)
    text = re.sub(r"\b([A-Z]) '([A-Z])", r"\1'\2", text)
    return re.sub(r"s ' ", "s' ", text)


def _wikitext_articles(lines: Iterable[str]) -> Iterator[str]:
    """Group WikiText rows into articles, turning ``= = Heading = =`` rows into Markdown headings.

    Real headings always follow an empty row. Heading-like rows that do not are pieces of a
    table legend (``GP = Games played ; G = Goals``) that WikiText split at its ``=`` signs, so
    they are glued back onto the surrounding line.
    """
    paragraphs: list[str] = []
    previous, glue = "", False
    for line in lines:
        raw = (line or "").strip()
        heading = _WIKITEXT_HEADING.fullmatch(raw) if raw else None
        if not raw:
            glue = False
        elif heading and not previous:
            level = heading.group(1).count("=")
            if level == 1 and paragraphs:
                yield _clean_text("\n\n".join(map(_wikitext_line, paragraphs)))
                paragraphs = []
            paragraphs.append("#" * level + " " + heading.group(2))
        elif paragraphs and (heading or glue):
            paragraphs[-1] += " " + raw
            glue = heading is not None
        else:
            paragraphs.append(raw)
        previous = raw
    if paragraphs:
        yield _clean_text("\n\n".join(map(_wikitext_line, paragraphs)))


def _build_wikitext(job: _Job) -> Iterator[dict]:
    for text in _wikitext_articles(row["text"] for row in _rows(job, _WIKITEXT2, ["text"])):
        if len(text) >= 200:
            yield {"text": text}


def _build_simple_wikipedia(job: _Job) -> Iterator[dict]:
    for row in _rows(job, _SIMPLE_WIKI, ["title", "text"], partial=True):
        text = _clean_text(row["text"] or "")
        if len(text) >= 200:
            yield {"text": f"{_one_line(row['title'] or '')}\n\n{text}".strip()}


def _build_goodwiki(job: _Job) -> Iterator[dict]:
    for row in _rows(job, _GOODWIKI, ["title", "markdown"], partial=True):
        text = _clean_text(row["markdown"] or "")
        if len(text) >= 200:
            yield {"text": f"# {_one_line(row['title'] or '')}\n\n{text}"}


def _build_arxiv(job: _Job) -> Iterator[dict]:
    path = _prefix(job, _ARXIV, "source.jsonl.gz", _ARXIV_PREFIX)
    job.report("Reading the downloaded data", force=True)
    for line in _gzip_lines(path):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        title = _one_line(record.get("title") or "")
        abstract = _one_line(record.get("abstract") or "")
        if title and len(abstract) >= 100:
            yield {"text": f"{title}\n\n{abstract}"}


def _build_children_stories(job: _Job) -> Iterator[dict]:
    path = _prefix(job, _CHILDREN, "source.json", _CHILDREN_PREFIX)
    job.report("Reading the downloaded data", force=True)
    for item in _json_array_items(path):
        text = _clean_text(item.get("text") or "") if isinstance(item, dict) else ""
        if len(text) >= 200:
            yield {"text": text}


_GUTENBERG_DELAY = 0.5  # seconds between books, to be gentle with the volunteer-run mirrors
_GUTENBERG_MIRRORS = (
    "https://gutenberg.pglaf.org/cache/epub/{0}/pg{0}.txt",
    "http://aleph.gutenberg.org/cache/epub/{0}/pg{0}.txt",
)
_GUTENBERG_BOOKS = (
    (1342, "Pride and Prejudice", 772_386), (11, "Alice's Adventures in Wonderland", 174_311),
    (1661, "The Adventures of Sherlock Holmes", 607_606), (84, "Frankenstein", 448_885),
    (46, "A Christmas Carol", 182_250), (55, "The Wonderful Wizard of Oz", 237_232),
    (16, "Peter Pan", 288_769), (120, "Treasure Island", 399_873),
    (74, "The Adventures of Tom Sawyer", 434_357), (2591, "Grimms' Fairy Tales", 560_164),
    (236, "The Jungle Book", 303_640), (35, "The Time Machine", 204_384),
    (45, "Anne of Green Gables", 606_693), (514, "Little Women", 1_071_398),
    (98, "A Tale of Two Cities", 807_197), (1260, "Jane Eyre", 1_084_803),
    (345, "Dracula", 890_348), (174, "The Picture of Dorian Gray", 465_814),
    (768, "Wuthering Heights", 693_819), (36, "The War of the Worlds", 366_624),
    (113, "The Secret Garden", 475_315), (289, "The Wind in the Willows", 358_358),
    (215, "The Call of the Wild", 199_757), (76, "Adventures of Huckleberry Finn", 622_460),
    (103, "Around the World in Eighty Days", 403_712), (158, "Emma", 933_709),
    (161, "Sense and Sensibility", 712_852), (1400, "Great Expectations", 1_058_204),
    (2701, "Moby-Dick", 1_276_315),
)
_GUTENBERG_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG E-?BOOK.*$", re.M | re.I)
_GUTENBERG_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG E-?BOOK", re.M | re.I)
_ILLUSTRATION = re.compile(r"\[(Illustration|Frontispiece)[^\]]*\]", re.I)
_ITALICS = re.compile(r"(?<!\w)_((?:[^_\n]|\n(?!\n))+?)_(?!\w)")
_CREDITS = re.compile(r"(produced by|e-?text prepared by|transcribed (from|by)|transcriber'?s note)", re.I)


def _unwrap(paragraph: str) -> str:
    """Join hard-wrapped prose lines; verse, lists and indented blocks keep their line breaks."""
    lines = paragraph.split("\n")
    if len(lines) < 2 or any(line[:1].isspace() for line in lines):
        return paragraph
    if min(len(line) for line in lines[:-1]) < 45:
        return paragraph
    return " ".join(line.strip() for line in lines)


def _strip_gutenberg(raw: str) -> str:
    """Book text without the Project Gutenberg header, licence footer or illustration tags."""
    text = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if start := _GUTENBERG_START.search(text):
        text = text[start.end():]
    if end := _GUTENBERG_END.search(text):
        text = text[: end.start()]
    text = _ITALICS.sub(r"\1", _ILLUSTRATION.sub("", text))
    paragraphs = []
    for paragraph in re.split(r"\n[ \t]*\n", text):
        paragraph = paragraph.rstrip().strip("\n")
        if not paragraph.strip():
            continue
        if "gutenberg" in paragraph.lower() or _CREDITS.match(paragraph.strip()):
            continue
        if not paragraphs and (paragraph.strip().lower() == "cover" or not re.search(r"[^\W_]", paragraph)):
            continue  # leftovers of a cover image at the very start
        paragraphs.append(_unwrap(paragraph))
    return _clean_text("\n\n".join(paragraphs))


def _build_gutenberg(job: _Job) -> Iterator[dict]:
    for index, (book, title, _) in enumerate(_GUTENBERG_BOOKS):
        if index:
            job.pause(_GUTENBERG_DELAY)
        path = job.fetch_first([mirror.format(book) for mirror in _GUTENBERG_MIRRORS],
                               f"pg{book}.txt", label=title)
        text = _strip_gutenberg(path.read_text(encoding="utf-8", errors="replace"))
        path.unlink()
        if len(text) >= 10_000:
            yield {"text": text}


def _build_cosmopedia(job: _Job) -> Iterator[dict]:
    for row in _rows(job, _COSMOPEDIA, ["text"], partial=True):
        if text := _clean_text(row["text"] or ""):
            yield {"text": text}


def _messages_recipe(remote: _Remote, partial: bool = False) -> Callable[[_Job], Iterator[dict]]:
    def build(job: _Job) -> Iterator[dict]:
        for row in _rows(job, remote, ["messages"], partial=partial):
            if messages := _clean_messages(row["messages"] or []):
                yield {"messages": messages}
    return build


def _build_dolly(job: _Job) -> Iterator[dict]:
    for row in _jsonl_file(_fetch(job, _DOLLY, "source.jsonl")):
        prompt = str(row.get("instruction") or "").strip()
        if context := str(row.get("context") or "").strip():
            prompt = f"{prompt}\n\n{context}"
        if record := _chat(prompt, row.get("response")):
            yield record


def _build_oasst1(job: _Job) -> Iterator[dict]:
    """English conversations following the best-ranked reply at every turn of each tree."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = _fetch(job, _OASST1, "source.parquet")
    job.report("Reading the downloaded data", force=True)
    table = pq.read_table(path, columns=["message_id", "parent_id", "text", "role", "lang",
                                         "review_result", "deleted", "rank"])
    rows = table.filter(pc.equal(table["lang"], "en")).to_pylist()
    del table
    children: dict[str, list[dict]] = defaultdict(list)
    roots = []
    for row in rows:
        if row["deleted"] or row["review_result"] is False:
            continue
        if row["parent_id"] is None:
            roots.append(row)
        else:
            children[row["parent_id"]].append(row)
    for root in roots:
        job.check()
        thread = [root]
        while replies := children.get(thread[-1]["message_id"]):
            thread.append(min(replies, key=lambda reply: 99 if reply["rank"] is None else reply["rank"]))
        messages = _clean_messages({"role": "user" if message["role"] == "prompter" else "assistant",
                                    "content": message["text"]} for message in thread)
        if messages:
            yield {"messages": messages}


def _build_gsm8k(job: _Job) -> Iterator[dict]:
    for row in _rows(job, _GSM8K, ["question", "answer"]):
        answer = re.sub(r"<<[^>]*>>", "", row["answer"] or "")
        answer = re.sub(r"\s*####\s*(.+?)\s*$", lambda match: f"\nThe answer is {match.group(1)}.", answer)
        if record := _chat(row["question"], answer):
            yield record


def _build_coedit(job: _Job) -> Iterator[dict]:
    for row in _jsonl_file(_fetch(job, _COEDIT, "source.jsonl")):
        if record := _chat(row.get("src"), row.get("tgt")):
            yield record


def _build_code_alpaca(job: _Job) -> Iterator[dict]:
    for item in _json_array_items(_fetch(job, _CODE_ALPACA, "source.json")):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("instruction") or "").strip()
        if extra := str(item.get("input") or "").strip():
            prompt = f"{prompt}\n\n{extra}"
        if record := _chat(prompt, item.get("output")):
            yield record


_SQUAD_QUESTIONS = 5  # per conversation, so most fit a small model's context window


def _build_squad(job: _Job) -> Iterator[dict]:
    """The passage with its first question, then follow-ups (a new conversation every 5)."""
    context, messages = None, []
    for row in _rows(job, _SQUAD, ["context", "question", "answers"]):
        answers = (row["answers"] or {}).get("text") or []
        question = (row["question"] or "").strip()
        if not answers or not question:
            continue
        if row["context"] != context or len(messages) >= 2 * _SQUAD_QUESTIONS:
            if conversation := _clean_messages(messages):
                yield {"messages": conversation}
            context = row["context"]
            messages = []
            question = f"{(context or '').strip()}\n\n{question}"
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answers[0]}]
    if conversation := _clean_messages(messages):
        yield {"messages": conversation}


# --------------------------------------------------------------------------- catalog

_LICENSE_URLS = {
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "MIT": "https://opensource.org/license/mit",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CDLA-Sharing-1.0": "https://cdla.dev/sharing-1-0/",
    "Public domain": "https://creativecommons.org/publicdomain/mark/1.0/",
    "Public domain (US)": "https://www.gutenberg.org/policy/license.html",
}

_BUILTINS: dict[str, DatasetInfo] = {}
_RECIPES: dict[str, _Recipe] = {}


def _register(build: Callable[[_Job], Iterable[dict]], *, max_bytes: int = 48 * MB,
              license: str, paper: str | None = None, hf: str | None = None,
              **fields_: Any) -> None:
    if hf:
        fields_.setdefault("source_label", f"Hugging Face · {hf}")
        fields_.setdefault("source_url", f"https://huggingface.co/datasets/{hf}")
    info = DatasetInfo(
        license=license, license_url=_LICENSE_URLS[license],
        paper_label=f"arXiv {paper}" if paper else None,
        paper_url=f"https://arxiv.org/abs/{paper}" if paper else None,
        recommended=fields_.pop("recommended", False), **fields_,
    )
    _BUILTINS[info.id] = info
    _RECIPES[info.id] = _Recipe(build, max_bytes)


# Pre-training corpora
_register(
    _build_tinystories, id="tinystories", kind=PRETRAIN, name="TinyStories",
    tagline="Simple stories that tiny models can master",
    description="Short stories that use only words a young child would know. Even a very small "
                "model can learn to write fluent, grammatical little tales from them, which makes "
                "this the best place to start.",
    license="CDLA-Sharing-1.0", paper="2305.07759", hf="roneneldan/TinyStories",
    size_label="23 MB · 27.6k stories", approx_bytes=22_502_601, icon="moon", color="#6e5bd8",
    recommended=True,
)
_register(
    _build_shakespeare, id="tiny-shakespeare", kind=PRETRAIN, name="Tiny Shakespeare",
    tagline="Speeches from Shakespeare's plays",
    description="A single megabyte of dialogue from Shakespeare's plays, written as speaker names "
                "followed by their lines. Your model learns the layout of a script and an "
                "Elizabethan turn of phrase; it trains in minutes.",
    license="Public domain", source_label="GitHub · karpathy/char-rnn",
    source_url="https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare",
    size_label="1.1 MB · 40k lines", approx_bytes=1_115_394, icon="theater", color="#b4532a",
)
_register(
    _build_wikitext, id="wikitext-2", kind=PRETRAIN, name="WikiText-2",
    tagline="Hand-picked, high-quality Wikipedia articles",
    description="About six hundred Wikipedia articles that editors judged to be good or featured, "
                "a classic benchmark for language models. We restore normal spacing and punctuation "
                "so your model learns encyclopedic writing with section headings.",
    license="CC-BY-SA-3.0", paper="1609.07843", hf="Salesforce/wikitext",
    size_label="11 MB · 600 articles", approx_bytes=6_357_543, icon="globe", color="#1f7ae0",
)
_register(
    _build_simple_wikipedia, id="simple-wikipedia", kind=PRETRAIN, name="Simple English Wikipedia",
    tagline="Encyclopedia articles in plain words", max_bytes=30 * MB,
    description="Articles from the Simple English Wikipedia, which explains topics with short "
                "sentences and common words. Good for teaching a small model factual, easy-to-read "
                "explanations about all kinds of subjects.",
    license="CC-BY-SA-3.0", hf="wikimedia/wikipedia",
    size_label="30 MB · 8.7k articles", approx_bytes=20_451_095, icon="school", color="#0f9d8a",
)
_register(
    _build_goodwiki, id="goodwiki", kind=PRETRAIN, name="GoodWiki",
    tagline="Wikipedia's best articles, in Markdown", max_bytes=30 * MB,
    description="Long Wikipedia articles that reached Good or Featured status, kept as Markdown with "
                "headings and lists. A model trained on it learns to write detailed, well-organised "
                "prose.",
    license="CC-BY-SA-4.0", hf="euirim/goodwiki",
    size_label="30 MB · 1.8k articles", approx_bytes=20_527_315, icon="sparkles", color="#d4a017",
)
_register(
    _build_arxiv, id="arxiv-abstracts", kind=PRETRAIN, name="arXiv Abstracts",
    tagline="Titles and summaries of research papers", max_bytes=30 * MB,
    description="The titles and abstracts of tens of thousands of scientific papers posted to arXiv, "
                "mostly from physics and mathematics. Your model picks up dense technical vocabulary "
                "and the way researchers summarise their results.",
    license="CC0-1.0", paper="1905.00075", hf="gfissore/arxiv-abstracts-2021",
    size_label="30 MB · 34k abstracts", approx_bytes=_ARXIV_PREFIX, icon="atom", color="#b31b1b",
)
_register(
    _build_children_stories, id="children-stories", kind=PRETRAIN, name="Children's Stories",
    tagline="Gentle stories for young readers", max_bytes=30 * MB,
    description="AI-written stories for young children, full of friendly characters, small "
                "adventures and simple lessons. They are longer and more varied than TinyStories, "
                "so they suit slightly bigger models.",
    license="Apache-2.0", hf="ajibawa-2023/Children-Stories-Collection",
    size_label="22 MB · 10.4k stories", approx_bytes=_CHILDREN_PREFIX, icon="heart", color="#e0559b",
)
_register(
    _build_gutenberg, id="gutenberg-classics", kind=PRETRAIN, name="Classic Novels",
    tagline="29 public-domain books from Project Gutenberg",
    description="Twenty-nine much-loved public-domain books, from Pride and Prejudice and Alice in "
                "Wonderland to Dracula and Moby-Dick, with the Project Gutenberg headers and licence "
                "text removed. Teaches long-form storytelling in rich, old-fashioned English.",
    license="Public domain (US)", source_label="Project Gutenberg (PGLAF mirror)",
    source_url="https://www.gutenberg.org/",
    size_label="16 MB · 29 books",
    approx_bytes=sum(size for _, _, size in _GUTENBERG_BOOKS), icon="books", color="#8b5e3c",
)
_register(
    _build_cosmopedia, id="cosmopedia-100k", kind=PRETRAIN, name="Cosmopedia 100k",
    tagline="Synthetic textbooks, stories and blog posts", max_bytes=30 * MB,
    description="AI-written textbook sections, how-to guides, blog posts and stories that cover a "
                "wide range of school and college topics. Your model learns to explain ideas step "
                "by step in clear, tidy prose.",
    license="Apache-2.0", hf="HuggingFaceTB/cosmopedia-100k",
    size_label="30 MB · 7.9k texts", approx_bytes=27_976_253, icon="book", color="#3d8f45",
)

# Fine-tuning conversations
_register(
    _messages_recipe(_EVERYDAY), id="everyday-conversations", kind=FINETUNE,
    name="Everyday Conversations", tagline="Friendly small talk with an assistant",
    description="A couple of thousand short, friendly chats about everyday things like the weather, "
                "hobbies and food. Ideal first fine-tuning data: your model learns to greet, answer "
                "and keep a simple conversation going.",
    license="Apache-2.0", hf="HuggingFaceTB/everyday-conversations-llama3.1-2k",
    size_label="2.2 MB · 2.3k conversations", approx_bytes=_EVERYDAY.size, icon="chat",
    color="#2f80ed",
    recommended=True,
)
_register(
    _messages_recipe(_NO_ROBOTS), id="no-robots", kind=FINETUNE, name="No Robots",
    tagline="Instructions and answers written by people",
    description="About ten thousand requests answered by skilled human writers: brainstorming, "
                "rewriting, summarising, open questions and more. Teaches natural, helpful replies. "
                "Its licence allows non-commercial use only.",
    license="CC-BY-NC-4.0", hf="HuggingFaceH4/no_robots",
    size_label="13 MB · 9.5k conversations", approx_bytes=_NO_ROBOTS.size, icon="pencil",
    color="#e67e22",
)
_register(
    _build_dolly, id="dolly-15k", kind=FINETUNE, name="Dolly 15k",
    tagline="Prompts and answers written by Databricks staff",
    description="Fifteen thousand prompts with answers written by Databricks employees, covering "
                "questions, brainstorming, classification, summaries and creative writing. Some "
                "prompts include a reference passage to answer from.",
    license="CC-BY-SA-3.0", hf="databricks/databricks-dolly-15k",
    size_label="13 MB · 15k conversations", approx_bytes=_DOLLY.size, icon="list", color="#16a085",
)
_register(
    _build_oasst1, id="oasst1", kind=FINETUNE, name="OpenAssistant Conversations",
    tagline="Multi-turn chats written by volunteers",
    description="Conversations written and rated by thousands of volunteers. We follow the "
                "best-rated reply at every step to build English conversations, many with several "
                "back-and-forth turns.",
    license="Apache-2.0", paper="2304.07327", hf="OpenAssistant/oasst1",
    size_label="6.4 MB · 3.5k conversations", approx_bytes=_OASST1.size, icon="people",
    color="#8e44ad",
)
_register(
    _messages_recipe(_SMOLTALK, partial=True), id="smoltalk", kind=FINETUNE, name="SmolTalk",
    tagline="Varied assistant chats sized for small models", max_bytes=30 * MB,
    description="A broad mix of synthetic assistant conversations prepared for training small chat "
                "models: rewriting, summarising, maths, coding help and everyday questions. We use "
                "a slice of the compact smol-smoltalk edition.",
    license="Apache-2.0", paper="2502.02737", hf="HuggingFaceTB/smol-smoltalk",
    size_label="30 MB · 7.3k conversations", approx_bytes=18_464_497, icon="robot", color="#5b6ee1",
)
_register(
    _build_gsm8k, id="gsm8k", kind=FINETUNE, name="GSM8K",
    tagline="Grade-school maths word problems",
    description="About seven and a half thousand maths word problems, each solved step by step "
                "before giving the final number. Teaches your model to reason through a problem "
                "instead of guessing.",
    license="MIT", paper="2110.14168", hf="openai/gsm8k",
    size_label="4.3 MB · 7.5k problems", approx_bytes=_GSM8K.size, icon="math", color="#d35400",
)
_register(
    _build_coedit, id="coedit", kind=FINETUNE, name="CoEdIT",
    tagline="Instructions for editing text",
    description="Tens of thousands of editing requests such as fixing grammar, simplifying, "
                "paraphrasing or making a sentence more formal, each paired with the edited text. "
                "Teaches your model to follow precise rewriting instructions.",
    license="Apache-2.0", paper="2305.09857", hf="grammarly/coedit",
    size_label="22 MB · 69k edits", approx_bytes=_COEDIT.size, icon="feather", color="#27ae60",
)
_register(
    _build_code_alpaca, id="code-alpaca", kind=FINETUNE, name="Code Alpaca 20k",
    tagline="Short programming tasks with solutions",
    description="Around twenty thousand small programming requests with code answers, created with "
                "a language model. Covers many languages and everyday coding chores, from string "
                "handling to SQL queries.",
    license="CC-BY-4.0", hf="sahil2801/CodeAlpaca-20k",
    size_label="7.8 MB · 20k tasks", approx_bytes=_CODE_ALPACA.size, icon="code", color="#34495e",
)
_register(
    _build_squad, id="squad", kind=FINETUNE, name="SQuAD",
    tagline="Questions about Wikipedia passages",
    description="Wikipedia paragraphs with crowd-written questions whose answers are short phrases "
                "taken from the text. Each conversation shows a paragraph followed by up to five "
                "questions, so your model learns to read a passage and answer questions about it.",
    license="CC-BY-SA-4.0", paper="1606.05250", hf="rajpurkar/squad",
    size_label="29 MB · 87.6k questions", approx_bytes=_SQUAD.size, icon="question",
    color="#f39c12",
)
_register(
    _messages_recipe(_ULTRACHAT, partial=True), id="ultrachat", kind=FINETUNE,
    name="UltraChat 200k", tagline="Long, detailed multi-turn chats", max_bytes=30 * MB,
    description="Long conversations between a user and an AI assistant about an enormous range of "
                "topics, from a filtered version of UltraChat. Teaches detailed answers and "
                "follow-up turns; we use a slice of it.",
    license="MIT", paper="2305.14233", hf="HuggingFaceH4/ultrachat_200k",
    size_label="30 MB · 5.1k conversations", approx_bytes=21_280_396, icon="bubbles",
    color="#0f766e",
)
