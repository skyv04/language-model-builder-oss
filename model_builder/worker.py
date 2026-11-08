"""The background job process: prepares data, then trains or calibrates one project stage.

Started by the server as `python -m model_builder worker --project ID --kind KIND --device DEV --job ID`
in its own session, so a run keeps going when the window (or the server) goes away.  It reports
through files in the stage's run folder (see training.py); SIGTERM pauses at the next step.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import signal
import threading
import time
import traceback

from . import datasets, paths, training
from . import project as projects
from .data import build_conversations, build_corpus, conversations_info, corpus_info
from .tokenizer import BPE_SIZES, TextTokenizer, ensure_gpt2, label, limited_texts, load_gpt2, train_bpe

BPE_TRAINING_CHARS = 12_000_000
MAX_CONVERSATIONS = 60_000


def _fingerprint(path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def ensure_dataset(dataset_id: str, files: training.RunFiles, stop: threading.Event) -> datasets.DatasetInfo:
    info = datasets.get(dataset_id)
    if datasets.status(dataset_id)["downloaded"]:
        return info
    files.phase("prepare", f"Downloading {info.name}…", 0.0)

    def progress(done: int, total: int | None, message: str) -> None:
        files.update(phase="prepare", phase_label=message or f"Downloading {info.name}…",
                     progress=(done / total) if total else None, received=done, total=total)

    datasets.download(dataset_id, progress, stop)
    return info


def ensure_tokenizer(project, info: datasets.DatasetInfo, files: training.RunFiles, stop: threading.Event) -> TextTokenizer:
    if project.tokenizer_trained():
        return TextTokenizer.load(project.tokenizer_path)
    choice = project.data["tokenizer"]
    if choice["kind"] == "gpt2":
        files.phase("tokenizer", "Fetching the GPT-2 tokenizer…", None)
        ensure_gpt2(lambda done, total: files.update(progress=(done / total) if total else None))
        tokenizer = load_gpt2()
    else:
        size = int(choice.get("vocab_size", 10000))
        size = size if size in BPE_SIZES else 10000
        files.phase("tokenizer", f"Learning a {label('bpe', size)} vocabulary from {info.name}…", None)
        started = time.monotonic()
        tokenizer = train_bpe(limited_texts(datasets.iter_documents(info.id), BPE_TRAINING_CHARS), size)
        files.update(force=True, tokenizer_seconds=round(time.monotonic() - started, 1))
    if stop.is_set():
        raise training.StopRequested()
    tokenizer.save(project.tokenizer_path)
    training.write_json(project.folder / "tokenizer.meta.json",
                        {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size, "dataset": info.id, "time": time.time()})
    return tokenizer


def prepare_pretrain(project, files: training.RunFiles, stop: threading.Event) -> TextTokenizer:
    dataset_id = project.data["data"].get("pretrain")
    if not dataset_id:
        raise training.TrainingError("Choose a pre-training dataset first.")
    info = ensure_dataset(dataset_id, files, stop)
    tokenizer = ensure_tokenizer(project, info, files, stop)
    folder = project.data_dir("pretrain")
    wanted = {"dataset": dataset_id, "tokenizer": _fingerprint(project.tokenizer_path)}
    existing = corpus_info(folder)
    if existing and all(existing.get(key) == value for key, value in wanted.items()):
        return tokenizer
    total = max(1, int(datasets.status(dataset_id).get("documents") or 0))
    files.phase("tokenize", f"Tokenizing {info.name}…", 0.0)

    def progress(documents: int, tokens: int) -> None:
        files.update(phase="tokenize", phase_label=f"Tokenizing {info.name}: {tokens:,} tokens so far",
                     progress=min(1.0, documents / total), corpus_tokens=tokens)

    build_corpus(datasets.iter_documents(dataset_id), tokenizer, folder, progress, stop.is_set, meta=wanted)
    return tokenizer


def prepare_sft(project, files: training.RunFiles, stop: threading.Event) -> TextTokenizer:
    dataset_id = project.data["data"].get("finetune")
    if not dataset_id:
        raise training.TrainingError("Choose a fine-tuning dataset first.")
    if not project.tokenizer_trained():
        raise training.TrainingError("Pre-train the model first; fine-tuning reuses its tokenizer.")
    info = ensure_dataset(dataset_id, files, stop)
    tokenizer = TextTokenizer.load(project.tokenizer_path)
    base = training.latest(project.run_dir("pretrain"))
    context = int(base["config"]["context"]) if base else project.model_config().context
    folder = project.data_dir("sft")
    wanted = {"dataset": dataset_id, "tokenizer": _fingerprint(project.tokenizer_path), "context": context}
    existing = conversations_info(folder)
    if existing and all(existing.get(key) == value for key, value in wanted.items()):
        return tokenizer
    total = max(1, min(MAX_CONVERSATIONS, int(datasets.status(dataset_id).get("rows") or 0)))
    files.phase("tokenize", f"Formatting {info.name} as chat transcripts…", 0.0)

    def limited():
        for index, row in enumerate(datasets.iter_conversations(dataset_id)):
            if index >= MAX_CONVERSATIONS:
                return
            yield row

    def progress(rows: int) -> None:
        files.update(progress=min(1.0, rows / total))

    build_conversations(limited(), tokenizer, folder, context, progress, stop.is_set, meta=wanted)
    return tokenizer


def friendly(error: BaseException) -> str:
    if isinstance(error, MemoryError):
        return "The machine ran out of memory. Choose a smaller model shape or batch size and try again."
    text = str(error).strip() or error.__class__.__name__
    if "out of memory" in text.lower():
        return "The device ran out of memory. Choose a smaller model shape or batch size and try again."
    return text[:500]


def acquire_lock():
    """One training job at a time across the whole machine (per data directory)."""
    handle = open(paths.runtime_dir() / "worker.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def run(project_id: str, kind: str, device: str, job_id: str, calibrate_devices: list[str] | None = None) -> int:
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda *_: stop.set())
    try:
        os.nice(5)  # keep the desktop responsive on small machines
    except OSError:
        pass
    training.configure_threads()
    project = projects.load(project_id)
    stage = kind if kind in projects.STAGES else "calibrate"
    files = training.RunFiles(project.run_dir(stage), job_id=job_id, kind=kind, stage=stage, device=device,
                              project=project_id, target=None, step=None)
    lock = acquire_lock()
    if lock is None:
        files.update(force=True, state="error", error="Another training job is already running.")
        return 1
    try:
        if kind == "pretrain":
            tokenizer = prepare_pretrain(project, files, stop)
            result = training.pretrain(project, tokenizer, device, files, stop)
        elif kind == "sft":
            tokenizer = prepare_sft(project, files, stop)
            result = training.finetune(project, tokenizer, device, files, stop)
        elif kind == "dpo":
            tokenizer = TextTokenizer.load(project.tokenizer_path)
            result = training.preference_tune(project, tokenizer, device, files, stop, projects.read_preferences(project))
        elif kind == "calibrate":
            training.calibrate(project, calibrate_devices or ["cpu"], files, stop)
            result = "complete"
        else:
            raise training.TrainingError(f"Unknown job kind {kind!r}.")
        files.update(force=True, state=result, phase="done",
                     phase_label="Finished" if result == "complete" else "Paused")
        return 0
    except (training.StopRequested, InterruptedError, datasets.DownloadCancelled):
        files.update(force=True, state="paused", phase="done", phase_label="Paused")
        return 0
    except BaseException as error:  # noqa: BLE001 - every failure must reach the UI
        traceback.print_exc()
        files.update(force=True, state="error", phase="done", phase_label="Stopped by an error", error=friendly(error))
        return 1
    finally:
        lock.close()
