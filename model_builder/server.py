"""Local web server for the app window: the UI files, a JSON API, server-sent events and
NDJSON token streams (see the API contract in the README).

Security: it binds to 127.0.0.1 only; every /api request needs the per-launch secret (an
HttpOnly cookie set by opening /?token=…, or the X-LMB-Token header); Host and Origin headers
are checked so ordinary web pages cannot reach the API (DNS rebinding, cross-site requests).
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import queue
import random
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__, datasets, devices, paths, system
from . import project as projects
from .project import STAGES, ProjectError

IDLE_SECONDS = 90
MAX_JSON_BODY = 4 << 20
POLL_SECONDS = 0.25
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".woff2": "font/woff2", ".json": "application/json", ".txt": "text/plain; charset=utf-8", ".ico": "image/x-icon",
}
DEFAULT_SETTINGS = {"device": "auto", "reading_pane": False, "completed": [], "last_page": "welcome",
                    "sidebar_hidden": False, "keep_training": True}
DEVICE_CHOICES = ("auto", "cpu", "vulkan", "cuda", "mps")
BALLOT_PROMPTS = [
    "What makes someone a good friend?", "Describe a rainy afternoon in two sentences.", "How do plants drink water?",
    "Tell me something surprising about the Moon.", "What is a simple way to remember new words?",
    "Why does bread go stale?", "Give me one tip for sleeping better.", "Suggest a name for a small orange cat.",
    "What is a map used for?", "How can I be a good neighbour?", "Which season do you like best, and why?",
    "Suggest an easy breakfast.", "What should I do if I feel bored?", "Explain what a library is.",
]
BALLOT_SETTINGS = {"temperature": 0.9, "top_k": 50, "top_p": 0.95, "min_p": 0.0, "max_tokens": 80}
STAGE_NAMES = {"pretrain": "pre-training", "sft": "fine-tuning", "dpo": "preference tuning"}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def conflict(message: str) -> ApiError:
    return ApiError(409, message)


def write_json(path: Path, payload: dict, mode: int | None = None) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if mode is not None:
        os.chmod(temporary, mode)
    temporary.replace(path)


def platform_label() -> str:
    return f"{system.os_label()} · {platform.machine()}"


def engine_label() -> str:
    try:
        from importlib.metadata import version

        return f"PyTorch {version('torch').split('+')[0]}"
    except Exception:  # noqa: BLE001
        return "PyTorch"


def open_folder(folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    return {"path": str(folder), "opened": system.open_target(str(folder), env=devices.engine_env())}


# ---------- settings ----------


class Settings:
    def __init__(self) -> None:
        self.path = paths.data_dir() / "settings.json"
        self.lock = threading.Lock()
        stored = projects.read_json(self.path) or {}
        self.values = deepcopy(DEFAULT_SETTINGS)
        self.values.update({key: value for key, value in stored.items() if key in DEFAULT_SETTINGS})
        self.current = stored.get("current")
        self.initialized = bool(stored.get("initialized"))

    def public(self) -> dict:
        with self.lock:
            return deepcopy(self.values)

    def update(self, patch: dict) -> dict:
        with self.lock:
            for key, value in (patch or {}).items():
                if key == "device" and value in DEVICE_CHOICES:
                    self.values["device"] = value
                elif key in {"reading_pane", "sidebar_hidden", "keep_training"}:
                    self.values[key] = bool(value)
                elif key == "completed" and isinstance(value, list):
                    self.values["completed"] = list(dict.fromkeys(str(item)[:64] for item in value))[:64]
                elif key == "last_page" and isinstance(value, str):
                    self.values["last_page"] = value[:64]
            self._save()
            return deepcopy(self.values)

    def set_current(self, project_id: str | None) -> None:
        with self.lock:
            self.current = project_id
            self.initialized = True
            self._save()

    def _save(self) -> None:
        write_json(self.path, {**self.values, "current": self.current, "initialized": self.initialized})


# ---------- events ----------


class EventHub:
    def __init__(self) -> None:
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        self.last_seen = time.monotonic()

    def subscribe(self) -> queue.Queue:
        client: queue.Queue = queue.Queue(maxsize=4000)
        with self.lock:
            self.clients.add(client)
        return client

    def unsubscribe(self, client: queue.Queue) -> None:
        with self.lock:
            self.clients.discard(client)
            self.last_seen = time.monotonic()

    def count(self) -> int:
        with self.lock:
            return len(self.clients)

    def emit(self, event_type: str, /, **payload) -> None:
        message = json.dumps({**payload, "type": event_type}, ensure_ascii=False)
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            try:
                client.put_nowait(message)
            except queue.Full:
                pass


# ---------- dataset downloads started from the UI ----------


class Downloads:
    def __init__(self, app: "App") -> None:
        self.app = app
        self.lock = threading.Lock()
        self.current: dict | None = None
        self.cancel_event: threading.Event | None = None

    def snapshot(self) -> dict | None:
        with self.lock:
            return dict(self.current) if self.current else None

    def active(self, dataset_id: str | None = None) -> bool:
        current = self.current
        return bool(current and current["state"] == "downloading" and dataset_id in (None, current["dataset"]))

    def start(self, dataset_id: str) -> dict:
        info = datasets.get(dataset_id)
        if self.app.jobs.uses_dataset(dataset_id):
            raise conflict(f"{info.name} is being prepared for the current training run.")
        with self.lock:
            if self.active():
                raise conflict(f"{self.current['name']} is still downloading. Wait for it to finish or cancel it first.")
            self.current = {"dataset": dataset_id, "name": info.name, "received": 0, "total": info.approx_bytes or None,
                            "state": "downloading", "message": f"Starting {info.name}…"}
            self.cancel_event = threading.Event()
            threading.Thread(target=self._run, args=(dataset_id, self.cancel_event), daemon=True, name="download").start()
        snapshot = self.snapshot()
        self.app.hub.emit("download", **snapshot)
        return snapshot

    def _run(self, dataset_id: str, cancel: threading.Event) -> None:
        last = 0.0

        def progress(done: int, total: int | None, message: str) -> None:
            nonlocal last
            with self.lock:
                self.current.update(received=done, total=total, message=message)
            now = time.monotonic()
            if now - last >= 0.25:
                last = now
                self.app.hub.emit("download", **self.snapshot())

        try:
            datasets.download(dataset_id, progress, cancel)
            outcome = {"state": "done", "message": "Ready"}
        except datasets.DownloadCancelled:
            outcome = {"state": "cancelled", "message": "Cancelled"}
        except Exception as error:  # noqa: BLE001
            outcome = {"state": "error", "error": str(error)[:400], "message": "Download failed"}
        with self.lock:
            self.current.update(outcome)
        self.app.hub.emit("download", **self.snapshot())
        self.app.hub.emit("state")

    def cancel(self, dataset_id: str) -> dict | None:
        with self.lock:
            if self.current and self.current["dataset"] == dataset_id and self.current["state"] == "downloading":
                self.cancel_event.set()
                self.current["message"] = "Cancelling…"
        return self.snapshot()


# ---------- background jobs ----------


class Tail:
    """New JSON lines appended to a file, skipping records at or before `min_step`."""

    def __init__(self, path: Path, min_step: int) -> None:
        self.path, self.min_step = path, min_step
        try:
            stat = path.stat()
            self.inode, self.offset = stat.st_ino, stat.st_size
        except OSError:
            self.inode, self.offset = None, 0
        self.buffer = b""

    def read(self) -> list[dict]:
        try:
            stat = self.path.stat()
        except OSError:
            return []
        if stat.st_ino != self.inode or stat.st_size < self.offset:
            self.inode, self.offset, self.buffer = stat.st_ino, 0, b""
        if stat.st_size <= self.offset:
            return []
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read(min(stat.st_size - self.offset, 8 << 20))
        self.offset += len(chunk)
        lines = (self.buffer + chunk).split(b"\n")
        self.buffer = lines.pop()
        records = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if int(record.get("step", 0)) > self.min_step:
                records.append(record)
        return records


def _process_alive(pid: int) -> bool:
    """True while `pid` is still a running training worker (not a zombie, not a recycled pid)."""
    if pid <= 0:
        return False
    if not os.path.exists("/proc/self/stat"):
        return _process_alive_ps(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    state = stat.rsplit(")", 1)[-1].split()[0] if ")" in stat else "?"
    return state not in {"Z", "X"} and b"model_builder" in cmdline and b"worker" in cmdline


def _process_alive_ps(pid: int) -> bool:
    """The same check through ps, for systems without /proc such as macOS."""
    try:
        completed = subprocess.run(["ps", "-o", "stat=", "-o", "command=", "-p", str(pid)], stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    state, _, command = completed.stdout.strip().partition(" ")
    return completed.returncode == 0 and bool(state) and not state.startswith("Z") \
        and "model_builder" in command and "worker" in command


class JobManager:
    def __init__(self, app: "App") -> None:
        self.app = app
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.job: dict | None = None
        self.status: dict = {}
        self.status_stamp = None
        self.signature = None
        self.latest_stamp = None
        self.tails: list[tuple[str, Tail]] = []
        stored = projects.read_json(self.path)
        if stored and _process_alive(int(stored.get("pid", 0))):
            self._attach(stored)
        elif stored:
            self.path.unlink(missing_ok=True)

    @property
    def path(self) -> Path:
        return paths.runtime_dir() / "job.json"

    def _run_dir(self, job: dict) -> Path:
        return paths.projects_dir() / job["project"] / "runs" / job["stage"]

    def _attach(self, job: dict) -> None:
        from .training import latest

        self.job = job
        folder = self._run_dir(job)
        checkpoint = latest(folder) if job["stage"] in STAGES else None
        min_step = int(checkpoint["step"]) if checkpoint and not job.get("fresh") else -1
        if job.get("resume_step") is not None:
            min_step = int(job["resume_step"])
        self.tails = [("metrics", Tail(folder / "metrics.jsonl", min_step)), ("samples", Tail(folder / "samples.jsonl", min_step))]
        self.status, self.status_stamp, self.signature = {}, None, None
        self.latest_stamp = self._stamp(folder / "latest.json")

    @staticmethod
    def _stamp(path: Path):
        try:
            stat = path.stat()
            return (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None

    def alive(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return bool(self.job) and _process_alive(int(self.job["pid"]))

    def running(self) -> dict | None:
        job = self.job
        return {"project": job["project"], "kind": job["kind"], "stage": job["stage"]} if job else None

    def uses_dataset(self, dataset_id: str) -> bool:
        job = self.job
        if not job or job["kind"] not in {"pretrain", "sft"} or self.status.get("phase") not in {None, "prepare"}:
            return False
        try:
            project = projects.load(job["project"])
        except KeyError:
            return False
        return dataset_id in project.data["data"].values()

    def snapshot(self) -> dict | None:
        job, status = self.job, self.status
        if not job:
            return None
        state = "stopping" if job.get("stopping") else ("running" if status.get("phase") else "starting")
        return {
            "id": job["id"], "kind": job["kind"], "stage": job["stage"], "project": job["project"], "state": state,
            "phase": status.get("phase") or "prepare", "phase_label": status.get("phase_label") or "Starting…",
            "progress": status.get("progress"), "step": status.get("step"), "target": status.get("target"),
            "device": status.get("device") or job["device"], "started": job["started"], "eta": status.get("eta"),
        }

    def start(self, kind: str) -> dict:
        from .training import latest

        project = self.app.current_project()
        with self.lock:
            if self.job:
                raise conflict("A job is already running. Pause it before starting another one.")
            if kind in STAGES:
                reason = project.missing_for(kind)
                if reason:
                    raise conflict(reason)
                info = project.stage_info(kind)
                if info["state"] == "complete":
                    raise conflict(f"{STAGE_NAMES[kind].capitalize()} already reached its step target. Raise the "
                                   "target to keep training, or reset the stage to start over.")
            elif kind != "calibrate":
                raise ApiError(400, f"Unknown job kind {kind!r}.")
            for dataset_id in project.data["data"].values():
                if dataset_id and self.app.downloads.active(dataset_id):
                    raise conflict("That dataset is still downloading. Wait for the download to finish first.")
            setting = self.app.settings.public()["device"]
            device = devices.resolve(setting, project.estimates())
            if kind == "dpo" and device == "vulkan":
                device = "cpu"
            stage = kind if kind in STAGES else "calibrate"
            folder = project.run_dir(stage)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "status.json").unlink(missing_ok=True)
            checkpoint = latest(folder) if stage in STAGES else None
            job_id = uuid.uuid4().hex[:12]
            command = [sys.executable, "-m", "model_builder", "worker", "--project", project.id, "--kind", kind,
                       "--device", device, "--job", job_id]
            if kind == "calibrate":
                command += ["--devices", ",".join(devices.available_ids())]
            with open(folder / "worker.log", "w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env=devices.engine_env(), cwd=str(Path(__file__).resolve().parent.parent),
                )
            job = {"id": job_id, "kind": kind, "stage": stage, "project": project.id, "pid": process.pid,
                   "device": device, "started": time.time(), "resume_step": int(checkpoint["step"]) if checkpoint else -1}
            write_json(self.path, job)
            self.process = process
            self._attach(job)
        snapshot = self.snapshot()
        self.app.hub.emit("job", job=snapshot)
        self.app.hub.emit("state")
        return snapshot

    def stop(self) -> dict | None:
        with self.lock:
            if not self.job:
                return None
            self.job["stopping"] = True
            try:
                os.kill(int(self.job["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass
        snapshot = self.snapshot()
        self.app.hub.emit("job", job=snapshot)
        return snapshot

    def stop_for_project(self, project_id: str) -> None:
        if self.job and self.job["project"] == project_id:
            raise conflict("Pause the running job for this project first.")

    def poll(self) -> None:
        with self.lock:
            if not self.job:
                return
            job = self.job
            folder = self._run_dir(job)
            stamp = self._stamp(folder / "status.json")
            if stamp != self.status_stamp:
                self.status_stamp = stamp
                status = projects.read_json(folder / "status.json") or {}
                if status.get("job_id") == job["id"]:
                    self.status = status
                    signature = (status.get("phase"), status.get("state"), job.get("stopping"))
                    if signature != self.signature:
                        self.signature = signature
                        self.app.hub.emit("job", job=self.snapshot())
                    if status.get("phase") != "train":
                        self._emit_progress(status)
            for name, tail in self.tails:
                records = tail.read()
                if name == "metrics":
                    train = [record for record in records if "loss" in record]
                    for record in train[-40:]:
                        self._emit_progress({**self.status, **record, "phase": "train"})
                    for record in records:
                        if "val_loss" in record:
                            self.app.hub.emit("eval", stage=job["stage"], step=record["step"], val_loss=record["val_loss"])
                else:
                    for record in records:
                        self.app.hub.emit("sample", stage=job["stage"], **record)
            latest_stamp = self._stamp(folder / "latest.json")
            if latest_stamp != self.latest_stamp:
                self.latest_stamp = latest_stamp
                checkpoint = projects.read_json(folder / "latest.json") or {}
                self.app.hub.emit("checkpoint", stage=job["stage"], step=checkpoint.get("step"))
            if not self.alive():
                self._finish(folder)

    def _emit_progress(self, record: dict) -> None:
        job = self.job
        target = record.get("target", self.status.get("target"))
        step = record.get("step")
        progress = record.get("progress")
        if record.get("phase") == "train" and target and step is not None:
            progress = step / target
        self.app.hub.emit(
            "progress", stage=job["stage"], kind=job["kind"], step=step, target=target, loss=record.get("loss"),
            lr=record.get("lr"), tok_per_s=record.get("tok_per_s"), elapsed=record.get("elapsed"),
            tokens_seen=record.get("tokens_seen"), phase=record.get("phase"), phase_label=self.status.get("phase_label"),
            progress=progress, eta=self.status.get("eta"), device=self.status.get("device") or job["device"],
            accuracy=record.get("accuracy"), reward_margin=record.get("reward_margin"),
        )

    def _finish(self, folder: Path) -> None:
        job = self.job
        if self.process is not None:
            self.process.wait()
        status = projects.read_json(folder / "status.json") or {}
        if status.get("job_id") != job["id"]:
            status = {}
        result = status.get("state") if status.get("state") in {"complete", "paused", "error"} else "error"
        error = status.get("error") if result == "error" else None
        if result == "error" and not error:
            error = "The training process stopped unexpectedly"
            try:
                lines = [line for line in (folder / "worker.log").read_text(errors="replace").splitlines() if line.strip()]
                if lines:
                    error += f": {lines[-1][:300]}"
            except OSError:
                pass
            if self.process is not None and self.process.returncode in (-9, 137):
                error = "The system stopped the training process, probably because memory ran out. Try a smaller model or batch size."
        snapshot = self.snapshot()
        snapshot["state"] = "finished"
        if job["kind"] == "calibrate" and status.get("results"):
            self.app.hub.emit("calibration", results=status["results"])
        self.job, self.process, self.tails, self.status = None, None, [], {}
        self.path.unlink(missing_ok=True)
        self.app.hub.emit("job_end", job=snapshot, result=result, error=error)
        self.app.hub.emit("state")


# ---------- loaded models for sampling and chat ----------


class ModelCache:
    def __init__(self, size: int = 2) -> None:
        self.size = size
        self.items: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.tokenizers: dict = {}

    def tokenizer(self, project):
        from .tokenizer import TextTokenizer

        path = project.tokenizer_path
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            raise conflict("This model has no tokenizer yet; it is created when pre-training starts.") from None
        key = (project.id, stamp)
        if key not in self.tokenizers:
            self.tokenizers = {key: TextTokenizer.load(path)}
        return self.tokenizers[key]

    def get(self, project, stage: str):
        from . import training

        if stage not in STAGES:
            raise ApiError(400, "Unknown stage.")
        folder = project.run_dir(stage)
        for attempt in range(3):
            info = training.latest(folder)
            if info is None:
                raise conflict(f"There is no {STAGE_NAMES[stage]} checkpoint yet.")
            key = (project.id, stage, info["weights"])
            with self.lock:
                if key in self.items:
                    self.items.move_to_end(key)
                    return self.items[key]
            try:
                model, info = training.load_model(folder)
            except (FileNotFoundError, OSError):
                time.sleep(0.2)  # a new checkpoint replaced this one while we read it
                continue
            entry = (model, info, self.tokenizer(project))
            with self.lock:
                self.items[key] = entry
                while len(self.items) > self.size:
                    self.items.popitem(last=False)
            return entry
        raise conflict("The checkpoint is being rewritten; try again in a moment.")

    def best_stage(self, project) -> str:
        from .training import latest

        for stage in ("dpo", "sft", "pretrain"):
            if latest(project.run_dir(stage)):
                return stage
        raise conflict("Train the model first: there is no checkpoint to look inside yet.")


_torch_ready = False


def _prepare_torch() -> None:
    global _torch_ready
    if not _torch_ready:
        import torch

        torch.set_num_threads(max(1, devices.cpu_cores() // 2))
        _torch_ready = True


# ---------- the application ----------


class App:
    def __init__(self, persist: bool = False) -> None:
        self.persist = persist
        self.token = secrets.token_urlsafe(24)
        self.settings = Settings()
        self.hub = EventHub()
        self.models = ModelCache()
        self.downloads = Downloads(self)
        self.jobs = JobManager(self)
        self.last_request = time.monotonic()
        self.port = 0
        self.stopping = threading.Event()
        if not self.settings.initialized and not projects.list_projects():
            self.settings.set_current(projects.create("My First Model").id)

    def current_project(self, required: bool = True):
        project_id = self.settings.current
        if project_id:
            try:
                return projects.load(project_id)
            except KeyError:
                pass
        available = projects.list_projects()
        if available:
            self.settings.set_current(available[0].id)
            return available[0]
        if required:
            raise conflict("Create a project first.")
        return None

    def state(self) -> dict:
        running = self.jobs.running()
        current = self.current_project(required=False)
        return {
            "app": {"version": __version__, "engine": engine_label(), "data_dir": str(paths.data_dir()),
                    "platform": platform_label(), "file_manager": system.file_manager_name(),
                    "gpu_probe": "running" if devices.probing() else "done"},
            "settings": self.settings.public(),
            "devices": devices.available(),
            "auto_device": devices.resolve("auto"),  # what Auto trains on before Project setup has timed the devices
            "projects": [project.summary(running) for project in projects.list_projects()],
            "current": current.id if current else None,
            "job": self.jobs.snapshot(),
            "download": self.downloads.snapshot(),
        }

    def project_detail(self, project=None) -> dict:
        project = project or self.current_project()
        detail = project.detail(self.jobs.running())
        # Report the device "Auto" would really use here (e.g. CUDA before calibrating, never a vanished GPU).
        detail["estimates"]["auto_device"] = devices.resolve("auto", detail["estimates"])
        return detail

    def monitor(self, server: ThreadingHTTPServer) -> None:
        last_error = None
        while not self.stopping.is_set():
            try:
                self.jobs.poll()
                last_error = None
            except Exception as error:  # noqa: BLE001 - keep monitoring whatever happens
                if repr(error) != last_error:
                    last_error = repr(error)
                    traceback.print_exc()
            idle = (not self.persist and self.hub.count() == 0 and not self.jobs.job and not self.downloads.active()
                    and time.monotonic() - max(self.hub.last_seen, self.last_request) > IDLE_SECONDS)
            if idle:
                self.stopping.set()
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            time.sleep(POLL_SECONDS if self.jobs.job else 1.0)


# ---------- API handlers ----------


def _dataset_entry(info) -> dict:
    entry = asdict(info)
    entry["status"] = datasets.status(info.id)
    return entry


def _int_list(values, limit: int) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ApiError(400, "prompt_ids must be a non-empty list of token ids.")
    ids = []
    for value in values[-4096:]:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < limit:
            raise ApiError(400, "prompt_ids contains an id outside the vocabulary.")
        ids.append(value)
    return ids


def _fit_chat(tokenizer, messages: list[dict], system: str | None, context: int, reserve: int) -> list[int]:
    """Encode the transcript, dropping the oldest turns until it leaves room for a reply."""
    from .data import encode_chat

    turns = [message for message in messages if isinstance(message, dict) and message.get("role") in {"user", "assistant"}]
    while True:
        ids, _ = encode_chat(tokenizer, turns, add_generation_prompt=True, system=system)
        if len(ids) <= context - reserve or len(turns) <= 1:
            return ids[-(context - 1):]
        turns = turns[2:] if len(turns) > 2 else turns[1:]


class Api:
    def __init__(self, app: App) -> None:
        self.app = app
        self.routes = [
            ("GET", r"/api/ping", self.ping),
            ("GET", r"/api/state", self.state),
            ("POST", r"/api/settings", self.settings),
            ("POST", r"/api/projects", self.create_project),
            ("POST", r"/api/projects/select", self.select_project),
            ("GET", r"/api/projects/current", self.current_project),
            ("PATCH", r"/api/projects/current", self.update_project),
            ("POST", r"/api/projects/current/reset", self.reset_stage),
            ("POST", r"/api/projects/current/reveal", self.reveal_project),
            ("GET", r"/api/projects/current/runs/(pretrain|sft|dpo)/metrics", self.metrics),
            ("GET", r"/api/projects/current/runs/(pretrain|sft|dpo)/samples", self.samples),
            ("POST", r"/api/projects/current/generate", self.generate),
            ("POST", r"/api/projects/current/attention", self.attention),
            ("POST", r"/api/projects/current/ballot", self.ballot),
            ("GET", r"/api/projects/current/preferences", self.preferences),
            ("POST", r"/api/projects/current/preferences", self.add_preference),
            ("DELETE", r"/api/projects/current/preferences", self.clear_preferences),
            ("POST", r"/api/projects/current/export", self.export),
            ("DELETE", r"/api/projects/([a-z0-9][a-z0-9-]*)", self.delete_project),
            ("POST", r"/api/jobs", self.start_job),
            ("POST", r"/api/jobs/stop", self.stop_job),
            ("GET", r"/api/datasets", self.list_datasets),
            ("POST", r"/api/datasets/import", self.import_dataset),
            ("GET", r"/api/datasets/([a-z0-9][a-z0-9-]*)/preview", self.preview_dataset),
            ("POST", r"/api/datasets/([a-z0-9][a-z0-9-]*)/download", self.download_dataset),
            ("POST", r"/api/datasets/([a-z0-9][a-z0-9-]*)/cancel", self.cancel_download),
            ("DELETE", r"/api/datasets/([a-z0-9][a-z0-9-]*)", self.delete_dataset),
            ("POST", r"/api/tokenize", self.tokenize),
            ("POST", r"/api/reveal-data", self.reveal_data),
            ("POST", r"/api/devices/refresh", self.refresh_devices),
            ("POST", r"/api/window-closed", self.window_closed),
        ]
        self.compiled = [(method, re.compile(pattern + r"/?"), handler) for method, pattern, handler in self.routes]

    def find(self, method: str, path: str):
        allowed = False
        for route_method, pattern, handler in self.compiled:
            match = pattern.fullmatch(path)
            if match:
                if route_method == method:
                    return handler, match.groups()
                allowed = True
        raise ApiError(405 if allowed else 404, "Not found." if not allowed else "Method not allowed.")

    # ----- app state -----

    def ping(self, request) -> dict:
        return {"ok": True, "version": __version__, "pid": os.getpid()}

    def state(self, request) -> dict:
        return self.app.state()

    def settings(self, request) -> dict:
        return {"settings": self.app.settings.update(request.json())}

    def window_closed(self, request) -> dict:
        """The window went away: pause training unless the user wants runs to continue."""
        if self.app.jobs.job and not self.app.settings.public()["keep_training"]:
            self.app.jobs.stop()
            return {"stopped": True}
        return {"stopped": False}

    def refresh_devices(self, request) -> dict:
        devices.refresh(lambda: self.app.hub.emit("state"), background=True)
        return {"devices": devices.available(), "probing": True}

    # ----- projects -----

    def create_project(self, request) -> dict:
        body = request.json()
        project = projects.create(str(body.get("name") or "My Model"), str(body.get("preset") or "starter"))
        self.app.settings.set_current(project.id)
        self.app.hub.emit("state")
        return {"project": self.app.project_detail(project)}

    def select_project(self, request) -> dict:
        project = projects.load(str(request.json().get("id", "")))
        self.app.settings.set_current(project.id)
        self.app.hub.emit("state")
        return {"project": self.app.project_detail(project)}

    def current_project(self, request) -> dict:
        return self.app.project_detail()

    def update_project(self, request) -> dict:
        project = self.app.current_project()
        body = request.json()
        running = self.app.jobs.running()
        if running and running["project"] == project.id and any(key in body for key in ("config", "tokenizer", "data")):
            raise conflict("Pause the running job before changing the model or its data.")
        project.update(body)
        self.app.hub.emit("state")
        return self.app.project_detail(project)

    def reset_stage(self, request) -> dict:
        project = self.app.current_project()
        self.app.jobs.stop_for_project(project.id)
        project.reset(str(request.json().get("stage", "")))
        self.app.hub.emit("state")
        return self.app.project_detail(project)

    def reveal_project(self, request) -> dict:
        return open_folder(self.app.current_project().folder)

    def reveal_data(self, request) -> dict:
        return open_folder(paths.data_dir())

    def delete_project(self, request, project_id: str) -> dict:
        self.app.jobs.stop_for_project(project_id)
        projects.delete(project_id)
        if self.app.settings.current == project_id:
            remaining = projects.list_projects()
            self.app.settings.set_current(remaining[0].id if remaining else None)
        self.app.hub.emit("state")
        return {"ok": True, "current": self.app.settings.current}

    # ----- runs -----

    def metrics(self, request, stage: str) -> dict:
        from .training import read_jsonl

        project = self.app.current_project()
        records = read_jsonl(project.run_dir(stage) / "metrics.jsonl")
        train = [record for record in records if "loss" in record]
        if len(train) > 4000:
            stride = len(train) / 4000
            train = [train[int(index * stride)] for index in range(4000)] + [train[-1]]
        result = {
            "train": [[record["step"], record["loss"]] for record in train],
            "val": [[record["step"], record["val_loss"]] for record in records if "val_loss" in record],
            "lr": [[record["step"], record["lr"]] for record in train],
        }
        if stage == "dpo":
            result["accuracy"] = [[record["step"], record.get("accuracy")] for record in train]
            result["reward_margin"] = [[record["step"], record.get("reward_margin")] for record in train]
        return result

    def samples(self, request, stage: str) -> dict:
        from .training import read_jsonl

        project = self.app.current_project()
        return {"samples": read_jsonl(project.run_dir(stage) / "samples.jsonl")[-200:]}

    # ----- jobs -----

    def start_job(self, request) -> dict:
        return {"job": self.app.jobs.start(str(request.json().get("kind", "")))}

    def stop_job(self, request) -> dict:
        return {"job": self.app.jobs.stop()}

    # ----- datasets -----

    def list_datasets(self, request) -> dict:
        kind = request.query.get("kind")
        return {"datasets": [_dataset_entry(info) for info in datasets.catalog(kind or None)]}

    def preview_dataset(self, request, dataset_id: str) -> dict:
        info = datasets.get(dataset_id)
        if not datasets.status(info.id)["downloaded"]:
            raise conflict(f"Download {info.name} to see what is inside.")
        return datasets.preview(info.id)

    def download_dataset(self, request, dataset_id: str) -> dict:
        return {"download": self.app.downloads.start(dataset_id)}

    def cancel_download(self, request, dataset_id: str) -> dict:
        return {"download": self.app.downloads.cancel(dataset_id)}

    def import_dataset(self, request) -> dict:
        kind = request.query.get("kind", "pretrain")
        filename = unquote(request.headers.get("X-Filename") or "upload.txt")
        name = request.query.get("name") or Path(filename).stem
        data = request.raw(limit=getattr(datasets, "MAX_IMPORT_BYTES", 200 << 20))
        info = datasets.import_custom(name, kind, filename, data)
        self.app.hub.emit("state")
        return {"dataset": _dataset_entry(info)}

    def delete_dataset(self, request, dataset_id: str) -> dict:
        info = datasets.get(dataset_id)
        if self.app.jobs.uses_dataset(dataset_id) or self.app.downloads.active(dataset_id):
            raise conflict(f"{info.name} is in use right now.")
        if info.custom:
            datasets.delete_custom(info.id)
        else:
            datasets.delete_download(info.id)
        self.app.hub.emit("state")
        return {"ok": True}

    # ----- tokenizers -----

    def tokenize(self, request) -> dict:
        from .tokenizer import byte_tokens, label, shared_gpt2

        body = request.json()
        text = str(body.get("text", ""))[:20000]
        which = body.get("tokenizer", "project")
        if which == "bytes":
            return {"tokens": byte_tokens(text), "tokenizer": {"kind": "bytes", "vocab_size": 256, "label": "UTF-8 bytes"}}
        if which == "gpt2":
            try:
                tokenizer = shared_gpt2()
            except OSError as error:
                raise ApiError(503, f"Couldn't download the GPT-2 tokenizer ({error}).") from None
        else:
            project = self.app.current_project()
            tokenizer = self.app.models.tokenizer(project)
        return {"tokens": tokenizer.tokens(text, special=True),
                "tokenizer": {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size, "label": label(tokenizer.kind, tokenizer.vocab_size)}}

    # ----- generation -----

    def generate(self, request) -> None:
        from .sampling import GenerationStats, SamplerSettings, generate

        _prepare_torch()
        body = request.json()
        project = self.app.current_project()
        stage = body.get("stage") or "pretrain"
        model, info, tokenizer = self.app.models.get(project, stage)
        settings = SamplerSettings.from_dict(body.get("settings"))
        if settings.seed is None:
            settings.seed = random.randrange(1, 2**31 - 1)
        context = model.config.context
        if body.get("messages") is not None:
            reserve = min(settings.max_tokens, context // 3)
            ids = _fit_chat(tokenizer, body["messages"], body.get("system"), context, reserve)
            stop_ids = {tokenizer.end_id, tokenizer.eot_id}
        elif body.get("prompt_ids") is not None:
            ids = _int_list(body["prompt_ids"], model.config.vocab_size)
            stop_ids = {tokenizer.eot_id}
        else:
            prompt = str(body.get("prompt") or "")
            ids = tokenizer.encode(prompt) if prompt else [tokenizer.eot_id]
            ids = ids or [tokenizer.eot_id]
            stop_ids = {tokenizer.eot_id}
        ids = ids[-(context - 1):]
        request.start_stream("application/x-ndjson")
        request.write_line({
            "type": "start", "prompt_tokens": [{"id": i, "text": tokenizer.piece(i)} for i in ids[-1024:]],
            "prompt_text": tokenizer.decode(ids), "context_tokens": len(ids), "context_limit": context,
            "seed": settings.seed, "step": info["step"], "stage": stage,
        })
        stats = GenerationStats()
        try:
            for record in generate(model, tokenizer, ids, settings, stop_ids, stats, should_stop=lambda: request.closed):
                request.write_line({"type": "token", "token": record})
            request.write_line({"type": "end", **stats.to_dict()})
        except (BrokenPipeError, ConnectionResetError):
            request.closed = True
        except Exception as error:  # noqa: BLE001 - report errors inside the stream
            traceback.print_exc()
            request.write_line({"type": "error", "error": str(error)[:300]})

    def attention(self, request) -> dict:
        from .sampling import attention

        _prepare_torch()
        body = request.json()
        project = self.app.current_project()
        stage = body.get("stage") or self.app.models.best_stage(project)
        model, info, tokenizer = self.app.models.get(project, stage)
        result = attention(model, tokenizer, str(body.get("text") or "")[:2000])
        return {**result, "stage": stage, "step": info["step"], "heads": model.config.heads, "blocks": model.config.blocks}

    def ballot(self, request) -> dict:
        from .sampling import SamplerSettings, generate

        _prepare_torch()
        body = request.json()
        project = self.app.current_project()
        model, info, tokenizer = self.app.models.get(project, "sft")
        prompt = str(body.get("prompt") or random.choice(BALLOT_PROMPTS)).strip()[:1000]
        ids = _fit_chat(tokenizer, [{"role": "user", "content": prompt}], None, model.config.context, 32)
        answers = []
        seed = random.randrange(1, 2**31 - 1)
        for attempt in range(4):
            settings = SamplerSettings.from_dict({**BALLOT_SETTINGS, "seed": seed + attempt})
            records = list(generate(model, tokenizer, ids, settings, {tokenizer.end_id, tokenizer.eot_id}))
            tokens = [record["id"] for record in records if record["id"] not in tokenizer.special_ids]
            text = tokenizer.decode(tokens).strip()
            if text and text not in answers:
                answers.append(text)
            if len(answers) == 2:
                break
        while len(answers) < 2:
            answers.append(answers[0] if answers else "…")
        return {"prompt": prompt, "a": {"text": answers[0]}, "b": {"text": answers[1]}, "step": info["step"]}

    def preferences(self, request) -> dict:
        project = self.app.current_project()
        pairs = projects.read_preferences(project)
        return {"pairs": pairs, "count": len(pairs), "required": 4}

    def add_preference(self, request) -> dict:
        body = request.json()
        values = [str(body.get(key) or "").strip() for key in ("prompt", "chosen", "rejected")]
        if not all(values):
            raise ApiError(400, "A preference needs a prompt, a chosen answer and a rejected answer.")
        if values[1] == values[2]:
            raise ApiError(400, "The two answers are identical, so there is nothing to prefer.")
        project = self.app.current_project()
        count = projects.add_preference(project, *(value[:4000] for value in values))
        self.app.hub.emit("state")
        return {"count": count, "required": 4}

    def clear_preferences(self, request) -> dict:
        projects.clear_preferences(self.app.current_project())
        self.app.hub.emit("state")
        return {"count": 0, "required": 4}

    # ----- export -----

    def export(self, request) -> dict:
        from . import training
        from .data import DEFAULT_SYSTEM

        project = self.app.current_project()
        stage = str(request.json().get("stage") or "")
        stage = stage if stage in STAGES else self.app.models.best_stage(project)
        folder = project.run_dir(stage)
        info = training.latest(folder)
        if info is None:
            raise conflict(f"There is no {STAGE_NAMES[stage]} checkpoint to export yet.")
        try:
            target_dir = paths.downloads_dir()
        except OSError:
            target_dir = paths.data_dir() / "exports"
            target_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", project.name.lower()).strip("-") or "model"
        target = target_dir / f"{slug}-{stage}-step{info['step']}.zip"
        counter = 2
        while target.exists():
            target = target_dir / f"{slug}-{stage}-step{info['step']}-{counter}.zip"
            counter += 1
        tokenizer = json.loads(project.tokenizer_path.read_text(encoding="utf-8"))
        kind = tokenizer.pop("lmb_kind", "bpe")
        config = {
            "architecture": "gpt2-style decoder (tied embeddings, pre-LayerNorm, tanh GELU)",
            **info["config"], "layer_norm_eps": 1e-5, "dtype": "float32",
            "stage": stage, "step": info["step"], "tokens_seen": info.get("tokens_seen"), "base": info.get("base"),
            "tokenizer": {"kind": kind, "file": "tokenizer.json"},
            "special_tokens": {"end_of_text": "<|endoftext|>", "system": "<|system|>", "user": "<|user|>",
                               "assistant": "<|assistant|>", "end_of_turn": "<|end|>"},
            "chat_template": "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>\n",
            "default_system_prompt": DEFAULT_SYSTEM,
            "created_with": f"Language Model Builder, open-source edition {__version__}",
        }
        readme = EXPORT_README.format(name=project.name, stage=STAGE_NAMES[stage], step=info["step"], **info["config"])
        temporary = target.with_suffix(".part")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder / info["weights"], "model.safetensors", compress_type=zipfile.ZIP_STORED)
            archive.writestr("config.json", json.dumps(config, indent=2))
            archive.writestr("tokenizer.json", json.dumps(tokenizer, ensure_ascii=False))
            archive.writestr("README.md", readme)
            source = Path(__file__).with_name("model.py").read_text(encoding="utf-8")
            archive.writestr("model.py", source.replace("from .config import", "from config import"))
            archive.write(Path(__file__).with_name("config.py"), "config.py")
        temporary.replace(target)
        return {"path": str(target), "folder": str(target_dir)}


EXPORT_README = """# {name}

A small GPT-style language model trained from scratch with the open-source edition of Language Model Builder.

* Stage: {stage}, step {step}
* Shape: {blocks} blocks, width {width}, {heads} attention heads, {context}-token context, vocabulary {vocab_size}
* Files: `model.safetensors` (float32 weights), `config.json`, `tokenizer.json` (Hugging Face
  `tokenizers` format), and `model.py`/`config.py` (the PyTorch architecture definition)

## Load it

```python
import json, torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from config import ModelConfig
from model import GPT

settings = json.load(open("config.json"))
model = GPT(ModelConfig.from_dict(settings)).eval()
model.load_state_dict(load_file("model.safetensors"))
tokenizer = Tokenizer.from_file("tokenizer.json")
```

Chat models expect the transcript format in `config.json` (`chat_template`): each message is a
role token, a newline, the message text, `<|end|>` and a newline. Generation should stop at `<|end|>`.
"""


# ---------- HTTP plumbing ----------


class Request:
    def __init__(self, handler: "Handler", query: dict) -> None:
        self.handler = handler
        self.query = query
        self.headers = handler.headers
        self.closed = False
        self.streaming = False
        self._json = None

    def raw(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError(400, "Bad Content-Length.") from None
        if length > limit:
            raise ApiError(413, f"That file is larger than {limit // (1 << 20)} MB.")
        return self.handler.rfile.read(length) if length else b""

    def json(self) -> dict:
        if self._json is None:
            raw = self.raw(MAX_JSON_BODY)
            try:
                value = json.loads(raw) if raw.strip() else {}
            except ValueError:
                raise ApiError(400, "The request body is not valid JSON.") from None
            self._json = value if isinstance(value, dict) else {}
        return self._json

    def start_stream(self, content_type: str) -> None:
        self.streaming = True
        handler = self.handler
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.end_headers()

    def write_line(self, payload: dict) -> None:
        if self.closed:
            return
        try:
            self.handler.wfile.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            self.handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.closed = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "LanguageModelBuilder"
    app: App
    api: Api

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if os.environ.get("LMB_DEBUG"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def do_GET(self) -> None:  # noqa: N802
        self.dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self.dispatch("POST")

    def do_PATCH(self) -> None:  # noqa: N802
        self.dispatch("PATCH")

    def do_DELETE(self) -> None:  # noqa: N802
        self.dispatch("DELETE")

    def send_json(self, status: int, payload, extra: dict | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in {f"127.0.0.1:{self.app.port}", f"localhost:{self.app.port}"}

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://127.0.0.1:{self.app.port}", f"http://localhost:{self.app.port}"}

    def _authorised(self) -> bool:
        supplied = self.headers.get("X-LMB-Token")
        if not supplied:
            for part in (self.headers.get("Cookie") or "").split(";"):
                name, _, value = part.strip().partition("=")
                if name == "lmb_token":
                    supplied = value
        return bool(supplied) and secrets.compare_digest(supplied, self.app.token)

    def dispatch(self, method: str) -> None:
        self.app.last_request = time.monotonic()
        url = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        try:
            if not self._host_ok():
                raise ApiError(403, "Unexpected Host header.")
            if not url.path.startswith("/api/"):
                if method != "GET":
                    raise ApiError(405, "Method not allowed.")
                self.static(url.path, query)
                return
            if not self._authorised():
                raise ApiError(401, "This window is no longer connected to the engine. Close it and open the app again.")
            if method != "GET" and not self._origin_ok():
                raise ApiError(403, "Cross-origin requests are not allowed.")
            if url.path == "/api/events" and method == "GET":
                self.event_stream()
                return
            handler, groups = self.api.find(method, url.path)
            request = Request(self, query)
            result = handler(request, *groups)
            if not request.streaming:
                self.send_json(200, result if result is not None else {"ok": True})
        except ApiError as error:
            self._error(error.status, error.message)
        except ProjectError as error:
            self._error(409, str(error))
        except KeyError as error:
            detail = str(error.args[0]) if error.args else ""
            self._error(404, detail if " " in detail else "That project or dataset no longer exists.")
        except (ValueError, TypeError) as error:
            self._error(400, str(error) or "Invalid request.")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._error(500, f"Something went wrong: {error}")

    def _error(self, status: int, message: str) -> None:
        try:
            self.send_json(status, {"error": message})
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass

    def event_stream(self) -> None:
        client = self.app.hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            while not self.app.stopping.is_set():
                try:
                    message = client.get(timeout=15)
                    self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.hub.unsubscribe(client)

    def static(self, path: str, query: dict) -> None:
        if query.get("token"):
            if secrets.compare_digest(query["token"], self.app.token):
                self.send_response(302)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", f"lmb_token={self.app.token}; Path=/; HttpOnly; SameSite=Strict")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            raise ApiError(401, "Invalid token.")
        root = paths.ui_dir().resolve()
        relative = unquote(path).lstrip("/") or "index.html"
        target = (root / relative).resolve()
        if root not in target.parents and target != root or not target.is_file():
            raise ApiError(404, "Not found.")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", STATIC_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if target.suffix in {".html", ".js", ".css"} else "max-age=86400")
        self.send_header("X-Content-Type-Options", "nosniff")
        if target.suffix == ".html":
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; "
                             "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; "
                             "object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def server_info_path() -> Path:
    return paths.runtime_dir() / "server.json"


def read_server_info() -> dict | None:
    return projects.read_json(server_info_path())


def serve(port: int = 0, persist: bool = False, ready=None) -> int:
    """Run the server in this process until it goes idle (or forever with persist=True)."""
    lock = open(paths.runtime_dir() / "server.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another Language Model Builder server is already running for this data folder.", file=sys.stderr)
        return 1
    app = App(persist=persist)
    handler = type("BoundHandler", (Handler,), {"app": app, "api": Api(app)})
    server = Server(("127.0.0.1", port), handler)
    app.port = server.server_address[1]
    info = {"port": app.port, "token": app.token, "pid": os.getpid(), "version": __version__, "started": time.time()}
    write_json(server_info_path(), info, mode=0o600)
    devices.refresh(lambda: app.hub.emit("state"), background=True)
    monitor = threading.Thread(target=app.monitor, args=(server,), daemon=True, name="monitor")
    monitor.start()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (app.stopping.set(), threading.Thread(target=server.shutdown, daemon=True).start()))
    if ready:
        ready(info)
    print(f"Language Model Builder engine listening on http://127.0.0.1:{app.port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        app.stopping.set()
        server.server_close()
        current = read_server_info()
        if current and current.get("pid") == os.getpid():
            server_info_path().unlink(missing_ok=True)
        lock.close()
    return 0
