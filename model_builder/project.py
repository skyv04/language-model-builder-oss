"""Projects: one folder per model with its settings, tokenizer, data caches and runs.

Layout of projects/<id>/:
    project.json            name, model shape, tokenizer choice, datasets, recipes
    calibration.json        measured seconds per step, keyed by model shape and batch size
    tokenizer.json          trained (or downloaded) tokenizer
    data/pretrain/          train.bin, val.bin, corpus.json
    data/sft/               *_tokens.npy, *_targets.npy, *_offsets.npy, conversations.json
    preferences.jsonl       DPO ballot results
    runs/<stage>/           latest.json, checkpoints/, metrics.jsonl, samples.jsonl, status.json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from copy import deepcopy
from pathlib import Path

from . import paths, system
from .config import LIMITS, ModelConfig

STAGES = ("pretrain", "sft", "dpo")
STAGE_PARENT = {"sft": "pretrain", "dpo": "sft"}
SCHEMA = 2

PRESETS = {
    "pocket": {"context": 128, "width": 128, "blocks": 4, "heads": 4},
    "starter": {"context": 256, "width": 256, "blocks": 6, "heads": 8},
    "large": {"context": 512, "width": 512, "blocks": 8, "heads": 8},
}
PRESET_LABELS = {"pocket": "Pocket", "starter": "Starter", "large": "Large", "custom": "Custom"}

DEFAULT_RECIPES = {
    "pretrain": {
        "batch_size": 16, "learning_rate": 6e-4, "warmup": 100, "steps": 2000, "mode": "steps",
        "eval_every": 100, "sample_every": 250, "checkpoint_every": 250,
    },
    "sft": {
        "batch_size": 8, "learning_rate": 1e-4, "warmup": 20, "steps": 400, "mode": "steps",
        "eval_every": 50, "sample_every": 100, "checkpoint_every": 100,
    },
    "dpo": {"beta": 0.1, "learning_rate": 2e-5, "steps": 60, "batch_size": 4},
}
RECIPE_LIMITS = {
    "batch_size": (1, 256), "learning_rate": (1e-7, 1e-1), "warmup": (0, 100_000), "steps": (1, 10_000_000),
    "eval_every": (1, 1_000_000), "sample_every": (1, 1_000_000), "checkpoint_every": (1, 1_000_000), "beta": (0.001, 10.0),
}
SHAPE_LIMITS = {key: LIMITS[key] for key in ("context", "width", "blocks", "heads")}


class ProjectError(Exception):
    """A request that cannot be satisfied in the project's current state (HTTP 409)."""


def _now() -> float:
    return round(time.time(), 3)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "model"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def _meminfo(field: str) -> int | None:
    """MemTotal or MemAvailable in bytes: /proc/meminfo on Linux, sysctl on macOS."""
    if system.is_macos():
        try:
            total = int(system.sysctl("hw.memsize"))
            if field == "MemTotal":
                return total
            # The share of memory the kernel counts as free, the same figure `memory_pressure` reports.
            return total * int(system.sysctl("kern.memorystatus_level")) // 100 if field == "MemAvailable" else None
        except ValueError:
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def memory_total() -> int:
    return _meminfo("MemTotal") or 8 << 30


def memory_available() -> int | None:
    """Memory free for new work right now; on phones this is far below the total."""
    return _meminfo("MemAvailable")


def memory_warning(needed: int, batch: int) -> str | None:
    total, available = memory_total(), memory_available()
    if available and needed > 0.8 * available:
        return (
            f"Training this shape at batch {batch} needs roughly {needed / 1e9:.1f} GB, but only about "
            f"{available / 1e9:.1f} GB is free right now. Pick a smaller shape or batch size, or close other apps."
        )
    if needed > 0.45 * total:
        return (
            f"Training this shape at batch {batch} needs roughly {needed / 1e9:.1f} GB, but this machine has "
            f"{total / 1e9:.1f} GB in total. Pick a smaller shape or batch size."
        )
    return None


class Project:
    def __init__(self, folder: Path, data: dict) -> None:
        self.folder = folder
        self.data = data

    # ---------- basic properties ----------

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def tokenizer_path(self) -> Path:
        return self.folder / "tokenizer.json"

    @property
    def preferences_path(self) -> Path:
        return self.folder / "preferences.jsonl"

    def data_dir(self, kind: str) -> Path:
        return self.folder / "data" / kind

    def run_dir(self, stage: str) -> Path:
        if stage not in (*STAGES, "calibrate"):
            raise ValueError(f"unknown stage {stage!r}")
        return self.folder / "runs" / stage

    def save(self) -> None:
        self.data["updated"] = _now()
        _write_json(self.folder / "project.json", self.data)

    # ---------- model shape and tokenizer ----------

    def vocab_size(self) -> int:
        tokenizer = self.data["tokenizer"]
        if tokenizer["kind"] == "gpt2":
            from .tokenizer import GPT2_VOCAB

            return GPT2_VOCAB
        trained = read_json(self.folder / "tokenizer.meta.json")
        return int(trained["vocab_size"]) if trained else int(tokenizer["vocab_size"])

    def model_config(self) -> ModelConfig:
        shape = self.data["config"]
        return ModelConfig(
            vocab_size=self.vocab_size(), context=int(shape["context"]), width=int(shape["width"]),
            blocks=int(shape["blocks"]), heads=int(shape["heads"]),
        )

    def tokenizer_trained(self) -> bool:
        return self.tokenizer_path.exists()

    def locked(self) -> bool:
        return latest_checkpoint(self, "pretrain") is not None

    def tokenizer_label(self) -> str:
        from .tokenizer import label

        return label(self.data["tokenizer"]["kind"], self.vocab_size())

    # ---------- runs ----------

    def stage_info(self, stage: str, running: dict | None = None) -> dict:
        latest = latest_checkpoint(self, stage)
        recipe = self.data["recipes"][stage]
        status = read_json(self.run_dir(stage) / "status.json") or {}
        is_running = bool(running and running.get("stage") == stage and running.get("project") == self.id)
        target = None if recipe.get("mode") == "until_stop" else recipe.get("steps")
        info = {
            "state": "empty",
            "step": 0,
            "target": target,
            "loss": None,
            "val_loss": None,
            "lr": None,
            "tok_per_s": None,
            "elapsed": 0.0,
            "tokens_seen": 0,
            "checkpoint_step": None,
            "device": None,
            "base": None,
            "ready_reason": None,
        }
        if latest:
            info.update(
                step=latest["step"], loss=latest.get("loss"), val_loss=latest.get("val_loss"), lr=latest.get("lr"),
                elapsed=latest.get("elapsed", 0.0), tokens_seen=latest.get("tokens_seen", 0),
                checkpoint_step=latest["step"], device=latest.get("device"), base=latest.get("base"),
                tok_per_s=latest.get("tok_per_s"),
            )
        if is_running and status:
            for key in ("step", "loss", "lr", "tok_per_s", "elapsed", "tokens_seen", "device"):
                if status.get(key) is not None:
                    info[key] = status[key]
            if status.get("val_loss") is not None:
                info["val_loss"] = status["val_loss"]
        if is_running:
            info["state"] = "running"
        elif latest:
            # Raising the step target turns a finished stage back into a resumable one.
            done = target is not None and latest["step"] >= target
            info["state"] = "complete" if done else "paused"
        else:
            reason = self.missing_for(stage)
            info["state"] = "ready" if reason is None else "empty"
            info["ready_reason"] = reason
        if stage in STAGE_PARENT and info["base"] is None:
            parent = latest_checkpoint(self, STAGE_PARENT[stage])
            if parent:
                info["base"] = {"stage": STAGE_PARENT[stage], "step": parent["step"]}
        return info

    def missing_for(self, stage: str) -> str | None:
        """Why a stage cannot start yet, or None if it can."""
        if stage == "pretrain":
            if not self.data["data"].get("pretrain"):
                return "Choose a pre-training dataset first."
            return None
        if stage == "sft":
            if latest_checkpoint(self, "pretrain") is None:
                return "Pre-train the model first; fine-tuning starts from its latest checkpoint."
            if not self.data["data"].get("finetune"):
                return "Choose a fine-tuning dataset first."
            return None
        if stage == "dpo":
            if latest_checkpoint(self, "sft") is None:
                return "Fine-tune the model first; preference tuning starts from the SFT checkpoint."
            if preference_count(self) < 4:
                return "Bank at least four preference pairs to unlock a tuning run."
            return None
        return None

    def status(self, running: dict | None = None) -> dict:
        if running and running.get("project") == self.id and running.get("kind") in STAGES:
            label = {"pretrain": "Pre-training…", "sft": "Fine-tuning…", "dpo": "Preference tuning…"}[running["kind"]]
            return {"label": label, "tone": "red", "pulse": True}
        if running and running.get("project") == self.id and running.get("kind") == "calibrate":
            return {"label": "Measuring speed…", "tone": "blue", "pulse": True}
        for stage, done_label in (("dpo", "Preference-tuned"), ("sft", "Fine-tuned"), ("pretrain", "Pre-trained")):
            latest = latest_checkpoint(self, stage)
            if latest:
                info = self.stage_info(stage)
                if info["state"] == "complete":
                    return {"label": done_label, "tone": "green", "pulse": False}
                return {"label": f"Paused at step {latest['step']:,}", "tone": "orange", "pulse": False}
        if self.data["data"].get("pretrain"):
            return {"label": "Data selected", "tone": "blue", "pulse": False}
        return {"label": "Not trained yet", "tone": "grey", "pulse": False}

    def summary(self, running: dict | None = None) -> dict:
        return {"id": self.id, "name": self.name, "created": self.data.get("created"), "status": self.status(running)}

    def estimates(self) -> dict:
        config = self.model_config()
        batch = int(self.data["recipes"]["pretrain"]["batch_size"])
        memory = config.training_memory_bytes(batch)
        warning = memory_warning(memory, batch)
        entry = (read_json(self.folder / "calibration.json") or {}).get(calibration_key(self)) or {}
        results = entry.get("results", {})
        step_seconds = {device: (results.get(device) or {}).get("step_seconds") for device in ("cpu", "vulkan", "cuda", "mps")}
        measured = {device: seconds for device, seconds in step_seconds.items() if seconds}
        auto = min(measured, key=measured.get) if measured else "cpu"
        return {
            "parameters": config.parameter_count(),
            "weights_bytes": config.parameter_count() * 4,
            "training_memory_bytes": memory,
            "memory_warning": warning,
            "tokens_per_step": batch * config.context,
            "step_seconds": step_seconds,
            "calibrated": bool(measured),
            "calibrated_at": entry.get("time"),
            "auto_device": auto,
        }

    def detail(self, running: dict | None = None) -> dict:
        corpus = read_json(self.data_dir("pretrain") / "corpus.json")
        payload = self.summary(running)
        tokenizer = dict(self.data["tokenizer"])
        tokenizer.update(trained=self.tokenizer_trained(), label=self.tokenizer_label(), vocab_size=self.vocab_size())
        payload.update(
            config=dict(self.data["config"]),
            tokenizer=tokenizer,
            data={
                "pretrain": self.data["data"].get("pretrain"),
                "finetune": self.data["data"].get("finetune"),
                "corpus_tokens": (corpus["train_tokens"] + corpus["val_tokens"]) if corpus else None,
                "corpus_documents": corpus["documents"] if corpus else None,
                "corpus_dataset": corpus.get("dataset") if corpus else None,
            },
            recipes=deepcopy(self.data["recipes"]),
            estimates=self.estimates(),
            locked=self.locked(),
            stages={stage: self.stage_info(stage, running) for stage in STAGES},
            limits={key: list(value) for key, value in SHAPE_LIMITS.items()},
            presets=deepcopy(PRESETS),
            preferences=preference_count(self),
            folder=str(self.folder),
        )
        return payload

    # ---------- updates ----------

    def update(self, patch: dict) -> None:
        if "name" in patch:
            name = str(patch["name"]).strip()
            if not name:
                raise ProjectError("The project needs a name.")
            self.data["name"] = name[:80]
        shape_change = "config" in patch or "tokenizer" in patch
        if shape_change and self.locked():
            raise ProjectError("This model already has pre-training checkpoints. Reset pre-training to change its shape or tokenizer.")
        if "config" in patch:
            self._update_config(patch["config"] or {})
        if "tokenizer" in patch:
            self._update_tokenizer(patch["tokenizer"] or {})
        if "data" in patch:
            self._update_data(patch["data"] or {})
        if "recipes" in patch:
            self._update_recipes(patch["recipes"] or {})
        self.save()

    def _update_config(self, changes: dict) -> None:
        config = dict(self.data["config"])
        preset = changes.get("preset")
        if preset in PRESETS:
            config.update(PRESETS[preset])
            config["preset"] = preset
        for key in SHAPE_LIMITS:
            if key in changes:
                low, high = SHAPE_LIMITS[key]
                config[key] = int(min(high, max(low, int(changes[key]))))
                if preset not in PRESETS:
                    config["preset"] = "custom"
        if config["width"] % config["heads"] or (config["width"] // config["heads"]) % 2:
            raise ProjectError("The width must split evenly into attention heads with an even size per head.")
        if config.get("preset") in PRESETS and any(config[key] != PRESETS[config["preset"]][key] for key in SHAPE_LIMITS):
            config["preset"] = "custom"
        ModelConfig(vocab_size=self.vocab_size(), **{key: config[key] for key in SHAPE_LIMITS})
        self.data["config"] = config

    def _update_tokenizer(self, changes: dict) -> None:
        from .tokenizer import BPE_SIZES

        tokenizer = dict(self.data["tokenizer"])
        kind = changes.get("kind", tokenizer["kind"])
        if kind not in {"bpe", "gpt2"}:
            raise ProjectError("Unknown tokenizer kind.")
        tokenizer["kind"] = kind
        if "vocab_size" in changes:
            size = int(changes["vocab_size"])
            if size not in BPE_SIZES:
                raise ProjectError(f"Vocabulary size must be one of {', '.join(map(str, BPE_SIZES))}.")
            tokenizer["vocab_size"] = size
        if tokenizer != self.data["tokenizer"]:
            self.data["tokenizer"] = tokenizer
            self.invalidate_tokenizer()

    def _update_data(self, changes: dict) -> None:
        for kind in ("pretrain", "finetune"):
            if kind not in changes:
                continue
            value = changes[kind] or None
            if value is not None:
                from . import datasets

                info = datasets.get(value)
                expected = "pretrain" if kind == "pretrain" else "finetune"
                if info.kind != expected:
                    raise ProjectError(f"{info.name} is not a {'pre-training' if kind == 'pretrain' else 'fine-tuning'} dataset.")
            if value != self.data["data"].get(kind):
                self.data["data"][kind] = value
                if kind == "pretrain":
                    shutil.rmtree(self.data_dir("pretrain"), ignore_errors=True)
                    if not self.locked() and self.data["tokenizer"]["kind"] == "bpe":
                        self.invalidate_tokenizer()
                else:
                    shutil.rmtree(self.data_dir("sft"), ignore_errors=True)

    def _update_recipes(self, changes: dict) -> None:
        for stage, values in changes.items():
            if stage not in self.data["recipes"] or not isinstance(values, dict):
                continue
            recipe = self.data["recipes"][stage]
            for key, value in values.items():
                if key == "mode":
                    if value in {"steps", "until_stop"}:
                        recipe["mode"] = value
                    continue
                if key not in recipe or key not in RECIPE_LIMITS:
                    continue
                low, high = RECIPE_LIMITS[key]
                number = float(value)
                number = min(high, max(low, number))
                recipe[key] = number if key in {"learning_rate", "beta"} else int(round(number))

    def invalidate_tokenizer(self) -> None:
        self.tokenizer_path.unlink(missing_ok=True)
        (self.folder / "tokenizer.meta.json").unlink(missing_ok=True)
        shutil.rmtree(self.folder / "data", ignore_errors=True)

    def reset(self, stage: str) -> None:
        if stage not in STAGES:
            raise ProjectError("Unknown stage.")
        for name in STAGES[STAGES.index(stage):]:
            shutil.rmtree(self.run_dir(name), ignore_errors=True)
        if stage == "pretrain":
            shutil.rmtree(self.data_dir("sft"), ignore_errors=True)
        self.save()


# ---------- module-level helpers ----------


def calibration_key(project: Project) -> str:
    config = project.data["config"]
    batch = project.data["recipes"]["pretrain"]["batch_size"]
    return f"{config['context']}x{config['width']}x{config['blocks']}x{config['heads']}-v{project.vocab_size()}-b{batch}"


def latest_checkpoint(project: Project, stage: str) -> dict | None:
    latest = read_json(project.run_dir(stage) / "latest.json")
    if not latest:
        return None
    if not (project.run_dir(stage) / latest.get("weights", "")).exists():
        return None
    return latest


def preference_count(project: Project) -> int:
    try:
        with open(project.preferences_path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def read_preferences(project: Project) -> list[dict]:
    pairs = []
    try:
        with open(project.preferences_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return pairs


def add_preference(project: Project, prompt: str, chosen: str, rejected: str) -> int:
    record = {"prompt": prompt, "chosen": chosen, "rejected": rejected, "time": _now()}
    with open(project.preferences_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return preference_count(project)


def clear_preferences(project: Project) -> None:
    project.preferences_path.unlink(missing_ok=True)


def create(name: str, preset: str = "starter") -> Project:
    name = (name or "My Model").strip()[:80] or "My Model"
    project_id = slugify(name)
    folder = paths.projects_dir() / project_id
    folder.mkdir(parents=True)
    shape = dict(PRESETS.get(preset, PRESETS["starter"]))
    shape["preset"] = preset if preset in PRESETS else "starter"
    data = {
        "schema": SCHEMA,
        "id": project_id,
        "name": name,
        "created": _now(),
        "config": shape,
        "tokenizer": {"kind": "bpe", "vocab_size": 10000 if shape["width"] >= 256 else 4096},
        "data": {"pretrain": None, "finetune": None},
        "recipes": deepcopy(DEFAULT_RECIPES),
    }
    project = Project(folder, data)
    project.save()
    return project


def load(project_id: str) -> Project:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", project_id or ""):
        raise KeyError(project_id)
    folder = paths.projects_dir() / project_id
    data = read_json(folder / "project.json")
    if not data or data.get("schema") != SCHEMA:
        raise KeyError(project_id)
    for stage, defaults in DEFAULT_RECIPES.items():
        recipe = data.setdefault("recipes", {}).setdefault(stage, {})
        for key, value in defaults.items():
            recipe.setdefault(key, value)
    data.setdefault("data", {"pretrain": None, "finetune": None})
    return Project(folder, data)


def list_projects() -> list[Project]:
    projects = []
    for folder in sorted(paths.projects_dir().iterdir()):
        if folder.is_dir():
            try:
                projects.append(load(folder.name))
            except KeyError:
                continue
    projects.sort(key=lambda project: project.data.get("created", 0))
    return projects


def delete(project_id: str) -> None:
    project = load(project_id)
    shutil.rmtree(project.folder)
