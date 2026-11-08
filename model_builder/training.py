"""Training runs: pre-training, supervised fine-tuning (SFT), preference tuning (DPO), calibration.

Every stage keeps its files in projects/<id>/runs/<stage>/:
    status.json     live state, rewritten atomically at most twice a second
    metrics.jsonl   {"step", "loss", "lr", "tok_per_s", "elapsed", "tokens_seen"} or {"step", "val_loss"}
    samples.jsonl   {"step", "prompt", "text", "time"} written at every sample point
    latest.json     the newest checkpoint (weights + optimizer state under checkpoints/)
Runs can be paused (SIGTERM) and resumed; progress after the newest checkpoint is discarded.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .data import IGNORE, ChatRows, PreferencePair, TokenStream, encode_chat, encode_preference, pad_rows
from .model import GPT, ModelConfig, learning_rate_at, make_optimizer
from .sampling import sample_text
from .tokenizer import TextTokenizer

STATUS_INTERVAL = 0.5
EVAL_BATCHES = 6
CHECKPOINT_SECONDS = 600
SAMPLE_TOKENS = 48
SAMPLE_SEED = 1234
SFT_QUESTIONS = [
    "What is a good way to start the morning?",
    "Can you explain what clouds are made of?",
    "Give me two tips for learning to cook.",
    "Why do cats like boxes?",
    "Write one sentence about the ocean.",
    "What should I pack for a picnic?",
]
DEVICE_LABELS = {"cpu": "CPU", "vulkan": "GPU (Vulkan)", "cuda": "GPU (CUDA)", "mps": "GPU (Metal)"}
TORCH_GPUS = ("cuda", "mps")  # GPUs PyTorch trains on directly (NVIDIA CUDA; Apple Metal through MPS)


class StopRequested(Exception):
    """The user paused the run."""


class TrainingError(RuntimeError):
    """A run cannot continue; the message is shown to the user."""


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return records


def configure_threads() -> None:
    """Leave a core or two to the desktop: on a 6-core phone 4 threads also train faster than 6."""
    from .devices import cpu_cores

    cores = cpu_cores()
    try:
        threads = int(os.environ["LMB_THREADS"])
    except (KeyError, ValueError):
        threads = cores - 2 if cores >= 6 else cores - 1 if cores >= 3 else cores
    torch.set_num_threads(max(1, threads))


# ---------- run files ----------


class RunFiles:
    """Status, metrics, samples and checkpoints of one stage."""

    def __init__(self, folder: Path, **status) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.state = {"state": "running", "phase": "prepare", "phase_label": "Getting ready…", "progress": None,
                      "pid": os.getpid(), "started": time.time(), **status}
        self._written = 0.0
        self._metrics = None
        self.update(force=True)

    def update(self, force: bool = False, **fields) -> None:
        self.state.update(fields)
        now = time.monotonic()
        if force or now - self._written >= STATUS_INTERVAL:
            self._written = now
            self.state["time"] = time.time()
            write_json(self.folder / "status.json", self.state)

    def phase(self, phase: str, label: str, progress: float | None = None) -> None:
        self.update(force=True, phase=phase, phase_label=label, progress=progress)

    def _append(self, name: str, record: dict) -> None:
        with open(self.folder / name, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def metric(self, record: dict) -> None:
        if self._metrics is None:
            self._metrics = open(self.folder / "metrics.jsonl", "a", encoding="utf-8")
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def sample(self, record: dict) -> None:
        self._append("samples.jsonl", record)

    def keep_until(self, step: int) -> None:
        """Forget metrics and samples recorded after `step` (they belong to lost progress)."""
        for name in ("metrics.jsonl", "samples.jsonl"):
            path = self.folder / name
            if not path.exists():
                continue
            kept = [record for record in read_jsonl(path) if record.get("step", 0) <= step]
            temporary = path.with_name(f".{name}.tmp")
            temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in kept), encoding="utf-8")
            temporary.replace(path)

    def close(self) -> None:
        if self._metrics is not None:
            self._metrics.close()
            self._metrics = None


def latest(folder: Path) -> dict | None:
    info = read_json(folder / "latest.json")
    if not info or not (folder / info.get("weights", "missing")).exists():
        return None
    return info


def save_checkpoint(folder: Path, model: GPT, optimizer: torch.optim.Optimizer | None, info: dict) -> dict:
    """Write weights (+ optimizer) under a step-specific name, then point latest.json at them."""
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    stem = f"step-{int(info['step']):07d}"
    weights = checkpoints / f"{stem}.safetensors"
    state = {name: tensor.detach().to("cpu").contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, str(weights.with_suffix(".tmp")), metadata={"config": json.dumps(model.config.to_dict())})
    weights.with_suffix(".tmp").replace(weights)
    record = {**info, "weights": f"checkpoints/{weights.name}", "config": model.config.to_dict(), "time": time.time()}
    if optimizer is not None:
        optimizer_path = checkpoints / f"{stem}.optim.pt"
        torch.save(optimizer.state_dict(), optimizer_path.with_suffix(".tmp"))
        optimizer_path.with_suffix(".tmp").replace(optimizer_path)
        record["optimizer"] = f"checkpoints/{optimizer_path.name}"
    write_json(folder / "latest.json", record)
    keep = {weights.name, f"{stem}.optim.pt"}
    for old in checkpoints.iterdir():
        if old.name not in keep:
            old.unlink(missing_ok=True)
    return record


def load_model(folder: Path, device: str = "cpu") -> tuple[GPT, dict]:
    info = latest(folder)
    if info is None:
        raise FileNotFoundError(f"no checkpoint in {folder}")
    model = GPT(ModelConfig.from_dict(info["config"]))
    model.load_state_dict(load_file(str(folder / info["weights"])))
    return model.to(torch_device(device)).eval(), info


def torch_device(device: str) -> torch.device:
    return torch.device(device) if device in TORCH_GPUS else torch.device("cpu")


def optimizer_to(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Move optimizer state (loaded on the CPU) to the parameters' device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(torch_device(device))


# ---------- steppers: one optimisation step on a device ----------


class TorchStepper:
    """AdamW steps with PyTorch (CPU, CUDA or Metal)."""

    def __init__(self, model: GPT, optimizer: torch.optim.Optimizer, device: str) -> None:
        self.model, self.optimizer = model, optimizer
        self.device = torch_device(device)

    def _tensors(self, tokens: np.ndarray, targets: np.ndarray):
        return torch.from_numpy(tokens).long().to(self.device), torch.from_numpy(targets).long().to(self.device)

    def train_step(self, tokens: np.ndarray, targets: np.ndarray, lr: float) -> float:
        x, y = self._tensors(tokens, targets)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        loss = self.model.loss(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def eval_loss(self, tokens: np.ndarray, targets: np.ndarray) -> float:
        self.model.eval()
        x, y = self._tensors(tokens, targets)
        value = float(self.model.loss(x, y))
        self.model.train()
        return value

    def sync(self) -> None:
        pass

    def close(self) -> None:
        pass


class VulkanStepper:
    """Steps on the GPU through tinygrad; the torch model is refreshed on sync()."""

    def __init__(self, model: GPT, optimizer: torch.optim.Optimizer, step: int) -> None:
        from .gpu_model import GpuTrainer

        self.model, self.optimizer = model, optimizer
        self.trainer = GpuTrainer(model, optimizer, step)

    def train_step(self, tokens: np.ndarray, targets: np.ndarray, lr: float) -> float:
        return float(self.trainer.train_step(tokens.astype(np.int32), targets.astype(np.int32), lr))

    def eval_loss(self, tokens: np.ndarray, targets: np.ndarray) -> float:
        return float(self.trainer.eval_loss(tokens.astype(np.int32), targets.astype(np.int32)))

    def sync(self) -> None:
        self.trainer.sync_to_torch(self.model, self.optimizer)

    def close(self) -> None:
        self.trainer.close()


def make_stepper(model: GPT, optimizer: torch.optim.Optimizer, device: str, step: int):
    if device == "vulkan":
        return VulkanStepper(model, optimizer, step)
    return TorchStepper(model, optimizer, device)


def warmup_steps(device: str) -> int:
    """Steps after a (re)start that compile kernels or warm caches, so they say nothing about speed."""
    return 2 if device == "vulkan" else 1


# ---------- the shared training loop ----------


@dataclass
class LoopPlan:
    stage: str
    steps: int | None  # None = until the user stops it
    batch_size: int
    learning_rate: float
    warmup: int
    eval_every: int
    sample_every: int
    checkpoint_every: int

    @classmethod
    def from_recipe(cls, stage: str, recipe: dict) -> "LoopPlan":
        return cls(
            stage=stage,
            steps=None if recipe.get("mode") == "until_stop" else int(recipe["steps"]),
            batch_size=int(recipe["batch_size"]),
            learning_rate=float(recipe["learning_rate"]),
            warmup=int(recipe.get("warmup", 0)),
            eval_every=max(1, int(recipe.get("eval_every", 100))),
            sample_every=max(1, int(recipe.get("sample_every", 250))),
            checkpoint_every=max(1, int(recipe.get("checkpoint_every", 250))),
        )


def train_loop(
    *,
    plan: LoopPlan,
    files: RunFiles,
    model: GPT,
    device: str,
    next_batch: Callable[[], tuple[np.ndarray, np.ndarray]],
    eval_batches: list[tuple[np.ndarray, np.ndarray]],
    make_sample: Callable[[GPT, int], dict],
    stop: threading.Event,
    base: dict | None = None,
    init_from: Path | None = None,
) -> str:
    """Train until the target step or a stop request; returns "complete" or "paused"."""
    resume = latest(files.folder)
    optimizer = make_optimizer(model, plan.learning_rate)
    if resume:
        model.load_state_dict(load_file(str(files.folder / resume["weights"])))
        if resume.get("optimizer") and (files.folder / resume["optimizer"]).exists():
            optimizer.load_state_dict(torch.load(files.folder / resume["optimizer"], map_location="cpu", weights_only=True))
        step = int(resume["step"])
        tokens_seen = int(resume.get("tokens_seen", 0))
        elapsed = float(resume.get("elapsed", 0.0))
        base = resume.get("base", base)
        files.keep_until(step)
    else:
        if init_from is not None:
            model.load_state_dict(load_file(str(init_from)))
        step, tokens_seen, elapsed = 0, 0, 0.0
        files.keep_until(-1)
    model.to(torch_device(device))
    if device != "vulkan":
        optimizer_to(optimizer, device)
    if plan.steps is not None and step >= plan.steps:
        return "complete"

    files.phase("train", f"Starting on {DEVICE_LABELS.get(device, device)}…" if device != "vulkan" else "Compiling GPU kernels (first run can take a few minutes)…")
    stepper = make_stepper(model, optimizer, device, step)
    last = {"loss": resume.get("loss") if resume else None, "val_loss": resume.get("val_loss") if resume else None,
            "lr": None, "tok_per_s": resume.get("tok_per_s") if resume else None}
    step_time = None
    session_steps = 0
    last_checkpoint = time.monotonic()
    sample_index = len(read_jsonl(files.folder / "samples.jsonl"))
    session_start = time.monotonic()
    elapsed_base = elapsed

    def checkpoint(complete: bool) -> None:
        nonlocal last_checkpoint
        stepper.sync()
        save_checkpoint(files.folder, model, optimizer, {
            "step": step, "tokens_seen": tokens_seen, "elapsed": round(elapsed, 2), "loss": last["loss"],
            "val_loss": last["val_loss"], "lr": last["lr"], "tok_per_s": last["tok_per_s"], "device": device,
            "base": base, "complete": complete, "target": plan.steps,
        })
        last_checkpoint = time.monotonic()
        files.update(force=True, checkpoint_step=step)

    def evaluate() -> None:
        if not eval_batches:
            return
        value = sum(stepper.eval_loss(x, y) for x, y in eval_batches) / len(eval_batches)
        last["val_loss"] = round(value, 4)
        files.metric({"step": step, "val_loss": last["val_loss"]})
        files.update(val_loss=last["val_loss"])

    def sample() -> None:
        nonlocal sample_index
        stepper.sync()
        record = make_sample(model.to("cpu") if device in TORCH_GPUS else model, sample_index)
        if device in TORCH_GPUS:
            model.to(torch_device(device))
        sample_index += 1
        files.sample({"step": step, **record, "time": time.time()})

    try:
        if step == 0 and not resume:
            evaluate()
        while plan.steps is None or step < plan.steps:
            if stop.is_set():
                checkpoint(False)
                return "paused"
            lr = learning_rate_at(step, plan.learning_rate, plan.warmup, plan.steps)
            tokens, targets = next_batch()
            started = time.perf_counter()
            loss = stepper.train_step(tokens, targets, lr)
            seconds = time.perf_counter() - started
            if not math.isfinite(loss):
                raise TrainingError("The loss became NaN or infinite, so training stopped before saving broken weights. "
                                    "Try a lower learning rate or a longer warmup, then resume.")
            step += 1
            session_steps += 1
            tokens_seen += int(tokens.size)
            elapsed = elapsed_base + (time.monotonic() - session_start)
            if session_steps > warmup_steps(device):
                step_time = seconds if step_time is None else 0.9 * step_time + 0.1 * seconds
                last["tok_per_s"] = round(tokens.size / max(step_time, 1e-9), 1)
            last.update(loss=round(loss, 4), lr=lr)
            files.metric({"step": step, "loss": last["loss"], "lr": lr, "tok_per_s": last["tok_per_s"],
                          "elapsed": round(elapsed, 2), "tokens_seen": tokens_seen})
            remaining = None if plan.steps is None else plan.steps - step
            files.update(
                phase="train", phase_label=f"Training on {DEVICE_LABELS.get(device, device)}", step=step,
                target=plan.steps, loss=last["loss"], lr=lr, tok_per_s=last["tok_per_s"], elapsed=round(elapsed, 2),
                tokens_seen=tokens_seen, progress=None if plan.steps is None else step / plan.steps,
                eta=None if remaining is None or step_time is None else round(remaining * step_time, 1), device=device,
            )
            finished = plan.steps is not None and step >= plan.steps
            if step % plan.eval_every == 0 or finished:
                evaluate()
            if step % plan.sample_every == 0 or finished:
                sample()
            if step % plan.checkpoint_every == 0 or finished or time.monotonic() - last_checkpoint > CHECKPOINT_SECONDS:
                checkpoint(finished)
        return "complete"
    finally:
        stepper.close()
        files.close()


# ---------- stage drivers ----------


def pretrain(project, tokenizer: TextTokenizer, device: str, files: RunFiles, stop: threading.Event) -> str:
    corpus = project.data_dir("pretrain")
    meta = read_json(corpus / "corpus.json") or {}
    config = project.model_config()
    plan = LoopPlan.from_recipe("pretrain", project.data["recipes"]["pretrain"])
    seed = int(time.time()) & 0xFFFF
    train = TokenStream(corpus / "train.bin", meta["dtype"], config.context, seed=seed)
    val = TokenStream(corpus / "val.bin", meta["dtype"], config.context)
    eval_batches = val.fixed_batches(plan.batch_size, EVAL_BATCHES)

    def make_sample(model: GPT, index: int) -> dict:
        text = sample_text(model, tokenizer, [tokenizer.eot_id], SAMPLE_TOKENS, SAMPLE_SEED, {tokenizer.eot_id})
        return {"prompt": "", "text": text}

    torch.manual_seed(0)
    return train_loop(
        plan=plan, files=files, model=GPT(config), device=device, next_batch=lambda: train.batch(plan.batch_size),
        eval_batches=eval_batches, make_sample=make_sample, stop=stop,
    )


def sft_sample(tokenizer: TextTokenizer, model: GPT, question: str, seed: int = SAMPLE_SEED, max_tokens: int = 64) -> str:
    prompt, _ = encode_chat(tokenizer, [{"role": "user", "content": question}], add_generation_prompt=True)
    prompt = prompt[-(model.config.context - 8):]
    return sample_text(model, tokenizer, prompt, max_tokens, seed, {tokenizer.end_id, tokenizer.eot_id}).strip()


def finetune(project, tokenizer: TextTokenizer, device: str, files: RunFiles, stop: threading.Event) -> str:
    folder = project.data_dir("sft")
    base_folder = project.run_dir("pretrain")
    base_info = latest(base_folder)
    if base_info is None:
        raise TrainingError("Fine-tuning needs a pre-training checkpoint.")
    config = ModelConfig.from_dict(base_info["config"])
    plan = LoopPlan.from_recipe("sft", project.data["recipes"]["sft"])
    fixed = device == "vulkan"  # one compiled shape on the GPU
    train = ChatRows(folder, "train", config.context, tokenizer.eot_id, seed=int(time.time()) & 0xFFFF)
    val_rows = ChatRows(folder, "val", config.context, tokenizer.eot_id) if (folder / "val_offsets.npy").exists() else None
    eval_batches = val_rows.fixed_batches(plan.batch_size, min(EVAL_BATCHES, max(1, len(val_rows) // plan.batch_size)), fixed) if val_rows else []

    def make_sample(model: GPT, index: int) -> dict:
        question = SFT_QUESTIONS[index % len(SFT_QUESTIONS)]
        return {"prompt": question, "text": sft_sample(tokenizer, model, question)}

    return train_loop(
        plan=plan, files=files, model=GPT(config), device=device,
        next_batch=lambda: train.batch(plan.batch_size, fixed), eval_batches=eval_batches, make_sample=make_sample,
        stop=stop, base={"stage": "pretrain", "step": base_info["step"]}, init_from=base_folder / base_info["weights"],
    )


def preference_tune(project, tokenizer: TextTokenizer, device: str, files: RunFiles, stop: threading.Event, pairs: list[dict]) -> str:
    """Direct Preference Optimisation against a frozen copy of the SFT model (CPU, CUDA or Metal)."""
    if len(pairs) < 2:
        raise TrainingError("Preference tuning needs at least a few preference pairs.")
    device = device if device in TORCH_GPUS else "cpu"
    recipe = project.data["recipes"]["dpo"]
    sft_folder = project.run_dir("sft")
    sft_info = latest(sft_folder)
    if sft_info is None:
        raise TrainingError("Preference tuning needs a fine-tuned (SFT) checkpoint.")
    config = ModelConfig.from_dict(sft_info["config"])
    reference = GPT(config)
    reference.load_state_dict(load_file(str(sft_folder / sft_info["weights"])))
    reference.to(torch_device(device)).eval().requires_grad_(False)
    policy = GPT(config)
    steps, batch, beta = int(recipe["steps"]), int(recipe["batch_size"]), float(recipe["beta"])
    peak = float(recipe["learning_rate"])
    optimizer = make_optimizer(policy, peak, weight_decay=0.0)
    resume = latest(files.folder)
    if resume:
        policy.load_state_dict(load_file(str(files.folder / resume["weights"])))
        if resume.get("optimizer"):
            optimizer.load_state_dict(torch.load(files.folder / resume["optimizer"], map_location="cpu", weights_only=True))
        step, elapsed = int(resume["step"]), float(resume.get("elapsed", 0.0))
        files.keep_until(step)
    else:
        policy.load_state_dict(reference.state_dict())
        step, elapsed = 0, 0.0
        files.keep_until(-1)
    policy.to(torch_device(device)).train()
    optimizer_to(optimizer, device)
    base = {"stage": "sft", "step": sft_info["step"]}
    if step >= steps:
        return "complete"
    encoded = [encode_preference(tokenizer, PreferencePair(p["prompt"], p["chosen"], p["rejected"]), config.context) for p in pairs]
    rng = random.Random(len(pairs) * 7919 + step)
    warmup = min(10, max(1, steps // 10))
    last = {"loss": resume.get("loss") if resume else None, "lr": None, "tok_per_s": None}
    session_start, elapsed_base = time.monotonic(), elapsed
    tokens_seen = int(resume.get("tokens_seen", 0)) if resume else 0
    label = f"Preference tuning on {DEVICE_LABELS[device]}"
    files.phase("train", label, step / steps)

    def checkpoint(complete: bool) -> None:
        save_checkpoint(files.folder, policy, optimizer, {
            "step": step, "tokens_seen": tokens_seen, "elapsed": round(elapsed, 2), "loss": last["loss"], "val_loss": None,
            "lr": last["lr"], "tok_per_s": last["tok_per_s"], "device": device, "base": base, "complete": complete,
            "target": steps, "pairs": len(pairs), "beta": beta,
        })
        files.update(force=True, checkpoint_step=step)

    try:
        while step < steps:
            if stop.is_set():
                checkpoint(False)
                return "paused"
            started = time.perf_counter()
            picks = rng.sample(range(len(encoded)), min(batch, len(encoded)))
            rows = [encoded[i][0] for i in picks] + [encoded[i][1] for i in picks]
            tokens, targets = pad_rows(rows, tokenizer.eot_id)
            x = torch.from_numpy(tokens).to(torch_device(device))
            y = torch.from_numpy(targets).to(torch_device(device))
            lr = learning_rate_at(step, peak, warmup, None)
            for group in optimizer.param_groups:
                group["lr"] = lr
            policy_logps = policy.sequence_logprobs(x, y)
            with torch.no_grad():
                reference_logps = reference.sequence_logprobs(x, y)
            count = len(picks)
            margin = (policy_logps[:count] - reference_logps[:count]) - (policy_logps[count:] - reference_logps[count:])
            loss = -F.logsigmoid(beta * margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise TrainingError("The preference loss became NaN; try a lower learning rate.")
            step += 1
            tokens_seen += int((targets != IGNORE).sum())
            elapsed = elapsed_base + time.monotonic() - session_start
            seconds = time.perf_counter() - started
            last.update(loss=round(value, 4), lr=lr, tok_per_s=round(tokens.size / max(seconds, 1e-9), 1))
            margin = margin.detach()
            accuracy = float((margin > 0).float().mean())
            files.metric({"step": step, "loss": last["loss"], "lr": lr, "tok_per_s": last["tok_per_s"],
                          "elapsed": round(elapsed, 2), "tokens_seen": tokens_seen,
                          "reward_margin": round(float(beta * margin.mean()), 4), "accuracy": round(accuracy, 3)})
            files.update(phase="train", phase_label=label, step=step, target=steps, loss=last["loss"], lr=lr,
                         tok_per_s=last["tok_per_s"], elapsed=round(elapsed, 2), tokens_seen=tokens_seen,
                         progress=step / steps, eta=round((steps - step) * seconds, 1), device=device)
        policy.eval()
        question = SFT_QUESTIONS[0]
        cpu_policy = policy.to("cpu")
        files.sample({"step": step, "prompt": question, "text": sft_sample(tokenizer, cpu_policy, question), "time": time.time()})
        checkpoint(True)
        return "complete"
    finally:
        files.close()


# ---------- calibration ----------


def time_device(config: ModelConfig, batch: int, device: str, stop: threading.Event,
                report: Callable[[str], None], budget: float = 25.0) -> dict:
    """Seconds per optimisation step for this shape on one device (random data)."""
    torch.manual_seed(0)
    model = GPT(config)
    optimizer = make_optimizer(model, 1e-4)
    model.to(torch_device(device))
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, config.vocab_size, size=(batch, config.context + 1)).astype(np.int64)
    x, y = tokens[:, :-1], tokens[:, 1:]
    stepper = make_stepper(model, optimizer, device, 0)
    try:
        for index in range(warmup_steps(device)):
            if stop.is_set():
                raise StopRequested()
            report("Compiling GPU kernels (first run only)…" if device == "vulkan" and index == 0 else f"Warming up the {DEVICE_LABELS[device]}…")
            stepper.train_step(x, y, 1e-4)
        times = []
        started = time.monotonic()
        while len(times) < 5 and (len(times) < 2 or time.monotonic() - started < budget):
            if stop.is_set():
                raise StopRequested()
            report(f"Timing the {DEVICE_LABELS[device]}…")
            begin = time.perf_counter()
            stepper.train_step(x, y, 1e-4)
            times.append(time.perf_counter() - begin)
        seconds = float(np.median(times))
        return {"step_seconds": round(seconds, 4), "tok_per_s": round(batch * config.context / seconds, 1), "steps": len(times)}
    finally:
        stepper.close()
        del model, optimizer


def calibrate(project, devices: list[str], files: RunFiles, stop: threading.Event) -> dict:
    from .project import calibration_key

    config = project.model_config()
    batch = int(project.data["recipes"]["pretrain"]["batch_size"])
    results: dict[str, dict] = {}
    for index, device in enumerate(devices):
        def report(label: str, index=index) -> None:
            files.phase("calibrate", label, index / len(devices))

        try:
            results[device] = time_device(config, batch, device, stop, report)
        except StopRequested:
            raise
        except Exception as error:  # a device that cannot run this shape is reported, not fatal
            results[device] = {"error": str(error)[:300]}
    path = project.folder / "calibration.json"
    stored = read_json(path) or {}
    stored[calibration_key(project)] = {"time": time.time(), "results": results}
    write_json(path, stored)
    files.update(force=True, results=results, progress=1.0)
    return results
