"""Train model.GPT on a Vulkan GPU through tinygrad, with the same math as the torch reference.

``GpuTrainer`` mirrors ``model.GPT`` (tied embeddings, tanh GELU, pre-LayerNorm blocks), the
masked mean cross-entropy, global gradient-norm clipping and ``torch.optim.AdamW`` including
both parameter groups.  One training step is a single TinyJit replay: the host uploads the
batch plus a few per-parameter AdamW scalars (so learning-rate changes never recompile) and
reads back one float, the loss.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import torch

from .model import GPT

DEVICE = "WEBGPU"
IGNORE_INDEX = -100
CLIP_NORM = 1.0
LN_EPS = 1e-5
MAX_KERNEL_BUFFERS = 32  # WebGPU's default of 8 splits fused kernels; the Vulkan adapter allows far more
EMBEDDING = "tok_emb.weight"
# The tied embedding/output matrix gets zero rows up to a multiple of this on the GPU: tinygrad
# cannot tile a prime vocabulary (50261 made single kernels run for seconds).  Padded logits
# are -inf, so padded rows get exactly zero gradients and stay zero under AdamW.
VOCAB_MULTIPLE = 128
SPLIT_SQUARES = 32768
# From this (padded) vocabulary size the output layer runs as HEAD_SPLITS chunks (see GpuTrainer._head);
# measured: 50304 x 256 went from 92 to 49 ms per step, 4096 and 10112 gained nothing.
HEAD_SPLIT_VOCAB = 32768
HEAD_SPLITS = 16  # divides any multiple of VOCAB_MULTIPLE
ONEHOT_MULTIPLE = 64  # row padding of the embedding-gradient one-hot matmul (see _embedding_gradient)
# Kernel-compile worker processes during warm-up (~25 MB each).  Measured on 30M parameters:
# 3 workers instead of 6 lower the warm-up memory peak by ~70 MB for ~10% longer warm-up.
COMPILE_WORKERS = 3


@functools.lru_cache(maxsize=1)
def gpu_available() -> dict | None:
    """``{"name", "backend", "adapter_type"}`` for a physical GPU adapter, else None.  Never raises."""
    try:
        import tinygrad  # noqa: F401  (the trainer needs it, so a missing install means no GPU)

        from .gpu_runtime import adapter_summary

        return adapter_summary()
    except Exception:  # noqa: BLE001 - any failure simply means "no usable GPU"
        return None


def _context():
    from tinygrad.helpers import PARALLEL, Context

    # SPLIT_REDUCEOP=0: tinygrad's automatic two-kernel split of long reductions turned the
    # vocabulary-length matmul of the logits backward into a 4x slower strided kernel; the only
    # long full reductions left (gradient norms) are split explicitly in ``_square_sum``.
    return Context(DEV=DEVICE, MAX_KERNEL_BUFFERS=MAX_KERNEL_BUFFERS, SPLIT_REDUCEOP=0,
                   PARALLEL=min(PARALLEL.value, COMPILE_WORKERS))


def _square_sum(gradient):
    """Sum of squares; large tensors reduce per row first (a single long reduction runs on few threads)."""
    squares = gradient * gradient
    if gradient.ndim >= 2 and gradient.numel() >= SPLIT_SQUARES:
        squares = squares.reshape(gradient.shape[0], -1).sum(axis=1).contiguous()
    return squares.sum().contiguous()


class _Parameter:
    """One trainable tensor with its AdamW moments and hyper-parameters."""

    def __init__(self, name: str, value, exp_avg, exp_avg_sq, step: int, group: dict):
        self.name = name
        self.value, self.exp_avg, self.exp_avg_sq = value, exp_avg, exp_avg_sq
        self.step = step
        self.weight_decay = float(group["weight_decay"])
        self.beta1, self.beta2 = (float(beta) for beta in group["betas"])
        self.eps = float(group["eps"])

    def scalars(self, lr: float) -> tuple[float, float, float]:
        """(decay factor, step size, sqrt of the second-moment bias correction) for the next step."""
        step = self.step + 1
        return (
            1.0 - lr * self.weight_decay,
            lr / (1.0 - self.beta1**step),
            math.sqrt(1.0 - self.beta2**step),
        )


class _StepProgram:
    """Persistent input buffers plus one TinyJit for a fixed (batch, length)."""

    def __init__(self, trainer: "GpuTrainer", batch: int, length: int, train: bool):
        from tinygrad import Tensor, TinyJit, dtypes

        self.trainer, self.batch, self.length, self.train = trainer, batch, length, train
        count = batch * length
        scalars = 3 * len(trainer.trainable) if train else 0
        # int inputs: tokens, targets and, when training, the distinct-token map of the embedding backward
        ints = 2 * count + (count + trainer.padded_vocab if train else 0)
        self.ints = Tensor.zeros(ints, dtype=dtypes.int32, device=DEVICE).contiguous().realize()
        self.floats = Tensor.zeros(count + scalars, dtype=dtypes.float32, device=DEVICE).contiguous().realize()
        self.host_ints = np.zeros(ints, dtype=np.int32)
        self.host_floats = np.zeros(count + scalars, dtype=np.float32)
        self.jit = TinyJit(self._run)
        # Capture on the first call.  Every input is already a realized buffer, so TinyJit's usual
        # uncaptured first run would only execute the step once more, with its own unplanned
        # intermediates (for 30M parameters about 0.4 GB extra at the warm-up peak).
        self.jit.cnt = 1
        self.settled = False

    def _run(self):
        from tinygrad import Tensor

        trainer, count = self.trainer, self.batch * self.length
        tokens = self.ints[:count].reshape(self.batch, self.length)
        targets = self.ints[count : 2 * count].reshape(self.batch, self.length)
        weights = self.floats[:count].reshape(self.batch, self.length)
        loss, embedded = trainer._loss(tokens, targets, weights)
        if not self.train:
            return loss.realize()
        trainable = trainer.trainable
        *gradients, embedded_gradient = loss.gradient(*(parameter.value for parameter in trainable), embedded)
        names = [parameter.name for parameter in trainable]
        if EMBEDDING in names:
            index = names.index(EMBEDDING)
            gradients[index] = gradients[index] + self._embedding_gradient(embedded_gradient)
        # Materialize gradients and the clip factor: left lazy, tinygrad would fuse the global norm
        # (and with it every gradient) into each parameter's update kernel.  Realizing the gradients
        # as their own schedule also keeps tinygrad's scheduler from re-walking the whole
        # forward/backward graph for each of the AdamW stores (about 10% less warm-up).
        gradients = [gradient.contiguous() for gradient in gradients]
        Tensor.realize(loss, *gradients)
        squares = Tensor.stack(*[_square_sum(gradient) for gradient in gradients]).sum()
        scale = (CLIP_NORM / (squares.sqrt() + 1e-6)).minimum(1.0).contiguous()
        scalars = self.floats[count:].reshape(len(trainable), 3)
        updated = []
        for index, (parameter, gradient) in enumerate(zip(trainable, gradients)):
            gradient = gradient * scale
            decay, step_size, correction = scalars[index, 0], scalars[index, 1], scalars[index, 2]
            exp_avg = parameter.exp_avg + (gradient - parameter.exp_avg) * (1.0 - parameter.beta1)
            exp_avg_sq = parameter.exp_avg_sq * parameter.beta2 + gradient * gradient * (1.0 - parameter.beta2)
            value = parameter.value * decay - step_size * exp_avg / (exp_avg_sq.sqrt() / correction + parameter.eps)
            updated += [parameter.value.assign(value), parameter.exp_avg.assign(exp_avg), parameter.exp_avg_sq.assign(exp_avg_sq)]
        Tensor.realize(*updated)
        return loss

    def _embedding_gradient(self, embedded_gradient):
        """Embedding-lookup gradient: per-position gradients summed per distinct token, gathered into rows.

        tinygrad's own backward of ``table[tokens]`` is a vocabulary x positions x width one-hot
        reduction, as expensive as the output projection.  With the host-computed ``inverse``
        (position -> distinct-token index) and ``slot`` (vocabulary row -> distinct-token index,
        or ``count`` for the zero row) it costs a positions² x width matmul plus a gather.
        """
        from tinygrad import Tensor, dtypes

        count, width = self.batch * self.length, self.trainer.config.width
        inverse = self.ints[2 * count : 3 * count]
        slot = self.ints[3 * count :]
        # More than count rows: inverse < count, so row ``count`` (the one unused vocabulary rows read)
        # stays zero.  Rounded up because an odd row count (count + 1) tiled ~10x slower.
        rows = -(-(count + 1) // ONEHOT_MULTIPLE) * ONEHOT_MULTIPLE
        onehot = (Tensor.arange(rows).unsqueeze(1) == inverse.unsqueeze(0)).cast(dtypes.float32)
        per_token = onehot.matmul(embedded_gradient.reshape(count, width).contiguous()).contiguous()
        return per_token[slot].contiguous()

    def run(self, tokens: np.ndarray, targets: np.ndarray, lr: float | None) -> float:
        trainer, count = self.trainer, self.batch * self.length
        mask = targets != IGNORE_INDEX
        self.host_ints[:count] = tokens.reshape(-1)
        self.host_ints[count : 2 * count] = np.where(mask, targets, 0).reshape(-1)
        total = int(mask.sum())
        self.host_floats[:count] = mask.reshape(-1) / max(total, 1)
        if self.train:
            self.host_floats[count:] = np.array([parameter.scalars(lr) for parameter in trainer.trainable], dtype=np.float64).reshape(-1)
            distinct, inverse = np.unique(tokens.reshape(-1), return_inverse=True)
            self.host_ints[2 * count : 3 * count] = inverse.reshape(-1)
            slot = self.host_ints[3 * count :]
            slot.fill(count)
            slot[distinct] = np.arange(distinct.size)
        device = trainer.device
        device.write(self.ints.uop.buffer._buf, memoryview(self.host_ints))
        device.write(self.floats.uop.buffer._buf, memoryview(self.host_floats))
        with _context():
            loss = self.jit()
        if not self.settled and self.jit.captured is not None:
            self.settled = True
            _release_compile_resources(device)
        result = np.zeros(1, dtype=np.float32)
        device.read(memoryview(result), loss.uop.buffer._buf)
        return float(result[0])


def _release_compile_resources(device) -> None:
    """After a JIT capture: free buffers cached from the uncaptured first run, stop compile workers."""
    from tinygrad.engine.worker import terminate_worker_pool

    device.allocator.free_cache()
    terminate_worker_pool()


class GpuTrainer:
    """AdamW training of a ``model.GPT`` on the GPU; weights and optimizer state live on the device."""

    def __init__(self, model: GPT, optimizer: torch.optim.AdamW, step: int):
        from tinygrad import Device, Tensor

        from .gpu_runtime import register_runtime

        register_runtime()
        self.config = model.config
        self.vocab = model.config.vocab_size
        self.padded_vocab = -(-self.vocab // VOCAB_MULTIPLE) * VOCAB_MULTIPLE
        self.step = int(step)
        self.last_lr: float | None = None
        groups = {}
        for group in optimizer.param_groups:
            if group.get("amsgrad") or group.get("maximize"):
                raise ValueError("the GPU trainer supports plain AdamW only (no amsgrad/maximize)")
            for parameter in group["params"]:
                groups[id(parameter)] = group
        with _context():
            self.device = Device[DEVICE]
            self.values: dict[str, Tensor] = {}
            self.trainable: list[_Parameter] = []
            for name, parameter in model.named_parameters():
                rows = self.padded_vocab if name == EMBEDDING else None
                value = self._upload(parameter, rows)
                self.values[name] = value
                group = groups.get(id(parameter))
                if group is None or not parameter.requires_grad:
                    continue
                state = optimizer.state.get(parameter, {})
                moments = [
                    self._upload(state[key], rows) if key in state else Tensor.zeros(*value.shape, device=DEVICE).contiguous()
                    for key in ("exp_avg", "exp_avg_sq")
                ]
                step_count = int(state["step"]) if "step" in state else 0
                self.trainable.append(_Parameter(name, value, *moments, step_count, group))
            Tensor.realize(*self.values.values(), *(p.exp_avg for p in self.trainable), *(p.exp_avg_sq for p in self.trainable))
            self.device.allocator.free_cache()  # upload and padding temporaries
        self.programs: dict[tuple[int, int, bool], _StepProgram] = {}

    @staticmethod
    def _upload(tensor: torch.Tensor, rows: int | None = None):
        from tinygrad import Tensor

        # No host copy: tinygrad reads the array when __init__ realizes, before the torch tensor can change.
        array = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
        value = Tensor(array, device=DEVICE)
        if rows is not None and rows > array.shape[0]:
            value = value.pad(((0, rows - array.shape[0]), *[(0, 0)] * (array.ndim - 1)))
        return value.contiguous()

    @staticmethod
    def _download(tensor, shape: torch.Size) -> torch.Tensor:
        """Device tensor to torch, dropping vocabulary padding rows."""
        array = tensor.numpy().reshape(-1, *shape[1:])[: shape[0]]
        return torch.from_numpy(array.reshape(shape).copy())

    # ---- model
    def _layer_norm(self, x, name: str):
        return x.layernorm(eps=LN_EPS) * self.values[f"{name}.weight"] + self.values[f"{name}.bias"]

    def _linear(self, x, name: str):
        return x.matmul(self.values[f"{name}.weight"].T) + self.values[f"{name}.bias"]

    def _logits(self, tokens):
        from tinygrad import Tensor

        batch, length = tokens.shape
        width, heads = self.config.width, self.config.heads
        # Detached lookup: the train step adds the embedding gradient itself (see _embedding_gradient).
        embedded = self.values[EMBEDDING].detach()[tokens]
        x = embedded + self.values["pos_emb.weight"][:length]
        for index in range(self.config.blocks):
            prefix = f"blocks.{index}"
            q, k, v = self._linear(self._layer_norm(x, f"{prefix}.ln1"), f"{prefix}.attn.qkv").split(width, dim=2)
            # Separate materialized gradients: the gradient of the split is a concatenation, and
            # with lazy inputs each of its elements would evaluate all three attention matmuls.
            q, k, v = (t.contiguous_backward().reshape(batch, length, heads, width // heads).transpose(1, 2) for t in (q, k, v))
            y = self._attention(q, k, v).transpose(1, 2).reshape(batch, length, width)
            x = x + self._linear(y, f"{prefix}.attn.proj")
            hidden = self._linear(self._layer_norm(x, f"{prefix}.ln2"), f"{prefix}.mlp.fc").gelu()
            x = x + self._linear(hidden, f"{prefix}.mlp.proj")
        # Materialized logits (and their gradient): otherwise every softmax reduction and both
        # backward matmuls recompute the vocabulary projection.
        logits = self._head(self._layer_norm(x, "ln_f"))
        if self.padded_vocab != self.vocab:
            logits = (Tensor.arange(self.padded_vocab) < self.vocab).where(logits, float("-inf"))
        return logits.contiguous().contiguous_backward(), embedded

    def _head(self, hidden):
        """hidden @ tok_embᵀ, computed per vocabulary chunk so its input gradient is a split-K matmul.

        The gradient w.r.t. ``hidden`` reduces over the whole vocabulary for only batch x length x
        width outputs, which leaves most of the GPU idle; the expanded copy's materialized gradient
        holds one partial sum per chunk, and the expand's backward adds them up.
        """
        table = self.values[EMBEDDING]
        if self.padded_vocab < HEAD_SPLIT_VOCAB:
            return hidden.matmul(table.T)
        batch, length, width = hidden.shape
        splits = HEAD_SPLITS
        chunks = table.reshape(splits, self.padded_vocab // splits, width).transpose(1, 2)
        expanded = hidden.reshape(1, batch * length, width).expand(splits, batch * length, width).contiguous_backward()
        return expanded.matmul(chunks).permute(1, 0, 2).reshape(batch, length, self.padded_vocab)

    @staticmethod
    def _attention(q, k, v):
        """Causal softmax(q kᵀ/√d) v with the score and probability matrices (and gradients) materialized."""
        from tinygrad import dtypes

        scores = q.matmul(k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
        causal = scores.const_like(True, dtypes.bool).tril()
        scores = causal.where(scores, float("-inf")).contiguous().contiguous_backward()
        probs = scores.softmax(-1).contiguous().contiguous_backward()
        return probs.matmul(v)

    def _loss(self, tokens, targets, weights):
        """(Cross-entropy weighted per position (mask / number of counted targets) and summed, embedded tokens)."""
        logits, embedded = self._logits(tokens)
        picked = targets.unsqueeze(-1)._one_hot_along_dim(self.padded_vocab).where(logits, 0).sum(-1)
        return ((logits.logsumexp(-1) - picked) * weights).sum(), embedded

    # ---- public API
    def _program(self, tokens: np.ndarray, targets: np.ndarray, train: bool) -> _StepProgram:
        tokens, targets = np.asarray(tokens), np.asarray(targets)
        if tokens.ndim != 2 or tokens.shape != targets.shape:
            raise ValueError("tokens and targets must both have shape (batch, length)")
        if tokens.shape[1] > self.config.context:
            raise ValueError(f"sequence of {tokens.shape[1]} tokens exceeds the {self.config.context}-token context")
        vocab = self.config.vocab_size
        if tokens.size and (tokens.min() < 0 or tokens.max() >= vocab):
            raise ValueError("token id outside the vocabulary")
        counted = targets[targets != IGNORE_INDEX]
        if counted.size and (counted.min() < 0 or counted.max() >= vocab):
            raise ValueError("target id outside the vocabulary")
        key = (int(tokens.shape[0]), int(tokens.shape[1]), train)
        if key not in self.programs:
            with _context():
                self.programs[key] = _StepProgram(self, key[0], key[1], train)
        return self.programs[key]

    def train_step(self, tokens: np.ndarray, targets: np.ndarray, lr: float) -> float:
        program = self._program(tokens, targets, train=True)
        loss = program.run(np.asarray(tokens), np.asarray(targets), float(lr))
        for parameter in self.trainable:
            parameter.step += 1
        self.step += 1
        self.last_lr = float(lr)
        return loss

    def eval_loss(self, tokens: np.ndarray, targets: np.ndarray) -> float:
        program = self._program(tokens, targets, train=False)
        return program.run(np.asarray(tokens), np.asarray(targets), None)

    def sync_to_torch(self, model: GPT, optimizer: torch.optim.AdamW) -> None:
        trainable = {parameter.name: parameter for parameter in self.trainable}
        with torch.no_grad(), _context():
            for name, parameter in model.named_parameters():
                parameter.copy_(self._download(self.values[name], parameter.shape))
                if (entry := trainable.get(name)) is None:
                    continue
                state = optimizer.state[parameter]
                for key, tensor in (("exp_avg", entry.exp_avg), ("exp_avg_sq", entry.exp_avg_sq)):
                    state[key] = self._download(tensor, parameter.shape)
                state["step"] = torch.tensor(float(entry.step), dtype=torch.float32)
        if self.last_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = self.last_lr

    def close(self) -> None:
        self.programs.clear()
        self.values.clear()
        self.trainable.clear()
        device = getattr(self, "device", None)
        if device is not None:
            device.synchronize()
            device.allocator.free_cache()
