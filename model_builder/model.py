"""GPT-style decoder shared by pre-training, fine-tuning, preference tuning, and chat.

The architecture is the familiar GPT-2 recipe: learned token and position embeddings,
pre-LayerNorm transformer blocks (causal multi-head attention plus a 4x MLP with the
tanh-approximated GELU), a final LayerNorm, and an output layer tied to the token
embedding table.  Every tensor is float32 so CPU, CUDA, Metal and Vulkan backends agree.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import LIMITS, ModelConfig

__all__ = ["LIMITS", "ModelConfig", "GPT", "KVCache", "make_optimizer", "learning_rate_at", "parameter_groups"]


class KVCache:
    """Per-layer key/value tensors for incremental decoding."""

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, capture: list | None = None) -> torch.Tensor:
        batch, length, width = x.shape
        head_dim = width // self.heads
        q, k, v = self.qkv(x).split(width, dim=2)
        q = q.view(batch, length, self.heads, head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, head_dim).transpose(1, 2)
        if cache is not None:
            if cache.keys is not None:
                k = torch.cat((cache.keys, k), dim=2)
                v = torch.cat((cache.values, v), dim=2)
            cache.keys, cache.values = k, v
        total = k.shape[2]
        past = total - length
        if capture is not None:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            allowed = torch.ones(length, total, dtype=torch.bool, device=x.device).tril(diagonal=past)
            weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
            capture.append(weights.detach())
            y = weights @ v
        elif past == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif length == 1:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            allowed = torch.ones(length, total, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.width, 4 * config.width)
        self.proj = nn.Linear(4 * config.width, config.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.width)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.width)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, capture: list | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache, capture)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.width)
        self.pos_emb = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.blocks))
        self.ln_f = nn.LayerNorm(config.width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = residual_std if name.endswith(".proj") else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        caches: list[KVCache] | None = None,
        capture: list | None = None,
    ) -> torch.Tensor:
        length = tokens.shape[1]
        start = caches[0].length if caches else 0
        if start + length > self.config.context:
            raise ValueError(f"sequence of {start + length} tokens exceeds the {self.config.context}-token context")
        positions = torch.arange(start, start + length, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(positions)
        for index, block in enumerate(self.blocks):
            x = block(x, caches[index] if caches else None, capture)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over targets that are not -100."""
        logits = self(tokens)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)

    def sequence_logprobs(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Sum of log-probabilities of every target that is not -100, per sequence."""
        logits = self(tokens)
        logprobs = logits.log_softmax(dim=-1)
        mask = targets != -100
        picked = logprobs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (picked * mask).sum(dim=-1)

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    @torch.no_grad()
    def attention_maps(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Attention weights for one sequence: a list with one (heads, T, T) tensor per block."""
        capture: list[torch.Tensor] = []
        self(tokens[:, -self.config.context:], capture=capture)
        return [weights[0] for weights in capture]


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """AdamW groups: decay matrices and embeddings, never biases or LayerNorm gains."""
    decay, plain = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.dim() >= 2 else plain).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float = 0.1) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameter_groups(model, weight_decay), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
    )


def learning_rate_at(step: int, peak: float, warmup: int, total: int | None, floor_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay to floor_ratio * peak by `total` steps.

    `step` counts completed optimisation steps, so the first update uses step 0.  With no
    total (train until stopped) the rate stays at its peak after warmup.
    """
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if not total or total <= warmup:
        return peak
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    floor = peak * floor_ratio
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))
