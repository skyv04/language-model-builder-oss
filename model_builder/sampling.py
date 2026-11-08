"""Text generation with per-token probabilities, and attention maps for inspection."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from .model import GPT
from .tokenizer import TextTokenizer

PRESETS = {
    "predictable": {"temperature": 0.3, "top_k": 20, "top_p": 0.9, "min_p": 0.0},
    "balanced": {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.0},
    "creative": {"temperature": 1.15, "top_k": 0, "top_p": 1.0, "min_p": 0.05},
}


@dataclass
class SamplerSettings:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.0
    max_tokens: int = 128
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "SamplerSettings":
        data = data or {}

        def number(key, default, cast, low, high):
            try:
                value = cast(data.get(key, default))
            except (TypeError, ValueError):
                value = default
            return min(high, max(low, value))

        seed = data.get("seed")
        try:
            seed = None if seed in (None, "") else int(seed) & 0x7FFFFFFF
        except (TypeError, ValueError):
            seed = None
        return cls(
            temperature=number("temperature", 0.8, float, 0.0, 3.0),
            top_k=number("top_k", 40, int, 0, 100000),
            top_p=number("top_p", 0.95, float, 0.0, 1.0),
            min_p=number("min_p", 0.0, float, 0.0, 1.0),
            max_tokens=number("max_tokens", 128, int, 1, 4096),
            seed=seed,
        )


@dataclass
class GenerationStats:
    tokens: int = 0
    stop_reason: str = "length"
    started: float = field(default_factory=time.perf_counter)
    first_token: float | None = None
    finished: float | None = None

    def to_dict(self) -> dict:
        end = self.finished or time.perf_counter()
        decode_time = max(1e-6, end - (self.first_token or self.started))
        rate = (self.tokens - 1) / decode_time if self.tokens > 1 else (1 / decode_time if self.tokens else 0.0)
        return {
            "stop_reason": self.stop_reason,
            "tokens": self.tokens,
            "tok_per_s": round(rate, 1),
            "ttft": round((self.first_token or end) - self.started, 3),
        }


def decoding_distribution(logits: torch.Tensor, settings: SamplerSettings) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (display probabilities, sampling probabilities) for one position's logits."""
    temperature = settings.temperature if settings.temperature > 0 else 1.0
    display = torch.softmax(logits.float() / temperature, dim=-1)
    if settings.temperature <= 0:
        greedy = torch.zeros_like(display)
        greedy[int(torch.argmax(display))] = 1.0
        return display, greedy
    probs = display.clone()
    if settings.top_k and settings.top_k < probs.numel():
        threshold = torch.topk(probs, settings.top_k).values[-1]
        probs[probs < threshold] = 0.0
    if settings.min_p > 0:
        probs[probs < settings.min_p * probs.max()] = 0.0
    if settings.top_p < 1.0:
        sorted_probs, order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        drop = cumulative - sorted_probs > settings.top_p * cumulative[-1]
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        probs = display
    else:
        probs = probs / total
    return display, probs


def _token_record(tokenizer: TextTokenizer, token_id: int, display: torch.Tensor, alternatives: int = 5) -> dict:
    values, ids = torch.topk(display, min(alternatives, display.numel()))
    return {
        "id": token_id,
        "text": tokenizer.piece(token_id),
        "prob": round(float(display[token_id]), 5),
        "alternatives": [
            {"id": int(index), "text": tokenizer.piece(int(index)), "prob": round(float(value), 5)}
            for value, index in zip(values.tolist(), ids.tolist())
        ],
    }


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: TextTokenizer,
    prompt_ids: list[int],
    settings: SamplerSettings,
    stop_ids: set[int] | None = None,
    stats: GenerationStats | None = None,
    should_stop=None,
) -> Iterator[dict]:
    """Yield one record per generated token: {id, text, prob, alternatives}."""
    stats = stats or GenerationStats()
    model.eval()
    device = next(model.parameters()).device
    context = model.config.context
    prompt_ids = list(prompt_ids) or [tokenizer.eot_id]
    if len(prompt_ids) >= context:
        prompt_ids = prompt_ids[-(context - 1) :]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.seed if settings.seed is not None else int(time.time() * 1000) & 0x7FFFFFFF)
    caches = model.new_caches()
    logits = model(torch.tensor([prompt_ids], device=device), caches)[0, -1].cpu()
    position = len(prompt_ids)
    stats.stop_reason = "length"
    for _ in range(settings.max_tokens):
        if should_stop and should_stop():
            stats.stop_reason = "stopped"
            break
        display, probs = decoding_distribution(logits, settings)
        token_id = int(torch.multinomial(probs, 1, generator=generator)) if settings.temperature > 0 else int(torch.argmax(probs))
        stats.tokens += 1
        if stats.first_token is None:
            stats.first_token = time.perf_counter()
        yield _token_record(tokenizer, token_id, display)
        if stop_ids and token_id in stop_ids:
            stats.stop_reason = "end"
            break
        if position >= context:
            stats.stop_reason = "context"
            break
        logits = model(torch.tensor([[token_id]], device=device), caches)[0, -1].cpu()
        position += 1
    stats.finished = time.perf_counter()


def sample_text(model: GPT, tokenizer: TextTokenizer, prompt_ids: list[int], max_tokens: int = 48, seed: int = 1234, stop_ids: set[int] | None = None) -> str:
    """Short deterministic sample for the run timeline."""
    settings = SamplerSettings(temperature=0.8, top_k=40, top_p=0.95, max_tokens=max_tokens, seed=seed)
    ids = [record["id"] for record in generate(model, tokenizer, prompt_ids, settings, stop_ids=stop_ids)]
    if stop_ids and ids and ids[-1] in stop_ids:
        ids = ids[:-1]
    return tokenizer.decode(ids)


@torch.no_grad()
def attention(model: GPT, tokenizer: TextTokenizer, text: str, max_tokens: int = 32) -> dict:
    ids = tokenizer.encode(text, special=True)[:max_tokens] or [tokenizer.eot_id]
    model.eval()
    device = next(model.parameters()).device
    maps = model.attention_maps(torch.tensor([ids], device=device))
    return {
        "tokens": [{"id": token_id, "text": tokenizer.piece(token_id)} for token_id in ids],
        "layers": [[[[round(float(value), 4) for value in row] for row in head] for head in layer.cpu()] for layer in maps],
    }
