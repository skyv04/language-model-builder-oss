"""Model shape settings, kept free of PyTorch so the UI server can start quickly."""

from __future__ import annotations

from dataclasses import asdict, dataclass

LIMITS = {
    "vocab_size": (64, 65536),
    "context": (16, 1024),
    "width": (32, 1024),
    "blocks": (1, 16),
    "heads": (1, 16),
}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context: int = 256
    width: int = 256
    blocks: int = 4
    heads: int = 4

    def __post_init__(self) -> None:
        for name, (low, high) in LIMITS.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")
        if self.width % self.heads:
            raise ValueError("width must be divisible by the number of attention heads")
        if self.head_dim % 2:
            raise ValueError("each attention head needs an even number of dimensions")

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    def parameter_count(self) -> int:
        d = self.width
        per_block = 12 * d * d + 13 * d
        return self.vocab_size * d + self.context * d + self.blocks * per_block + 2 * d

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{key: int(data[key]) for key in ("vocab_size", "context", "width", "blocks", "heads")})

    def training_flops_per_token(self) -> float:
        """Forward plus backward cost, in the usual 6N + attention approximation."""
        return 6.0 * self.parameter_count() + 12.0 * self.blocks * self.width * self.context

    def training_memory_bytes(self, batch_size: int) -> int:
        """Rough upper bound on resident memory for one optimisation step."""
        weights = self.parameter_count() * 4 * 4  # weights, gradients, two AdamW moments
        tokens = batch_size * self.context
        activations = tokens * self.blocks * self.width * 4 * 20
        attention = batch_size * self.heads * self.context * self.context * self.blocks * 4 * 3
        logits = tokens * self.vocab_size * 4 * 3
        return int(weights + activations + attention + logits)
