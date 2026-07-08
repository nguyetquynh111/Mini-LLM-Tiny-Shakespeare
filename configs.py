"""Configuration presets for the Tiny Shakespeare Transformer models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def get_default_device() -> str:
    """Return the best available PyTorch device name."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class TransformerConfig:
    """Hyperparameters used by the model, training loop, and generation."""

    name: str
    vocab_size: int
    block_size: int
    batch_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    learning_rate: float
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    device: str = "cpu"
    seed: int = 1337

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a plain dictionary for checkpoint saving."""
        return asdict(self)


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "model_a": {
        "name": "model_a",
        "vocab_size": 256,
        "block_size": 64,
        "batch_size": 32,
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 2,
        "dropout": 0.2,
        "learning_rate": 3e-4,
    },
    "model_b": {
        "name": "model_b",
        "vocab_size": 256,
        "block_size": 128,
        "batch_size": 32,
        "n_embd": 256,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.2,
        "learning_rate": 3e-4,
    },
}


def get_config(config_name: str, **overrides: Any) -> TransformerConfig:
    """Build a config by name and apply optional command-line overrides."""
    if config_name not in MODEL_PRESETS:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available}")

    values = dict(MODEL_PRESETS[config_name])
    values["device"] = get_default_device()
    values.update({key: value for key, value in overrides.items() if value is not None})
    return TransformerConfig(**values)


def config_from_dict(values: dict[str, Any]) -> TransformerConfig:
    """Recreate a config object from a checkpoint dictionary."""
    return TransformerConfig(**values)
