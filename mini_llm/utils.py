"""Shared repository paths, artifact helpers, and reproducibility utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"
LOG_DIR = PACKAGE_DIR / "logs"
EVALUATION_DIR = REPO_ROOT / "evaluation"
EVALUATION_RESULTS_DIR = EVALUATION_DIR / "results"
GENERATION_DIR = EVALUATION_DIR / "generations"
PLOT_DIR = EVALUATION_DIR / "plots"
DATA_DIR = REPO_ROOT / "data"
DATA_FILE = DATA_DIR / "tiny_shakespeare.txt"


def ensure_artifact_dirs() -> None:
    """Create the standard training and evaluation artifact directories."""
    for directory in (CHECKPOINT_DIR, LOG_DIR, EVALUATION_RESULTS_DIR, GENERATION_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA when available."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenizer_metadata() -> dict[str, object]:
    """Return metadata for the byte-level tokenizer used by this project."""
    return {
        "type": "byte-level utf-8",
        "vocab_size": 256,
        "decode_errors": "replace",
    }


def build_checkpoint(
    *,
    config,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: Optional[dict[str, object]],
    optimizer_config: dict[str, object],
    step: int,
    train_loss: Optional[float],
    val_loss: Optional[float],
    loss_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Build a checkpoint dictionary with explicit training metadata."""
    from datetime import datetime, timezone

    return {
        "config_name": config.name,
        "config": config.to_dict(),
        "model_config": config.to_dict(),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "optimizer_config": optimizer_config,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_metadata(),
        "loss_rows": loss_rows,
    }


def save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    """Save a checkpoint, creating its parent directory if needed."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def missing_checkpoint_message(model_name: str, checkpoint_path: Path) -> str:
    """Return a clear remediation message for a missing model checkpoint."""
    return (
        f"Missing checkpoint for {model_name}: {checkpoint_path}\n"
        f"Run this first from the repository root: python -m mini_llm.train --config {model_name}"
    )


def missing_logs_message(model_a_log: Path, model_b_log: Path) -> str:
    """Return a clear remediation message for missing loss logs."""
    return (
        "Missing loss logs required for plotting.\n"
        f"Expected Model A log: {model_a_log}\n"
        f"Expected Model B log: {model_b_log}\n"
        "Train both models first from the repository root:\n"
        "python -m mini_llm.train --config model_a\n"
        "python -m mini_llm.train --config model_b"
    )
