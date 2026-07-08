"""Shared filesystem paths and small CLI helpers."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
EVALUATION_DIR = REPO_ROOT / "evaluation"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"
GENERATION_DIR = EVALUATION_OUTPUT_DIR / "generations"
PLOT_DIR = EVALUATION_OUTPUT_DIR / "plots"
DATA_DIR = REPO_ROOT / "data"
DATA_FILE = DATA_DIR / "tiny_shakespeare.txt"


def ensure_output_dirs() -> None:
    """Create all standard output directories used by training and evaluation."""
    for directory in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, EVALUATION_OUTPUT_DIR, GENERATION_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


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
