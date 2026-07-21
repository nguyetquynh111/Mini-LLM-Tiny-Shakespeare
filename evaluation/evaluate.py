"""Deterministic full-split evaluation for saved Tiny Shakespeare checkpoints.

Every validation target byte is scored exactly once. Windows overlap so that,
after the opening window, scored targets are primed with preceding context. This
avoids both Monte Carlo noise and the boundary bias of disjoint chunks while
remaining much cheaper than a stride-one sweep.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn import functional as F

from mini_llm.configs import config_from_dict, get_default_device
from mini_llm.data import load_data
from mini_llm.model import GPTLanguageModel
from mini_llm.utils import (
    CHECKPOINT_DIR,
    EVALUATION_RESULTS_DIR,
    ensure_artifact_dirs,
    missing_checkpoint_message,
    seed_everything,
)


DEFAULT_CHECKPOINT_DIR = CHECKPOINT_DIR
DEFAULT_OUTPUT_PATH = EVALUATION_RESULTS_DIR / "metrics.csv"


def perplexity_from_loss(loss: float) -> float:
    """Convert mean token-level cross-entropy in nats to perplexity."""
    return math.exp(loss)


def bits_per_byte_from_loss(loss: float) -> float:
    """Convert byte-level cross-entropy in nats to bits per byte."""
    return loss / math.log(2.0)


def compute_baselines(train_data: torch.Tensor, val_data: torch.Tensor) -> dict[str, float | int]:
    """Return context-free reference losses fitted without validation leakage."""
    if train_data.numel() == 0 or val_data.numel() == 0:
        raise ValueError("Training and validation splits must be non-empty")

    counts = torch.bincount(train_data.cpu(), minlength=256).to(torch.float64)
    observed_vocab_size = int((counts > 0).sum().item())
    probabilities = (counts + 1.0) / (counts.sum() + 256.0)
    unigram_val_loss = -torch.log(probabilities[val_data.cpu()]).mean().item()
    unseen_mask = counts == 0
    unseen_val_tokens = int(unseen_mask[val_data.cpu()].sum().item())

    return {
        "uniform_256_loss": math.log(256),
        "uniform_observed_loss": math.log(observed_vocab_size),
        "train_unigram_val_loss": unigram_val_loss,
        "observed_vocab_size": observed_vocab_size,
        "unseen_val_tokens": unseen_val_tokens,
    }


def _scoring_windows(
    data: torch.Tensor,
    block_size: int,
    stride: int,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Build overlapping windows whose scored suffixes partition all targets."""
    if data.ndim != 1:
        raise ValueError("Full-split evaluation expects a one-dimensional token tensor")
    if data.numel() < 2:
        raise ValueError("Evaluation data must contain at least two tokens")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if stride <= 0 or stride > block_size:
        raise ValueError("stride must satisfy 1 <= stride <= block_size")

    windows: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    token_count = int(data.numel())
    for target_start in range(1, token_count, stride):
        target_end = min(target_start + stride, token_count)
        context_start = max(0, target_end - 1 - block_size)
        x = data[context_start : target_end - 1]
        y = data[context_start + 1 : target_end]
        score_from = target_start - (context_start + 1)
        windows.append((x, y, score_from))
    return windows


@torch.no_grad()
def full_split_loss(
    model: GPTLanguageModel,
    data: torch.Tensor,
    device: str,
    *,
    stride: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> dict[str, float | int]:
    """Score every next-token target exactly once with context-primed windows."""
    block_size = model.config.block_size
    stride = stride or max(1, block_size // 2)
    batch_size = batch_size or model.config.batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Windows with the same length and score boundary can be safely stacked.
    grouped: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for x, y, score_from in _scoring_windows(data, block_size, stride):
        grouped[(int(x.numel()), score_from)].append((x, y))

    was_training = model.training
    model.eval()
    total_nll = 0.0
    tokens_scored = 0
    for (_, score_from), windows in grouped.items():
        for offset in range(0, len(windows), batch_size):
            chunk = windows[offset : offset + batch_size]
            x = torch.stack([pair[0] for pair in chunk]).to(device)
            y = torch.stack([pair[1] for pair in chunk]).to(device)
            logits, _ = model(x)
            per_token_nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="none",
            ).view(len(chunk), -1)
            scored = per_token_nll[:, score_from:]
            total_nll += scored.to(torch.float64).sum().item()
            tokens_scored += scored.numel()
    model.train(was_training)

    expected_tokens = int(data.numel()) - 1
    if tokens_scored != expected_tokens:
        raise RuntimeError(f"Scored {tokens_scored} targets; expected {expected_tokens}")

    loss = total_nll / tokens_scored
    return {
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "bits_per_byte": bits_per_byte_from_loss(loss),
        "tokens_scored": tokens_scored,
        "coverage": tokens_scored / expected_tokens,
        "stride": stride,
    }


@torch.no_grad()
def evaluate_checkpoint_details(
    checkpoint_path: Path,
    device: str,
    *,
    stride: Optional[int] = None,
    batch_size: Optional[int] = None,
    val_data: Optional[torch.Tensor] = None,
) -> dict[str, float | int | str]:
    """Load one checkpoint and return deterministic full-validation metrics."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(missing_checkpoint_message(checkpoint_path.stem, checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = config_from_dict(checkpoint["config"])
    config.device = device
    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if val_data is None:
        val_data = load_data()["val"]
    metrics = full_split_loss(model, val_data, device, stride=stride, batch_size=batch_size)
    step = int(checkpoint.get("step", config.max_iters))
    tokens_per_step = config.batch_size * config.block_size
    return {
        "model": config.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "training_tokens": step * tokens_per_step,
        "tokens_per_step": tokens_per_step,
        **metrics,
    }


def evaluate_checkpoint(checkpoint_path: Path, eval_iters: Optional[int], device: str) -> float:
    """Backward-compatible wrapper returning deterministic full-validation loss.

    ``eval_iters`` is retained so existing callers keep working; random-batch
    evaluation is intentionally no longer used.
    """
    del eval_iters
    return float(evaluate_checkpoint_details(checkpoint_path, device)["loss"])


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write deterministic evaluation metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint_step",
        "training_tokens",
        "tokens_per_step",
        "full_val_loss",
        "perplexity",
        "bits_per_byte",
        "tokens_scored",
        "coverage",
        "context_stride",
        "uniform_256_loss",
        "uniform_observed_loss",
        "train_unigram_val_loss",
        "observed_vocab_size",
        "unseen_val_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints over the complete validation split.")
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=None,
        help="Deprecated compatibility option; complete deterministic evaluation is always used.",
    )
    parser.add_argument("--stride", type=int, default=None, help="Scoring stride; default is half the context length.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation window batch size.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_default_device()
    seed_everything(args.seed)
    ensure_artifact_dirs()
    splits = load_data()
    baselines = compute_baselines(splits["train"], splits["val"])

    rows: list[dict[str, object]] = []
    for model_name in ("model_a", "model_b"):
        checkpoint_path = args.checkpoint_dir / f"{model_name}.pt"
        try:
            details = evaluate_checkpoint_details(
                checkpoint_path,
                device,
                stride=args.stride,
                batch_size=args.batch_size,
                val_data=splits["val"],
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        row = {
            "model": details["model"],
            "checkpoint_step": details["checkpoint_step"],
            "training_tokens": details["training_tokens"],
            "tokens_per_step": details["tokens_per_step"],
            "full_val_loss": f"{float(details['loss']):.6f}",
            "perplexity": f"{float(details['perplexity']):.6f}",
            "bits_per_byte": f"{float(details['bits_per_byte']):.6f}",
            "tokens_scored": details["tokens_scored"],
            "coverage": f"{float(details['coverage']):.6f}",
            "context_stride": details["stride"],
            **{
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in baselines.items()
            },
        }
        rows.append(row)
        print(
            f"{model_name}: full_val_loss={row['full_val_loss']}, "
            f"perplexity={row['perplexity']}, bits_per_byte={row['bits_per_byte']}, "
            f"coverage={row['coverage']}"
        )

    write_metrics(args.output, rows)
    print(f"Saved deterministic metrics to {args.output}")


if __name__ == "__main__":
    main()
