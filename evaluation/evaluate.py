"""Evaluate saved Tiny Shakespeare checkpoints on validation loss."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from configs import config_from_dict, get_default_device
from data import get_batch, load_data
from model import GPTLanguageModel


DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "evaluation" / "metrics.csv"


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, eval_iters: Optional[int], device: str) -> float:
    """Return average validation cross-entropy loss for one checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = config_from_dict(checkpoint["config"])
    config.device = device
    if eval_iters is not None:
        config.eval_iters = eval_iters

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    losses = torch.empty(config.eval_iters, device=device)
    for index in range(config.eval_iters):
        x, y = get_batch("val", config.batch_size, config.block_size, device)
        _, loss = model(x, y)
        losses[index] = loss.item()

    return losses.mean().item()


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    """Write evaluation metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "final_val_loss", "perplexity"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Model A and Model B checkpoints.")
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_default_device()
    load_data()

    rows: list[dict[str, str]] = []
    for model_name in ("model_a", "model_b"):
        checkpoint_path = args.checkpoint_dir / f"{model_name}.pt"
        val_loss = evaluate_checkpoint(checkpoint_path, args.eval_iters, device)
        perplexity = math.exp(val_loss)
        rows.append(
            {
                "model": model_name,
                "final_val_loss": f"{val_loss:.6f}",
                "perplexity": f"{perplexity:.6f}",
            }
        )
        print(f"{model_name}: final_val_loss={val_loss:.6f}, perplexity={perplexity:.6f}")

    write_metrics(args.output, rows)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
