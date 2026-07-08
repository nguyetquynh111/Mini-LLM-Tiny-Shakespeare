"""Train a byte-level Transformer on Tiny Shakespeare.

Smoke-test examples:
python train.py --config model_a --max_iters 2 --eval_interval 1 --eval_iters 1
python train.py --config model_b --max_iters 2 --eval_interval 1 --eval_iters 1
python -c "from configs import get_config; from model import GPTLanguageModel; import torch; c=get_config('model_a'); m=GPTLanguageModel(c); x=torch.zeros((2,c.block_size),dtype=torch.long); print(m(x)[0].shape)"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Union

import torch

from configs import get_config
from data import get_batch, load_data
from model import GPTLanguageModel


ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Tiny Shakespeare Transformer.")
    parser.add_argument("--config", choices=["model_a", "model_b"], required=True)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, config) -> dict[str, float]:
    """Estimate average train and validation loss."""
    model.eval()
    losses_by_split: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.empty(config.eval_iters, device=config.device)
        for index in range(config.eval_iters):
            x, y = get_batch(split, config.batch_size, config.block_size, config.device)
            _, loss = model(x, y)
            losses[index] = loss.item()
        losses_by_split[split] = losses.mean().item()
    model.train()
    return losses_by_split


def write_loss_log(path: Path, rows: list[dict[str, Union[float, int]]]) -> None:
    """Write training losses to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = get_config(
        args.config,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=args.device,
        seed=args.seed,
    )

    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    load_data()
    model = GPTLanguageModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_rows: list[dict[str, Union[float, int]]] = []

    print(f"Training {config.name} on {config.device}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    for step in range(config.max_iters + 1):
        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = estimate_loss(model, config)
            row = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
            }
            loss_rows.append(row)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        if step == config.max_iters:
            break

        x, y = get_batch("train", config.batch_size, config.block_size, config.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    log_path = LOG_DIR / f"{config.name}_loss.csv"
    checkpoint_path = CHECKPOINT_DIR / f"{config.name}.pt"

    write_loss_log(log_path, loss_rows)
    torch.save(
        {
            "config_name": config.name,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": config.max_iters,
            "loss_rows": loss_rows,
        },
        checkpoint_path,
    )

    print(f"Saved loss log to {log_path}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
