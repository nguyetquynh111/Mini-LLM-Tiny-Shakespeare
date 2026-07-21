"""Train a byte-level Transformer on Tiny Shakespeare.

Smoke-test examples:
python -m mini_llm.train --config model_a --max_iters 2 --eval_interval 1 --eval_iters 1
python -m mini_llm.train --config model_b --max_iters 2 --eval_interval 1 --eval_iters 1
python -c "from mini_llm.configs import get_config; from mini_llm.model import GPTLanguageModel; import torch; c=get_config('model_a'); m=GPTLanguageModel(c); x=torch.zeros((2,c.block_size),dtype=torch.long); print(m(x)[0].shape)"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Union

import torch

from mini_llm.configs import get_config
from mini_llm.data import get_batch, load_data
from mini_llm.model import GPTLanguageModel
from mini_llm.utils import (
    CHECKPOINT_DIR,
    LOG_DIR,
    build_checkpoint,
    ensure_artifact_dirs,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Tiny Shakespeare Transformer.")
    parser.add_argument("--config", choices=["model_a", "model_b"], required=True)
    parser.add_argument("--max-iters", "--max_iters", dest="max_iters", type=int, default=None)
    parser.add_argument("--eval-interval", "--eval_interval", dest="eval_interval", type=int, default=None)
    parser.add_argument("--eval-iters", "--eval_iters", dest="eval_iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training arguments with clear messages."""
    if args.max_iters is not None and args.max_iters < 0:
        raise ValueError("--max-iters must be non-negative")
    if args.eval_interval is not None and args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    if args.eval_iters is not None and args.eval_iters <= 0:
        raise ValueError("--eval-iters must be positive")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive when provided")


def load_resume_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[int, list[dict[str, Union[float, int]]]]:
    """Load model/optimizer state and return the saved step and loss rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    step = int(checkpoint.get("step", 0))
    loss_rows = checkpoint.get("loss_rows", [])
    if not isinstance(loss_rows, list):
        loss_rows = []
    return step, loss_rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = get_config(
        args.config,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=args.device,
        seed=args.seed,
        grad_clip=args.grad_clip,
    )

    seed_everything(config.seed)

    ensure_artifact_dirs()

    load_data()
    model = GPTLanguageModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_rows: list[dict[str, Union[float, int]]] = []
    start_step = 0
    best_val_loss = float("inf")

    if args.resume_from is not None:
        start_step, loss_rows = load_resume_checkpoint(args.resume_from, model, optimizer, config.device)
        if loss_rows:
            previous_val_losses = [float(row["val_loss"]) for row in loss_rows if "val_loss" in row]
            if previous_val_losses:
                best_val_loss = min(previous_val_losses)
        print(f"Resumed from {args.resume_from} at step {start_step}")

    print(f"Training {config.name} on {config.device}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    final_train_loss = None
    final_val_loss = None
    optimizer_config = {
        "type": "AdamW",
        "learning_rate": config.learning_rate,
        "grad_clip": config.grad_clip,
    }
    final_checkpoint_path = args.checkpoint_dir / f"{config.name}.pt"
    best_checkpoint_path = args.checkpoint_dir / f"{config.name}_best.pt"

    for step in range(start_step, config.max_iters + 1):
        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = estimate_loss(model, config)
            row = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
            }
            loss_rows.append(row)
            final_train_loss = losses["train"]
            final_val_loss = losses["val"]
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    best_checkpoint_path,
                    build_checkpoint(
                        config=config,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        optimizer_config=optimizer_config,
                        step=step,
                        train_loss=losses["train"],
                        val_loss=losses["val"],
                        loss_rows=loss_rows,
                    ),
                )

        if step == config.max_iters:
            break

        x, y = get_batch("train", config.batch_size, config.block_size, config.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    log_path = args.log_dir / f"{config.name}_loss.csv"

    write_loss_log(log_path, loss_rows)
    save_checkpoint(
        final_checkpoint_path,
        build_checkpoint(
            config=config,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            optimizer_config=optimizer_config,
            step=config.max_iters,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            loss_rows=loss_rows,
        ),
    )

    print(f"Saved loss log to {log_path}")
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    print(f"Saved best checkpoint to {best_checkpoint_path}")


if __name__ == "__main__":
    main()
