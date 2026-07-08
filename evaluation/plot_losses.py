"""Plot training and validation loss curves for both Transformer models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "evaluation" / "loss_convergence.png"


def read_loss_log(path: Path) -> dict[str, list[float]]:
    """Read one loss CSV file produced by train.py."""
    if not path.exists():
        raise FileNotFoundError(f"Missing loss log: {path}")

    steps: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"step", "train_loss", "val_loss"}
        if set(reader.fieldnames or []) != required_columns:
            raise ValueError(f"Expected columns step,train_loss,val_loss in {path}")

        for row in reader:
            steps.append(float(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not steps:
        raise ValueError(f"Loss log is empty: {path}")

    return {
        "step": steps,
        "train_loss": train_losses,
        "val_loss": val_losses,
    }


def plot_losses(log_dir: Path, output_path: Path) -> None:
    """Create a clean loss convergence plot for the README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_a = read_loss_log(log_dir / "model_a_loss.csv")
    model_b = read_loss_log(log_dir / "model_b_loss.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    ax.plot(model_a["step"], model_a["train_loss"], label="Model A train", color="#1f77b4", linewidth=2.0)
    ax.plot(model_a["step"], model_a["val_loss"], label="Model A validation", color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(model_b["step"], model_b["train_loss"], label="Model B train", color="#d62728", linewidth=2.0)
    ax.plot(model_b["step"], model_b["val_loss"], label="Model B validation", color="#d62728", linestyle="--", linewidth=2.0)

    ax.set_title("Tiny Shakespeare Loss Convergence", fontsize=15, pad=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=True)
    ax.margins(x=0.02)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Model A and Model B loss curves.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_losses(args.log_dir, args.output)


if __name__ == "__main__":
    main()
