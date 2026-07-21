"""Create publication-quality convergence and budget-comparison plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from evaluation.training_analysis import load_default_logs, write_analysis_outputs
from mini_llm.utils import LOG_DIR, PLOT_DIR, ensure_artifact_dirs


DEFAULT_OUTPUT = PLOT_DIR / "loss_convergence.png"
DEFAULT_EQUAL_TOKEN_OUTPUT = PLOT_DIR / "equal_token_comparison.png"
COLORS = {"model_a": "#2563EB", "model_b": "#DC2626"}
LABELS = {"model_a": "Model A", "model_b": "Model B"}


def _millions(value: float, _position: int) -> str:
    return f"{value / 1_000_000:.0f}M"


def plot_convergence(logs: dict[str, list[dict[str, float | int]]], output_path: Path) -> None:
    """Plot train/validation loss against steps and actual tokens consumed."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True, constrained_layout=True)
    for model_name in ("model_a", "model_b"):
        rows = logs[model_name]
        steps = [int(row["step"]) for row in rows]
        tokens = [int(row["tokens_seen"]) for row in rows]
        train = [float(row["train_loss"]) for row in rows]
        val = [float(row["val_loss"]) for row in rows]
        color = COLORS[model_name]
        label = LABELS[model_name]
        for axis, x_values in zip(axes, (steps, tokens)):
            axis.plot(x_values, train, linestyle="--", linewidth=2.0, color=color, alpha=0.8, label=f"{label} train")
            axis.plot(x_values, val, linewidth=2.5, color=color, label=f"{label} validation")
            best_index = min(range(len(val)), key=val.__getitem__)
            axis.scatter(x_values[best_index], val[best_index], s=55, facecolor="white", edgecolor=color, linewidth=2, zorder=3)

    axes[0].set_title("Fixed optimization-step budget")
    axes[0].set_xlabel("Optimizer step")
    axes[1].set_title("Actual training-token budget")
    axes[1].set_xlabel("Byte tokens processed")
    axes[1].xaxis.set_major_formatter(FuncFormatter(_millions))
    common_tokens = set(int(row["tokens_seen"]) for row in logs["model_a"]) & set(
        int(row["tokens_seen"]) for row in logs["model_b"]
    )
    equal_token_budget = max(common_tokens)
    axes[1].axvline(equal_token_budget, color="#6B7280", linestyle=":", linewidth=1.5)
    axes[1].annotate(
        "equal-token budget",
        xy=(equal_token_budget, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        rotation=90,
        fontsize=8,
        color="#4B5563",
    )
    axes[0].set_ylabel("Cross-entropy loss (nats/byte)")
    axes[0].legend(frameon=True, fontsize=9)
    for axis in axes:
        axis.set_ylim(bottom=1.1)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tiny Shakespeare: Training and Validation Convergence", fontsize=16, fontweight="semibold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_budget_comparison(comparisons: list[dict[str, object]], output_path: Path) -> None:
    """Compare logged validation loss under fixed-step and equal-token budgets."""
    lookup = {(str(row["scenario"]), str(row["model"])): row for row in comparisons}
    scenarios = ["fixed_step", "equal_tokens"]
    scenario_labels = ["Same steps\n(5,000 each)", "Same tokens\n(10.24M each)"]
    x_positions = [0, 1]
    width = 0.34

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(7.8, 5.2))
    for index, model_name in enumerate(("model_a", "model_b")):
        values = [float(lookup[(scenario, model_name)]["val_loss"]) for scenario in scenarios]
        positions = [x + (index - 0.5) * width for x in x_positions]
        bars = axis.bar(positions, values, width=width, color=COLORS[model_name], label=LABELS[model_name])
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)

    axis.set_xticks(x_positions, scenario_labels)
    axis.set_ylabel("Logged validation loss (nats/byte)")
    axis.set_title("Scaling Comparison: Fixed Steps vs Equal Training Tokens", fontweight="semibold")
    axis.set_ylim(0, max(float(row["val_loss"]) for row in comparisons) * 1.22)
    axis.legend(frameon=True)
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.025,
        "Source: saved training-log validation estimates; final full-split metrics are reported separately.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training convergence and equal-token comparison.")
    parser.add_argument("--model-a-log", type=Path, default=LOG_DIR / "model_a_loss.csv")
    parser.add_argument("--model-b-log", type=Path, default=LOG_DIR / "model_b_loss.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--equal-token-output", type=Path, default=DEFAULT_EQUAL_TOKEN_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    logs = load_default_logs(args.model_a_log, args.model_b_log)
    _, comparisons = write_analysis_outputs(logs)
    plot_convergence(logs, args.output)
    plot_budget_comparison(comparisons, args.equal_token_output)
    print(f"Saved convergence plot to {args.output}")
    print(f"Saved equal-token comparison plot to {args.equal_token_output}")


if __name__ == "__main__":
    main()
