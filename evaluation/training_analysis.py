"""Reproducible convergence analysis from the saved training logs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_llm.configs import MODEL_PRESETS
from mini_llm.utils import EVALUATION_OUTPUT_DIR, LOG_DIR, missing_logs_message


DEFAULT_STATS_PATH = EVALUATION_OUTPUT_DIR / "convergence_stats.csv"
DEFAULT_COMPARISONS_PATH = EVALUATION_OUTPUT_DIR / "training_comparisons.csv"


def read_loss_log(path: Path, tokens_per_step: int) -> list[dict[str, float | int]]:
    """Read and validate one loss log, adding its cumulative token budget."""
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            step = int(raw["step"])
            rows.append(
                {
                    "step": step,
                    "tokens_seen": step * tokens_per_step,
                    "train_loss": float(raw["train_loss"]),
                    "val_loss": float(raw["val_loss"]),
                }
            )
    if not rows:
        raise ValueError(f"Loss log is empty: {path}")
    if any(current["step"] <= previous["step"] for previous, current in zip(rows, rows[1:])):
        raise ValueError(f"Steps must be strictly increasing: {path}")
    return rows


def divergence_onset(
    rows: list[dict[str, float | int]],
    *,
    threshold: float = 0.05,
    patience: int = 3,
) -> dict[str, float | int] | None:
    """Find the first sustained train/validation gap above a stated threshold."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if patience <= 0:
        raise ValueError("patience must be positive")
    for index, row in enumerate(rows):
        window = rows[index : index + patience]
        if len(window) < patience:
            break
        if all(float(item["val_loss"]) - float(item["train_loss"]) > threshold for item in window):
            return row
    return None


def summarize_convergence(
    model_name: str,
    rows: list[dict[str, float | int]],
    *,
    gap_threshold: float = 0.05,
    gap_patience: int = 3,
) -> dict[str, object]:
    """Summarize best validation point, final gap, and sustained divergence."""
    best = min(rows, key=lambda row: float(row["val_loss"]))
    final = rows[-1]
    onset = divergence_onset(rows, threshold=gap_threshold, patience=gap_patience)
    return {
        "model": model_name,
        "final_step": final["step"],
        "final_tokens_seen": final["tokens_seen"],
        "final_train_loss": f"{float(final['train_loss']):.6f}",
        "final_val_loss": f"{float(final['val_loss']):.6f}",
        "final_generalization_gap": f"{float(final['val_loss']) - float(final['train_loss']):.6f}",
        "best_val_step": best["step"],
        "best_val_tokens_seen": best["tokens_seen"],
        "best_val_loss": f"{float(best['val_loss']):.6f}",
        "overfit_cost_after_best": f"{float(final['val_loss']) - float(best['val_loss']):.6f}",
        "divergence_step": "" if onset is None else onset["step"],
        "divergence_tokens_seen": "" if onset is None else onset["tokens_seen"],
        "gap_threshold": gap_threshold,
        "gap_patience": gap_patience,
    }


def _row_at(rows: Iterable[dict[str, float | int]], key: str, value: int) -> dict[str, float | int]:
    for row in rows:
        if int(row[key]) == value:
            return row
    raise ValueError(f"No logged row has {key}={value}")


def comparison_rows(
    logs: dict[str, list[dict[str, float | int]]],
) -> list[dict[str, object]]:
    """Return both the original fixed-step and fair equal-token comparisons."""
    if set(logs) != {"model_a", "model_b"}:
        raise ValueError("Expected logs for model_a and model_b")

    common_steps = set(int(row["step"]) for row in logs["model_a"]) & set(
        int(row["step"]) for row in logs["model_b"]
    )
    common_tokens = set(int(row["tokens_seen"]) for row in logs["model_a"]) & set(
        int(row["tokens_seen"]) for row in logs["model_b"]
    )
    if not common_steps or not common_tokens:
        raise ValueError("Logs do not contain a common fixed-step and equal-token comparison point")

    scenarios = {
        "fixed_step": ("step", max(common_steps)),
        "equal_tokens": ("tokens_seen", max(common_tokens)),
    }
    output: list[dict[str, object]] = []
    for scenario, (key, value) in scenarios.items():
        for model_name in ("model_a", "model_b"):
            row = _row_at(logs[model_name], key, value)
            output.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    "step": row["step"],
                    "tokens_seen": row["tokens_seen"],
                    "train_loss": f"{float(row['train_loss']):.6f}",
                    "val_loss": f"{float(row['val_loss']):.6f}",
                    "metric_source": "saved_training_log_sampled_batches",
                }
            )
    return output


def load_default_logs(model_a_log: Path, model_b_log: Path) -> dict[str, list[dict[str, float | int]]]:
    """Load both standard logs using their architecture-derived token budgets."""
    if not model_a_log.exists() or not model_b_log.exists():
        raise FileNotFoundError(missing_logs_message(model_a_log, model_b_log))
    logs: dict[str, list[dict[str, float | int]]] = {}
    for model_name, path in (("model_a", model_a_log), ("model_b", model_b_log)):
        preset = MODEL_PRESETS[model_name]
        tokens_per_step = int(preset["batch_size"]) * int(preset["block_size"])
        logs[model_name] = read_loss_log(path, tokens_per_step)
    return logs


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_outputs(
    logs: dict[str, list[dict[str, float | int]]],
    stats_path: Path = DEFAULT_STATS_PATH,
    comparisons_path: Path = DEFAULT_COMPARISONS_PATH,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute and save all convergence tables."""
    stats = [summarize_convergence(model_name, logs[model_name]) for model_name in ("model_a", "model_b")]
    comparisons = comparison_rows(logs)
    _write_csv(stats_path, stats)
    _write_csv(comparisons_path, comparisons)
    return stats, comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze convergence and training-budget fairness.")
    parser.add_argument("--model-a-log", type=Path, default=LOG_DIR / "model_a_loss.csv")
    parser.add_argument("--model-b-log", type=Path, default=LOG_DIR / "model_b_loss.csv")
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--comparisons-output", type=Path, default=DEFAULT_COMPARISONS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = load_default_logs(args.model_a_log, args.model_b_log)
    stats, comparisons = write_analysis_outputs(logs, args.stats_output, args.comparisons_output)
    for row in stats:
        print(
            f"{row['model']}: best_val={row['best_val_loss']} at step {row['best_val_step']}; "
            f"final_gap={row['final_generalization_gap']}"
        )
    equal_token_rows = [row for row in comparisons if row["scenario"] == "equal_tokens"]
    budget = equal_token_rows[0]["tokens_seen"]
    print(f"Equal-token comparison saved at {int(budget):,} training tokens")
    print(f"Saved {args.stats_output} and {args.comparisons_output}")


if __name__ == "__main__":
    main()
