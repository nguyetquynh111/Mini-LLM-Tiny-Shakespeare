import csv

import pytest

from evaluation.training_analysis import (
    comparison_rows,
    divergence_onset,
    read_loss_log,
    summarize_convergence,
)


def rows(tokens_per_step=10):
    return [
        {"step": 0, "tokens_seen": 0, "train_loss": 5.5, "val_loss": 5.5},
        {"step": 1, "tokens_seen": tokens_per_step, "train_loss": 2.0, "val_loss": 2.04},
        {"step": 2, "tokens_seen": 2 * tokens_per_step, "train_loss": 1.8, "val_loss": 1.87},
        {"step": 3, "tokens_seen": 3 * tokens_per_step, "train_loss": 1.7, "val_loss": 1.78},
        {"step": 4, "tokens_seen": 4 * tokens_per_step, "train_loss": 1.6, "val_loss": 1.69},
    ]


def test_read_loss_log_adds_training_token_budget(tmp_path):
    path = tmp_path / "loss.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerow({"step": 0, "train_loss": 5.5, "val_loss": 5.6})
        writer.writerow({"step": 2, "train_loss": 2.0, "val_loss": 2.1})

    loaded = read_loss_log(path, tokens_per_step=128)

    assert loaded[1]["tokens_seen"] == 256


def test_divergence_requires_sustained_gap():
    onset = divergence_onset(rows(), threshold=0.05, patience=3)

    assert onset is not None
    assert onset["step"] == 2
    assert divergence_onset(rows(), threshold=0.2, patience=2) is None


def test_convergence_summary_reports_best_and_final_gap():
    summary = summarize_convergence("model_a", rows())

    assert summary["best_val_step"] == 4
    assert summary["final_generalization_gap"] == "0.090000"
    assert summary["divergence_step"] == 2


def test_comparisons_include_fixed_step_and_largest_common_token_budget():
    model_a = rows(tokens_per_step=10)
    model_b = [
        {**row, "tokens_seen": int(row["step"]) * 20}
        for row in rows(tokens_per_step=20)
    ]

    comparisons = comparison_rows({"model_a": model_a, "model_b": model_b})
    fixed = [row for row in comparisons if row["scenario"] == "fixed_step"]
    equal = [row for row in comparisons if row["scenario"] == "equal_tokens"]

    assert {row["step"] for row in fixed} == {4}
    assert {row["tokens_seen"] for row in equal} == {40}
    assert {(row["model"], row["step"]) for row in equal} == {("model_a", 4), ("model_b", 2)}


def test_comparisons_reject_missing_model():
    with pytest.raises(ValueError, match="Expected logs"):
        comparison_rows({"model_a": rows()})
