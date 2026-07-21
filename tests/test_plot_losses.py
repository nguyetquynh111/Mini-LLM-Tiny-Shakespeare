import math

from evaluation.plot_losses import validation_perplexities


def test_validation_perplexities_are_exponentiated_validation_losses():
    rows = [
        {"step": 0, "train_loss": 2.5, "val_loss": 2.0},
        {"step": 1, "train_loss": 1.5, "val_loss": 1.0},
    ]

    perplexities = validation_perplexities(rows)

    assert perplexities == [math.exp(2.0), math.exp(1.0)]
