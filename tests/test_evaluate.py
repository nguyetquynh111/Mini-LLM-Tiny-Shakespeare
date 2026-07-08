import math

import pytest

pytest.importorskip("torch")

from evaluation.evaluate import perplexity_from_loss


def test_perplexity_from_loss_uses_exponential():
    assert perplexity_from_loss(0.0) == 1.0
    assert math.isclose(perplexity_from_loss(2.0), math.exp(2.0))
