import math

import pytest

torch = pytest.importorskip("torch")

from evaluation.evaluate import (
    _scoring_windows,
    bits_per_byte_from_loss,
    compute_baselines,
    full_split_loss,
    perplexity_from_loss,
)
from mini_llm.configs import TransformerConfig
from mini_llm.model import GPTLanguageModel


def test_perplexity_from_loss_uses_exponential():
    assert perplexity_from_loss(0.0) == 1.0
    assert math.isclose(perplexity_from_loss(2.0), math.exp(2.0))


def test_bits_per_byte_converts_nats():
    assert math.isclose(bits_per_byte_from_loss(math.log(2.0)), 1.0)


def test_scoring_windows_cover_every_target_once():
    data = torch.arange(19)
    windows = _scoring_windows(data, block_size=8, stride=4)

    scored = sum(len(y) - score_from for _, y, score_from in windows)

    assert scored == len(data) - 1
    assert all(len(x) <= 8 for x, _, _ in windows)


def test_full_split_loss_is_exact_and_deterministic_for_uniform_model():
    config = TransformerConfig(
        name="uniform",
        vocab_size=256,
        block_size=8,
        batch_size=3,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        learning_rate=1e-3,
        device="cpu",
    )
    model = GPTLanguageModel(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    data = torch.arange(23) % 7

    first = full_split_loss(model, data, "cpu", stride=4, batch_size=2)
    second = full_split_loss(model, data, "cpu", stride=4, batch_size=5)

    assert first["tokens_scored"] == len(data) - 1
    assert first["coverage"] == 1.0
    assert math.isclose(first["loss"], math.log(256), abs_tol=1e-6)
    assert math.isclose(first["loss"], second["loss"], abs_tol=1e-9)


def test_baselines_fit_unigram_only_on_training_data():
    train = torch.tensor([1, 1, 1, 2], dtype=torch.long)
    val = torch.tensor([1, 2, 3], dtype=torch.long)

    baselines = compute_baselines(train, val)

    assert baselines["observed_vocab_size"] == 2
    assert baselines["unseen_val_tokens"] == 1
    assert math.isclose(baselines["uniform_256_loss"], math.log(256))
