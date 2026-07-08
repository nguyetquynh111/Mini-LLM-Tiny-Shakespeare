import pytest

torch = pytest.importorskip("torch")

from mini_llm.configs import TransformerConfig
from mini_llm.model import CausalSelfAttention, GPTLanguageModel


def tiny_config(**overrides):
    values = {
        "name": "tiny",
        "vocab_size": 256,
        "block_size": 8,
        "batch_size": 2,
        "n_embd": 16,
        "n_head": 4,
        "n_layer": 2,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "device": "cpu",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_attention_requires_embedding_dimension_divisible_by_heads():
    with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
        CausalSelfAttention(tiny_config(n_embd=10, n_head=4))


def test_forward_returns_logits_and_loss_with_expected_shapes():
    torch.manual_seed(0)
    model = GPTLanguageModel(tiny_config())
    idx = torch.randint(0, 256, (2, 5))
    targets = torch.randint(0, 256, (2, 5))

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 5, 256)
    assert loss is not None
    assert loss.ndim == 0


def test_forward_without_targets_omits_loss():
    model = GPTLanguageModel(tiny_config())
    idx = torch.randint(0, 256, (2, 5))

    _, loss = model(idx)

    assert loss is None


def test_forward_rejects_sequences_longer_than_block_size():
    model = GPTLanguageModel(tiny_config(block_size=4))
    idx = torch.randint(0, 256, (1, 5))

    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)


def test_causal_mask_prevents_future_tokens_from_changing_prefix_logits():
    torch.manual_seed(0)
    model = GPTLanguageModel(tiny_config(block_size=6))
    model.eval()
    left = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed_future = torch.tensor([[1, 2, 3, 99, 98, 97]], dtype=torch.long)

    with torch.no_grad():
        left_logits, _ = model(left)
        changed_logits, _ = model(changed_future)

    assert torch.allclose(left_logits[:, :3, :], changed_logits[:, :3, :], atol=1e-6)


def test_generate_appends_exact_number_of_tokens_and_preserves_prompt():
    torch.manual_seed(0)
    model = GPTLanguageModel(tiny_config(block_size=4))
    model.eval()
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.long)

    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (1, 8)
    assert torch.equal(generated[:, :3], prompt)
