import json

import pytest

torch = pytest.importorskip("torch")

from evaluation import generate_samples
from mini_llm.configs import TransformerConfig
from mini_llm.model import GPTLanguageModel


def tiny_config(**overrides):
    values = {
        "name": "tiny",
        "vocab_size": 256,
        "block_size": 8,
        "batch_size": 2,
        "n_embd": 16,
        "n_head": 4,
        "n_layer": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "device": "cpu",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_generate_for_prompts_records_exact_requested_new_tokens():
    torch.manual_seed(0)
    model = GPTLanguageModel(tiny_config())
    model.eval()

    outputs = generate_samples.generate_for_prompts(
        model,
        ["To be"],
        device="cpu",
        max_new_tokens=5,
        temperature=0.9,
        top_k=10,
    )

    assert outputs[0]["prompt"] == "To be"
    assert outputs[0]["requested_new_token_count"] == 5
    assert outputs[0]["actual_generated_new_token_count"] == 5
    assert outputs[0]["sampling"] == {"temperature": 0.9, "top_k": 10}


def test_verify_generated_length_accepts_exact_byte_count():
    generate_samples.verify_generated_length("abc", "abcde", 2)


def test_verify_generated_length_rejects_wrong_byte_count():
    with pytest.raises(ValueError, match="Expected 3 new byte tokens, found 2"):
        generate_samples.verify_generated_length("abc", "abcde", 3)


def test_verify_generation_record_length_uses_saved_byte_count():
    generate_samples.verify_generation_record_length({"actual_generated_new_token_count": 150}, 150)
    with pytest.raises(ValueError, match="Expected 150 new byte tokens, found 149"):
        generate_samples.verify_generation_record_length({"actual_generated_new_token_count": 149}, 150)


def test_write_generations_jsonl_includes_checkpoint_and_model(tmp_path):
    path = tmp_path / "model.jsonl"
    outputs = [
        {
            "prompt": "To be",
            "generated_text": "To be!",
            "requested_new_token_count": 1,
            "actual_generated_new_token_count": 1,
            "sampling": {"temperature": 1.0, "top_k": None},
        }
    ]

    generate_samples.write_generations_jsonl(path, "model_a", tmp_path / "model_a.pt", outputs)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["model_config_name"] == "model_a"
    assert record["checkpoint_path"].endswith("model_a.pt")
    assert record["actual_generated_new_token_count"] == 1
