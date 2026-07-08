import sys

import pytest

torch = pytest.importorskip("torch")

from mini_llm.configs import TransformerConfig
from mini_llm.model import GPTLanguageModel
from mini_llm import generate


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


def test_main_exits_with_clear_message_for_missing_checkpoint(tmp_path, monkeypatch, capsys):
    missing_checkpoint = tmp_path / "missing.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(missing_checkpoint),
            "--prompt",
            "To be",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate.main()

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "Missing checkpoint" in stderr
    assert "python -m mini_llm.train --config missing" in stderr


def test_main_loads_checkpoint_and_prints_generated_text(tmp_path, monkeypatch, capsys):
    config = tiny_config()
    model = GPTLanguageModel(config)
    checkpoint_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "To be",
            "--max_new_tokens",
            "1",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "generation.json"),
        ],
    )

    generate.main()

    output = capsys.readouterr().out.rstrip("\n")
    assert len(output.encode("utf-8")) >= len("To be".encode("utf-8")) + 1


def test_main_rejects_empty_prompt_after_loading_checkpoint(tmp_path, monkeypatch):
    config = tiny_config()
    model = GPTLanguageModel(config)
    checkpoint_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "",
            "--max_new_tokens",
            "1",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate.main()


def test_main_rejects_non_positive_max_new_tokens(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "tiny.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "To be",
            "--max-new-tokens",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="--max-new-tokens must be positive"):
        generate.main()


def test_main_writes_structured_generation_json(tmp_path, monkeypatch):
    config = tiny_config()
    model = GPTLanguageModel(config)
    checkpoint_path = tmp_path / "tiny.pt"
    output_path = tmp_path / "generation.json"
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "To be",
            "--max-new-tokens",
            "3",
            "--temperature",
            "0.8",
            "--top-k",
            "5",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
    )

    generate.main()

    import json

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["prompt"] == "To be"
    assert record["checkpoint_path"] == str(checkpoint_path)
    assert record["model_config_name"] == "tiny"
    assert record["requested_new_token_count"] == 3
    assert record["actual_generated_new_token_count"] == 3
    assert record["sampling"] == {"temperature": 0.8, "top_k": 5, "seed": 1337}
