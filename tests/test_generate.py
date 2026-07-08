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


def test_main_loads_checkpoint_and_prints_prompt_when_zero_new_tokens(tmp_path, monkeypatch, capsys):
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
            "0",
            "--device",
            "cpu",
        ],
    )

    generate.main()

    assert capsys.readouterr().out.strip() == "To be"


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
            "0",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate.main()
