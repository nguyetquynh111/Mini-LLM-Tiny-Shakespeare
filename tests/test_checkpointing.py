import pytest

torch = pytest.importorskip("torch")

from mini_llm.configs import TransformerConfig
from mini_llm.model import GPTLanguageModel
from mini_llm.train import load_resume_checkpoint
from mini_llm.utils import build_checkpoint, save_checkpoint


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


def test_checkpoint_save_load_roundtrip_includes_metadata(tmp_path):
    config = tiny_config(grad_clip=1.0)
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    checkpoint = build_checkpoint(
        config=config,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        optimizer_config={"type": "AdamW", "learning_rate": config.learning_rate, "grad_clip": 1.0},
        step=7,
        train_loss=2.5,
        val_loss=2.7,
        loss_rows=[{"step": 7, "train_loss": 2.5, "val_loss": 2.7}],
    )
    path = tmp_path / "tiny.pt"

    save_checkpoint(path, checkpoint)
    loaded = torch.load(path, map_location="cpu")

    assert loaded["config_name"] == "tiny"
    assert loaded["model_config"] == config.to_dict()
    assert loaded["optimizer_config"]["grad_clip"] == 1.0
    assert loaded["step"] == 7
    assert loaded["train_loss"] == 2.5
    assert loaded["val_loss"] == 2.7
    assert loaded["tokenizer"]["vocab_size"] == 256
    assert "timestamp" in loaded


def test_load_resume_checkpoint_restores_model_optimizer_and_step(tmp_path):
    config = tiny_config()
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    checkpoint = build_checkpoint(
        config=config,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        optimizer_config={"type": "AdamW", "learning_rate": config.learning_rate},
        step=3,
        train_loss=3.0,
        val_loss=3.1,
        loss_rows=[{"step": 3, "train_loss": 3.0, "val_loss": 3.1}],
    )
    path = tmp_path / "resume.pt"
    save_checkpoint(path, checkpoint)

    resumed_model = GPTLanguageModel(config)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=config.learning_rate)
    step, loss_rows = load_resume_checkpoint(path, resumed_model, resumed_optimizer, "cpu")

    assert step == 3
    assert loss_rows == [{"step": 3, "train_loss": 3.0, "val_loss": 3.1}]
    for original, resumed in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(original, resumed)
    assert resumed_optimizer.state_dict()["param_groups"][0]["lr"] == config.learning_rate
