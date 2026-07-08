import pytest

from mini_llm.configs import config_from_dict, get_config


def test_get_config_applies_non_none_overrides(monkeypatch):
    monkeypatch.setattr("mini_llm.configs.get_default_device", lambda: "cpu")

    config = get_config("model_a", max_iters=3, eval_iters=None, device="cpu", seed=7)

    assert config.name == "model_a"
    assert config.max_iters == 3
    assert config.eval_iters == 200
    assert config.device == "cpu"
    assert config.seed == 7


def test_get_config_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown config 'missing'"):
        get_config("missing")


def test_config_from_dict_recreates_transformer_config():
    values = get_config("model_a", device="cpu").to_dict()

    config = config_from_dict(values)

    assert config.to_dict() == values
