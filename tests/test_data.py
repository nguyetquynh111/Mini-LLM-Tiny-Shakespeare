import pytest

torch = pytest.importorskip("torch")

from mini_llm import data


def test_encode_decode_round_trips_utf8_text():
    text = "To be, or not to be. O Romeo!"

    tokens = data.encode(text)

    assert all(isinstance(token, int) for token in tokens)
    assert all(0 <= token <= 255 for token in tokens)
    assert data.decode(tokens) == text


def test_decode_replaces_invalid_utf8_bytes():
    assert data.decode([0xFF]) == "\ufffd"


def test_load_data_splits_and_caches_encoded_text(monkeypatch):
    calls = []

    def fake_load_text():
        calls.append("called")
        return "abcdefghij"

    monkeypatch.setattr(data, "_DATA_CACHE", None)
    monkeypatch.setattr(data, "load_text", fake_load_text)

    first = data.load_data()
    second = data.load_data()

    assert first is second
    assert calls == ["called"]
    assert first["train"].tolist() == data.encode("abcdefghi")
    assert first["val"].tolist() == data.encode("j")


def test_get_batch_returns_shifted_targets_from_requested_split(monkeypatch):
    dataset = torch.arange(20, dtype=torch.long)
    monkeypatch.setattr(data, "load_data", lambda: {"train": dataset, "val": dataset + 100})
    torch.manual_seed(0)

    x, y = data.get_batch("train", batch_size=4, block_size=5)

    assert x.shape == (4, 5)
    assert y.shape == (4, 5)
    assert torch.equal(y, x + 1)


def test_get_batch_rejects_invalid_split():
    with pytest.raises(ValueError, match="split must be either 'train' or 'val'"):
        data.get_batch("test", batch_size=1, block_size=1)


def test_get_batch_rejects_too_small_split(monkeypatch):
    monkeypatch.setattr(data, "load_data", lambda: {"train": torch.arange(4), "val": torch.arange(4)})

    with pytest.raises(ValueError, match="too small"):
        data.get_batch("train", batch_size=1, block_size=4)
