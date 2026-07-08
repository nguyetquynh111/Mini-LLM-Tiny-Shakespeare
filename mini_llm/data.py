"""Byte-level tokenizer and data loading utilities for Tiny Shakespeare."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.error import URLError
from urllib.request import urlretrieve

import torch

from mini_llm.utils import DATA_FILE


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TRAIN_FRACTION = 0.9
VOCAB_SIZE = 256

_DATA_CACHE: Optional[dict[str, torch.Tensor]] = None


def encode(text: str) -> list[int]:
    """Encode text to byte tokens in the range 0 to 255."""
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    """Decode byte tokens back to text, replacing invalid byte sequences."""
    return bytes(tokens).decode("utf-8", errors="replace")


def ensure_dataset(data_file: Path = DATA_FILE) -> Path:
    """Download Tiny Shakespeare if it is not already present."""
    if data_file.exists():
        return data_file

    data_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(DATA_URL, data_file)
    except (OSError, URLError) as exc:
        raise RuntimeError(
            "Tiny Shakespeare dataset is missing and could not be downloaded. "
            f"Place the dataset at {data_file} or check network access."
        ) from exc

    return data_file


def load_text(data_file: Path = DATA_FILE) -> str:
    """Load the Tiny Shakespeare text dataset as UTF-8 text."""
    path = ensure_dataset(data_file)
    return path.read_text(encoding="utf-8")


def load_data() -> dict[str, torch.Tensor]:
    """Load and cache train and validation tensors."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    text = load_text()
    tokens = torch.tensor(encode(text), dtype=torch.long)
    split_index = int(TRAIN_FRACTION * len(tokens))
    _DATA_CACHE = {
        "train": tokens[:split_index],
        "val": tokens[split_index:],
    }
    return _DATA_CACHE


def get_batch(
    split: str,
    batch_size: int,
    block_size: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return input and target tensors with shape (B, T)."""
    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'")

    data = load_data()[split]
    if len(data) <= block_size:
        raise ValueError("Dataset split is too small for the requested block_size")

    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    offsets = torch.arange(block_size)
    positions = starts[:, None] + offsets[None, :]
    x = data[positions]
    y = data[positions + 1]

    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y


if __name__ == "__main__":
    sample = "To be, or not to be."
    tokens = encode(sample)
    print("Encoded tokens:", tokens)
    print("Decoded text:", decode(tokens))
    xb, yb = get_batch("train", batch_size=4, block_size=8)
    print("Input batch shape:", tuple(xb.shape))
    print("Target batch shape:", tuple(yb.shape))
