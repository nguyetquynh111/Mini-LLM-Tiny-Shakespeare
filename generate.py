"""Generate text from a saved Tiny Shakespeare Transformer checkpoint.

Smoke-test example:
python generate.py --checkpoint checkpoints/model_a.pt --prompt "To be, or not to " --max_new_tokens 150
python generate.py --checkpoint checkpoints/model_b.pt --prompt "To be, or not to " --max_new_tokens 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs import config_from_dict, get_default_device
from data import decode, encode
from model import GPTLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_default_device()
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = config_from_dict(checkpoint["config"])
    config.device = device

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_tokens = encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("prompt must not be empty")

    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(idx, args.max_new_tokens)
    text = decode(generated[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
