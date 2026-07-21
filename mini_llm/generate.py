"""Generate text from a saved Tiny Shakespeare Transformer checkpoint.

Smoke-test example:
python -m mini_llm.generate --checkpoint mini_llm/checkpoints/model_a.pt --prompt "To be, or not to " --max-new-tokens 150
python -m mini_llm.generate --checkpoint mini_llm/checkpoints/model_b.pt --prompt "To be, or not to " --max-new-tokens 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from mini_llm.configs import config_from_dict, get_default_device
from mini_llm.data import decode, encode
from mini_llm.model import GPTLanguageModel
from mini_llm.utils import GENERATION_DIR, ensure_artifact_dirs, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid generation arguments with clear messages."""
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive when provided")


def default_output_path(checkpoint_path: Path) -> Path:
    """Return the default JSON output path for a generation run."""
    return GENERATION_DIR / f"{checkpoint_path.stem}_generation.json"


def write_generation_json(path: Path, record: dict[str, object]) -> None:
    """Write one structured generation record as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = args.device or get_default_device()
    seed_everything(args.seed)
    ensure_artifact_dirs()

    if not args.checkpoint.exists():
        model_name = args.checkpoint.stem
        print(
            f"Missing checkpoint: {args.checkpoint}\n"
            f"Run this first from the repository root: python -m mini_llm.train --config {model_name}",
            file=sys.stderr,
        )
        raise SystemExit(1)

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
    generated = model.generate(
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    generated_tokens = generated[0].tolist()
    actual_new_tokens = len(generated_tokens) - len(prompt_tokens)
    if actual_new_tokens != args.max_new_tokens:
        raise RuntimeError(
            f"Generated {actual_new_tokens} new byte tokens, expected {args.max_new_tokens}"
        )

    text = decode(generated_tokens)
    output_path = args.output or default_output_path(args.checkpoint)
    write_generation_json(
        output_path,
        {
            "prompt": args.prompt,
            "checkpoint_path": str(args.checkpoint),
            "model_config_name": config.name,
            "requested_new_token_count": args.max_new_tokens,
            "actual_generated_new_token_count": actual_new_tokens,
            "generated_text": text,
            "sampling": {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "seed": args.seed,
            },
        },
    )
    print(text)


if __name__ == "__main__":
    main()
