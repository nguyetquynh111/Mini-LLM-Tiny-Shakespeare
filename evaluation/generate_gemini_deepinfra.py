"""Generate Gemini Flash samples through DeepInfra and update comparison notes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = REPO_ROOT / "evaluation"
DEFAULT_PROMPTS_PATH = EVALUATION_DIR / "prompts.txt"
DEFAULT_OUTPUT_PATH = EVALUATION_DIR / "generations_gemini.txt"
DEFAULT_COMPARISON_PATH = EVALUATION_DIR / "comparison_table.md"
DEFAULT_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEFAULT_MODEL = "google/gemini-3.5-flash"
MAX_NEW_TOKENS = 150


def read_prompts(path: Path) -> List[str]:
    """Read non-empty prompt lines."""
    if not path.exists():
        raise FileNotFoundError(f"Missing prompts file: {path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"Prompts file is empty: {path}")
    return prompts


def get_api_key() -> str:
    """Load the DeepInfra API key from environment variables."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("DEEPINFRA_API_KEY") or os.getenv("DEEPINFRA_TOKEN")
    if not api_key:
        raise RuntimeError("Set DEEPINFRA_API_KEY in .env before running this script.")
    return api_key


def chat_completion(messages: List[Dict[str, str]], api_key: str, model: str, api_url: str, max_tokens: int) -> str:
    """Call the DeepInfra OpenAI-compatible chat completions endpoint."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_gemini_samples(prompts: List[str], api_key: str, model: str, api_url: str) -> List[Dict[str, str]]:
    """Generate one Gemini continuation for every prompt."""
    samples: List[Dict[str, str]] = []
    system_prompt = (
        "You are generating text for a Tiny Shakespeare language-model comparison. "
        "Continue the user prompt in a Shakespeare-inspired dramatic style. "
        "Return only the continuation text, with no explanation."
    )
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Continue this prompt with exactly 150 new tokens if possible. "
                    f"Prompt: {prompt}"
                ),
            },
        ]
        output = chat_completion(messages, api_key, model, api_url, MAX_NEW_TOKENS)
        samples.append({"prompt": prompt, "output": output})
        print(f"Generated Gemini sample for prompt: {prompt}")
    return samples


def write_gemini_outputs(path: Path, samples: List[Dict[str, str]], model: str) -> None:
    """Write Gemini generations in the same readable format as local samples."""
    lines = [
        "Gemini Flash generations",
        f"Provider model: {model}",
        f"Each sample requests {MAX_NEW_TOKENS} new generated tokens.",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                "=" * 80,
                f"Prompt: {sample['prompt']}",
                "-" * 80,
                sample["output"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Gemini generations to {path}")


def read_text_or_placeholder(path: Path) -> str:
    """Read a text file or return a short placeholder if it is missing."""
    if not path.exists():
        return f"Missing file: {path.name}"
    return path.read_text(encoding="utf-8")


def update_comparison_table(api_key: str, model: str, api_url: str, comparison_path: Path) -> None:
    """Ask Gemini to update the qualitative comparison table from saved samples."""
    model_a_text = read_text_or_placeholder(EVALUATION_DIR / "generations_model_a.txt")
    model_b_text = read_text_or_placeholder(EVALUATION_DIR / "generations_model_b.txt")
    gemini_text = read_text_or_placeholder(EVALUATION_DIR / "generations_gemini.txt")

    messages = [
        {
            "role": "system",
            "content": (
                "You write concise, honest project evaluation notes in English. "
                "Return only a Markdown table with these columns: Model, Structural stability, "
                "Shakespearean style accuracy, Degenerative repetition loops, Readability, Overall analysis. "
                "Do not invent strong local-model performance if the samples are placeholders or fractured."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a qualitative comparison table for Custom Model A, Custom Model B, and Gemini Flash. "
                "Use the following saved generation files as evidence.\n\n"
                f"Custom Model A samples:\n{model_a_text[:4000]}\n\n"
                f"Custom Model B samples:\n{model_b_text[:4000]}\n\n"
                f"Gemini Flash samples:\n{gemini_text[:4000]}"
            ),
        },
    ]
    table = chat_completion(messages, api_key, model, api_url, max_tokens=900)
    comparison_path.write_text("# Qualitative Generation Comparison\n\n" + table + "\n", encoding="utf-8")
    print(f"Updated comparison table at {comparison_path}")


def parse_args() -> argparse.Namespace:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Generate Gemini samples through DeepInfra.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--model", type=str, default=os.getenv("DEEPINFRA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api_url", type=str, default=os.getenv("DEEPINFRA_API_URL", DEFAULT_API_URL))
    parser.add_argument("--skip_comparison", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        api_key = get_api_key()
        prompts = read_prompts(args.prompts)
        samples = generate_gemini_samples(prompts, api_key, args.model, args.api_url)
        write_gemini_outputs(args.output, samples, args.model)
        if not args.skip_comparison:
            update_comparison_table(api_key, args.model, args.api_url, args.comparison)
    except requests.HTTPError as exc:
        print(f"DeepInfra request failed: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
