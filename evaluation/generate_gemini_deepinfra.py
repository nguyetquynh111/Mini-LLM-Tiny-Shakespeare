"""Generate Gemini Flash samples through DeepInfra."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_llm.utils import EVALUATION_DIR, GENERATION_DIR, ensure_output_dirs


DEFAULT_PROMPTS_PATH = EVALUATION_DIR / "prompts.txt"
DEFAULT_OUTPUT_PATH = GENERATION_DIR / "gemini_flash.jsonl"
DEFAULT_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEFAULT_MODEL = "google/gemini-3.5-flash"
MAX_NEW_TOKENS = 150
EMPTY_RESPONSE_RETRY_MULTIPLIER = 4
TOKENIZATION_NOTE = (
    "Gemini tokenization is not byte-level and is not directly comparable to the local "
    "byte-level models; this is a qualitative comparison only."
)


class EmptyAssistantTextError(RuntimeError):
    """Raised when DeepInfra returns a choice without assistant text."""


def content_to_text(content: Any) -> str:
    """Normalize OpenAI-compatible message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                text = content_to_text(content[key])
                if text:
                    return text
    return ""


def extract_chat_text(data: Dict[str, Any]) -> str:
    """Extract assistant text or raise a useful error for empty responses."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"DeepInfra response did not include choices: {json.dumps(data)[:1200]}")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"DeepInfra returned an invalid choice: {json.dumps(choice)[:1200]}")

    message = choice.get("message")
    text = ""
    if isinstance(message, dict):
        text = content_to_text(message.get("content")).strip()
    if not text:
        text = content_to_text(choice.get("text")).strip()
    if text:
        return text

    finish_reason = choice.get("finish_reason", "unknown")
    response_summary = json.dumps(
        {
            "finish_reason": finish_reason,
            "message": message,
            "choice_keys": sorted(choice.keys()),
        },
        ensure_ascii=True,
    )
    raise EmptyAssistantTextError(f"DeepInfra returned no assistant text: {response_summary[:1200]}")


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
        raise RuntimeError("Missing API key. Set DEEPINFRA_API_KEY in the environment or .env.")
    return api_key


def build_retry_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ask the model to answer directly when the first response used all tokens."""
    retry_messages = [dict(message) for message in messages]
    retry_instruction = (
        "Answer directly with final text only. Do not include hidden reasoning, analysis, "
        "or an explanation before the requested text."
    )
    if retry_messages and retry_messages[0].get("role") == "system":
        retry_messages[0]["content"] = f"{retry_messages[0].get('content', '')} {retry_instruction}"
    else:
        retry_messages.insert(0, {"role": "system", "content": retry_instruction})
    return retry_messages


def post_chat_completion(
    messages: List[Dict[str, str]], api_key: str, model: str, api_url: str, max_tokens: int
) -> Dict[str, Any]:
    """Post one DeepInfra OpenAI-compatible chat completions request."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "reasoning": {"enabled": False},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    return response.json()


def chat_completion(messages: List[Dict[str, str]], api_key: str, model: str, api_url: str, max_tokens: int) -> str:
    """Call DeepInfra and retry once if the provider returns an empty final message."""
    data = post_chat_completion(messages, api_key, model, api_url, max_tokens)
    try:
        return extract_chat_text(data)
    except EmptyAssistantTextError:
        retry_tokens = max_tokens * EMPTY_RESPONSE_RETRY_MULTIPLIER
        retry_messages = build_retry_messages(messages)
        retry_data = post_chat_completion(retry_messages, api_key, model, api_url, retry_tokens)
        return extract_chat_text(retry_data)


def approximate_whitespace_token_count(text: str) -> int:
    """Return a rough whitespace-token count for qualitative comparison."""
    return len(text.split())


def generate_gemini_samples(prompts: List[str], api_key: str, model: str, api_url: str) -> List[Dict[str, Any]]:
    """Generate one Gemini continuation for every prompt."""
    samples: List[Dict[str, Any]] = []
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
                    "Continue this prompt with roughly 150 tokens. "
                    f"Prompt: {prompt}"
                ),
            },
        ]
        output = chat_completion(messages, api_key, model, api_url, MAX_NEW_TOKENS)
        samples.append(
            {
                "prompt": prompt,
                "requested_token_count": MAX_NEW_TOKENS,
                "returned_text": output,
                "approximate_whitespace_token_count": approximate_whitespace_token_count(output),
                "tokenization_note": TOKENIZATION_NOTE,
                "provider_model": model,
            }
        )
        print(f"Generated Gemini sample for prompt: {prompt}")
    return samples


def write_gemini_outputs(path: Path, samples: List[Dict[str, Any]], model: str) -> None:
    """Write Gemini generations as structured JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            record = {**sample, "provider_model": model}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved Gemini generations to {path}")


def parse_args() -> argparse.Namespace:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Generate Gemini samples through DeepInfra.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", type=str, default=os.getenv("DEEPINFRA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api_url", type=str, default=os.getenv("DEEPINFRA_API_URL", DEFAULT_API_URL))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_output_dirs()
        api_key = get_api_key()
        prompts = read_prompts(args.prompts)
        samples = generate_gemini_samples(prompts, api_key, args.model, args.api_url)
        write_gemini_outputs(args.output, samples, args.model)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            print("DeepInfra rate limit reached. Wait and retry later.", file=sys.stderr)
        else:
            print(f"DeepInfra API request failed: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        raise SystemExit(1) from exc
    except requests.RequestException as exc:
        print(f"DeepInfra API request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
