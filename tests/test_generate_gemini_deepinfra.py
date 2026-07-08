import json
import pytest

from evaluation import generate_gemini_deepinfra as gemini


def test_read_prompts_ignores_blank_lines(tmp_path):
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("\nTo be, or not to\n\nO Romeo, Romeo\n  \n", encoding="utf-8")

    assert gemini.read_prompts(prompts_path) == ["To be, or not to", "O Romeo, Romeo"]


def test_read_prompts_rejects_empty_file(tmp_path):
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("\n   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="Prompts file is empty"):
        gemini.read_prompts(prompts_path)


def test_extract_chat_text_from_openai_compatible_response():
    data = {"choices": [{"message": {"content": "  Henceforth, my lord.  "}}]}

    assert gemini.extract_chat_text(data) == "Henceforth, my lord."


def test_extract_chat_text_from_content_parts():
    data = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "First line."},
                        {"type": "text", "text": "Second line."},
                    ]
                }
            }
        ]
    }

    assert gemini.extract_chat_text(data) == "First line.\nSecond line."


def test_extract_chat_text_raises_for_empty_choices():
    with pytest.raises(RuntimeError, match="did not include choices"):
        gemini.extract_chat_text({"choices": []})


def test_extract_chat_text_raises_custom_error_for_empty_assistant_text():
    data = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": None, "reasoning_content": None},
            }
        ]
    }

    with pytest.raises(gemini.EmptyAssistantTextError, match="finish_reason"):
        gemini.extract_chat_text(data)


def test_chat_completion_posts_expected_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "A noble answer."}}]}

    def fake_post(api_url, headers, data, timeout):
        captured["api_url"] = api_url
        captured["headers"] = headers
        captured["payload"] = json.loads(data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(gemini.requests, "post", fake_post)

    output = gemini.chat_completion(
        [{"role": "user", "content": "Continue this."}],
        api_key="test-key",
        model="provider/model",
        api_url="https://example.test/chat/completions",
        max_tokens=150,
    )

    assert output == "A noble answer."
    assert captured["api_url"] == "https://example.test/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["timeout"] == 120
    assert captured["payload"] == {
        "model": "provider/model",
        "messages": [{"role": "user", "content": "Continue this."}],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 150,
        "reasoning_effort": "none",
        "reasoning": {"enabled": False},
    }


def test_chat_completion_retries_empty_assistant_text(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            if len(calls) == 1:
                return {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"role": "assistant", "content": None, "reasoning_content": None},
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "A retried answer."}}]}

    def fake_post(api_url, headers, data, timeout):
        payload = json.loads(data)
        calls.append(payload)
        return FakeResponse(payload)

    monkeypatch.setattr(gemini.requests, "post", fake_post)

    output = gemini.chat_completion(
        [{"role": "user", "content": "Continue this."}],
        api_key="test-key",
        model="provider/model",
        api_url="https://example.test/chat/completions",
        max_tokens=150,
    )

    assert output == "A retried answer."
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 150
    assert calls[1]["max_tokens"] == 150 * gemini.EMPTY_RESPONSE_RETRY_MULTIPLIER
    assert calls[1]["messages"][0]["role"] == "system"
    assert "Answer directly" in calls[1]["messages"][0]["content"]


def test_generate_gemini_samples_calls_once_per_prompt(monkeypatch):
    calls = []

    def fake_chat_completion(messages, api_key, model, api_url, max_tokens):
        calls.append(
            {
                "messages": messages,
                "api_key": api_key,
                "model": model,
                "api_url": api_url,
                "max_tokens": max_tokens,
            }
        )
        return f"continuation {len(calls)}"

    monkeypatch.setattr(gemini, "chat_completion", fake_chat_completion)

    samples = gemini.generate_gemini_samples(
        ["To be, or not to", "O Romeo, Romeo"],
        api_key="test-key",
        model="provider/model",
        api_url="https://example.test/chat/completions",
    )

    assert samples == [
        {"prompt": "To be, or not to", "output": "continuation 1"},
        {"prompt": "O Romeo, Romeo", "output": "continuation 2"},
    ]
    assert len(calls) == 2
    assert all(call["api_key"] == "test-key" for call in calls)
    assert all(call["max_tokens"] == gemini.MAX_NEW_TOKENS for call in calls)
    assert "Return only the continuation text" in calls[0]["messages"][0]["content"]
    assert "Prompt: To be, or not to" in calls[0]["messages"][1]["content"]


def test_write_gemini_outputs_matches_generation_file_format(tmp_path):
    output_path = tmp_path / "gemini_flash.txt"

    gemini.write_gemini_outputs(
        output_path,
        [{"prompt": "To be, or not to", "output": "Whether 'tis nobler still."}],
        model="provider/model",
    )

    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("Gemini Flash generations\nProvider model: provider/model\n")
    assert f"Each sample requests {gemini.MAX_NEW_TOKENS} new generated tokens." in text
    assert "Prompt: To be, or not to" in text
    assert "Whether 'tis nobler still." in text
