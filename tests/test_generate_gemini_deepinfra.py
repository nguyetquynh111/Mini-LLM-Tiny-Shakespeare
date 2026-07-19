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


def test_extract_chat_result_reads_provider_metadata():
    data = {
        "model": "provider/model",
        "choices": [{"finish_reason": "stop", "message": {"content": "A noble answer."}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }

    result = gemini.extract_chat_result(data)

    assert result.returned_text == "A noble answer."
    assert result.provider_model == "provider/model"
    assert result.completion_tokens == 4
    assert result.finish_reason == "stop"


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
            return {
                "model": "provider/model",
                "choices": [{"finish_reason": "stop", "message": {"content": "A noble answer."}}],
                "usage": {"completion_tokens": 4},
            }

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
                    "model": "provider/model",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"role": "assistant", "content": None, "reasoning_content": None},
                        }
                    ],
                }
            return {
                "model": "provider/model",
                "choices": [{"finish_reason": "stop", "message": {"content": "A retried answer."}}],
                "usage": {"completion_tokens": 4},
            }

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
    assert calls[1]["max_tokens"] == 150
    assert calls[1]["messages"][0]["role"] == "system"
    assert "Answer directly" in calls[1]["messages"][0]["content"]


def test_generate_gemini_samples_calls_once_per_prompt(monkeypatch):
    calls = []

    def fake_chat_completion_result(messages, api_key, model, api_url, max_tokens):
        calls.append(
            {
                "messages": messages,
                "api_key": api_key,
                "model": model,
                "api_url": api_url,
                "max_tokens": max_tokens,
            }
        )
        return gemini.ChatCompletionResult(
            returned_text=f"continuation {len(calls)}",
            completion_tokens=10 + len(calls),
            finish_reason="stop",
            provider_model=model,
        )

    monkeypatch.setattr(gemini, "chat_completion_result", fake_chat_completion_result)

    samples = gemini.generate_gemini_samples(
        ["To be, or not to", "O Romeo, Romeo"],
        api_key="test-key",
        model="provider/model",
        api_url="https://example.test/chat/completions",
    )

    assert samples[0]["prompt"] == "To be, or not to"
    assert samples[0]["provider"] == gemini.PROVIDER
    assert samples[0]["provider_model"] == "provider/model"
    assert samples[0]["requested_completion_tokens"] == gemini.MAX_NEW_TOKENS
    assert samples[0]["completion_tokens"] == 11
    assert samples[0]["finish_reason"] == "stop"
    assert samples[0]["generated_at_utc"].endswith("Z")
    assert samples[0]["returned_text"] == "continuation 1"
    assert "not directly comparable" in samples[0]["tokenization_note"]
    assert samples[1]["returned_text"] == "continuation 2"
    assert len(calls) == 2
    assert all(call["api_key"] == "test-key" for call in calls)
    assert all(call["max_tokens"] == gemini.MAX_NEW_TOKENS for call in calls)
    assert "Return only the continuation text" in calls[0]["messages"][0]["content"]
    assert "requested 150 provider-token limit" in calls[0]["messages"][1]["content"]
    assert "Prompt: To be, or not to" in calls[0]["messages"][1]["content"]


def test_write_gemini_outputs_writes_structured_jsonl(tmp_path):
    output_path = tmp_path / "gemini_flash.jsonl"

    gemini.write_gemini_outputs(
        output_path,
        [
            {
                "prompt": "To be, or not to",
                "provider": gemini.PROVIDER,
                "provider_model": "provider/model",
                "requested_completion_tokens": 150,
                "completion_tokens": 4,
                "finish_reason": "stop",
                "generated_at_utc": "2026-07-18T00:00:00Z",
                "returned_text": "Whether 'tis nobler still.",
                "tokenization_note": gemini.TOKENIZATION_NOTE,
            }
        ],
        model="provider/model",
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["provider_model"] == "provider/model"
    assert record["prompt"] == "To be, or not to"
    assert record["provider"] == gemini.PROVIDER
    assert record["requested_completion_tokens"] == 150
    assert record["completion_tokens"] == 4
    assert record["finish_reason"] == "stop"
    assert record["generated_at_utc"] == "2026-07-18T00:00:00Z"
    assert record["returned_text"] == "Whether 'tis nobler still."
    assert "not byte-level" in record["tokenization_note"]


def test_generate_gemini_samples_warns_when_completion_tokens_unavailable(monkeypatch):
    def fake_chat_completion_result(messages, api_key, model, api_url, max_tokens):
        return gemini.ChatCompletionResult(
            returned_text="A noble answer.",
            completion_tokens=None,
            finish_reason="stop",
            provider_model=model,
        )

    monkeypatch.setattr(gemini, "chat_completion_result", fake_chat_completion_result)

    with pytest.warns(RuntimeWarning, match="usage.completion_tokens"):
        samples = gemini.generate_gemini_samples(
            ["To be, or not to"],
            api_key="test-key",
            model="provider/model",
            api_url="https://example.test/chat/completions",
        )

    assert samples[0]["completion_tokens"] is None
