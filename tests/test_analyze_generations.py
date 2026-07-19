import math

import pytest

from evaluation.analyze_generations import (
    ANALYSIS_BYTES,
    analyze_records,
    continuation_bytes,
    duplicate_ngram_rate,
    longest_corpus_match,
)


def test_duplicate_ngram_rate_detects_repeated_occurrences():
    assert duplicate_ngram_rate(b"abcd", 4) == 0.0
    assert math.isclose(duplicate_ngram_rate(b"aaaaa", 2), 0.75)


def test_longest_corpus_match_finds_exact_span():
    assert longest_corpus_match(b"zzcdefyy", b"abcdefgh") == 4


def test_local_record_requires_exactly_150_generated_tokens():
    record = {
        "actual_generated_new_token_count": ANALYSIS_BYTES,
        "generated_token_ids": [65] * ANALYSIS_BYTES,
    }
    assert continuation_bytes(record, "model_a") == b"A" * ANALYSIS_BYTES

    record["actual_generated_new_token_count"] = 149
    with pytest.raises(ValueError, match="exactly 150"):
        continuation_bytes(record, "model_a")


def test_analysis_is_normalized_to_first_150_bytes():
    local = {
        "prompt": "P",
        "actual_generated_new_token_count": ANALYSIS_BYTES,
        "generated_token_ids": list(range(ANALYSIS_BYTES)),
    }
    gemini = {"prompt": "P", "returned_text": "x" * 500}
    rows = analyze_records(
        {"model_a": [local], "model_b": [local], "gemini_flash": [gemini]},
        training_corpus=b"abcdefghijklmnopqrstuvwxyz" * 20,
    )

    assert len(rows) == 3
    assert all(row["analysis_bytes"] == ANALYSIS_BYTES for row in rows)
    assert rows[-1]["source_output_bytes"] == 500
