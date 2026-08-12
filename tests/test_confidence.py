"""Confidence-scoring unit tests — heuristic and combined."""

from __future__ import annotations

from doc_intelligence.confidence import (
    LLM_CONFIDENCE_CAP,
    combine_scores,
    heuristic_score,
    needs_review,
    overall_confidence,
)


def test_value_present_in_source_is_high_confidence():
    score = heuristic_score("invoice_no", "INV-1138", "...invoice number INV-1138 dated...")
    assert score >= 0.9


def test_value_absent_from_source_lowers_score():
    score = heuristic_score("invoice_no", "INV-9999", "no such invoice here")
    # Numeric-looking value not found → low
    assert score < 0.7


def test_null_field_is_neutral_not_zero():
    # Treating null as 0 would flood the queue with "I don't know" fields.
    assert heuristic_score("due_date", None, "any text") == 0.5


def test_list_field_partial_match():
    items = [
        {"description": "Stainless mixing bowl"},
        {"description": "Whisk attachment"},
        {"description": "Phantom item not present"},
    ]
    src = "Stainless mixing bowl ... Whisk attachment ..."
    score = heuristic_score("line_items", items, src)
    # 2/3 hits → ~0.67
    assert 0.55 <= score <= 0.75


def test_llm_self_confidence_is_capped():
    """A model returning 1.0 shouldn't dominate — heuristic still moderates."""
    llm_scores = {"vendor": 1.0}  # over-confident
    extracted = {"vendor": "Totally Made Up Co"}
    source = "completely different text"
    combined = combine_scores(llm_scores, extracted, source)
    # 0.4 * 0.95 (capped) + 0.6 * heuristic(~0.4) ≈ 0.62
    assert combined["vendor"] < LLM_CONFIDENCE_CAP


def test_combine_strips_underscore_fields():
    combined = combine_scores(
        {"vendor": 0.9}, {"vendor": "ACME", "_confidence": {"x": 1.0}}, "ACME invoice"
    )
    assert "_confidence" not in combined
    assert "vendor" in combined


def test_overall_confidence_is_min():
    assert overall_confidence({"a": 0.9, "b": 0.5, "c": 0.8}) == 0.5


def test_overall_confidence_empty():
    assert overall_confidence({}) == 0.0


def test_needs_review_below_threshold():
    assert needs_review({"a": 0.6}, threshold=0.75) is True
    assert needs_review({"a": 0.9, "b": 0.8}, threshold=0.75) is False


def test_combine_scores_handles_missing_llm_scores():
    """Missing LLM scores should default to a reasonable middle value, not crash."""
    combined = combine_scores(None, {"vendor": "ACME"}, "ACME widgets")
    assert "vendor" in combined
    assert 0 <= combined["vendor"] <= 1
