"""Per-field confidence scoring.

Two signals are combined:

1. **LLM self-reported confidence**: the extraction prompts instruct the model
   to return a ``_confidence`` map alongside the data. We trust the model here
   but cap it — a model returning 1.0 for everything is suspicious.
2. **Post-hoc heuristics**: we look for the extracted value as a substring of
   the source text. Found → boost; not found → penalise. We also penalise
   ``null`` fields and short-string vendor names.

The final per-field score is clamped to ``[0.0, 1.0]``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


# Anything below this triggers human review.
DEFAULT_REVIEW_THRESHOLD = 0.75

# Cap on raw LLM-reported confidence before combining with heuristics.
LLM_CONFIDENCE_CAP = 0.95


def _normalise(value: Any) -> str:
    """Convert values to a normalised string for substring search."""
    if value is None:
        return ""
    s = str(value)
    # Strip currency symbols, whitespace, common punctuation
    s = re.sub(r"[\s,]", "", s)
    return s.lower()


def heuristic_score(field_name: str, value: Any, source_text: str) -> float:
    """Heuristic confidence for a single field, independent of LLM signal."""
    if value is None or value == "" or value == []:
        # Null fields are explicit "don't know" — neutral, not zero.
        return 0.5

    norm_value = _normalise(value)
    norm_source = _normalise(source_text)

    # Numeric / date / id-like fields: exact substring presence is strong signal.
    if norm_value and norm_value in norm_source:
        return 0.95

    # Lists (line_items): score by proportion of items whose key tokens appear.
    if isinstance(value, list) and value:
        hits = 0
        for item in value:
            token = _normalise(
                item.get("description") if isinstance(item, dict) else item
            )
            if token and token[:20] in norm_source:
                hits += 1
        return max(0.4, hits / len(value))

    # Free-text fields (vendor, parties): partial match is acceptable.
    if isinstance(value, str) and len(value) >= 3:
        # Try first few significant tokens
        tokens = [t for t in re.split(r"\W+", value.lower()) if len(t) >= 3]
        if not tokens:
            return 0.5
        hits = sum(1 for t in tokens if t in source_text.lower())
        return 0.4 + 0.5 * (hits / len(tokens))

    return 0.5


def combine_scores(
    llm_scores: Mapping[str, float] | None,
    extracted: Mapping[str, Any],
    source_text: str,
) -> dict[str, float]:
    """Per-field combined confidence scores.

    Returns a dict keyed by field name with floats in [0, 1].
    """
    llm_scores = llm_scores or {}
    out: dict[str, float] = {}
    for field, value in extracted.items():
        if field.startswith("_"):
            continue
        llm = min(float(llm_scores.get(field, 0.7)), LLM_CONFIDENCE_CAP)
        heur = heuristic_score(field, value, source_text)
        # Weighted average — heuristics get 60% weight because they're grounded
        # in the source text, LLM gets 40%.
        out[field] = round(0.4 * llm + 0.6 * heur, 3)
    return out


def overall_confidence(scores: Mapping[str, float]) -> float:
    """Single scalar for routing decisions — minimum field score."""
    if not scores:
        return 0.0
    return min(scores.values())


def needs_review(
    scores: Mapping[str, float],
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> bool:
    """True if any field falls below the review threshold."""
    return overall_confidence(scores) < threshold
