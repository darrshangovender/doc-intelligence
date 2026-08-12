"""Base extractor — shared LLM call, JSON parsing, Pydantic validation, confidence scoring.

Concrete extractors subclass :class:`BaseExtractor` and supply:

* a ``schema`` class attribute (a :class:`pydantic.BaseModel` subclass)
* ``doc_type`` string
* ``build_prompt(text)`` returning the user prompt

The base orchestrates the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from doc_intelligence.confidence import combine_scores, needs_review
from doc_intelligence.llm_client import LLMClient, parse_json_response


class ExtractionStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass
class ExtractionResult:
    """Container returned by every extractor."""

    doc_type: str
    status: ExtractionStatus
    data: dict[str, Any]
    confidence: dict[str, float]
    source_text: str
    errors: list[str] = field(default_factory=list)
    raw_llm_response: str | None = None

    @property
    def overall_confidence(self) -> float:
        if not self.confidence:
            return 0.0
        return min(self.confidence.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "status": self.status.value,
            "data": self.data,
            "confidence": self.confidence,
            "overall_confidence": self.overall_confidence,
            "errors": self.errors,
        }


SYSTEM_PROMPT = """You are a document data extraction engine. \
You read OCR'd text from invoices, receipts, contracts and similar business documents \
and emit ONE strict JSON object that matches the schema described in the user message.

Rules:
1. Output ONLY a single JSON object. No prose, no markdown fences.
2. If a field is not present on the document, set it to null. Do NOT guess.
3. All monetary amounts are decimal numbers (no currency symbols inside strings).
4. Dates must be ISO 8601 (YYYY-MM-DD).
5. Include a `_confidence` map with a float in [0,1] for every top-level field, \
indicating how sure you are that the value is correct given the source text.
"""


class BaseExtractor:
    """Shared extraction logic. Subclass and set ``schema``, ``doc_type``, ``prompt_intro``."""

    schema: ClassVar[type[BaseModel]]
    doc_type: ClassVar[str]
    prompt_intro: ClassVar[str] = ""

    def __init__(self, llm: LLMClient, review_threshold: float = 0.75) -> None:
        self.llm = llm
        self.review_threshold = review_threshold

    # --- subclasses override this if they need custom prompting ---
    def build_prompt(self, source_text: str) -> str:
        schema_json = self.schema.model_json_schema()
        return (
            f"{self.prompt_intro}\n\n"
            f"Extract fields matching this JSON schema:\n"
            f"```json\n{schema_json}\n```\n\n"
            f"--- DOCUMENT TEXT ---\n{source_text}\n--- END DOCUMENT ---\n\n"
            f"Return JSON only."
        )

    def extract(self, source_text: str) -> ExtractionResult:
        """Run the full extract → validate → score pipeline on already-loaded text."""
        prompt = self.build_prompt(source_text)
        raw = self.llm.complete(system=SYSTEM_PROMPT, user=prompt)

        try:
            payload = parse_json_response(raw)
        except (ValueError, Exception) as exc:  # noqa: BLE001
            return ExtractionResult(
                doc_type=self.doc_type,
                status=ExtractionStatus.FAILED,
                data={},
                confidence={},
                source_text=source_text,
                errors=[f"JSON parse failed: {exc}"],
                raw_llm_response=raw,
            )

        # Separate LLM-reported confidence from real fields
        llm_conf = payload.pop("_confidence", None) or {}

        try:
            validated = self.schema.model_validate(payload)
        except ValidationError as exc:
            return ExtractionResult(
                doc_type=self.doc_type,
                status=ExtractionStatus.FAILED,
                data=payload,
                confidence={},
                source_text=source_text,
                errors=[str(err) for err in exc.errors()],
                raw_llm_response=raw,
            )

        data = validated.model_dump(mode="json")
        scores = combine_scores(llm_conf, data, source_text)
        review = needs_review(scores, threshold=self.review_threshold)
        status = ExtractionStatus.NEEDS_REVIEW if review else ExtractionStatus.AUTO_APPROVED

        return ExtractionResult(
            doc_type=self.doc_type,
            status=status,
            data=data,
            confidence=scores,
            source_text=source_text,
            raw_llm_response=raw,
        )
