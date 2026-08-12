"""Contract metadata extractor + Pydantic schema."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from doc_intelligence.extractors.base import BaseExtractor


class ContractMeta(BaseModel):
    parties: list[str] = Field(min_length=1, description="Legal names of contracting parties")
    effective_date: date | None = None
    term: str | None = Field(default=None, description="e.g. '12 months', 'indefinite'")
    renewal_notice_days: int | None = Field(default=None, ge=0, le=365)
    governing_law: str | None = None
    contract_type: str | None = Field(
        default=None, description="e.g. NDA, MSA, SOW, employment agreement"
    )

    @field_validator("parties")
    @classmethod
    def strip_each_party(cls, v: list[str]) -> list[str]:
        cleaned = [p.strip() for p in v if p and p.strip()]
        if not cleaned:
            raise ValueError("at least one party required")
        return cleaned


class ContractExtractor(BaseExtractor):
    schema: ClassVar[type[BaseModel]] = ContractMeta
    doc_type: ClassVar[str] = "contract"
    prompt_intro: ClassVar[str] = (
        "You are extracting metadata from a legal CONTRACT.\n"
        "Identify: the contracting parties (usually in the preamble — 'between X and Y'), "
        "the effective date, the term (length of the agreement), the renewal notice period in "
        "days if specified, the governing law jurisdiction, and the contract type (NDA, MSA, "
        "SOW, employment, etc.). Set fields to null if not stated."
    )
