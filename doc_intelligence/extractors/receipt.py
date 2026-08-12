"""Receipt extractor + Pydantic schema."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from doc_intelligence.extractors.base import BaseExtractor


class ReceiptLine(BaseModel):
    description: str = Field(min_length=1)
    amount: Decimal


class ReceiptData(BaseModel):
    merchant: str = Field(min_length=1)
    purchase_date: date | None = None
    items: list[ReceiptLine] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal
    payment_method: str | None = None
    currency: str | None = Field(default=None, max_length=8)

    @field_validator("merchant")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class ReceiptExtractor(BaseExtractor):
    schema: ClassVar[type[BaseModel]] = ReceiptData
    doc_type: ClassVar[str] = "receipt"
    prompt_intro: ClassVar[str] = (
        "You are extracting fields from a retail RECEIPT.\n"
        "Look for: merchant name (usually top, often all caps), date of purchase, "
        "the list of purchased items (description + amount), tax, total, payment method "
        "(cash/credit/debit), and the currency code if discernible."
    )
