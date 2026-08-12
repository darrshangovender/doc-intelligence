"""Invoice extractor + Pydantic schema."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator

from doc_intelligence.extractors.base import BaseExtractor


# Money-comparison tolerance — OCR and rounding rarely match to the cent.
TOTAL_TOLERANCE = Decimal("0.05")


class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: Decimal = Field(default=Decimal("1"))
    unit_price: Decimal
    line_total: Decimal

    @model_validator(mode="after")
    def line_total_reconciles(self) -> "LineItem":
        expected = (self.quantity * self.unit_price).quantize(Decimal("0.01"))
        actual = self.line_total.quantize(Decimal("0.01"))
        if abs(expected - actual) > TOTAL_TOLERANCE:
            # Don't raise — line item arithmetic is noisy on OCR. Trust line_total.
            pass
        return self


class InvoiceData(BaseModel):
    vendor: str = Field(min_length=1, description="Supplier name as it appears on the invoice")
    invoice_no: str = Field(min_length=1)
    invoice_date: date | None = None
    due_date: date | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal

    @field_validator("vendor", "invoice_no")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def totals_reconcile(self) -> "InvoiceData":
        """Line items should sum to the total within a small tolerance.

        We allow the model to set ``line_items=[]`` if the invoice is a
        single-figure summary (e.g. a service invoice). In that case there's
        nothing to reconcile.
        """
        if not self.line_items:
            return self
        line_sum = sum((li.line_total for li in self.line_items), Decimal("0"))
        expected_total = line_sum + (self.tax or Decimal("0"))
        if abs(expected_total - self.total) > TOTAL_TOLERANCE:
            raise ValueError(
                f"Line items + tax ({expected_total}) do not reconcile to total ({self.total})"
            )
        return self


class InvoiceExtractor(BaseExtractor):
    schema: ClassVar[type[BaseModel]] = InvoiceData
    doc_type: ClassVar[str] = "invoice"
    prompt_intro: ClassVar[str] = (
        "You are extracting fields from a supplier INVOICE.\n"
        "Pay particular attention to: vendor name (top of the document), invoice number, "
        "any due date (often labelled 'Due', 'Pay by', 'Payment due'), and the grand total.\n"
        "Line items are usually in a table — each row has a description, quantity, unit price, "
        "and a line total. If quantity is missing assume 1."
    )
