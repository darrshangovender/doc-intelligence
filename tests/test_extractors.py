"""Golden-file tests: feed each sample doc through the matching extractor and
assert the extracted Pydantic model matches expectations."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from doc_intelligence.extractors.base import ExtractionStatus
from doc_intelligence.extractors.contract import ContractExtractor
from doc_intelligence.extractors.invoice import InvoiceExtractor
from doc_intelligence.extractors.receipt import ReceiptExtractor
from doc_intelligence.llm_client import StubLLM


# ---------------- Invoice ----------------

def test_invoice_extracts_known_fields(stub_llm, sample_text):
    text = sample_text("demo/sample_docs/invoice_001.txt")
    result = InvoiceExtractor(llm=stub_llm).extract(text)

    assert result.status == ExtractionStatus.AUTO_APPROVED
    assert result.data["vendor"] == "ACME WIDGETS PTY LTD"
    assert result.data["invoice_no"] == "INV-2025-00042"
    assert result.data["invoice_date"] == "2025-03-14"
    assert result.data["due_date"] == "2025-04-13"
    assert Decimal(result.data["total"]) == Decimal("994.75")
    assert len(result.data["line_items"]) == 3


def test_invoice_line_items_reconcile_to_total(stub_llm, sample_text):
    text = sample_text("demo/sample_docs/invoice_002.txt")
    result = InvoiceExtractor(llm=stub_llm).extract(text)
    assert result.status == ExtractionStatus.AUTO_APPROVED

    line_sum = sum(Decimal(li["line_total"]) for li in result.data["line_items"])
    assert line_sum + Decimal(result.data["tax"]) == Decimal(result.data["total"])


def test_invoice_rejects_when_totals_disagree():
    """Reconciliation validator catches a model that misread the total."""
    bad_payload = {
        "vendor": "Bad Co",
        "invoice_no": "X-1",
        "invoice_date": "2025-01-01",
        "line_items": [
            {"description": "A", "quantity": 1, "unit_price": "100", "line_total": "100"},
        ],
        "tax": "15",
        "total": "999.99",  # ← should be 115
    }
    llm = StubLLM(default=json.dumps(bad_payload))
    result = InvoiceExtractor(llm=llm).extract("doesn't matter")
    assert result.status == ExtractionStatus.FAILED
    assert any("reconcile" in e for e in result.errors)


# ---------------- Receipt ----------------

def test_receipt_extracts_known_fields(stub_llm, sample_text):
    text = sample_text("demo/sample_docs/receipt_001.txt")
    result = ReceiptExtractor(llm=stub_llm).extract(text)
    assert result.status == ExtractionStatus.AUTO_APPROVED
    assert result.data["merchant"] == "CHECKERS HYPER"
    assert result.data["total"] == "226.50"
    assert result.data["payment_method"] == "VISA"


def test_receipt_currency_optional(stub_llm, sample_text):
    text = sample_text("demo/sample_docs/receipt_003.txt")
    result = ReceiptExtractor(llm=stub_llm).extract(text)
    assert result.data["currency"] == "ZAR"


# ---------------- Contract ----------------

def test_contract_parses_parties_and_renewal(stub_llm, sample_text):
    text = sample_text("demo/sample_docs/contract_001.txt")
    result = ContractExtractor(llm=stub_llm).extract(text)
    assert result.status == ExtractionStatus.AUTO_APPROVED
    assert result.data["parties"] == ["Agulhas Code (Pty) Ltd", "Velocity Contact Hub (Pty) Ltd"]
    assert result.data["renewal_notice_days"] == 30
    assert result.data["contract_type"] == "NDA"


def test_contract_allows_null_renewal(stub_llm, sample_text):
    """SOW has no renewal clause — should still parse cleanly with None."""
    text = sample_text("demo/sample_docs/contract_003.txt")
    result = ContractExtractor(llm=stub_llm).extract(text)
    assert result.status == ExtractionStatus.AUTO_APPROVED
    assert result.data["renewal_notice_days"] is None


# ---------------- Error paths ----------------

def test_unparseable_json_returns_failed_status():
    llm = StubLLM(default="this is not JSON, just prose")
    result = InvoiceExtractor(llm=llm).extract("anything")
    assert result.status == ExtractionStatus.FAILED
    assert result.errors


def test_validation_error_returns_failed_status():
    # Missing required `vendor` and `invoice_no`
    llm = StubLLM(default=json.dumps({"total": "100"}))
    result = InvoiceExtractor(llm=llm).extract("anything")
    assert result.status == ExtractionStatus.FAILED


@pytest.mark.parametrize(
    "sample,extractor_cls,key",
    [
        ("demo/sample_docs/invoice_001.txt", InvoiceExtractor, "vendor"),
        ("demo/sample_docs/receipt_001.txt", ReceiptExtractor, "merchant"),
        ("demo/sample_docs/contract_001.txt", ContractExtractor, "parties"),
    ],
)
def test_every_extractor_returns_confidence_scores(stub_llm, sample_text, sample, extractor_cls, key):
    text = sample_text(sample)
    result = extractor_cls(llm=stub_llm).extract(text)
    assert key in result.confidence
    assert 0.0 <= result.confidence[key] <= 1.0
