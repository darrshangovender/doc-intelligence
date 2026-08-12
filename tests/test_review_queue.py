"""Review-queue lifecycle tests."""

from __future__ import annotations

import json

import pytest

from doc_intelligence.extractors.base import ExtractionResult, ExtractionStatus
from doc_intelligence.extractors.invoice import InvoiceExtractor
from doc_intelligence.llm_client import StubLLM
from doc_intelligence.review_queue import ReviewQueue


@pytest.fixture
def queue(tmp_path):
    return ReviewQueue(tmp_path / "test.db")


def _make_result(status: ExtractionStatus = ExtractionStatus.NEEDS_REVIEW) -> ExtractionResult:
    return ExtractionResult(
        doc_type="invoice",
        status=status,
        data={"vendor": "ACME", "invoice_no": "X1", "total": "100"},
        confidence={"vendor": 0.6, "invoice_no": 0.7, "total": 0.9},
        source_text="ACME ... invoice X1 ... total 100",
        errors=[],
    )


def test_add_and_get(queue):
    rid = queue.add(_make_result(), source_path="/tmp/x.pdf")
    rec = queue.get(rid)
    assert rec is not None
    assert rec.doc_type == "invoice"
    assert rec.status == "pending"
    assert rec.data["vendor"] == "ACME"
    assert rec.source_path == "/tmp/x.pdf"


def test_list_pending_returns_only_pending(queue):
    rid_a = queue.add(_make_result())
    rid_b = queue.add(_make_result())
    queue.approve(rid_a)

    pending = queue.list(status="pending")
    assert {r.id for r in pending} == {rid_b}

    approved = queue.list(status="approved")
    assert {r.id for r in approved} == {rid_a}


def test_approve_lifecycle(queue):
    rid = queue.add(_make_result())
    assert queue.pending_count() == 1

    queue.approve(rid, reviewer="darrshan", notes="looks fine")

    assert queue.pending_count() == 0
    rec = queue.get(rid)
    assert rec.status == "approved"
    assert rec.reviewer == "darrshan"
    assert rec.review_notes == "looks fine"


def test_approve_with_corrected_data(queue):
    """Reviewer can override the extracted data when approving."""
    rid = queue.add(_make_result())
    queue.approve(rid, corrected_data={"vendor": "FIXED", "invoice_no": "X1", "total": "100"})
    rec = queue.get(rid)
    assert rec.data["vendor"] == "FIXED"


def test_reject_lifecycle(queue):
    rid = queue.add(_make_result())
    queue.reject(rid, reviewer="darrshan", notes="bogus extract")
    rec = queue.get(rid)
    assert rec.status == "rejected"
    assert rec.review_notes == "bogus extract"


def test_auto_approved_results_cannot_be_enqueued(queue):
    with pytest.raises(ValueError, match="Auto-approved"):
        queue.add(_make_result(ExtractionStatus.AUTO_APPROVED))


def test_approve_unknown_id_raises(queue):
    with pytest.raises(KeyError):
        queue.approve("does-not-exist")


def test_filter_by_doc_type(queue):
    rid_inv = queue.add(_make_result())
    contract_result = _make_result()
    contract_result.doc_type = "contract"
    rid_con = queue.add(contract_result)

    inv_items = queue.list(doc_type="invoice")
    assert {r.id for r in inv_items} == {rid_inv}
    con_items = queue.list(doc_type="contract")
    assert {r.id for r in con_items} == {rid_con}


def test_full_extract_to_queue_pipeline(queue):
    """End-to-end: a low-confidence extraction enqueues itself via the façade."""
    # Force low confidence by returning data that doesn't appear in source text.
    payload = {
        "vendor": "Phantom Co",
        "invoice_no": "X-Y-Z-9999",
        "invoice_date": "2025-01-01",
        "line_items": [],
        "total": "12345.67",
        "_confidence": {"vendor": 0.5, "invoice_no": 0.5, "total": 0.5},
    }
    llm = StubLLM(default=json.dumps(payload))
    result = InvoiceExtractor(llm=llm).extract("totally unrelated source text")

    assert result.status == ExtractionStatus.NEEDS_REVIEW
    rid = queue.add(result)
    rec = queue.get(rid)
    assert rec.overall_confidence < 0.75
