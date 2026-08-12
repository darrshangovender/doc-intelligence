"""doc-intelligence: OCR + LLM document extraction with a human-review queue.

Public surface:

* :class:`Extractor` — high-level façade that picks an extractor by document type,
  runs OCR/PDF parsing, calls the LLM, validates the result with a Pydantic schema,
  scores per-field confidence, and routes low-confidence results to the review queue.
* :class:`ReviewQueue` — SQLite-backed queue for human triage.
* :class:`ExtractionResult` — extraction container (data, confidence, raw text, status).
"""

from doc_intelligence.extractors.base import (
    BaseExtractor,
    ExtractionResult,
    ExtractionStatus,
)
from doc_intelligence.extractors.contract import ContractExtractor, ContractMeta
from doc_intelligence.extractors.invoice import InvoiceData, InvoiceExtractor, LineItem
from doc_intelligence.extractors.receipt import ReceiptData, ReceiptExtractor, ReceiptLine
from doc_intelligence.review_queue import ReviewQueue, ReviewRecord
from doc_intelligence.facade import Extractor

__all__ = [
    "Extractor",
    "BaseExtractor",
    "ExtractionResult",
    "ExtractionStatus",
    "ReviewQueue",
    "ReviewRecord",
    "InvoiceExtractor",
    "InvoiceData",
    "LineItem",
    "ReceiptExtractor",
    "ReceiptData",
    "ReceiptLine",
    "ContractExtractor",
    "ContractMeta",
]

__version__ = "0.2.0"
