"""Per-document-type extractors. Each defines a Pydantic schema + prompt."""

from doc_intelligence.extractors.base import BaseExtractor, ExtractionResult, ExtractionStatus
from doc_intelligence.extractors.contract import ContractExtractor, ContractMeta
from doc_intelligence.extractors.invoice import InvoiceData, InvoiceExtractor, LineItem
from doc_intelligence.extractors.receipt import ReceiptData, ReceiptExtractor, ReceiptLine

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ExtractionStatus",
    "InvoiceExtractor",
    "InvoiceData",
    "LineItem",
    "ReceiptExtractor",
    "ReceiptData",
    "ReceiptLine",
    "ContractExtractor",
    "ContractMeta",
]
