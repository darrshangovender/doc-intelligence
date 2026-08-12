"""High-level ``Extractor`` façade — picks the right extractor by doc_type and
optionally routes results into the review queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from doc_intelligence import ocr
from doc_intelligence.extractors.base import BaseExtractor, ExtractionResult, ExtractionStatus
from doc_intelligence.extractors.contract import ContractExtractor
from doc_intelligence.extractors.invoice import InvoiceExtractor
from doc_intelligence.extractors.receipt import ReceiptExtractor
from doc_intelligence.llm_client import LLMClient, get_default_client
from doc_intelligence.review_queue import ReviewQueue


EXTRACTOR_REGISTRY: dict[str, type[BaseExtractor]] = {
    "invoice": InvoiceExtractor,
    "receipt": ReceiptExtractor,
    "contract": ContractExtractor,
}


class Extractor:
    """High-level façade.

    Usage::

        ext = Extractor(llm=StubLLM(...), queue=ReviewQueue("queue.db"))
        result = ext.run("invoice.pdf", doc_type="invoice")
        if result.status == ExtractionStatus.AUTO_APPROVED:
            persist(result.data)

    If ``queue`` is supplied, low-confidence / failed results are auto-enqueued.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        queue: ReviewQueue | None = None,
        provider: str = "anthropic",
        review_threshold: float = 0.75,
    ) -> None:
        self.llm = llm or get_default_client(provider=provider)
        self.queue = queue
        self.review_threshold = review_threshold

    def _build_extractor(self, doc_type: str) -> BaseExtractor:
        try:
            cls = EXTRACTOR_REGISTRY[doc_type]
        except KeyError as exc:
            known = ", ".join(EXTRACTOR_REGISTRY)
            raise ValueError(f"Unknown doc_type {doc_type!r}. Known: {known}") from exc
        return cls(llm=self.llm, review_threshold=self.review_threshold)

    def run(
        self,
        source: Union[str, Path],
        *,
        doc_type: str,
        enqueue: bool = True,
    ) -> ExtractionResult:
        """End-to-end: load → extract → validate → score → optionally enqueue."""
        text = ocr.read(source)
        extractor = self._build_extractor(doc_type)
        result = extractor.extract(text)

        if enqueue and self.queue is not None and result.status != ExtractionStatus.AUTO_APPROVED:
            source_path = str(source) if isinstance(source, (str, Path)) else None
            self.queue.add(result, source_path=source_path)

        return result

    @staticmethod
    def known_doc_types() -> list[str]:
        return sorted(EXTRACTOR_REGISTRY)
