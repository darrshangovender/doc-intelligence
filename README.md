# Document-Intelligence Pipeline

> Automated extraction pipeline for a client receiving structured data from semi-structured PDFs (invoices and statements). OCR → LLM-based field extraction with schema validation → Postgres write → exception queue for human review. Replaced manual data entry on a recurring high-volume workflow.

**Stack:** Python · OpenAI · Tesseract / pdfplumber · Pydantic · PostgreSQL · FastAPI · Docker

---

## The problem

The client received hundreds of supplier invoices and bank statements per week as PDFs. Three formats from each supplier, and they'd quietly change the layout every few months. A team of two was keying these into the accounting system by hand.

A pure OCR + regex approach was tried and abandoned: every layout change broke it. A pure LLM approach was prone to hallucinating values that weren't on the page. The pipeline below is the design that survived.

## How it works

1. **Pre-process.** PDF page → text via `pdfplumber` (for digital PDFs) with Tesseract OCR fallback (for scans).
2. **Type-classify.** A small first-pass LLM call decides "is this an invoice, a bank statement, or unknown?"
3. **Extract.** Type-specific prompt with a strict Pydantic schema. The model is instructed to output JSON only and to set fields to `null` rather than guess.
4. **Validate.** Pydantic parses + validates: required fields, types, plausibility checks (e.g. line-item totals must sum to the stated total within R0.01).
5. **Reconcile.** Documents that fail validation, or whose extracted total deviates from the OCR-extracted "total" line, go to an **exception queue** for human review rather than into Postgres.
6. **Persist.** Clean rows go into Postgres tables tied back to the original PDF (hash + page number) for audit.

## Why an exception queue is the whole game

The thing that kills doc-extraction pipelines is silent errors — a value extracted slightly wrong that nobody catches until the books don't reconcile six weeks later.

So the policy is: **be willing to fail loudly and often.** It's cheaper for the bookkeeper to clear 30 exceptions a day than for the auditor to find one wrong invoice three months later. Exception triage is the cost of correctness.

The exception queue surfaces:
- The original PDF page (rendered as image)
- The extracted JSON
- The validation failure(s)
- A "fix and approve" form that writes the corrected record back to the training set

That last bit matters: every human correction becomes a few-shot example for the prompt. The system gets *less* exception-prone over time.

## Architecture

```
PDF in (S3) ──▶ pdfplumber / Tesseract
              ──▶ type classifier (LLM)
              ──▶ extractor (LLM with Pydantic schema)
              ──▶ validator
                    ├─ pass ──▶ Postgres
                    └─ fail ──▶ exception queue
                                  └─▶ human review
                                       └─▶ correction → few-shot store
```

## Repo structure

```
.
├── ingest/
│   ├── pdf_to_text.py
│   └── classifier.py
├── extract/
│   ├── prompts/
│   │   ├── invoice.md
│   │   └── statement.md
│   ├── schemas.py        # Pydantic models
│   └── extractor.py
├── validate/
│   └── checks.py
├── api/
│   └── exceptions.py     # FastAPI for the review UI
├── web/
│   └── (Next.js review UI)
└── eval/
    └── extraction_accuracy.py
```

## Schema example

```python
class Invoice(BaseModel):
    invoice_number: str
    invoice_date: date
    supplier_name: str
    supplier_vat_number: str | None
    line_items: list[LineItem]
    subtotal_excl_vat: Decimal
    vat_amount: Decimal
    total_incl_vat: Decimal

    @model_validator(mode="after")
    def totals_must_reconcile(self):
        expected = sum(li.total for li in self.line_items)
        if abs(expected - self.subtotal_excl_vat) > Decimal("0.01"):
            raise ValueError("Line items do not reconcile to subtotal")
        return self
```

That `model_validator` is the line of code that catches more silent extraction errors than any prompt change.

## Results

- Replaced manual data entry on a recurring high-volume workflow.
- Exception rate trended down month-on-month as the few-shot store grew.
- Auditable: every Postgres row links back to the source PDF hash + page.

## Local setup

```bash
docker compose up -d
uv sync
uv run python ingest/pdf_to_text.py --in ./samples
uv run uvicorn api.main:app --reload
cd web && pnpm dev
```

## Author

Darrshan Govender · Founder, [Agulhas Code](https://agulhascode.co.za)
