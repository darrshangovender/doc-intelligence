"""Canned LLM responses for the sample documents.

Used by both the eval harness and the unit tests so we can exercise the full
pipeline offline. In production these would come from a real LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path

from doc_intelligence.llm_client import StubLLM


# Each entry: substring of the document text → JSON payload string the model would emit.
# The keys are picked to be unique-enough to match by ``in`` substring search.
FIXTURE_RESPONSES: dict[str, dict] = {
    # ---------------- Invoices ----------------
    "INV-2025-00042": {
        "vendor": "ACME WIDGETS PTY LTD",
        "invoice_no": "INV-2025-00042",
        "invoice_date": "2025-03-14",
        "due_date": "2025-04-13",
        "line_items": [
            {
                "description": "Stainless mixing bowl",
                "quantity": 4,
                "unit_price": "120.00",
                "line_total": "480.00",
            },
            {
                "description": "Whisk attachment, large",
                "quantity": 2,
                "unit_price": "95.00",
                "line_total": "190.00",
            },
            {
                "description": "Silicone spatula set",
                "quantity": 3,
                "unit_price": "65.00",
                "line_total": "195.00",
            },
        ],
        "subtotal": "865.00",
        "tax": "129.75",
        "total": "994.75",
        "_confidence": {
            "vendor": 0.95,
            "invoice_no": 0.98,
            "invoice_date": 0.95,
            "due_date": 0.95,
            "line_items": 0.9,
            "subtotal": 0.95,
            "tax": 0.95,
            "total": 0.98,
        },
    },
    "BMC-7781": {
        "vendor": "BLUE MOUNTAIN COFFEE ROASTERS",
        "invoice_no": "BMC-7781",
        "invoice_date": "2025-05-02",
        "due_date": "2025-05-16",
        "line_items": [
            {
                "description": "Single-origin Ethiopia Yirgacheffe",
                "quantity": 5,
                "unit_price": "180.00",
                "line_total": "900.00",
            },
            {
                "description": "House blend espresso (1kg)",
                "quantity": 3,
                "unit_price": "220.00",
                "line_total": "660.00",
            },
            {
                "description": "Decaf Colombia",
                "quantity": 2,
                "unit_price": "240.00",
                "line_total": "480.00",
            },
        ],
        "subtotal": "2040.00",
        "tax": "306.00",
        "total": "2346.00",
        "_confidence": {
            "vendor": 0.95,
            "invoice_no": 0.98,
            "invoice_date": 0.95,
            "due_date": 0.95,
            "line_items": 0.92,
            "subtotal": 0.95,
            "tax": 0.95,
            "total": 0.97,
        },
    },
    "INV-CTL-1138": {
        "vendor": "Cape Town Logistics CC",
        "invoice_no": "INV-CTL-1138",
        "invoice_date": "2025-06-01",
        "due_date": "2025-06-15",
        "line_items": [
            {
                "description": "Delivery service Cape Town -> Durban",
                "quantity": 1,
                "unit_price": "1500.00",
                "line_total": "1500.00",
            },
            {
                "description": "Fuel surcharge",
                "quantity": 1,
                "unit_price": "150.00",
                "line_total": "150.00",
            },
        ],
        "subtotal": "1650.00",
        "tax": "247.50",
        "total": "1897.50",
        "_confidence": {
            "vendor": 0.93,
            "invoice_no": 0.97,
            "invoice_date": 0.95,
            "due_date": 0.95,
            "line_items": 0.9,
            "subtotal": 0.94,
            "tax": 0.95,
            "total": 0.97,
        },
    },
    # ---------------- Receipts ----------------
    "CHECKERS HYPER": {
        "merchant": "CHECKERS HYPER",
        "purchase_date": "2025-04-18",
        "items": [
            {"description": "Brown bread loaf", "amount": "18.99"},
            {"description": "Free range eggs 6pk", "amount": "42.50"},
            {"description": "Whole milk 2L", "amount": "29.99"},
            {"description": "Bananas 1kg", "amount": "15.49"},
            {"description": "Toilet paper 9pk", "amount": "89.99"},
        ],
        "subtotal": "196.96",
        "tax": "29.54",
        "total": "226.50",
        "payment_method": "VISA",
        "currency": None,
        "_confidence": {
            "merchant": 0.95,
            "purchase_date": 0.92,
            "items": 0.9,
            "subtotal": 0.95,
            "tax": 0.95,
            "total": 0.97,
            "payment_method": 0.9,
            "currency": 0.5,
        },
    },
    "WOOLWORTHS V&A WATERFRONT": {
        "merchant": "WOOLWORTHS V&A WATERFRONT",
        "purchase_date": "2025-05-09",
        "items": [
            {"description": "Greek yoghurt 1kg", "amount": "54.99"},
            {"description": "Sourdough loaf", "amount": "42.00"},
            {"description": "Aged cheddar 250g", "amount": "89.50"},
            {"description": "Olive oil 500ml", "amount": "79.99"},
        ],
        "subtotal": "266.48",
        "tax": "39.97",
        "total": "306.45",
        "payment_method": "CASH",
        "currency": None,
        "_confidence": {
            "merchant": 0.96,
            "purchase_date": 0.94,
            "items": 0.9,
            "subtotal": 0.94,
            "tax": 0.94,
            "total": 0.96,
            "payment_method": 0.93,
            "currency": 0.5,
        },
    },
    "PICK N PAY HOPE STREET": {
        "merchant": "PICK N PAY HOPE STREET",
        "purchase_date": "2025-03-26",
        "items": [
            {"description": "Coffee beans 250g", "amount": "85.00"},
            {"description": "Croissant x2", "amount": "34.00"},
            {"description": "Sparkling water 1L", "amount": "18.50"},
        ],
        "subtotal": "137.50",
        "tax": "20.63",
        "total": "158.13",
        "payment_method": "DEBIT CARD",
        "currency": "ZAR",
        "_confidence": {
            "merchant": 0.96,
            "purchase_date": 0.92,
            "items": 0.9,
            "subtotal": 0.94,
            "tax": 0.94,
            "total": 0.97,
            "payment_method": 0.93,
            "currency": 0.95,
        },
    },
    # ---------------- Contracts ----------------
    "MUTUAL NON-DISCLOSURE AGREEMENT": {
        "parties": ["Agulhas Code (Pty) Ltd", "Velocity Contact Hub (Pty) Ltd"],
        "effective_date": "2025-02-01",
        "term": "24 months",
        "renewal_notice_days": 30,
        "governing_law": "Republic of South Africa",
        "contract_type": "NDA",
        "_confidence": {
            "parties": 0.95,
            "effective_date": 0.95,
            "term": 0.9,
            "renewal_notice_days": 0.95,
            "governing_law": 0.95,
            "contract_type": 0.95,
        },
    },
    "MASTER SERVICES AGREEMENT": {
        "parties": ["MediBridge Health Solutions", "LeasEase Property Tech (Pty) Ltd"],
        "effective_date": "2025-03-15",
        "term": "12 months",
        "renewal_notice_days": 60,
        "governing_law": "Western Cape, South Africa",
        "contract_type": "MSA",
        "_confidence": {
            "parties": 0.95,
            "effective_date": 0.95,
            "term": 0.92,
            "renewal_notice_days": 0.95,
            "governing_law": 0.93,
            "contract_type": 0.95,
        },
    },
    "STATEMENT OF WORK NO. 1": {
        "parties": ["MediBridge Health Solutions", "Trekova Mobility (Pty) Ltd"],
        "effective_date": "2025-04-01",
        "term": "6 months",
        "renewal_notice_days": None,
        "governing_law": "Republic of South Africa",
        "contract_type": "SOW",
        "_confidence": {
            "parties": 0.95,
            "effective_date": 0.95,
            "term": 0.92,
            "renewal_notice_days": 0.6,
            "governing_law": 0.93,
            "contract_type": 0.95,
        },
    },
}


def build_stub_llm() -> StubLLM:
    """LLM that returns the fixture JSON if any of its keys appears in the prompt."""
    mapping = {needle: json.dumps(payload) for needle, payload in FIXTURE_RESPONSES.items()}
    # Default catches doc types we haven't fixtured (returns an empty object).
    return StubLLM(mapping=mapping, default=json.dumps({}))


def load_sample(rel_path: str) -> str:
    """Load a sample doc relative to the repo root."""
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / rel_path).read_text(encoding="utf-8")
