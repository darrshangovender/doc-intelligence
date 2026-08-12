# Extending: adding a new extractor

A new extractor is three small changes: a Pydantic schema, an `Extractor`
subclass, and a registry entry.

## 1. Define the schema

Schemas live in `doc_intelligence/extractors/`. Use `pydantic.BaseModel` and
prefer typed fields with validators over free-form `dict` blobs.

```python
# doc_intelligence/extractors/id_card.py
from datetime import date
from pydantic import BaseModel, Field, field_validator

class IDCardData(BaseModel):
    full_name: str = Field(min_length=2)
    id_number: str = Field(min_length=6)
    date_of_birth: date | None = None
    nationality: str | None = None
    issued_date: date | None = None
    expiry_date: date | None = None

    @field_validator("id_number")
    @classmethod
    def strip_spaces(cls, v: str) -> str:
        return v.replace(" ", "")
```

## 2. Subclass `BaseExtractor`

```python
from typing import ClassVar
from pydantic import BaseModel
from doc_intelligence.extractors.base import BaseExtractor

class IDCardExtractor(BaseExtractor):
    schema: ClassVar[type[BaseModel]] = IDCardData
    doc_type: ClassVar[str] = "id_card"
    prompt_intro: ClassVar[str] = (
        "You are extracting fields from a government-issued ID card. "
        "Look for: full name, ID number, date of birth, nationality, "
        "and the issued/expiry dates if printed."
    )
```

The base class handles the LLM call, JSON parsing, Pydantic validation, and
per-field confidence scoring. You only override `build_prompt` if you need
something more elaborate than the default schema-introspection prompt.

## 3. Register it

Add the new extractor to `doc_intelligence/facade.py`:

```python
from doc_intelligence.extractors.id_card import IDCardExtractor

EXTRACTOR_REGISTRY: dict[str, type[BaseExtractor]] = {
    "invoice": InvoiceExtractor,
    "receipt": ReceiptExtractor,
    "contract": ContractExtractor,
    "id_card": IDCardExtractor,  # new
}
```

And re-export from `doc_intelligence/__init__.py` if you want it on the
public surface.

## 4. Add eval coverage

Drop a sample document in `demo/sample_docs/`, add an entry to
`eval/golden_extractions.yml`, and add a fixture response in
`eval/fixtures.py` so the offline test suite covers it.

```yaml
- id: id_card_001
  file: demo/sample_docs/id_card_001.txt
  doc_type: id_card
  expected:
    full_name: Darrshan Govender
    id_number: "9001011234080"
    date_of_birth: "1990-01-01"
```

```python
# eval/fixtures.py
"DARRSHAN GOVENDER": {
    "full_name": "Darrshan Govender",
    "id_number": "9001011234080",
    "date_of_birth": "1990-01-01",
    "nationality": "South African",
    "issued_date": "2015-06-12",
    "expiry_date": "2025-06-11",
    "_confidence": {"full_name": 0.97, "id_number": 0.98, ...},
},
```

Run `python eval/run_eval.py` to confirm.

## Tips

* **Lean on validators.** A schema-level reconciliation check (line items sum
  to total) catches a whole category of silent LLM errors that prompt tweaks
  alone never will. The invoice schema's `totals_reconcile` is the model.
* **Set fields to `None` rather than guessing.** The system prompt already
  instructs the model to do this. Mirror it in your `prompt_intro`.
* **Anchor on what's distinctive.** Fixture-matching keys (`"BMC-7781"`,
  `"MUTUAL NON-DISCLOSURE AGREEMENT"`) should be uniquely identifying within
  the sample corpus.
