"""Eval harness — runs every golden case through the extractor and reports
field-level precision / recall per doc_type.

Run::

    python eval/run_eval.py
    python eval/run_eval.py --json   # machine-readable

The default LLM is the offline stub. Pass ``--provider anthropic|openai`` to
hit a real model (requires the relevant API key + extra installed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

# Make the repo importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doc_intelligence.facade import Extractor  # noqa: E402
from doc_intelligence.llm_client import get_default_client  # noqa: E402
from eval.fixtures import build_stub_llm, load_sample  # noqa: E402


GOLDEN_PATH = REPO_ROOT / "eval" / "golden_extractions.yml"


def _norm(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "|".join(sorted(_norm(x) for x in v))
    # try money compare
    try:
        return str(Decimal(str(v)).normalize())
    except (InvalidOperation, ValueError):
        return str(v).strip().lower()


def _field_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list) and isinstance(actual, list):
        # Order-insensitive normalised comparison
        return {_norm(x) for x in expected} == {_norm(x) for x in actual}
    return _norm(expected) == _norm(actual)


def _extract_line_item_descriptions(data: dict) -> list[str]:
    items = data.get("line_items") or data.get("items") or []
    return [li.get("description", "") for li in items if isinstance(li, dict)]


def evaluate_case(case: dict, llm) -> dict[str, Any]:
    source = load_sample(case["file"])
    extractor = Extractor(llm=llm)
    result = extractor.run(source, doc_type=case["doc_type"], enqueue=False)

    expected = case["expected"]
    field_results: dict[str, bool] = {}
    actual = result.data

    for field, expected_value in expected.items():
        if field == "line_item_descriptions":
            actual_value = _extract_line_item_descriptions(actual)
        else:
            actual_value = actual.get(field)
        field_results[field] = _field_match(expected_value, actual_value)

    return {
        "id": case["id"],
        "doc_type": case["doc_type"],
        "status": result.status.value,
        "fields": field_results,
        "overall_confidence": result.overall_confidence,
    }


def aggregate(case_results: list[dict]) -> dict[str, Any]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in case_results:
        by_type[r["doc_type"]].append(r)

    summary: dict[str, Any] = {}
    for doc_type, results in by_type.items():
        total_fields = sum(len(r["fields"]) for r in results)
        correct_fields = sum(sum(r["fields"].values()) for r in results)
        accuracy = correct_fields / total_fields if total_fields else 0.0
        summary[doc_type] = {
            "n_docs": len(results),
            "n_fields": total_fields,
            "n_correct": correct_fields,
            "field_accuracy": round(accuracy, 4),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument(
        "--provider",
        choices=["stub", "anthropic", "openai"],
        default="stub",
        help="LLM backend (default: offline stub).",
    )
    args = parser.parse_args()

    cases = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]
    llm = build_stub_llm() if args.provider == "stub" else get_default_client(args.provider)

    case_results = [evaluate_case(c, llm) for c in cases]
    summary = aggregate(case_results)

    if args.json:
        print(json.dumps({"cases": case_results, "summary": summary}, indent=2))
        return

    print(f"\n  Ran {len(case_results)} cases via provider={args.provider}\n")
    for r in case_results:
        missed = [f for f, ok in r["fields"].items() if not ok]
        marker = "OK " if not missed else "FAIL"
        miss_s = f"  (missed: {', '.join(missed)})" if missed else ""
        print(f"  [{marker}] {r['id']:<14} {r['doc_type']:<10}  conf={r['overall_confidence']:.2f}{miss_s}")

    print("\n  --- per-type field accuracy ---")
    for doc_type, s in sorted(summary.items()):
        print(
            f"  {doc_type:<10} n={s['n_docs']:>2}  "
            f"fields={s['n_correct']}/{s['n_fields']}  "
            f"accuracy={s['field_accuracy']:.1%}"
        )

    overall_fields = sum(s["n_fields"] for s in summary.values())
    overall_correct = sum(s["n_correct"] for s in summary.values())
    if overall_fields:
        print(
            f"\n  OVERALL    fields={overall_correct}/{overall_fields}  "
            f"accuracy={overall_correct/overall_fields:.1%}\n"
        )


if __name__ == "__main__":
    main()
