# The human-review queue

## Why it exists

The thing that kills document-extraction pipelines is **silent errors** — a
value extracted slightly wrong that nobody catches until the books don't
reconcile six weeks later. A line item misread as `R1,200` instead of
`R12,000` will happily pass every type check.

So the policy is: **be willing to fail loudly and often.** It is cheaper for
a bookkeeper to clear 30 exceptions a day than for an auditor to find one
wrong invoice three months later. Exception triage is the cost of
correctness.

## What gets queued

An extraction lands in the queue if **any one** of these is true:

1. **Schema validation failed.** Pydantic rejected the LLM's JSON — missing
   required field, line items don't sum to total, date isn't parseable.
   Status: `FAILED`.
2. **A field's combined confidence is below the threshold** (default `0.75`).
   Status: `NEEDS_REVIEW`.
3. The LLM output wasn't parseable as JSON at all. Status: `FAILED`.

Auto-approved results (every field above threshold, validation passed) bypass
the queue entirely and flow straight to your downstream system.

## Confidence: how it's scored

Two signals are combined per field:

| Signal | Weight | Source |
|---|---|---|
| LLM self-reported confidence (capped at 0.95) | 0.4 | Model emits a `_confidence` map |
| Post-hoc heuristic (does the value appear in the source text?) | 0.6 | `doc_intelligence/confidence.py` |

The heuristic is the more trustworthy signal. The cap on LLM
self-confidence exists because a model returning `1.0` on every field is
suspicious, not impressive.

The overall extraction confidence is the **minimum** field score — one weak
field is enough to queue the whole document. This is deliberate: it's easier
to skim and approve than to chase down a single misread line in production.

## CLI lifecycle

```bash
# Extract — auto-queues if confidence is low
doc-intel extract demo/sample_docs/invoice_001.txt --type invoice

# Look at the queue
doc-intel review
doc-intel review --type invoice

# Inspect one item
doc-intel show <id>

# Triage
doc-intel approve <id> --reviewer darrshan --notes "vendor name confirmed"
doc-intel reject  <id> --reviewer darrshan --notes "wrong doc type"
```

`approve` can also accept corrected data via the Python API
(`queue.approve(id, corrected_data={...})`), which is how the original
production pipeline fed corrections back into a few-shot store.

## What the queue does NOT do

This reference implementation deliberately stops short of:

- A web UI for review (the production system used a Next.js page with a
  PDF preview alongside the JSON).
- Few-shot learning from approved corrections (an interesting extension —
  store approved diffs and inject the closest 2-3 into the next prompt).
- SLAs or assignment routing — single-queue, single-reviewer here.

If you're extending this for production use, add those on top — but the
core "queue rather than guess" discipline is the load-bearing piece.
