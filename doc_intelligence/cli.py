"""CLI: ``doc-intel extract``, ``doc-intel review``, ``doc-intel approve``, ``doc-intel reject``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from doc_intelligence.facade import EXTRACTOR_REGISTRY, Extractor
from doc_intelligence.review_queue import ReviewQueue


DEFAULT_DB = "review_queue.db"


@click.group()
@click.version_option()
def main() -> None:
    """doc-intel: OCR + LLM document extraction with human-review queue."""


@main.command("extract")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--type",
    "doc_type",
    type=click.Choice(sorted(EXTRACTOR_REGISTRY)),
    required=True,
    help="Document type — picks the schema and prompt.",
)
@click.option("--db", default=DEFAULT_DB, show_default=True, help="Review-queue SQLite path.")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    show_default=True,
)
@click.option(
    "--no-enqueue",
    is_flag=True,
    help="Don't add low-confidence results to the review queue.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.75,
    show_default=True,
    help="Minimum per-field confidence to auto-approve.",
)
def extract_cmd(
    source: Path,
    doc_type: str,
    db: str,
    provider: str,
    no_enqueue: bool,
    threshold: float,
) -> None:
    """Extract structured data from a document."""
    queue = None if no_enqueue else ReviewQueue(db)
    extractor = Extractor(provider=provider, queue=queue, review_threshold=threshold)
    result = extractor.run(source, doc_type=doc_type, enqueue=not no_enqueue)
    click.echo(json.dumps(result.to_dict(), indent=2, default=str))
    if result.status.value == "failed":
        sys.exit(2)


@main.command("review")
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--status", default="pending", show_default=True)
@click.option("--type", "doc_type", default=None, help="Filter by doc_type.")
@click.option("--limit", default=50, show_default=True, type=int)
def review_cmd(db: str, status: str, doc_type: str | None, limit: int) -> None:
    """List items in the review queue."""
    q = ReviewQueue(db)
    items = q.list(status=status, doc_type=doc_type, limit=limit)
    if not items:
        click.echo("No items.")
        return
    click.echo(f"{'ID':<14} {'TYPE':<10} {'CONF':>6}  {'CREATED':<26} ERRORS")
    click.echo("-" * 80)
    for item in items:
        errs = "; ".join(item.errors)[:30] if item.errors else ""
        click.echo(
            f"{item.id:<14} {item.doc_type:<10} "
            f"{item.overall_confidence:>6.2f}  {item.created_at:<26} {errs}"
        )


@main.command("show")
@click.argument("record_id")
@click.option("--db", default=DEFAULT_DB, show_default=True)
def show_cmd(record_id: str, db: str) -> None:
    """Show a single queue item in detail."""
    q = ReviewQueue(db)
    rec = q.get(record_id)
    if rec is None:
        click.echo(f"No such record: {record_id}", err=True)
        sys.exit(1)
    click.echo(
        json.dumps(
            {
                "id": rec.id,
                "doc_type": rec.doc_type,
                "status": rec.status,
                "data": rec.data,
                "confidence": rec.confidence,
                "overall_confidence": rec.overall_confidence,
                "errors": rec.errors,
                "source_path": rec.source_path,
                "created_at": rec.created_at,
            },
            indent=2,
            default=str,
        )
    )


@main.command("approve")
@click.argument("record_id")
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--reviewer", default=None)
@click.option("--notes", default=None)
def approve_cmd(record_id: str, db: str, reviewer: str | None, notes: str | None) -> None:
    """Approve a queue item."""
    q = ReviewQueue(db)
    q.approve(record_id, reviewer=reviewer, notes=notes)
    click.echo(f"approved {record_id}")


@main.command("reject")
@click.argument("record_id")
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--reviewer", default=None)
@click.option("--notes", default=None)
def reject_cmd(record_id: str, db: str, reviewer: str | None, notes: str | None) -> None:
    """Reject a queue item."""
    q = ReviewQueue(db)
    q.reject(record_id, reviewer=reviewer, notes=notes)
    click.echo(f"rejected {record_id}")


if __name__ == "__main__":
    main()
