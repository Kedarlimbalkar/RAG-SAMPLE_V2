"""
Command-Line Interface (CLI) Runner for Supreme Court Legal RAG Pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from scrag.config import settings
from scrag.database import db
from scrag.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("scrag.cli")


@click.group()
@click.version_option(version="0.2.0")
def cli() -> None:
    """Supreme Court Legal RAG Pipeline CLI."""
    pass


@cli.command("run")
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=settings.raw_pdf_dir,
    help="Local path to raw PDF directory.",
)
@click.option(
    "--from-gcs",
    is_flag=True,
    default=False,
    help="Pull inbound raw PDFs from GCS bucket (gs://sc-legal-rag-data-v2/raw/).",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Maximum number of documents to process in this run.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume processing unfinished documents from database state.",
)
def run_pipeline(
    source: Path, from_gcs: bool, limit: Optional[int], resume: bool
) -> None:
    """
    Execute batch processing pipeline (parse, chunk, embed, pgvector, and export).
    """
    click.echo("=" * 70)
    click.echo("Starting Supreme Court Legal RAG Pipeline (V2)")
    click.echo(f"  Source Directory : {source}")
    click.echo(f"  GCS Sync Active  : {from_gcs} (gs://{settings.gcs_bucket_name})")
    click.echo(f"  Document Limit   : {limit or 'Unlimited'}")
    click.echo(f"  Resume Mode      : {resume}")
    click.echo("=" * 70)

    try:
        metrics = pipeline.run_batch(
            source_dir=source,
            from_gcs=from_gcs,
            limit=limit,
            resume=resume,
        )
        click.secho(
            f"\nPipeline Run Complete! Processed: {metrics['processed']} | "
            f"Failed: {metrics['failed']} | Total: {metrics['total']}",
            fg="green" if metrics["failed"] == 0 else "yellow",
            bold=True,
        )
    except Exception as e:
        click.secho(f"Pipeline execution error: {e}", fg="red", err=True)
        sys.exit(1)


@cli.command("init-db")
@click.option(
    "--schema-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=Path("schema.sql"),
    help="Path to schema.sql DDL definition file.",
)
def init_database(schema_file: Path) -> None:
    """Initialize PostgreSQL database schema and pgvector extensions."""
    click.echo(f"Applying schema DDL from {schema_file}...")
    try:
        db.run_schema_migration(schema_file)
        click.secho("Database schema initialized successfully!", fg="green", bold=True)
    except Exception as e:
        click.secho(f"Failed to initialize database: {e}", fg="red", err=True)
        sys.exit(1)


@cli.command("status")
def pipeline_status() -> None:
    """Display pipeline status metrics from PostgreSQL."""
    click.echo("Fetching pipeline status from PostgreSQL...")
    sql = """
        SELECT status, count(*) AS count
        FROM documents
        GROUP BY status
        ORDER BY count DESC;
    """
    try:
        with db.get_cursor(commit=False) as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            click.echo("\n--- Document Lifecycle Status ---")
            for r in rows:
                click.echo(f"  {r['status']:<15} : {r['count']} documents")
    except Exception as e:
        click.secho(f"Could not connect to database: {e}", fg="yellow")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
