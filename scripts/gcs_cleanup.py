"""
GCS V2 Storage Cleanup & Partition Hygiene Tool.

Cleans up test/temporary output folders (e.g. chroma/, chunks/, parsed/, logs/)
generated inside raw judgment directories (e.g. year=1950) to enforce strict separation:
- gs://sc-legal-rag-data-v2/raw/       (Untouched raw PDFs ONLY)
- gs://sc-legal-rag-data-v2/parsed-v2/ (Extracted & cleaned text / JSONL)
- gs://sc-legal-rag-data-v2/chunks-v2/ (Hierarchical parent/child chunks)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scrag.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("gcs_cleanup")

# Subdirectories considered temporary/misplaced inside raw partitions
TEMP_SUBFOLDERS = ["chroma", ".chroma", "chunks", "parsed", "logs", "temp", "cache"]


def cleanup_raw_directory(
    year: str = "1950",
    bucket_name: Optional[str] = None,
    dry_run: bool = False,
    subfolders_to_remove: Optional[List[str]] = None,
) -> int:
    """
    Remove temporary output folders located inside raw data directories in GCS.
    """
    bucket = bucket_name or settings.gcs_bucket_name
    subfolders = subfolders_to_remove or TEMP_SUBFOLDERS

    logger.info("=" * 70)
    logger.info(f"GCS Storage Cleanup: Bucket=gs://{bucket} | Year={year} | DryRun={dry_run}")
    logger.info("=" * 70)

    # Search paths where temporary folders might have been created:
    # 1. gs://<bucket>/raw/year=<year>/<folder>/
    # 2. gs://<bucket>/year=<year>/<folder>/
    candidate_prefixes = [
        f"raw/year={year}/",
        f"year={year}/",
    ]

    deleted_count = 0

    # Attempt cleanup via google-cloud-storage Python SDK
    try:
        from google.cloud import storage
        client = storage.Client(project=settings.gcp_project_id)
        gcs_bucket = client.bucket(bucket)

        for base_prefix in candidate_prefixes:
            for folder in subfolders:
                folder_prefix = f"{base_prefix}{folder}/"
                blobs = list(gcs_bucket.list_blobs(prefix=folder_prefix))

                if not blobs:
                    continue

                logger.info(f"Found {len(blobs)} temporary objects in gs://{bucket}/{folder_prefix}")
                for b in blobs:
                    if dry_run:
                        logger.info(f"[DRY-RUN] Would delete: gs://{bucket}/{b.name}")
                    else:
                        b.delete()
                        deleted_count += 1
                        logger.info(f"[DELETED] gs://{bucket}/{b.name}")

        logger.info(f"Cleanup complete via SDK. Total objects purged: {deleted_count}")
        return 0

    except Exception as e:
        logger.warning(f"Python SDK cleanup unavailable ({e}). Falling back to gcloud storage CLI...")

    # Fallback cleanup via gcloud storage rm CLI
    for base_prefix in candidate_prefixes:
        for folder in subfolders:
            folder_uri = f"gs://{bucket}/{base_prefix}{folder}/"
            cmd = ["gcloud", "storage", "rm", "--recursive", folder_uri]

            if dry_run:
                logger.info(f"[DRY-RUN] Would run: {' '.join(cmd)}")
                continue

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    logger.info(f"Successfully purged: {folder_uri}")
                    deleted_count += 1
                elif "matches no objects" not in res.stderr.lower():
                    logger.debug(f"gcloud notice for {folder_uri}: {res.stderr.strip()}")
            except FileNotFoundError:
                logger.error("gcloud CLI not found. Please install Google Cloud SDK.")
                return 1

    logger.info(f"Cleanup operation finished. Folders removed: {deleted_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean up temporary output folders inside GCS raw directories."
    )
    parser.add_argument(
        "--year",
        default="1950",
        help="Target year partition (default: 1950).",
    )
    parser.add_argument(
        "--bucket",
        default=settings.gcs_bucket_name,
        help="Target GCS bucket name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview objects to be deleted without executing removal.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=TEMP_SUBFOLDERS,
        help="Specific folder names to remove (default: chroma, chunks, parsed, logs).",
    )

    args = parser.parse_args()
    return cleanup_raw_directory(
        year=args.year,
        bucket_name=args.bucket,
        dry_run=args.dry_run,
        subfolders_to_remove=args.folders,
    )


if __name__ == "__main__":
    sys.exit(main())
