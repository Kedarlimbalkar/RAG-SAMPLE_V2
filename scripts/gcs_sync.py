"""
GCS V2 Storage Synchronization Utility.

Handles bidirectional synchronization and partition management between local
storage and Google Cloud Storage bucket: gs://sc-legal-rag-data-v2/
Partitions:
- /raw            : Inbound raw Supreme Court judgment PDFs
- /parsed-v2      : Parsed document text streams and page offset maps
- /chunks-v2      : Hierarchical parent/child chunk exports
- /snapshots-v2   : Compressed Parquet / JSONL pipeline snapshot archives
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scrag.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("gcs_sync")


def verify_bucket_access(bucket_name: Optional[str] = None) -> bool:
    """Check connectivity to target GCS bucket using google-cloud-storage or gcloud CLI."""
    bucket = bucket_name or settings.gcs_bucket_name
    logger.info(f"Verifying access to bucket gs://{bucket}...")

    # Attempt 1: Native google-cloud-storage
    try:
        from google.cloud import storage
        client = storage.Client(project=settings.gcp_project_id)
        gcs_bucket = client.bucket(bucket)
        if gcs_bucket.exists():
            logger.info(f"Successfully verified gs://{bucket} via Google Cloud Storage SDK.")
            return True
    except Exception as e:
        logger.debug(f"SDK verification fallback: {e}")

    # Attempt 2: gcloud storage CLI
    try:
        res = subprocess.run(
            ["gcloud", "storage", "ls", f"gs://{bucket}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            logger.info(f"Successfully verified gs://{bucket} via gcloud CLI.")
            return True
        else:
            logger.error(f"gcloud storage ls failed: {res.stderr.strip()}")
    except FileNotFoundError:
        logger.warning("Neither google-cloud-storage SDK nor gcloud CLI is available.")

    return False


def sync_from_gcs(
    remote_prefix: str,
    local_dir: Path,
    bucket_name: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Download objects from GCS bucket prefix to local directory."""
    bucket = bucket_name or settings.gcs_bucket_name
    gcs_uri = f"gs://{bucket}/{remote_prefix.strip('/')}/"
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Syncing from {gcs_uri} to {local_dir} (dry_run={dry_run})...")

    cmd = ["gcloud", "storage", "rsync", "--recursive", gcs_uri, str(local_dir)]
    if dry_run:
        cmd.append("--dry-run")

    try:
        res = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info(f"Sync complete:\n{res.stdout}")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Sync from GCS failed (exit code {e.returncode}):\n{e.stderr}")
        return e.returncode
    except FileNotFoundError:
        logger.warning("gcloud CLI not found; falling back to python storage client...")
        return _sdk_download(bucket, remote_prefix, local_dir, dry_run)


def sync_to_gcs(
    local_dir: Path,
    remote_prefix: str,
    bucket_name: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Upload objects from local directory to GCS bucket prefix."""
    bucket = bucket_name or settings.gcs_bucket_name
    gcs_uri = f"gs://{bucket}/{remote_prefix.strip('/')}/"

    if not local_dir.exists():
        logger.warning(f"Local directory {local_dir} does not exist. Nothing to upload.")
        return 0

    logger.info(f"Syncing from {local_dir} to {gcs_uri} (dry_run={dry_run})...")

    cmd = ["gcloud", "storage", "rsync", "--recursive", str(local_dir), gcs_uri]
    if dry_run:
        cmd.append("--dry-run")

    try:
        res = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info(f"Upload complete:\n{res.stdout}")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Upload to GCS failed (exit code {e.returncode}):\n{e.stderr}")
        return e.returncode
    except FileNotFoundError:
        logger.warning("gcloud CLI not found; falling back to python storage client...")
        return _sdk_upload(bucket, local_dir, remote_prefix, dry_run)


def _sdk_download(bucket_name: str, prefix: str, dest_dir: Path, dry_run: bool) -> int:
    """Fallback download using google-cloud-storage Python library."""
    try:
        from google.cloud import storage
        client = storage.Client(project=settings.gcp_project_id)
        bucket = client.bucket(bucket_name)
        prefix_clean = prefix.strip("/") + "/"
        blobs = bucket.list_blobs(prefix=prefix_clean)

        count = 0
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            relative = blob.name[len(prefix_clean):]
            target_path = dest_dir / relative
            logger.info(f"[SDK Sync] {blob.name} -> {target_path}")
            if not dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(target_path))
            count += 1
        logger.info(f"Downloaded {count} blobs via SDK.")
        return 0
    except Exception as e:
        logger.error(f"Python SDK download failed: {e}")
        return 1


def _sdk_upload(bucket_name: str, source_dir: Path, prefix: str, dry_run: bool) -> int:
    """Fallback upload using google-cloud-storage Python library."""
    try:
        from google.cloud import storage
        client = storage.Client(project=settings.gcp_project_id)
        bucket = client.bucket(bucket_name)
        prefix_clean = prefix.strip("/")

        count = 0
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(source_dir).as_posix()
            blob_name = f"{prefix_clean}/{rel}"
            logger.info(f"[SDK Upload] {file_path} -> gs://{bucket_name}/{blob_name}")
            if not dry_run:
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
            count += 1
        logger.info(f"Uploaded {count} files via SDK.")
        return 0
    except Exception as e:
        logger.error(f"Python SDK upload failed: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="GCS V2 Storage Partition Sync Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify
    subparsers.add_parser("verify", help="Verify GCS bucket access")

    # download-raw
    p_dl = subparsers.add_parser("download-raw", help="Download raw PDFs from GCS /raw partition")
    p_dl.add_argument("--dest", type=Path, default=settings.raw_pdf_dir)
    p_dl.add_argument("--dry-run", action="store_true")

    # upload-snapshots
    p_up = subparsers.add_parser("upload-snapshots", help="Upload local snapshots to /snapshots-v2")
    p_up.add_argument("--source", type=Path, default=settings.snapshot_dir)
    p_up.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "verify":
        return 0 if verify_bucket_access() else 1
    elif args.command == "download-raw":
        return sync_from_gcs(
            remote_prefix=settings.gcs_raw_prefix,
            local_dir=args.dest,
            dry_run=args.dry_run,
        )
    elif args.command == "upload-snapshots":
        return sync_to_gcs(
            local_dir=args.source,
            remote_prefix=settings.gcs_snapshots_prefix,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
