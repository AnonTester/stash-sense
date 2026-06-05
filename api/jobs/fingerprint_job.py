"""Fingerprint generation as a queue job."""
from __future__ import annotations

import json
import logging
from typing import Optional

from base_job import BaseJob, JobContext
from fingerprint_generator import SceneFingerprintGenerator
from recommendations_router import get_db_version, get_rec_db, get_stash_client

logger = logging.getLogger(__name__)


class FingerprintGenerationJob(BaseJob):
    """Wraps SceneFingerprintGenerator as a queue-managed job.

    Cursor format (JSON string):
        {"offset": <int>, "processed": <int>}

    The cursor is saved after each batch of 100 scenes so the job resumes
    from roughly where it was interrupted rather than restarting at offset 0.
    """

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return None

        db_version = get_db_version()
        if db_version is None:
            raise RuntimeError("No face recognition database loaded; cannot generate fingerprints")

        # Parse resumption cursor
        start_offset = 0
        start_processed = 0
        if cursor:
            try:
                c = json.loads(cursor)
                start_offset = int(c.get("offset", 0))
                start_processed = int(c.get("processed", 0))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning(
                    "Fingerprint job: could not parse cursor %r — starting from offset 0",
                    cursor,
                )

        logger.warning(
            "Fingerprint generation job starting (job_id=%d, db_version=%s, "
            "start_offset=%d, start_processed=%d)",
            context.job_id, db_version, start_offset, start_processed,
        )

        stash = get_stash_client()
        db = get_rec_db()
        generator = SceneFingerprintGenerator(
            stash_client=stash,
            rec_db=db,
            db_version=db_version,
        )

        async for progress in generator.generate_all(
            start_offset=start_offset,
            start_processed=start_processed,
        ):
            if progress.batch_completed:
                # Persist cursor after each full batch for crash recovery
                new_cursor = json.dumps({
                    "offset": progress.current_offset,
                    "processed": progress.processed_scenes,
                })
                await context.checkpoint(
                    cursor=new_cursor,
                    items_processed=progress.processed_scenes,
                )
                logger.debug(
                    "Fingerprint job checkpoint: offset=%d processed=%d/%d",
                    progress.current_offset,
                    progress.processed_scenes,
                    progress.total_scenes,
                )
            else:
                await context.report_progress(
                    progress.processed_scenes,
                    progress.total_scenes,
                )

            if context.is_stop_requested():
                generator.request_stop()
                break

        final = generator.progress
        logger.warning(
            "Fingerprint generation job finished (job_id=%d, status=%s): "
            "%d processed, %d successful, %d skipped, %d failed",
            context.job_id, final.status.value,
            final.processed_scenes,
            final.successful,
            final.skipped,
            final.failed,
        )

        # Return None so the queue marks the job completed (no further cursor needed)
        return None
