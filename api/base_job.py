"""BaseJob ABC and JobContext for the operation queue system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from recommendations_db import RecommendationsDB


class JobContext:
    """Execution context passed to jobs, providing stop signaling, yielding, and progress reporting."""

    def __init__(self, job_id: int, db: RecommendationsDB, queue_manager: Any) -> None:
        self._job_id = job_id
        self._db = db
        self._queue_manager = queue_manager
        self._stop_requested = False
        self._user_cancelled = False
        self._result_summary: Optional[str] = None

    @property
    def job_id(self) -> int:
        return self._job_id

    def request_stop(self) -> None:
        """Signal that this job should stop at its next checkpoint (system shutdown)."""
        self._stop_requested = True

    def request_user_cancel(self) -> None:
        """Signal that the user explicitly cancelled this job. Implies request_stop."""
        self._stop_requested = True
        self._user_cancelled = True

    def is_stop_requested(self) -> bool:
        """Check whether a stop has been requested."""
        return self._stop_requested

    def is_user_cancelled(self) -> bool:
        """True if the user explicitly cancelled this job (vs. system shutdown)."""
        return self._user_cancelled

    @property
    def result_summary(self) -> Optional[str]:
        return self._result_summary

    def set_result_summary(self, summary: str) -> None:
        self._result_summary = summary

    async def should_yield(self) -> bool:
        """Check whether this job should yield to a higher-priority job.

        Returns False if no queue manager is set; otherwise delegates to the manager.
        """
        if self._queue_manager is None:
            return False
        return await self._queue_manager.should_job_yield(self._job_id)

    async def checkpoint(self, cursor: str, items_processed: int) -> None:
        """Persist progress so the job can be resumed from this point."""
        self._db.update_job_progress(
            self._job_id, items_processed=items_processed, cursor=cursor
        )

    async def report_progress(self, items_processed: int, items_total: Optional[int] = None) -> None:
        """Report current progress without saving a resumption cursor."""
        self._db.update_job_progress(
            self._job_id, items_processed=items_processed, items_total=items_total
        )


class BaseJob(ABC):
    """Abstract base class for all queue-managed jobs."""

    @abstractmethod
    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        """Execute the job.

        Args:
            context: The JobContext providing stop signals, yielding, and progress reporting.
            cursor: Optional cursor string to resume from a previous checkpoint.

        Returns:
            A cursor string if the job was interrupted and can be resumed, or None if completed.
        """
        ...
