"""
Scene Fingerprint Generator

Generates face fingerprints for scenes in the Stash library.
Supports checkpointing for restart resilience and rate limiting.

This generator calls the /identify/scene endpoint internally, which
handles frame extraction, face detection, matching, and fingerprint
persistence automatically.

Usage:
    generator = SceneFingerprintGenerator(
        stash_client=stash,
        rec_db=db,
        db_version="2026.01.30",
        identify_endpoint="http://localhost:5000/identify/scene",
    )

    # Generate fingerprints for all scenes
    async for progress in generator.generate_all():
        print(f"Progress: {progress.processed}/{progress.total}")
"""

import httpx
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, AsyncIterator
from enum import Enum

import face_config

if TYPE_CHECKING:
    from stash_client_unified import StashClientUnified
    from recommendations_db import RecommendationsDB


logger = logging.getLogger(__name__)


class GeneratorStatus(str, Enum):
    """Status of the fingerprint generator."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class GeneratorProgress:
    """Progress information for fingerprint generation."""
    status: GeneratorStatus
    total_scenes: int
    processed_scenes: int
    successful: int
    failed: int
    skipped: int  # Already have current-version fingerprint
    current_scene_id: Optional[int] = None
    current_scene_title: Optional[str] = None
    error_message: Optional[str] = None
    # Cursor support: set at the end of each batch so the job can checkpoint.
    batch_completed: bool = False   # True only on the extra yield after a full batch
    current_offset: int = 0         # Pagination offset after this batch

    @property
    def progress_pct(self) -> float:
        if self.total_scenes == 0:
            return 0.0
        return (self.processed_scenes / self.total_scenes) * 100

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "total_scenes": self.total_scenes,
            "processed_scenes": self.processed_scenes,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "progress_pct": round(self.progress_pct, 1),
            "current_scene_id": self.current_scene_id,
            "current_scene_title": self.current_scene_title,
            "error_message": self.error_message,
        }


@dataclass
class FingerprintResult:
    """Result of fingerprinting a single scene."""
    scene_id: int
    success: bool
    fingerprint_id: Optional[int] = None
    performers_found: int = 0
    frames_analyzed: int = 0
    error: Optional[str] = None


class SceneFingerprintGenerator:
    """
    Generates face fingerprints for scenes with checkpointing.

    Features:
    - Processes scenes one at a time
    - Calls /identify/scene which saves fingerprint automatically
    - Respects rate limiting
    - Can be stopped gracefully
    - Skips scenes with up-to-date fingerprints
    - Supports cursor-based resumption via start_offset / start_processed
    """

    def __init__(
        self,
        stash_client: "StashClientUnified",
        rec_db: "RecommendationsDB",
        db_version: str,
        sidecar_url: str = "http://localhost:5000",
        num_frames: int = face_config.NUM_FRAMES,
        min_face_size: int = face_config.MIN_FACE_SIZE,
        max_distance: float = face_config.MAX_DISTANCE,
    ):
        self.stash = stash_client
        self.rec_db = rec_db
        self.db_version = db_version
        self.sidecar_url = sidecar_url.rstrip("/")

        # Identification config
        self.num_frames = num_frames
        self.min_face_size = min_face_size
        self.max_distance = max_distance

        # State
        self._status = GeneratorStatus.IDLE
        self._stop_requested = False
        self._progress = GeneratorProgress(
            status=GeneratorStatus.IDLE,
            total_scenes=0,
            processed_scenes=0,
            successful=0,
            failed=0,
            skipped=0,
        )

    @property
    def status(self) -> GeneratorStatus:
        return self._status

    @property
    def progress(self) -> GeneratorProgress:
        return self._progress

    def request_stop(self):
        """Request graceful stop. Generator will finish current scene then stop."""
        if self._status == GeneratorStatus.RUNNING:
            self._stop_requested = True
            self._status = GeneratorStatus.STOPPING
            self._progress.status = GeneratorStatus.STOPPING
            logger.info("Stop requested, will finish current scene")

    async def generate_all(
        self,
        refresh_outdated: bool = True,
        batch_size: int = 100,
        start_offset: int = 0,
        start_processed: int = 0,
    ) -> AsyncIterator[GeneratorProgress]:
        """
        Generate fingerprints for all scenes that need them.

        Args:
            refresh_outdated: Also regenerate fingerprints from older DB versions
            batch_size: Number of scenes to query at a time
            start_offset: Resume pagination from this offset (0 = start fresh)
            start_processed: Cumulative processed count before this run (from cursor)

        Yields:
            GeneratorProgress after each scene and once more at each batch boundary
            (batch_completed=True) so the job can save a resumption cursor.
        """
        self._status = GeneratorStatus.RUNNING
        self._progress.status = GeneratorStatus.RUNNING
        self._stop_requested = False
        self._progress.processed_scenes = start_processed
        self._progress.current_offset = start_offset

        resuming = start_offset > 0 or start_processed > 0

        try:
            # Get total scene count
            _, total = await self.stash.get_scenes_for_fingerprinting(limit=1, offset=0)
            self._progress.total_scenes = total

            if resuming:
                logger.warning(
                    "Fingerprint generation resuming from offset %d "
                    "(previously processed: %d / %d, db_version=%s)",
                    start_offset, start_processed, total, self.db_version,
                )
            else:
                logger.warning(
                    "Fingerprint generation starting: %d total scenes, db_version=%s, "
                    "refresh_outdated=%s, batch_size=%d",
                    total, self.db_version, refresh_outdated, batch_size,
                )

            yield self._progress

            offset = start_offset
            while offset < total and not self._stop_requested:
                # Fetch batch of scenes
                scenes, _ = await self.stash.get_scenes_for_fingerprinting(
                    limit=batch_size,
                    offset=offset,
                )

                if not scenes:
                    logger.debug(
                        "Batch at offset=%d returned no scenes (total=%d); stopping.",
                        offset, total,
                    )
                    break

                batch_successful = 0
                batch_failed = 0
                batch_skipped = 0

                logger.debug(
                    "Processing batch: offset=%d, scenes_in_batch=%d, "
                    "cumulative_processed=%d/%d (%.1f%%)",
                    offset, len(scenes),
                    self._progress.processed_scenes, total,
                    self._progress.progress_pct,
                )

                for scene in scenes:
                    if self._stop_requested:
                        break

                    scene_id = int(scene["id"])
                    scene_title = scene.get("title") or f"Scene {scene_id}"
                    self._progress.current_scene_id = scene_id
                    self._progress.current_scene_title = scene_title
                    self._progress.batch_completed = False

                    # Check if we need to process this scene
                    existing = self.rec_db.get_scene_fingerprint(scene_id)
                    if existing:
                        if existing.get("fingerprint_status") == "complete":
                            if not refresh_outdated or existing.get("db_version") == self.db_version:
                                logger.debug(
                                    "Scene %d (%s): skipped — already fingerprinted "
                                    "(db_version=%s, current=%s)",
                                    scene_id, scene_title,
                                    existing.get("db_version"), self.db_version,
                                )
                                self._progress.skipped += 1
                                self._progress.processed_scenes += 1
                                batch_skipped += 1
                                yield self._progress
                                continue

                        logger.debug(
                            "Scene %d (%s): re-fingerprinting "
                            "(status=%s, db_version=%s → %s)",
                            scene_id, scene_title,
                            existing.get("fingerprint_status"),
                            existing.get("db_version"), self.db_version,
                        )
                    else:
                        logger.debug("Scene %d (%s): no existing fingerprint", scene_id, scene_title)

                    # Generate fingerprint by calling /identify/scene
                    result = await self._identify_scene(scene_id)

                    self._progress.processed_scenes += 1
                    if result.success:
                        self._progress.successful += 1
                        batch_successful += 1
                        logger.debug(
                            "Scene %d (%s): fingerprinted — performers_found=%d, frames=%d",
                            scene_id, scene_title,
                            result.performers_found, result.frames_analyzed,
                        )
                    else:
                        self._progress.failed += 1
                        batch_failed += 1
                        logger.debug(
                            "Scene %d (%s): fingerprint failed — %s",
                            scene_id, scene_title, result.error,
                        )

                    yield self._progress

                offset += batch_size
                self._progress.current_offset = offset

                logger.debug(
                    "Batch complete: offset_now=%d, "
                    "batch_ok=%d skipped=%d failed=%d | "
                    "total processed=%d/%d (%.1f%%)",
                    offset,
                    batch_successful, batch_skipped, batch_failed,
                    self._progress.processed_scenes, total,
                    self._progress.progress_pct,
                )

                # Yield once more with batch_completed=True so the job can
                # save a resumption cursor without writing per-scene.
                self._progress.batch_completed = True
                self._progress.current_scene_id = None
                self._progress.current_scene_title = None
                yield self._progress
                self._progress.batch_completed = False

            if self._stop_requested:
                self._status = GeneratorStatus.PAUSED
                self._progress.status = GeneratorStatus.PAUSED
                logger.warning(
                    "Fingerprint generation paused at offset %d "
                    "(%d/%d processed, %d ok, %d skipped, %d failed)",
                    offset,
                    self._progress.processed_scenes, total,
                    self._progress.successful,
                    self._progress.skipped,
                    self._progress.failed,
                )
            else:
                self._status = GeneratorStatus.COMPLETED
                self._progress.status = GeneratorStatus.COMPLETED
                logger.warning(
                    "Fingerprint generation complete: %d/%d processed "
                    "(%d successful, %d skipped, %d failed), db_version=%s",
                    self._progress.processed_scenes, total,
                    self._progress.successful,
                    self._progress.skipped,
                    self._progress.failed,
                    self.db_version,
                )

        except Exception as e:
            self._status = GeneratorStatus.ERROR
            self._progress.status = GeneratorStatus.ERROR
            self._progress.error_message = str(e)
            logger.error(
                "Fingerprint generation error at offset ~%d (%d processed so far): %s",
                self._progress.current_offset,
                self._progress.processed_scenes,
                e,
                exc_info=True,
            )
            raise

        finally:
            self._progress.current_scene_id = None
            self._progress.current_scene_title = None
            self._progress.batch_completed = False
            yield self._progress

    async def generate_for_scene(self, scene_id: int) -> FingerprintResult:
        """Generate fingerprint for a single scene."""
        return await self._identify_scene(scene_id)

    async def _identify_scene(self, scene_id: int) -> FingerprintResult:
        """Call the /identify/scene endpoint to generate and save fingerprint."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.sidecar_url}/identify/scene",
                    json={
                        "scene_id": str(scene_id),
                        "num_frames": self.num_frames,
                        "min_face_size": self.min_face_size,
                        "max_distance": self.max_distance,
                        "matching_mode": "hybrid",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    persons = data.get("persons", [])
                    performers_found = sum(1 for p in persons if p.get("best_match"))

                    # Check if fingerprint was actually saved
                    fingerprint_saved = data.get("fingerprint_saved", False)
                    fingerprint_error = data.get("fingerprint_error")

                    if not fingerprint_saved and performers_found > 0:
                        # Identification succeeded but save failed
                        error_msg = fingerprint_error or "Fingerprint save failed"
                        logger.warning(f"Scene {scene_id} fingerprint save failed: {error_msg}")
                        return FingerprintResult(
                            scene_id=scene_id,
                            success=False,
                            error=f"Save failed: {error_msg}",
                            performers_found=performers_found,
                            frames_analyzed=data.get("frames_analyzed", 0),
                        )

                    return FingerprintResult(
                        scene_id=scene_id,
                        success=True,
                        performers_found=performers_found,
                        frames_analyzed=data.get("frames_analyzed", 0),
                    )
                else:
                    error_detail = response.json().get("detail", response.text)
                    logger.warning(f"Scene {scene_id} identification failed: {error_detail}")

                    # Mark as error in database
                    self.rec_db.create_scene_fingerprint(
                        stash_scene_id=scene_id,
                        total_faces=0,
                        frames_analyzed=0,
                        fingerprint_status="error",
                        db_version=self.db_version,
                    )

                    return FingerprintResult(
                        scene_id=scene_id,
                        success=False,
                        error=error_detail,
                    )

        except httpx.TimeoutException:
            error = "Timeout - scene may be too long or system too slow"
            logger.warning(f"Scene {scene_id} timed out")
            return FingerprintResult(scene_id=scene_id, success=False, error=error)

        except Exception as e:
            logger.error("Error identifying scene %s: %s", scene_id, e, exc_info=True)
            return FingerprintResult(scene_id=scene_id, success=False, error=str(e))


# Singleton generator instance for background processing
_generator_instance: Optional[SceneFingerprintGenerator] = None


def get_generator() -> Optional[SceneFingerprintGenerator]:
    """Get the current generator instance."""
    return _generator_instance


def set_generator(generator: SceneFingerprintGenerator):
    """Set the generator instance."""
    global _generator_instance
    _generator_instance = generator
