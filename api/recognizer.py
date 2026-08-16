"""Face recognition against the database.

Matches detected faces against the pre-built performer database.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

from voyager import Index

from config import DatabaseConfig
from embeddings import FaceEmbeddingGenerator, DetectedFace, FaceEmbedding
from matching import MatchingConfig, match_face, MatchingResult
from database_reader import PerformerDatabaseReader


@dataclass
class PerformerMatch:
    """A potential performer match."""
    universal_id: str  # e.g., "stashdb.org:50459d16-..."
    stashdb_id: str  # Just the UUID part
    name: str
    country: Optional[str]
    image_url: Optional[str]
    facenet_distance: float
    arcface_distance: float
    combined_score: float  # Lower is better (average of distances)
    # Set only for local-index matches (universal_id starts with "local:"),
    # to the local Stash performer id -- unambiguous even when stashdb_id
    # above is a real linked StashDB uuid rather than the local id itself,
    # so callers never have to guess which one stashdb_id actually holds.
    local_performer_id: Optional[str] = None


@dataclass
class RecognitionResult:
    """Result of face recognition on an image."""
    face: DetectedFace
    matches: list[PerformerMatch]  # Sorted by combined_score (best first)
    embedding: Optional["FaceEmbedding"] = None  # Stored for clustering (avoids recomputation)


class FaceRecognizer:
    """Recognize faces against the performer database."""

    def __init__(self, db_config: DatabaseConfig, models_dir: "Path | None" = None):
        """
        Initialize the recognizer.

        Args:
            db_config: Database configuration with paths to index files
            models_dir: Directory containing ONNX embedding models.
                If None, FaceEmbeddingGenerator will auto-detect
                (DATA_DIR/models first, then ./models).
        """
        self.db_config = db_config
        self.generator = FaceEmbeddingGenerator(models_dir=models_dir)

        # Load indices
        print(f"Loading FaceNet index from {db_config.facenet_index_path}...")
        with open(db_config.facenet_index_path, "rb") as f:
            self.facenet_index = Index.load(f)

        print(f"Loading ArcFace index from {db_config.arcface_index_path}...")
        with open(db_config.arcface_index_path, "rb") as f:
            self.arcface_index = Index.load(f)

        # Load metadata
        print(f"Loading faces mapping from {db_config.faces_json_path}...")
        with open(db_config.faces_json_path) as f:
            self.faces = json.load(f)  # index -> universal_id

        print(f"Loading performers from {db_config.performers_json_path}...")
        with open(db_config.performers_json_path) as f:
            self.performers = json.load(f)  # universal_id -> metadata

        print(f"Loaded {len(self.faces)} faces, {len(self.performers)} performers")

        # Optionally load AdaFace index
        self.adaface_index = None
        if db_config.adaface_index_path and db_config.adaface_index_path.exists():
            print(f"Loading AdaFace index from {db_config.adaface_index_path}...")
            with open(db_config.adaface_index_path, "rb") as f:
                self.adaface_index = Index.load(f)
            print(f"AdaFace index loaded: {len(self.adaface_index)} vectors")

        # Optionally load tattoo embedding index and mapping
        self.tattoo_index = None
        self.tattoo_mapping = None
        if (db_config.tattoo_index_path and db_config.tattoo_index_path.exists()
                and db_config.tattoo_json_path and db_config.tattoo_json_path.exists()):
            print(f"Loading tattoo embedding index from {db_config.tattoo_index_path}...")
            with open(db_config.tattoo_index_path, "rb") as f:
                self.tattoo_index = Index.load(f)
            with open(db_config.tattoo_json_path) as f:
                self.tattoo_mapping = json.load(f)  # index -> universal_id
            print(f"Tattoo embeddings loaded: {len(self.tattoo_index)} vectors, "
                  f"{len(self.tattoo_mapping)} mappings")

        # Optionally load the local performer index -- built from this
        # Stash instance's own performer cover images by the
        # local_performer_sync job, absent until that's run at least once.
        self.local_performer_index = None
        if db_config.local_faces_json_path and db_config.local_faces_json_path.exists():
            from local_performer_index import LocalPerformerIndex
            print(f"Loading local performer index from {db_config.local_faces_json_path}...")
            self.local_performer_index = LocalPerformerIndex(
                db_config.local_facenet_index_path,
                db_config.local_arcface_index_path,
                db_config.local_faces_json_path,
            )
            print(f"Local performer index loaded: {len(self.local_performer_index)} performers")

        # Initialize SQLite database reader for multi-signal data
        self.db_reader = None
        if db_config.sqlite_db_path and db_config.sqlite_db_path.exists():
            print(f"Loading SQLite database from {db_config.sqlite_db_path}...")
            self.db_reader = PerformerDatabaseReader(str(db_config.sqlite_db_path))

    def _get_performer_info(self, universal_id: str) -> dict:
        """Get performer info from universal ID."""
        return self.performers.get(universal_id, {})

    def recognize_face(
        self,
        face: DetectedFace,
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> list[PerformerMatch]:
        """
        Recognize a single detected face.

        Args:
            face: DetectedFace object with cropped face image
            top_k: Number of top matches to return
            max_distance: Maximum distance threshold (cosine distance, 0-2)

        Returns:
            List of PerformerMatch objects, sorted by combined score
        """
        # Generate embedding
        embedding = self.generator.get_embedding(face.image)

        # Query both indices
        facenet_neighbors, facenet_distances = self.facenet_index.query(
            embedding.facenet, k=top_k * 2  # Get more to allow filtering
        )
        arcface_neighbors, arcface_distances = self.arcface_index.query(
            embedding.arcface, k=top_k * 2
        )

        # Combine results - use the intersection or union approach
        # For now, let's average the distances for faces that appear in both
        matches = {}
        faces_count = len(self.faces)

        for i, (idx, dist) in enumerate(zip(facenet_neighbors, facenet_distances)):
            # Skip if index is out of bounds (can happen with index/metadata mismatch)
            if idx < 0 or idx >= faces_count:
                continue
            universal_id = self.faces[idx]
            # Skip null entries (gaps in index from deleted faces or missing stashbox IDs)
            if universal_id is None:
                continue
            if universal_id not in matches:
                matches[universal_id] = {"facenet": dist, "arcface": None}
            else:
                matches[universal_id]["facenet"] = dist

        for i, (idx, dist) in enumerate(zip(arcface_neighbors, arcface_distances)):
            # Skip if index is out of bounds (can happen with index/metadata mismatch)
            if idx < 0 or idx >= faces_count:
                continue
            universal_id = self.faces[idx]
            # Skip null entries (gaps in index from deleted faces or missing stashbox IDs)
            if universal_id is None:
                continue
            if universal_id not in matches:
                matches[universal_id] = {"facenet": None, "arcface": dist}
            else:
                matches[universal_id]["arcface"] = dist

        # Build match objects
        result = []
        for universal_id, distances in matches.items():
            facenet_dist = distances["facenet"]
            arcface_dist = distances["arcface"]

            # Skip if missing either distance (not in top results for both)
            if facenet_dist is None or arcface_dist is None:
                continue

            # Calculate combined score
            combined = (facenet_dist + arcface_dist) / 2

            # Filter by max distance
            if combined > max_distance:
                continue

            # Get performer info
            info = self._get_performer_info(universal_id)
            stashdb_id = universal_id.split(":", 1)[1] if ":" in universal_id else universal_id

            result.append(PerformerMatch(
                universal_id=universal_id,
                stashdb_id=stashdb_id,
                name=info.get("name", "Unknown"),
                country=info.get("country"),
                image_url=info.get("image_url"),
                facenet_distance=facenet_dist,
                arcface_distance=arcface_dist,
                combined_score=combined,
            ))

        # Sort by combined score
        result.sort(key=lambda m: m.combined_score)

        return result[:top_k]

    def recognize_face_v2(
        self,
        face: DetectedFace,
        config: MatchingConfig = None,
        embedding: "FaceEmbedding | None" = None,
    ) -> tuple[list[PerformerMatch], MatchingResult]:
        """
        Recognize a face using the robust matching strategy (V2).

        This uses intelligent health detection to determine when ArcFace
        is producing reliable vs degenerate output, and adapts the fusion
        strategy accordingly.

        Args:
            face: DetectedFace object with cropped face image
            config: Matching configuration (uses defaults if not provided)
            embedding: Pre-computed FaceEmbedding (skips inference if provided)

        Returns:
            Tuple of (matches, matching_result, embedding)
        """
        if config is None:
            config = MatchingConfig()

        # Use pre-computed embedding or generate one
        if embedding is None:
            embedding = self.generator.get_embedding(face.image)

        # Use new matching logic
        local_index = self.local_performer_index
        result = match_face(
            facenet_embedding=embedding.facenet,
            arcface_embedding=embedding.arcface,
            facenet_index=self.facenet_index,
            arcface_index=self.arcface_index,
            faces_mapping=self.faces,
            performers=self.performers,
            config=config,
            local_facenet_index=local_index.facenet_index if local_index else None,
            local_arcface_index=local_index.arcface_index if local_index else None,
            local_performers_mapping=local_index.mapping if local_index else None,
        )

        # Convert to PerformerMatch format for compatibility
        matches = []
        for candidate in result.matches:
            id_part = candidate.universal_id.split(":", 1)[1] if ":" in candidate.universal_id else candidate.universal_id

            if candidate.universal_id.startswith("local:"):
                # Local-index match: id_part is the local Stash performer
                # id, not a StashDB uuid. Use the real linked stashdb_id if
                # this performer has one (so "already tagged" checks and
                # StashBox linking still work for them), otherwise fall
                # back to the local id as the identifier.
                local_info = (self.local_performer_index.mapping.get(id_part, {})
                              if self.local_performer_index else {})
                stashdb_id = local_info.get("stashdb_id") or id_part
                country = None
                image_url = local_info.get("image_url")
                local_performer_id = id_part
            else:
                stashdb_id = id_part
                country = self.performers.get(candidate.universal_id, {}).get("country")
                image_url = self.performers.get(candidate.universal_id, {}).get("image_url")
                local_performer_id = None

            matches.append(PerformerMatch(
                universal_id=candidate.universal_id,
                stashdb_id=stashdb_id,
                name=candidate.name,
                country=country,
                image_url=image_url,
                facenet_distance=candidate.facenet_distance or 0.0,
                arcface_distance=candidate.arcface_distance or 0.0,
                combined_score=candidate.combined_distance,
                local_performer_id=local_performer_id,
            ))

        return matches, result, embedding

    def recognize_image(
        self,
        image: np.ndarray,
        top_k: int = 5,
        max_distance: float = 1.0,
        min_face_confidence: float = 0.5,
        min_face_size: int = 40,
    ) -> list[RecognitionResult]:
        """
        Detect and recognize all faces in an image.

        Uses batch embedding generation + V2 matching with adaptive health detection.

        Args:
            image: RGB image as numpy array
            top_k: Number of top matches per face
            max_distance: Maximum distance threshold
            min_face_confidence: Minimum face detection confidence
            min_face_size: Minimum face width/height in pixels

        Returns:
            List of RecognitionResult objects, one per detected face
        """
        # Detect faces
        all_faces = self.generator.detect_faces(image, min_confidence=min_face_confidence)

        # Filter small faces
        faces = [f for f in all_faces if f.bbox["w"] >= min_face_size and f.bbox["h"] >= min_face_size]

        if not faces:
            return []

        # Batch generate embeddings for all faces at once
        embeddings = self.generator.get_embeddings_batch([f.image for f in faces])

        # Configure matching
        config = MatchingConfig(
            max_results=top_k,
            max_distance=max_distance,
        )

        # Match each face using pre-computed embeddings
        results = []
        for face, emb in zip(faces, embeddings):
            matches, _, _ = self.recognize_face_v2(face, config, embedding=emb)
            results.append(RecognitionResult(face=face, matches=matches, embedding=emb))

        return results


if __name__ == "__main__":
    # Quick test
    import requests
    from embeddings import load_image

    db_config = DatabaseConfig(data_dir=Path("./data"))
    recognizer = FaceRecognizer(db_config)

    # Test with an image
    test_url = "https://stashdb.org/images/b0aef39d-a1d6-4e58-a136-293f02b84921"
    print(f"\nTesting with {test_url}...")

    response = requests.get(test_url)
    image = load_image(response.content)

    results = recognizer.recognize_image(image)
    print(f"\nFound {len(results)} face(s)")

    for i, result in enumerate(results):
        print(f"\nFace {i+1}: confidence={result.face.confidence:.2f}")
        for j, match in enumerate(result.matches[:3]):
            print(f"  {j+1}. {match.name} (score={match.combined_score:.3f})")
            print(f"     StashDB: {match.stashdb_id}")
