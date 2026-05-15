"""Tests for face alignment helper in embeddings.py."""

import sys
from unittest.mock import Mock

import numpy as np

# Ensure embeddings.py can import insightface symbols in lightweight test envs.
if "insightface" not in sys.modules:
    sys.modules["insightface"] = Mock()
if "insightface.app" not in sys.modules:
    sys.modules["insightface.app"] = Mock()
sys.modules["insightface.app"].FaceAnalysis = Mock()

from embeddings import align_face_with_similarity_transform


def _sample_landmarks() -> np.ndarray:
    return np.array([
        [100.0, 100.0],
        [140.0, 100.0],
        [120.0, 125.0],
        [105.0, 150.0],
        [135.0, 150.0],
    ], dtype=np.float32)


def test_align_face_returns_aligned_image(monkeypatch):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    lmk = _sample_landmarks()
    expected = np.ones((112, 112, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "embeddings.cv2.estimateAffinePartial2D",
        lambda src, dst, method: (np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), None),
    )
    monkeypatch.setattr(
        "embeddings.cv2.warpAffine",
        lambda img, M, size, borderValue=0.0: expected,
    )

    out = align_face_with_similarity_transform(image, lmk, image_size=112)

    assert out is expected
    assert out.shape == (112, 112, 3)


def test_align_face_returns_none_for_invalid_landmarks():
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    assert align_face_with_similarity_transform(image, np.array([]), image_size=112) is None
    assert align_face_with_similarity_transform(image, np.array([[1.0, 2.0]]), image_size=112) is None


def test_align_face_returns_none_when_transform_fails(monkeypatch):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    lmk = _sample_landmarks()

    monkeypatch.setattr(
        "embeddings.cv2.estimateAffinePartial2D",
        lambda src, dst, method: (None, None),
    )

    assert align_face_with_similarity_transform(image, lmk, image_size=112) is None


def test_align_face_returns_none_when_warp_raises(monkeypatch):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    lmk = _sample_landmarks()

    monkeypatch.setattr(
        "embeddings.cv2.estimateAffinePartial2D",
        lambda src, dst, method: (np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), None),
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("warp failed")

    monkeypatch.setattr("embeddings.cv2.warpAffine", _boom)

    assert align_face_with_similarity_transform(image, lmk, image_size=112) is None
