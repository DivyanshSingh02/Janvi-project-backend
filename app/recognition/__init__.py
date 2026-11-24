from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import get_settings
from app.recognition.base import (
    DetectedFace,
    FaceBox,
    InvalidEmbeddingError,
    ModelInfo,
    MultipleFacesError,
    NoFaceDetectedError,
    RecognitionError,
    RecognitionService,
    RecognitionUnavailableError,
    extract_single_face,
)
from app.recognition.facenet import FaceNetRecognitionService  # imports no torch: that happens on first use
from app.recognition.unconfigured import UnconfiguredRecognitionService

__all__ = [
    "DetectedFace",
    "FaceNetRecognitionService",
    "FaceBox",
    "InvalidEmbeddingError",
    "ModelInfo",
    "MultipleFacesError",
    "NoFaceDetectedError",
    "RecognitionDep",
    "RecognitionError",
    "RecognitionService",
    "RecognitionUnavailableError",
    "UnconfiguredRecognitionService",
    "extract_single_face",
    "get_recognition_service",
]


@lru_cache
def _build_service() -> RecognitionService:
    """Builds (never loads) the configured service. A FaceNet service only remembers its settings here: the models
    are loaded on first use, so a missing weights file or a missing PyTorch can never break application start-up."""
    backend = get_settings().recognition_backend
    if backend == "none":
        return UnconfiguredRecognitionService()
    if backend == "facenet":
        return FaceNetRecognitionService()
    raise RuntimeError("Unsupported RECOGNITION_BACKEND")          # unreachable: the setting only allows these two


def get_recognition_service() -> RecognitionService:
    """FastAPI dependency. Tests replace it with app.dependency_overrides[get_recognition_service]."""
    return _build_service()


RecognitionDep = Annotated[RecognitionService, Depends(get_recognition_service)]
