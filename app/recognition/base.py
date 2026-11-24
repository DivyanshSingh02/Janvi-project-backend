"""The face recognition contract. The rest of the application only talks to `RecognitionService`, so a real
model can be plugged in later without touching routes, services or the database (see docs/recognition.md).

Rules for every implementation:
- NEVER invent an embedding, a face or a match. If the model cannot do its job, raise RecognitionUnavailableError.
- Exactly ONE face per image is what the application accepts: use extract_single_face() to enforce it.
- Messages never contain image data, file names or personal data.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.recognition.vectors import InvalidVectorError, normalise


class RecognitionError(Exception):
    """Base class for every recognition failure."""


class RecognitionUnavailableError(RecognitionError):
    """No usable recognition model (HTTP 503). Nothing may be written because of it."""


class InvalidEmbeddingError(RecognitionUnavailableError):
    """The model returned something that is not a usable embedding (wrong size, NaN, all zeros). That is a
    fault of the model, not of the photo, so it is also a 503 and never marks a photo as failed."""


class NoFaceDetectedError(RecognitionError):
    """The image contains no face."""


class MultipleFacesError(RecognitionError):
    """The image contains more than one face."""


@dataclass(frozen=True)
class ModelInfo:
    name: str          # for example "example-face-model"
    version: str       # changes whenever the vectors change, so old and new vectors are never compared
    dimension: int     # length of every embedding


@dataclass(frozen=True)
class FaceBox:
    """Where the face is, in pixels of the image that was analysed."""

    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray        # float32, shape (dimension,)
    box: FaceBox
    detection_confidence: float  # how sure the DETECTOR is that this is a face (not a similarity, not identity)


class RecognitionService(ABC):
    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True only when this implementation can currently perform recognition."""

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Raises RecognitionUnavailableError when not configured."""

    @abstractmethod
    def extract_faces(self, image_jpeg_bytes: bytes) -> list[DetectedFace]:
        """All faces passing implementation quality gates (empty when none). Raises RecognitionUnavailableError
        when the implementation cannot perform inference."""


def extract_single_face(service: RecognitionService, image_jpeg_bytes: bytes) -> DetectedFace:
    """The ONE face of the image, checked: exactly one face, and an embedding of the model's dimension without
    NaN, inf or all zeros. Raises NoFaceDetectedError, MultipleFacesError or RecognitionUnavailableError
    (InvalidEmbeddingError for an unusable embedding)."""
    info = service.model_info()
    faces = service.extract_faces(image_jpeg_bytes)
    if not faces:
        raise NoFaceDetectedError
    if len(faces) > 1:
        raise MultipleFacesError
    face = faces[0]
    try:
        embedding = normalise(face.embedding)
    except InvalidVectorError:
        raise InvalidEmbeddingError("The recognition model returned an unusable embedding") from None
    if embedding.shape != (info.dimension,):
        raise InvalidEmbeddingError("The recognition model returned an embedding of the wrong size")
    return DetectedFace(embedding=embedding, box=face.box, detection_confidence=face.detection_confidence)
