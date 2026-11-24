"""A FAKE face recognition service for the tests. It lives ONLY under tests/ and must never be imported by app/.

It recognises nothing: tests register, per image, the exact vectors it should "find". Images are identified by
the SHA-256 of the JPEG bytes the application hands to extract_faces() (for stored photos that is CasePhoto.sha256).
An image nobody registered has no faces, so it behaves like a picture without a face."""
import hashlib

import numpy as np

from app.recognition import DetectedFace, FaceBox, ModelInfo, RecognitionService, RecognitionUnavailableError


class FakeRecognitionService(RecognitionService):
    def __init__(self, name: str = "fake-face-model", version: str = "1", dimension: int = 4):
        self.name = name
        self.version = version
        self.dimension = dimension
        self.unavailable = False             # True: every call raises RecognitionUnavailableError
        self.fail_on_call: int | None = None  # the n-th extract_faces call raises RecognitionUnavailableError
        self.crash_on_call: int | None = None  # the n-th extract_faces call raises a RuntimeError
        self.calls: list[str] = []           # the SHA-256 of every image extract_faces() was given
        self._faces: dict[str, list[DetectedFace]] = {}

    @property
    def is_configured(self) -> bool:
        return True

    def set_faces(self, image: bytes | str, *vectors) -> None:
        """Register what the 'model' finds in an image: no vectors = no face, one = one face, several = several."""
        key = image if isinstance(image, str) else hashlib.sha256(image).hexdigest()
        self._faces[key] = [
            DetectedFace(
                embedding=np.asarray(vector, dtype=np.float32), box=FaceBox(10, 10, 90, 90), detection_confidence=0.99
            )
            for vector in vectors
        ]

    def model_info(self) -> ModelInfo:
        if self.unavailable:
            raise RecognitionUnavailableError("The fake recognition service is switched off")
        return ModelInfo(name=self.name, version=self.version, dimension=self.dimension)

    def extract_faces(self, image_jpeg_bytes: bytes) -> list[DetectedFace]:
        if self.unavailable:
            raise RecognitionUnavailableError("The fake recognition service is switched off")
        self.calls.append(hashlib.sha256(image_jpeg_bytes).hexdigest())
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RecognitionUnavailableError("The fake recognition service failed on purpose")
        if self.crash_on_call is not None and len(self.calls) == self.crash_on_call:
            raise RuntimeError("The fake recognition service crashed on purpose")
        return list(self._faces.get(self.calls[-1], []))
