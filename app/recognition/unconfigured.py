from app.recognition.base import DetectedFace, ModelInfo, RecognitionService, RecognitionUnavailableError

NOT_CONFIGURED_MESSAGE = "No face recognition model is configured"


class UnconfiguredRecognitionService(RecognitionService):
    """Stands in while no model exists. Every method raises: it never returns an embedding, a face or a match."""

    @property
    def is_configured(self) -> bool:
        return False

    def model_info(self) -> ModelInfo:
        raise RecognitionUnavailableError(NOT_CONFIGURED_MESSAGE)

    def extract_faces(self, image_jpeg_bytes: bytes) -> list[DetectedFace]:
        raise RecognitionUnavailableError(NOT_CONFIGURED_MESSAGE)
