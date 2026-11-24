"""The recognition contract: the placeholder, the one-face rule and the embedding checks."""
import numpy as np
import pytest

from app.recognition import (
    DetectedFace,
    FaceBox,
    InvalidEmbeddingError,
    ModelInfo,
    MultipleFacesError,
    NoFaceDetectedError,
    RecognitionService,
    RecognitionUnavailableError,
    UnconfiguredRecognitionService,
    extract_single_face,
    get_recognition_service,
)
from fake_recognition import FakeRecognitionService

IMAGE = b"not a real image: the fake only looks at the hash"


def test_the_unconfigured_service_reports_it_is_not_configured_and_never_returns_anything():
    service = UnconfiguredRecognitionService()
    assert service.is_configured is False
    with pytest.raises(RecognitionUnavailableError) as info:
        service.model_info()
    assert str(info.value) == "No face recognition model is configured"
    with pytest.raises(RecognitionUnavailableError):
        service.extract_faces(IMAGE)
    with pytest.raises(RecognitionUnavailableError):
        extract_single_face(service, IMAGE)


def test_the_default_backend_is_the_unconfigured_service():
    service = get_recognition_service()
    assert isinstance(service, UnconfiguredRecognitionService)
    assert service is get_recognition_service()


def test_the_only_recognition_service_inside_app_is_the_unconfigured_placeholder():
    inside_app = {cls for cls in RecognitionService.__subclasses__() if cls.__module__.startswith("app.")}
    from app.recognition.facenet import FaceNetRecognitionService
    assert inside_app == {UnconfiguredRecognitionService, FaceNetRecognitionService}
    assert FakeRecognitionService.__module__ == "fake_recognition"          # the fake lives under tests/


def test_exactly_one_face_is_required():
    fake = FakeRecognitionService()
    with pytest.raises(NoFaceDetectedError):                                # nothing registered: no face
        extract_single_face(fake, IMAGE)
    fake.set_faces(IMAGE, [1, 0, 0, 0], [0, 1, 0, 0])
    with pytest.raises(MultipleFacesError):
        extract_single_face(fake, IMAGE)
    fake.set_faces(IMAGE, [3, 4, 0, 0])
    face = extract_single_face(fake, IMAGE)
    assert isinstance(face, DetectedFace)
    assert face.embedding.dtype == np.float32
    assert face.embedding.tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0], abs=1e-6)     # returned L2-normalised


def test_an_unusable_embedding_is_a_model_fault_not_a_missing_face():
    fake = FakeRecognitionService(dimension=4)
    for bad in ([1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0], [float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0]):
        fake.set_faces(IMAGE, bad)
        with pytest.raises(InvalidEmbeddingError):
            extract_single_face(fake, IMAGE)
    assert issubclass(InvalidEmbeddingError, RecognitionUnavailableError)        # so it answers 503


def test_the_contract_types_are_plain_data():
    info = ModelInfo(name="example", version="1", dimension=4)
    assert (info.name, info.version, info.dimension) == ("example", "1", 4)
    box = FaceBox(left=1, top=2, right=3, bottom=4)
    assert (box.left, box.top, box.right, box.bottom) == (1, 2, 3, 4)
