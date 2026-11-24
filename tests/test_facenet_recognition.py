"""FaceNet service tests that need NO torch: the detector, the network and (where needed) the torch module are fakes.
Nothing here says anything about real recognition quality; it checks the rules around the model."""
import importlib.metadata
import logging
import sys
import threading
import types
from contextlib import nullcontext
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.recognition import (
    FaceNetRecognitionService,
    InvalidEmbeddingError,
    MultipleFacesError,
    NoFaceDetectedError,
    RecognitionUnavailableError,
    extract_single_face,
)
from app.recognition import facenet
from app.recognition.base import ModelInfo
from app.recognition.facenet import MAX_MODEL_VERSION_LENGTH, WEIGHTS_FILENAME, _Loaded

BIG = 1_000_000          # the smallest file size the service treats as a plausible weights file


def _log_text(caplog) -> str:
    """The log messages AND the structured extra fields (caplog.text alone omits the extras)."""
    return caplog.text + "\n" + "\n".join(str(record.__dict__) for record in caplog.records)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
        self.shape = self.array.shape

    def __getitem__(self, item):
        return FakeTensor(self.array[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


class FakeDetector:
    def __init__(self, boxes, probabilities):
        self.boxes = boxes
        self.probabilities = probabilities
        self.min_face_size = None
        self.detected_size = None
        self.detected_mode = None
        self.extracted_boxes = None
        self.extracted_size = None

    def detect(self, image):
        self.detected_size = image.size
        self.detected_mode = image.mode
        return self.boxes, self.probabilities

    def extract(self, image, boxes, save_path):
        self.extracted_boxes = boxes
        self.extracted_size = image.size
        return FakeTensor(np.zeros((len(boxes), 3, 160, 160), dtype=np.float32))


class FakeResnet:
    def __init__(self, output=None):
        self.output = output
        self.batches = []

    def __call__(self, crops):
        self.batches.append(len(crops.array))
        if self.output is not None:
            return FakeTensor(self.output)
        return FakeTensor(np.ones((len(crops.array), 512), dtype=np.float32))


def _jpeg(size=(200, 160), exif=None, mode="RGB", fmt="JPEG"):
    stream = BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.new(mode, size, "white").save(stream, format=fmt, **kwargs)
    return stream.getvalue()


def _loaded(version="v"):
    return _Loaded(FakeTorch(), FakeDetector(None, None), FakeResnet(), ModelInfo("facenet-vggface2", version, 512))


def _service(monkeypatch, boxes, probabilities, output=None, version="facenet-pytorch-test-vgg2-abc-p1", **kwargs):
    service = FaceNetRecognitionService(
        model_dir=kwargs.pop("model_dir", "unused"),
        min_face_size=kwargs.pop("min_face_size", 40),
        min_detection_probability=kwargs.pop("min_detection_probability", 0.90),
    )
    detector = FakeDetector(boxes, probabilities)
    resnet = FakeResnet(output)
    loaded = _Loaded(FakeTorch(), detector, resnet, ModelInfo("facenet-vggface2", version, 512))
    monkeypatch.setattr(service, "_preflight", lambda: None)
    monkeypatch.setattr(service, "_load_models", lambda: loaded)
    return service, detector, resnet


def _arr(*rows):
    return np.array(rows, dtype=np.float32)


# ------------------------------------------------------------------ quality gates and the faces returned
def test_quality_gates_reject_small_and_low_probability_faces_and_return_every_other_face(monkeypatch):
    boxes = _arr([0, 0, 10, 10], [10, 10, 80, 80], [20, 20, 100, 100], [110, 20, 180, 90])
    probabilities = np.array([0.99, 0.50, 0.99, 0.95], dtype=np.float32)
    service, detector, _ = _service(monkeypatch, boxes, probabilities)
    faces = service.extract_faces(_jpeg())
    assert len(faces) == 2                                          # several faces are returned, not an error
    assert detector.extracted_boxes.shape == (2, 4)
    assert all(face.embedding.shape == (512,) and face.embedding.dtype == np.float32 for face in faces)
    assert all(float(np.linalg.norm(face.embedding)) == pytest.approx(1.0, abs=1e-6) for face in faces)
    assert [face.detection_confidence for face in faces] == pytest.approx([0.99, 0.95])
    corners = [(f.box.left, f.box.top, f.box.right, f.box.bottom) for f in faces]
    assert corners == [(20, 20, 100, 100), (110, 20, 180, 90)]


def test_the_size_gate_uses_the_visible_part_of_a_box_that_sticks_out_of_the_image(monkeypatch):
    # 200x160 image: the box is 80 wide but only 30 px of it are inside the image.
    service, _, _ = _service(monkeypatch, _arr([170, 20, 250, 100]), np.array([0.99]))
    assert service.extract_faces(_jpeg()) == []


def test_the_size_gate_is_inclusive_at_the_minimum(monkeypatch):
    service, _, _ = _service(monkeypatch, _arr([20, 20, 60, 60], [100, 20, 139, 59]), np.array([0.99, 0.99]))
    faces = service.extract_faces(_jpeg())
    assert [(f.box.left, f.box.right) for f in faces] == [(20, 60)]        # 40 px passes, 39 px does not


def test_the_probability_gate_is_inclusive(monkeypatch):
    service, _, _ = _service(monkeypatch, _arr([20, 20, 100, 100]), np.array([0.90]), min_detection_probability=0.90)
    assert len(service.extract_faces(_jpeg())) == 1


def test_no_face_gives_an_empty_list_and_the_one_face_rule_is_left_to_extract_single_face(monkeypatch):
    service, _, _ = _service(monkeypatch, None, None)
    assert service.extract_faces(_jpeg()) == []
    with pytest.raises(NoFaceDetectedError):
        extract_single_face(service, _jpeg())
    two, _, _ = _service(monkeypatch, _arr([20, 20, 100, 100], [110, 20, 180, 90]), np.array([0.99, 0.99]))
    with pytest.raises(MultipleFacesError):
        extract_single_face(two, _jpeg())


def test_a_second_small_face_does_not_break_the_one_face_rule(monkeypatch):
    service, _, _ = _service(monkeypatch, _arr([20, 20, 100, 100], [150, 120, 165, 135]), np.array([0.99, 0.99]))
    assert extract_single_face(service, _jpeg()).box.right == 100


def test_many_faces_are_embedded_in_batches(monkeypatch):
    boxes = _arr(*[[i * 12, 10, i * 12 + 50, 60] for i in range(3)])
    service, _, resnet = _service(monkeypatch, boxes, np.array([0.99] * 3))
    assert len(service.extract_faces(_jpeg(size=(400, 200)))) == 3
    assert resnet.batches == [3]
    many = _arr(*[[i * 5, 10, i * 5 + 50, 60] for i in range(20)])
    service2, _, resnet2 = _service(monkeypatch, many, np.array([0.99] * 20))
    assert len(service2.extract_faces(_jpeg(size=(400, 200)))) == 20
    assert resnet2.batches == [16, 4]


# ------------------------------------------------------------------ images: Pillow, EXIF, RGB, size limits
def test_exif_orientation_is_applied_before_detection(monkeypatch):
    service, detector, _ = _service(monkeypatch, _arr([10, 10, 100, 100]), np.array([0.99]))
    exif = Image.Exif()
    exif[274] = 6                                                   # rotate 90 degrees
    assert len(service.extract_faces(_jpeg(exif=exif))) == 1
    assert detector.detected_size == (160, 200)


def test_images_are_converted_to_rgb(monkeypatch):
    service, detector, _ = _service(monkeypatch, _arr([10, 10, 100, 100]), np.array([0.99]))
    service.extract_faces(_jpeg(mode="RGBA", fmt="PNG"))
    assert detector.detected_mode == "RGB"
    service.extract_faces(_jpeg(mode="L", fmt="PNG"))
    assert detector.detected_mode == "RGB"


def test_an_undecodable_image_has_no_face_and_the_log_has_only_the_error_type(monkeypatch, caplog):
    service, detector, _ = _service(monkeypatch, _arr([10, 10, 100, 100]), np.array([0.99]))
    with caplog.at_level(logging.WARNING):
        assert service.extract_faces(b"CANARY_NOT_AN_IMAGE") == []
        assert service.extract_faces(b"") == []
    assert detector.detected_size is None
    assert "CANARY_NOT_AN_IMAGE" not in _log_text(caplog)
    assert "UnidentifiedImageError" in _log_text(caplog)


def test_a_decompression_bomb_has_no_face(monkeypatch):
    service, detector, _ = _service(monkeypatch, _arr([10, 10, 100, 100]), np.array([0.99]))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    assert service.extract_faces(_jpeg(size=(200, 160))) == []
    assert detector.detected_size is None


def test_a_large_image_is_shrunk_for_detection_only_and_boxes_are_mapped_back(monkeypatch):
    # 4000x3000 is shrunk to 2000x1500 (factor 0.5): a 120 px face is 60 px there, a 38 px face is 19 px.
    boxes = _arr([100, 100, 160, 160], [400, 400, 419, 419])
    service, detector, _ = _service(monkeypatch, boxes, np.array([0.99, 0.99]))
    faces = service.extract_faces(_jpeg(size=(4000, 3000)))
    assert detector.detected_size == (2000, 1500)
    assert detector.min_face_size == 20                              # the 40 px gate, scaled by 0.5
    assert detector.extracted_size == (4000, 3000)                   # crops come from the ORIGINAL image
    assert len(faces) == 1                                           # the 38 px (original) face fails the 40 px gate
    assert (faces[0].box.left, faces[0].box.top, faces[0].box.right, faces[0].box.bottom) == (200, 200, 320, 320)
    assert detector.extracted_boxes.tolist() == [[200.0, 200.0, 320.0, 320.0]]


def test_the_detector_minimum_never_goes_below_the_mtcnn_floor(monkeypatch):
    service, detector, _ = _service(monkeypatch, None, None, min_face_size=12)
    service.extract_faces(_jpeg(size=(4000, 3000)))
    assert detector.min_face_size == 12


# ------------------------------------------------------------------ error mapping
def test_unusable_embeddings_are_a_model_fault(monkeypatch):
    box, prob = _arr([10, 10, 100, 100]), np.array([0.99])
    for bad in (np.zeros((1, 512)), np.full((1, 512), np.nan), np.ones((1, 511)), np.ones((2, 512))):
        service, _, _ = _service(monkeypatch, box, prob, output=bad)
        with pytest.raises(InvalidEmbeddingError):
            service.extract_faces(_jpeg())
    assert issubclass(InvalidEmbeddingError, RecognitionUnavailableError)


def test_an_unexpected_detector_output_is_unavailable(monkeypatch):
    service, _, _ = _service(monkeypatch, _arr([10, 10, 100, 100]), np.array([0.9, 0.9]))
    with pytest.raises(RecognitionUnavailableError, match="unexpected output"):
        service.extract_faces(_jpeg())


def test_a_crash_inside_the_model_is_unavailable_and_leaks_nothing(monkeypatch, caplog):
    service, detector, _ = _service(monkeypatch, None, None)

    def explode(image):
        raise RuntimeError("CANARY_PATH C:/secret/dir/CANARY_FILE.jpg")

    detector.detect = explode
    with caplog.at_level(logging.INFO):
        with pytest.raises(RecognitionUnavailableError) as info:
            service.extract_faces(_jpeg())
    assert str(info.value) == "FaceNet inference failed"
    assert "CANARY" not in _log_text(caplog) and "CANARY" not in repr(info.value)
    assert "RuntimeError" in _log_text(caplog)                        # the type alone is logged


def test_out_of_memory_is_unavailable(monkeypatch):
    service, detector, _ = _service(monkeypatch, None, None)

    def oom(image):
        raise MemoryError

    detector.detect = oom
    with pytest.raises(RecognitionUnavailableError, match="inference failed"):
        service.extract_faces(_jpeg())


# ------------------------------------------------------------------ model_info, version, readiness
def test_model_info_is_stable_and_the_models_load_once(monkeypatch):
    service, _, _ = _service(monkeypatch, None, None)
    calls = []
    original = service._load_models
    monkeypatch.setattr(service, "_load_models", lambda: calls.append(1) or original())
    first, second = service.model_info(), service.model_info()
    assert first == second and first.name == "facenet-vggface2" and first.dimension == 512
    assert service.is_configured is True and service.is_configured is True
    service.extract_faces(_jpeg())
    assert len(calls) == 1                                            # health checks and requests never reload


def test_the_model_version_includes_the_package_version_and_fits_the_database_column():
    digest = "0123456789abcdef" * 4
    version = facenet.build_model_version("2.6.0", digest)
    assert version.startswith("facenet-pytorch-2.6.0-")
    assert len(version) <= MAX_MODEL_VERSION_LENGTH
    assert version == facenet.build_model_version("2.6.0", digest)
    assert facenet.build_model_version("2.6.0", "a" * 64) != facenet.build_model_version("2.6.0", "b" * 64)
    for odd in ("", "x" * 200, "2.6.0\n+evil path/..", "9.99.999.dev12345+local"):
        assert 0 < len(facenet.build_model_version(odd, "f" * 64)) <= MAX_MODEL_VERSION_LENGTH


def test_missing_weights_are_a_clear_unavailable_error_that_needs_no_restart_to_fix(tmp_path, monkeypatch):
    service = FaceNetRecognitionService(model_dir=tmp_path)
    for _ in range(2):
        with pytest.raises(RecognitionUnavailableError, match="weights file is missing"):
            service.model_info()
        assert service.is_configured is False
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"\0" * BIG)             # the operator downloads the file ...
    monkeypatch.setattr(facenet, "_package_available", lambda name: True)
    monkeypatch.setattr(service, "_load_models", lambda: _loaded())
    assert service.is_configured is True                               # ... and the same process picks it up


def test_a_tiny_or_directory_weights_file_is_refused_without_loading(tmp_path, monkeypatch):
    service = FaceNetRecognitionService(model_dir=tmp_path)

    def must_not_load():
        pytest.fail("must not load")

    monkeypatch.setattr(service, "_load_models", must_not_load)
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"placeholder")
    with pytest.raises(RecognitionUnavailableError, match="empty, truncated or unreadable"):
        service.model_info()
    (tmp_path / WEIGHTS_FILENAME).unlink()
    (tmp_path / WEIGHTS_FILENAME).mkdir()
    with pytest.raises(RecognitionUnavailableError, match="weights file is missing"):
        service.model_info()


def test_missing_torch_install_is_a_clear_unavailable_error(tmp_path, monkeypatch):
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"\0" * BIG)
    monkeypatch.setitem(sys.modules, "torch", None)                    # "import torch" now fails
    service = FaceNetRecognitionService(model_dir=tmp_path)
    with pytest.raises(RecognitionUnavailableError, match="PyTorch or facenet-pytorch is not installed"):
        service.model_info()
    assert service.is_configured is False


def test_an_import_failure_after_the_cheap_check_is_reported_the_same_way(tmp_path, monkeypatch):
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"\0" * BIG)
    monkeypatch.setattr(facenet, "_package_available", lambda name: True)
    monkeypatch.setitem(sys.modules, "torch", None)
    service = FaceNetRecognitionService(model_dir=tmp_path)
    with pytest.raises(RecognitionUnavailableError, match="not installed"):
        service.model_info()


def test_a_failed_heavy_load_is_retried_only_after_the_cool_down(tmp_path, monkeypatch):
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"\0" * BIG)
    monkeypatch.setattr(facenet, "_package_available", lambda name: True)
    now = [100.0]
    service = FaceNetRecognitionService(model_dir=tmp_path, clock=lambda: now[0])
    attempts = []

    def failing():
        attempts.append(1)
        raise ValueError("CANARY third-party text")

    monkeypatch.setattr(service, "_load_models", failing)
    for _ in range(3):                                                 # a health-check loop
        with pytest.raises(RecognitionUnavailableError) as info:
            service.model_info()
        assert "CANARY" not in str(info.value)
    assert len(attempts) == 1
    now[0] += facenet.RETRY_AFTER_FAILED_LOAD_SECONDS + 1
    with pytest.raises(RecognitionUnavailableError):
        service.model_info()
    assert len(attempts) == 2
    monkeypatch.setattr(service, "_load_models", lambda: _loaded())
    now[0] += facenet.RETRY_AFTER_FAILED_LOAD_SECONDS + 1
    assert service.model_info().version == "v"


def test_concurrent_first_calls_load_the_models_exactly_once(monkeypatch):
    service, _, _ = _service(monkeypatch, None, None)
    loaded = service._load_models()
    calls = []
    release = threading.Event()

    def slow_load():
        calls.append(1)
        release.wait(5)
        return loaded

    monkeypatch.setattr(service, "_load_models", slow_load)
    results = []
    threads = [threading.Thread(target=lambda: results.append(service.model_info())) for _ in range(6)]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(10)
    assert len(calls) == 1 and len(results) == 6


def test_readiness_never_waits_for_a_load_that_is_in_progress(monkeypatch):
    service, _, _ = _service(monkeypatch, None, None)
    loaded = service._load_models()
    started, release = threading.Event(), threading.Event()

    def slow_load():
        started.set()
        release.wait(5)
        return loaded

    monkeypatch.setattr(service, "_load_models", slow_load)
    loader = threading.Thread(target=service.model_info)
    loader.start()
    assert started.wait(5)
    assert service.is_configured is False                              # returns at once: "not ready yet"
    release.set()
    loader.join(10)
    assert service.is_configured is True


# ------------------------------------------------------------------ the real loading code, with fake torch modules
class _FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, flag):
        self.requires_grad = flag


class _FakeNetwork:
    created = []

    def __init__(self, reject=False, **kwargs):
        self.kwargs = kwargs
        self.reject = reject
        self.loaded_state = None
        self.strict = None
        self.evaluating = False
        self.params = [_FakeParameter(), _FakeParameter()]
        _FakeNetwork.created.append(self)

    def load_state_dict(self, state, strict=True):
        if self.reject:
            raise RuntimeError("CANARY mismatch in C:/secret/path")
        self.loaded_state, self.strict = dict(state), strict

    def eval(self):
        self.evaluating = True
        return self

    def parameters(self):
        return self.params

    def __call__(self, batch):
        return FakeTensor(np.ones((len(batch.array), 512), dtype=np.float32))


class _FakeMtcnn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_stack(monkeypatch, checkpoint, tmp_path, reject=False):
    torch = types.ModuleType("torch")
    torch.load_args = None

    def load(path, map_location=None, weights_only=None):
        torch.load_args = (path, map_location, weights_only)
        return checkpoint

    def device(name):
        return name

    def zeros(shape):
        return FakeTensor(np.zeros(shape))

    torch.load, torch.device, torch.no_grad, torch.zeros = load, device, nullcontext, zeros
    module = types.ModuleType("facenet_pytorch")
    module.MTCNN = _FakeMtcnn

    def network_factory(**kwargs):
        return _FakeNetwork(reject=reject, **kwargs)

    module.InceptionResnetV1 = network_factory
    module.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "facenet_pytorch", module)
    monkeypatch.setattr(facenet, "_package_available", lambda name: True)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.6.0")
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"\x01" * BIG)
    return torch


def test_loading_drops_the_classifier_head_loads_strictly_offline_and_builds_the_detector(tmp_path, monkeypatch):
    _FakeNetwork.created.clear()
    checkpoint = {"conv2d_1a.conv.weight": 1, "last_bn.bias": 2, "logits.weight": 3, "logits.bias": 4}
    torch = _fake_stack(monkeypatch, checkpoint, tmp_path)
    service = FaceNetRecognitionService(model_dir=tmp_path, min_face_size=50, min_detection_probability=0.8)
    info = service.model_info()
    assert info.name == "facenet-vggface2" and info.dimension == 512
    assert info.version.startswith("facenet-pytorch-2.6.0-vgg2-") and len(info.version) <= MAX_MODEL_VERSION_LENGTH
    path, map_location, weights_only = torch.load_args
    assert path == tmp_path / WEIGHTS_FILENAME and map_location == "cpu" and weights_only is True
    network = _FakeNetwork.created[-1]
    assert network.kwargs == {"pretrained": None, "classify": False}   # pretrained=None: nothing is downloaded
    assert network.loaded_state == {"conv2d_1a.conv.weight": 1, "last_bn.bias": 2}
    assert network.strict is True and network.evaluating is True
    assert all(parameter.requires_grad is False for parameter in network.params)
    detector = service._loaded.detector
    assert detector.kwargs["min_face_size"] == 50 and detector.kwargs["keep_all"] is True
    assert detector.kwargs["image_size"] == 160 and detector.kwargs["device"] == "cpu"
    assert service.is_configured is True


def test_the_version_follows_the_weights_file(tmp_path, monkeypatch):
    _fake_stack(monkeypatch, {"a": 1}, tmp_path)
    first = FaceNetRecognitionService(model_dir=tmp_path).model_info().version
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"\x02" * BIG)
    assert FaceNetRecognitionService(model_dir=tmp_path).model_info().version != first


def test_weights_that_do_not_match_the_network_are_refused_without_leaking_text(tmp_path, monkeypatch, caplog):
    _fake_stack(monkeypatch, {"a": 1}, tmp_path, reject=True)
    service = FaceNetRecognitionService(model_dir=tmp_path)
    with caplog.at_level(logging.INFO):
        with pytest.raises(RecognitionUnavailableError, match="not a usable VGGFace2"):
            service.model_info()
    assert "CANARY" not in _log_text(caplog) and "secret" not in _log_text(caplog)
    assert "weights_invalid" in _log_text(caplog)
    assert service.is_configured is False


def test_a_checkpoint_that_is_not_a_state_dictionary_is_refused(tmp_path, monkeypatch):
    _fake_stack(monkeypatch, ["not", "a", "dict"], tmp_path)
    with pytest.raises(RecognitionUnavailableError, match="not a usable VGGFace2"):
        FaceNetRecognitionService(model_dir=tmp_path).model_info()


def test_clean_state_dict_unwraps_and_drops_only_the_classifier_head():
    assert facenet.clean_state_dict({"state_dict": {"a": 1, "logits.weight": 2}}) == {"a": 1}
    cleaned = facenet.clean_state_dict({"a": 1, "logits.bias": 2, "last_linear.weight": 3})
    assert cleaned == {"a": 1, "last_linear.weight": 3}
    for bad in (None, [], {}, {1: 2}, "text"):
        with pytest.raises(ValueError):
            facenet.clean_state_dict(bad)


# ------------------------------------------------------------------ construction and the factory
def test_constructor_arguments_are_validated():
    for kwargs in (
        {"min_face_size": 11}, {"min_face_size": 1001}, {"min_face_size": 40.5}, {"min_face_size": True},
        {"min_detection_probability": 0}, {"min_detection_probability": 1.5},
        {"min_detection_probability": float("nan")}, {"model_dir": "  "},
    ):
        base = {"model_dir": "x", "min_face_size": 40, "min_detection_probability": 0.9}
        base.update(kwargs)
        with pytest.raises(ValueError):
            FaceNetRecognitionService(**base)


def test_factory_selects_facenet_without_loading_or_touching_the_disk(monkeypatch, tmp_path):
    from app import recognition

    monkeypatch.setattr(
        recognition, "get_settings",
        lambda: types.SimpleNamespace(
            recognition_backend="facenet", recognition_model_dir=str(tmp_path / "does-not-exist"),
            recognition_min_face_size=40, recognition_min_detection_probability=0.90,
        ),
    )
    recognition._build_service.cache_clear()
    try:
        service = recognition.get_recognition_service()
        assert isinstance(service, FaceNetRecognitionService)
        assert service._loaded is None
        assert not (tmp_path / "does-not-exist").exists()
    finally:
        recognition._build_service.cache_clear()


# ------------------------------------------------------------------ the API with the real service and no weights
@pytest.fixture()
def facenet_without_weights(monkeypatch, tmp_path):
    """RECOGNITION_BACKEND=facenet with an empty model folder, built the way a real start-up builds it."""
    from app import recognition
    from app.core.config import get_settings

    monkeypatch.setenv("RECOGNITION_BACKEND", "facenet")
    monkeypatch.setenv("RECOGNITION_MODEL_DIR", str(tmp_path / "CANARY_MODEL_DIR"))
    get_settings.cache_clear()
    recognition._build_service.cache_clear()
    yield tmp_path
    monkeypatch.undo()
    get_settings.cache_clear()
    recognition._build_service.cache_clear()


def test_selecting_facenet_without_weights_does_not_break_startup_and_health_says_unconfigured(
    client, facenet_without_weights
):
    from app.main import create_app

    create_app()                                                       # start-up: no torch, no weights, no crash
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["recognition"] == "unconfigured"


def test_facenet_without_weights_answers_503_for_embeddings_and_search(
    client, make_user, auth_headers, make_case, make_photos, make_image_bytes, facenet_without_weights, caplog
):
    admin = make_user(email="noweights-admin@example.test", role="admin")
    owner = make_user(email="noweights-owner@example.test")
    case = make_case(owner)
    make_photos(owner, case, 1)
    with caplog.at_level(logging.INFO):
        approved = client.post(f"/admin/cases/{case.id}/approve", headers=auth_headers(admin))
        assert approved.status_code == 200                              # an approval is never undone by recognition
        assert approved.json()["embeddings"] == {"generated": 0, "pending": 1, "failed": 0}
        enrol = client.post(f"/admin/cases/{case.id}/embeddings", headers=auth_headers(admin))
        search = client.post(
            "/search", headers=auth_headers(admin),
            files={"file": ("q.png", make_image_bytes("PNG", size=(220, 220), seed=5), "image/png")},
        )
    for response in (enrol, search):
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "recognition_unavailable"
        assert response.json()["error"]["message"] == "The face recognition service is temporarily unavailable"
        assert "CANARY_MODEL_DIR" not in response.text
    assert "CANARY_MODEL_DIR" not in _log_text(caplog)
    assert "weights_missing" in _log_text(caplog)                        # the operator can see why, by code
