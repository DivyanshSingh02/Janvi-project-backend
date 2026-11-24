"""TORCH-ONLY tests: skipped (one by one) when torch / facenet-pytorch are not installed.

They use a RANDOMLY initialised network saved in the same layout as the real VGGFace2 checkpoint (including its
8631-class `logits.*` head). They check the loading mechanics against the real libraries; they say NOTHING about
recognition quality, and no real weights or real faces are needed or used."""
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.recognition import FaceNetRecognitionService, RecognitionUnavailableError
from app.recognition import facenet
from app.recognition.facenet import WEIGHTS_FILENAME


@pytest.fixture(scope="module")
def stack():
    torch = pytest.importorskip("torch")
    facenet_pytorch = pytest.importorskip("facenet_pytorch")
    return torch, facenet_pytorch


@pytest.fixture(scope="module")
def random_weights_dir(stack):
    """A folder with facenet-vggface2.pt holding RANDOM weights in the real checkpoint's layout."""
    torch, facenet_pytorch = stack
    directory = Path(tempfile.mkdtemp(prefix="missing_person_random_weights_"))
    torch.manual_seed(0)
    classifier = facenet_pytorch.InceptionResnetV1(pretrained=None, classify=True, num_classes=8631)
    state = classifier.state_dict()
    assert any(key.startswith("logits.") for key in state)
    torch.save(state, directory / WEIGHTS_FILENAME)
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


def test_the_real_loading_code_accepts_a_checkpoint_with_a_classifier_head(random_weights_dir):
    service = FaceNetRecognitionService(model_dir=random_weights_dir)
    info = service.model_info()
    assert info.name == "facenet-vggface2" and info.dimension == 512
    assert info.version.startswith("facenet-pytorch-") and len(info.version) <= facenet.MAX_MODEL_VERSION_LENGTH
    assert service.is_configured is True
    assert service.model_info() == info


def test_the_embedding_network_returns_unit_length_512_vectors_in_eval_mode(stack, random_weights_dir):
    torch, facenet_pytorch = stack
    recogniser = facenet._build_recogniser(torch, facenet_pytorch.InceptionResnetV1, random_weights_dir / WEIGHTS_FILENAME)
    with torch.no_grad():
        output = recogniser(torch.rand((3, 3, 160, 160)))
    array = output.numpy()
    assert array.shape == (3, 512) and array.dtype == np.float32
    assert np.linalg.norm(array, axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)
    assert all(parameter.requires_grad is False for parameter in recogniser.parameters())


def test_a_blank_image_has_no_face_with_the_real_detector(random_weights_dir):
    service = FaceNetRecognitionService(model_dir=random_weights_dir)
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (320, 240), "white").save(buffer, format="JPEG")
    assert service.extract_faces(buffer.getvalue()) == []


def test_verify_weights_file_accepts_the_layout_and_refuses_a_wrong_one(stack, random_weights_dir, tmp_path):
    torch, _ = stack
    assert facenet.verify_weights_file(random_weights_dir / WEIGHTS_FILENAME)
    wrong = tmp_path / "wrong.pt"
    torch.save({"conv2d_1a.conv.weight": torch.zeros(1)}, wrong)
    with pytest.raises(RecognitionUnavailableError, match="not a usable VGGFace2"):
        facenet.verify_weights_file(wrong)


def test_a_file_that_is_not_a_checkpoint_is_refused_without_leaking_text(random_weights_dir, tmp_path, caplog):
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"CANARY_NOT_A_CHECKPOINT" * 100_000)
    service = FaceNetRecognitionService(model_dir=tmp_path)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RecognitionUnavailableError, match="not a usable VGGFace2"):
            service.model_info()
    assert "CANARY" not in caplog.text
    assert service.is_configured is False
