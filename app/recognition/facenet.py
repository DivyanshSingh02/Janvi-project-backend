"""CPU FaceNet recognition: MTCNN face detection + InceptionResnetV1 (VGGFace2 weights), 512-dimensional embeddings.

What this module guarantees (see docs/recognition.md):
- Importing it never imports PyTorch. The models are loaded on first use, under a lock, from ONE local weights file
  (RECOGNITION_MODEL_DIR/facenet-vggface2.pt). Nothing is ever downloaded while the API runs.
- A missing or unreadable weights file, a missing PyTorch / facenet-pytorch, or a broken install raises
  RecognitionUnavailableError (the API answers 503). It never crashes start-up and never returns a made-up result.
- is_configured does not load the heavy models on every call: it is True only once they are loaded and usable. Cheap
  problems (no weights file, no packages) are re-checked every time, so fixing them needs no restart. A failed heavy
  load is remembered for RETRY_AFTER_FAILED_LOAD_SECONDS so a health-check loop cannot re-read 100 MB each time.
- extract_faces() returns ALL faces that pass the two quality gates (the one-face rule lives in extract_single_face()).
- Embeddings are float32 and L2-normalised. Exception text of third-party libraries, file paths and image data never
  appear in a message or a log line: only fixed texts, reason codes and counts.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import logging
import math
import re
import stat
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.core.config import RECOGNITION_MIN_FACE_SIZE_LOWER, RECOGNITION_MIN_FACE_SIZE_UPPER, get_settings
from app.recognition.base import (
    DetectedFace,
    FaceBox,
    InvalidEmbeddingError,
    ModelInfo,
    RecognitionService,
    RecognitionUnavailableError,
)
from app.recognition.vectors import InvalidVectorError, normalise

logger = logging.getLogger(__name__)

WEIGHTS_FILENAME = "facenet-vggface2.pt"
MODEL_NAME = "facenet-vggface2"
EMBEDDING_DIMENSION = 512
# Bump when detection, cropping or preprocessing changes in a way that changes the vectors, so old and new
# embeddings are never compared (it is part of the model version).
PIPELINE_REVISION = 1
# face_embeddings.model_version and face_search_logs.model_version are String(50): a longer value fails on INSERT.
MAX_MODEL_VERSION_LENGTH = 50
# A real VGGFace2 checkpoint is about 100 MB. Anything below this is empty or truncated. (No expected size or
# checksum is hard-coded: neither could be verified here.)
MIN_PLAUSIBLE_WEIGHTS_BYTES = 1_000_000

IMAGE_SIZE = 160                       # InceptionResnetV1 input size
DETECTION_MAX_SIDE = 2000              # longer sides are shrunk for DETECTION only (boxes are mapped back)
DETECTOR_MIN_FACE_SIZE_FLOOR = RECOGNITION_MIN_FACE_SIZE_LOWER   # MTCNN's smallest usable pyramid level
EMBED_BATCH_SIZE = 16                  # crops per forward pass: bounds memory for images with many faces
RETRY_AFTER_FAILED_LOAD_SECONDS = 30.0
CLASSIFIER_PREFIX = "logits."          # the VGGFace2 checkpoint also holds an 8631-class classifier head

MSG_WEIGHTS_MISSING = (
    "FaceNet weights file is missing; run: python -m scripts.download_recognition_model (see docs/recognition.md)"
)
MSG_WEIGHTS_UNREADABLE = "FaceNet weights file is empty, truncated or unreadable"
MSG_WEIGHTS_INVALID = "FaceNet weights could not be loaded: not a usable VGGFace2 InceptionResnetV1 checkpoint"
MSG_PACKAGES_MISSING = "PyTorch or facenet-pytorch is not installed; install requirements-facenet.txt"
MSG_PACKAGES_UNUSABLE = (
    "PyTorch or facenet-pytorch is installed but cannot be imported (on Windows this is often a missing Microsoft "
    "Visual C++ runtime); see docs/recognition.md"
)
MSG_DETECTOR_FAILED = "The FaceNet detector (MTCNN) could not be created"
MSG_SMOKE_FAILED = "The FaceNet model returned an unexpected result when it was loaded"
MSG_STILL_LOADING = "The FaceNet models are still loading"
MSG_INFERENCE_FAILED = "FaceNet inference failed"
MSG_DETECTOR_OUTPUT = "The FaceNet detector returned unexpected output"


class _LoadProblem(RecognitionUnavailableError):
    """A RecognitionUnavailableError that also carries a short reason code (for logs: never a path or a library's text)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Loaded:
    torch: Any
    detector: Any
    recogniser: Any
    info: ModelInfo


@dataclass(frozen=True)
class _FailedLoad:
    code: str
    message: str
    retry_at: float


# ---------------------------------------------------------------- pure helpers (no PyTorch needed)
def _package_available(name: str) -> bool:
    """True when the package can be found. Cheap: it does not import it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_version(package_version: str, weights_sha256: str) -> str:
    """The model version stored with every embedding: facenet-pytorch version, a prefix of the weights file's
    SHA-256 and the pipeline revision. It changes whenever the vectors can change, and is never longer than the
    database column (50 characters)."""
    safe_package = re.sub(r"[^0-9A-Za-z.+_-]", "", str(package_version))[:10] or "unknown"
    safe_digest = re.sub(r"[^0-9a-f]", "", str(weights_sha256).lower())[:12] or "unknown"
    version = f"facenet-pytorch-{safe_package}-vgg2-{safe_digest}-p{PIPELINE_REVISION}"
    if len(version) > MAX_MODEL_VERSION_LENGTH:       # cannot happen with the limits above; never store a longer one
        raise _LoadProblem("version_too_long", "The FaceNet model version is too long")
    return version


def clean_state_dict(state: Any) -> dict[str, Any]:
    """The weights an InceptionResnetV1 embedding network needs, from a loaded checkpoint.

    The VGGFace2 checkpoint was saved from a model that still had its 8631-class classifier head, so it holds
    `logits.*` tensors. The embedding network (classify=False) has no such layer and a strict load would fail on them,
    so they are dropped here, explicitly. Everything else must match exactly (the strict load checks that)."""
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) for key in state):
        raise ValueError("not a state dictionary")
    return {key: value for key, value in state.items() if not key.startswith(CLASSIFIER_PREFIX)}


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))      # transparency on white, like the upload pipeline
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        return flattened
    return image.convert("RGB")


def _decode(data: bytes) -> Image.Image | None:
    """Pillow decode, EXIF orientation applied, RGB. None when the bytes are not a usable image (corrupt, truncated,
    a decompression bomb, bad EXIF ...): such an image has no face we can use. Only the exception TYPE is logged."""
    try:
        with Image.open(io.BytesIO(data)) as source:
            oriented = ImageOps.exif_transpose(source)       # an independent, rotated copy
            rgb = _to_rgb(oriented)
            rgb.load()
        return rgb
    except Exception as exc:
        logger.warning("recognition_image_undecodable", extra={"reason": type(exc).__name__})
        return None


def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


# ---------------------------------------------------------------- loading the real models
def _import_stack():
    """(torch, facenet_pytorch, MTCNN, InceptionResnetV1). Importing PyTorch on Windows can fail with an OSError
    (DLL load failed) as well as an ImportError."""
    try:
        import torch
        import facenet_pytorch
        from facenet_pytorch import MTCNN, InceptionResnetV1
    except ImportError:
        raise _LoadProblem("packages_missing", MSG_PACKAGES_MISSING) from None
    except OSError:
        raise _LoadProblem("packages_unusable", MSG_PACKAGES_UNUSABLE) from None
    return torch, facenet_pytorch, MTCNN, InceptionResnetV1


def _package_version(facenet_module: Any) -> str:
    try:
        return importlib.metadata.version("facenet-pytorch")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(facenet_module, "__version__", "unknown"))


def _build_recogniser(torch: Any, inception_cls: Any, weights_path: Path) -> Any:
    """InceptionResnetV1 (embeddings only) with the VGGFace2 weights from the local file. pretrained=None means
    facenet-pytorch will not try to download anything. weights_only=True: the file is never allowed to run code."""
    try:
        state = clean_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model = inception_cls(pretrained=None, classify=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except Exception:
        raise _LoadProblem("weights_invalid", MSG_WEIGHTS_INVALID) from None
    return model


def _smoke_test(torch: Any, recogniser: Any) -> None:
    """One blank 160x160 image through the network: the output must be one finite 512-vector."""
    try:
        with torch.no_grad():
            output = _to_numpy(recogniser(torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))))
    except Exception:
        raise _LoadProblem("smoke_test_failed", MSG_SMOKE_FAILED) from None
    if output.shape != (1, EMBEDDING_DIMENSION) or not np.isfinite(output).all():
        raise _LoadProblem("smoke_test_failed", MSG_SMOKE_FAILED)


def verify_weights_file(weights_path: str | Path) -> str:
    """Used by scripts/download_recognition_model.py --check: loads the weights into the embedding network (no
    detector, no network access) and runs one blank image through it. Returns the facenet-pytorch version.
    Raises RecognitionUnavailableError (with a fixed message) when that is not possible."""
    torch, facenet_module, _mtcnn, inception_cls = _import_stack()
    recogniser = _build_recogniser(torch, inception_cls, Path(weights_path))
    _smoke_test(torch, recogniser)
    return _package_version(facenet_module)


class FaceNetRecognitionService(RecognitionService):
    """MTCNN + VGGFace2 InceptionResnetV1, CPU only. Thread-safe: loading is serialised by a lock, and inference is
    serialised by a second lock (one image at a time: bounded CPU and memory, and the detector's per-call settings
    can never race). A multi-worker deployment loads one copy of the models per worker process."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        min_face_size: int | None = None,
        min_detection_probability: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        settings = None
        if model_dir is None or min_face_size is None or min_detection_probability is None:
            settings = get_settings()
        directory = model_dir if model_dir is not None else settings.recognition_model_dir
        size = min_face_size if min_face_size is not None else settings.recognition_min_face_size
        probability = (
            min_detection_probability if min_detection_probability is not None
            else settings.recognition_min_detection_probability
        )
        if not str(directory).strip():
            raise ValueError("model_dir must not be empty")
        if (
            isinstance(size, bool) or not isinstance(size, int)
            or not RECOGNITION_MIN_FACE_SIZE_LOWER <= size <= RECOGNITION_MIN_FACE_SIZE_UPPER
        ):
            raise ValueError(
                f"min_face_size must be an integer from {RECOGNITION_MIN_FACE_SIZE_LOWER} "
                f"to {RECOGNITION_MIN_FACE_SIZE_UPPER}"
            )
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("min_detection_probability must be greater than 0 and at most 1")

        self._model_dir = Path(directory)
        self._min_face_size = size
        self._min_detection_probability = probability
        self._clock = clock
        self._lock = threading.Lock()              # loading
        self._inference_lock = threading.Lock()    # one image at a time
        self._loaded: _Loaded | None = None
        self._load_failure: _FailedLoad | None = None
        self._last_logged_code: str | None = None

    # ------------------------------------------------------------ the RecognitionService contract
    @property
    def is_configured(self) -> bool:
        """True only when the models are loaded and usable. The first call may load them (once); it never waits
        for another thread that is already loading (that is reported as not ready yet), and repeated calls never
        reload anything."""
        try:
            self._ensure_loaded(wait=False)
            return True
        except RecognitionUnavailableError:
            return False

    def model_info(self) -> ModelInfo:
        return self._ensure_loaded().info

    def extract_faces(self, image_jpeg_bytes: bytes) -> list[DetectedFace]:
        loaded = self._ensure_loaded()
        image = _decode(image_jpeg_bytes)
        if image is None:
            return []
        try:
            with self._inference_lock:
                return self._extract(loaded, image)
        except RecognitionUnavailableError:
            raise
        except MemoryError:
            logger.warning("recognition_inference_failed", extra={"reason": "MemoryError"})
            raise RecognitionUnavailableError(MSG_INFERENCE_FAILED) from None
        except Exception as exc:
            # Third-party exception text can contain paths or image details: only its type is logged.
            logger.warning("recognition_inference_failed", extra={"reason": type(exc).__name__})
            raise RecognitionUnavailableError(MSG_INFERENCE_FAILED) from None
        finally:
            image.close()

    # ------------------------------------------------------------ inference
    def _extract(self, loaded: _Loaded, image: Image.Image) -> list[DetectedFace]:
        width, height = image.size
        detector, recogniser, torch = loaded.detector, loaded.recogniser, loaded.torch

        # Detection runs on a copy that is shrunk when it is large (MTCNN's cost and memory grow with the pixel
        # count); the boxes are mapped back, so the gates, the reported boxes and the crops all use the ORIGINAL
        # image. The detector's minimum face size is scaled the same way, so the gate means the same in pixels of
        # the original whatever the shrink factor.
        longest = max(width, height)
        detect_image = image
        if longest > DETECTION_MAX_SIDE:
            factor = DETECTION_MAX_SIDE / longest
            detect_image = image.resize(
                (max(1, round(width * factor)), max(1, round(height * factor))), Image.Resampling.BILINEAR
            )
        scale_x, scale_y = detect_image.width / width, detect_image.height / height
        try:
            detector.min_face_size = max(
                DETECTOR_MIN_FACE_SIZE_FLOOR, int(round(self._min_face_size * min(scale_x, scale_y)))
            )
            with torch.no_grad():
                boxes, probabilities = detector.detect(detect_image)
        finally:
            if detect_image is not image:
                detect_image.close()
        if boxes is None or probabilities is None:
            return []

        accepted = self._apply_quality_gates(boxes, probabilities, scale_x, scale_y, width, height)
        if not accepted:
            return []

        with torch.no_grad():
            crops = detector.extract(image, np.asarray([box for box, _ in accepted], dtype=np.float32), None)
            embeddings = self._embed(recogniser, crops, len(accepted))

        faces: list[DetectedFace] = []
        for (box, probability), vector in zip(accepted, embeddings):
            try:
                unit = normalise(vector)                      # float32, length 1; refuses NaN, inf and all zeros
            except InvalidVectorError:
                raise InvalidEmbeddingError("The FaceNet model returned an unusable embedding") from None
            left, top, right, bottom = box
            faces.append(
                DetectedFace(
                    embedding=unit,
                    box=FaceBox(
                        left=int(math.floor(left)), top=int(math.floor(top)),
                        right=min(width, int(math.ceil(right))), bottom=min(height, int(math.ceil(bottom))),
                    ),
                    detection_confidence=probability,
                )
            )
        return faces

    def _apply_quality_gates(
        self, boxes: Any, probabilities: Any, scale_x: float, scale_y: float, width: int, height: int
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """The two gates: detection probability and face size (the smaller side of the box, in pixels of the
        original image, after clipping the box to the image). Biggest face first, so the order is deterministic."""
        try:
            box_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
            probability_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            raise RecognitionUnavailableError(MSG_DETECTOR_OUTPUT) from None
        if box_array.shape[0] != probability_array.shape[0]:
            raise RecognitionUnavailableError(MSG_DETECTOR_OUTPUT)

        accepted: list[tuple[tuple[float, float, float, float], float]] = []
        for raw_box, probability in zip(box_array, probability_array):
            if not (np.isfinite(raw_box).all() and np.isfinite(probability)):
                continue
            if probability < self._min_detection_probability:
                continue
            left = min(max(raw_box[0] / scale_x, 0.0), float(width))
            right = min(max(raw_box[2] / scale_x, 0.0), float(width))
            top = min(max(raw_box[1] / scale_y, 0.0), float(height))
            bottom = min(max(raw_box[3] / scale_y, 0.0), float(height))
            if min(right - left, bottom - top) < self._min_face_size:
                continue
            accepted.append(((float(left), float(top), float(right), float(bottom)), float(probability)))
        accepted.sort(key=lambda item: (-(item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), item[0][0], item[0][1]))
        return accepted

    @staticmethod
    def _embed(recogniser: Any, crops: Any, expected: int) -> np.ndarray:
        """The embeddings of the face crops, in batches of EMBED_BATCH_SIZE: a (expected, 512) float32 array."""
        try:
            crop_count = int(crops.shape[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            raise InvalidEmbeddingError("The FaceNet detector returned unusable face crops") from None
        if crop_count != expected:
            raise InvalidEmbeddingError("The FaceNet detector returned the wrong number of face crops")
        parts: list[np.ndarray] = []
        for start in range(0, expected, EMBED_BATCH_SIZE):
            part = _to_numpy(recogniser(crops[start:start + EMBED_BATCH_SIZE]))
            if part.ndim == 1:
                part = part.reshape(1, -1)
            if part.ndim != 2 or part.shape[1] != EMBEDDING_DIMENSION or part.shape[0] != min(
                EMBED_BATCH_SIZE, expected - start
            ):
                raise InvalidEmbeddingError("The FaceNet model returned an embedding with the wrong dimension")
            parts.append(part)
        return np.concatenate(parts, axis=0)

    # ------------------------------------------------------------ loading
    @property
    def _weights_path(self) -> Path:
        return self._model_dir / WEIGHTS_FILENAME

    def _ensure_loaded(self, *, wait: bool = True) -> _Loaded:
        loaded = self._loaded
        if loaded is not None:
            return loaded
        if not self._lock.acquire(blocking=wait):
            raise RecognitionUnavailableError(MSG_STILL_LOADING)      # another thread is loading: not ready yet
        try:
            if self._loaded is not None:
                return self._loaded
            try:
                self._preflight()                                      # cheap: never cached, repeated every call
                failure = self._load_failure
                if failure is not None and self._clock() < failure.retry_at:
                    raise _LoadProblem(failure.code, failure.message)
                try:
                    loaded = self._load_models()
                except RecognitionUnavailableError as exc:
                    code = getattr(exc, "code", "load_failed")
                    self._load_failure = _FailedLoad(code, str(exc), self._clock() + RETRY_AFTER_FAILED_LOAD_SECONDS)
                    raise
                except Exception:
                    self._load_failure = _FailedLoad(
                        "load_failed", MSG_WEIGHTS_INVALID, self._clock() + RETRY_AFTER_FAILED_LOAD_SECONDS
                    )
                    raise _LoadProblem("load_failed", MSG_WEIGHTS_INVALID) from None
            except _LoadProblem as problem:
                self._log_problem_once(problem.code)
                raise
            self._load_failure = None
            self._loaded = loaded
            self._last_logged_code = None
            logger.info(
                "recognition_model_loaded",
                extra={"model_name": loaded.info.name, "model_version": loaded.info.version,
                       "dimension": loaded.info.dimension},
            )
            return loaded
        finally:
            self._lock.release()

    def _log_problem_once(self, code: str) -> None:
        """One warning per change of problem, so a health-check loop does not flood the log."""
        if code != self._last_logged_code:
            self._last_logged_code = code
            logger.warning("recognition_unavailable", extra={"reason": code})

    def _preflight(self) -> None:
        """Cheap checks that need no PyTorch: the weights file exists and is plausible, the packages can be found."""
        path = self._weights_path
        try:
            info = path.stat()
        except FileNotFoundError:
            raise _LoadProblem("weights_missing", MSG_WEIGHTS_MISSING) from None
        except OSError:
            raise _LoadProblem("weights_unreadable", MSG_WEIGHTS_UNREADABLE) from None
        if not stat.S_ISREG(info.st_mode):
            raise _LoadProblem("weights_missing", MSG_WEIGHTS_MISSING)
        if info.st_size < MIN_PLAUSIBLE_WEIGHTS_BYTES:
            raise _LoadProblem("weights_unreadable", MSG_WEIGHTS_UNREADABLE)
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except OSError:
            raise _LoadProblem("weights_unreadable", MSG_WEIGHTS_UNREADABLE) from None
        if not (_package_available("torch") and _package_available("facenet_pytorch")):
            raise _LoadProblem("packages_missing", MSG_PACKAGES_MISSING)

    def _load_models(self) -> _Loaded:
        """The heavy part: hash the weights, import PyTorch, build the detector and the embedding network, load the
        weights, run one blank image. Called at most once per RETRY_AFTER_FAILED_LOAD_SECONDS after a failure."""
        weights_path = self._weights_path
        try:
            digest = _sha256_file(weights_path)
        except OSError:
            raise _LoadProblem("weights_unreadable", MSG_WEIGHTS_UNREADABLE) from None

        torch, facenet_module, mtcnn_cls, inception_cls = _import_stack()
        recogniser = _build_recogniser(torch, inception_cls, weights_path)
        try:
            detector = mtcnn_cls(
                image_size=IMAGE_SIZE,
                margin=0,
                min_face_size=self._min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                select_largest=True,
                keep_all=True,
                device=torch.device("cpu"),
            )
        except Exception:
            raise _LoadProblem("detector_failed", MSG_DETECTOR_FAILED) from None
        _smoke_test(torch, recogniser)
        info = ModelInfo(
            name=MODEL_NAME,
            version=build_model_version(_package_version(facenet_module), digest),
            dimension=EMBEDDING_DIMENSION,
        )
        return _Loaded(torch=torch, detector=detector, recogniser=recogniser, info=info)
