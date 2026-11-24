# Stage 6B — real recognition backend (review of the uploaded attempt, fixes, and final files)

The zip you uploaded already contained a ChatGPT version of Stage 6B. I read all 134 files, found real defects in it,
and rewrote the affected parts. **No database migration is needed** (embedding columns and the `String(50)` version
column already fit), so no HEAD constant and no migration test changes.

## 1. Scope and assumptions
Stage 6B only: FaceNet (MTCNN + InceptionResnetV1/VGGFace2, CPU) behind the existing `RecognitionService` contract,
settings, lazy thread-safe loading, optional requirements, three scripts, docs, tests. Nothing from Stage 7+ was touched.
I treated the uploaded zip as the current state of Stages 1–8B (the original Stage 8B zip was not provided to diff against).

## 2. Defects found in the uploaded attempt (all fixed)
| # | Defect | Effect |
|---|---|---|
| 1 | Weights were loaded with a strict `load_state_dict` into a network without a classifier head, but the VGGFace2 checkpoint also holds `logits.*` tensors (as I recall the facenet-pytorch layout) | The real backend could never load; always 503. Now `logits.*` are dropped explicitly and everything else must match |
| 2 | `model_version` was 55 characters; `face_embeddings.model_version` and `face_search_logs.model_version` are `String(50)` | First real enrolment would fail on INSERT (500). Now ≤ 50, tested |
| 3 | `.gitignore` ended with the literal text `\n` (`models/recognition/\nevaluation-datasets/\n`) | Weights and evaluation datasets were NOT ignored. Fixed, regression test added, `*.pt`/`*.pth`/`*.part` added |
| 4 | `requirements-facenet.txt` pinned `torch==2.6.0+cpu` (not installable from PyPI) and the documented `--no-deps` install skipped PyTorch's own dependencies | `import torch` would fail. New 3-step install (PyTorch with deps, facenet-pytorch with `--no-deps`); Windows PyPI torch wheels are CPU-only |
| 5 | A failed first load was cached forever; `is_configured` could block behind a load | Downloading the weights needed a restart; health could hang. Now: cheap checks every call, 30 s cool-down for heavy failures, health never waits |
| 6 | `DecompressionBombError` not caught; an undecodable image gave 503 for the whole case | Now "no face" for that photo; only the error type is logged |
| 7 | `tests/conftest.py` did not pin `RECOGNITION_BACKEND` | A developer `.env` with `facenet` would break tests / load torch. Pinned to `none` with an empty model dir |
| 8 | `evaluate_recognition`: O(n²) Python loop (LFW would take days), counted a *model failure* as "skipped images", thresholds from arbitrary quantiles | Rewritten: blocked numpy scoring, model failure stops the run, thresholds per target false-accept rate + equal-error point, exact false-accept/false-reject counts |
| 9 | `reembed_cases`: one bad case aborted the run; docs mentioned a count it never printed | Per-case isolation, `cases_skipped`/`cases_failed`/`photos_pending` printed, exit code |
| 10 | No bound on detection cost for 25-megapixel uploads; unserialised inference | Detection runs on an image shrunk to ≤ 2000 px (boxes mapped back, gate scaled), batches of 16, one inference at a time |
| 11 | Docs promised behaviour that did not exist (restart needed, "result cached") | Rewritten to match the code |

Honest caveat on #1: it rests on my knowledge of the facenet-pytorch checkpoint, not on a run. `tests/test_facenet_torch.py`
(and `--check`) verify it on your machine once torch is installed; the code is correct either way (it only drops keys
starting with `logits.`, and a strict load checks all the rest).

## 3. Files
**Created:** `tests/test_facenet_torch.py`, `tests/test_recognition_packaging.py`.
**Rewritten (Stage 6B files):** `app/recognition/facenet.py`, `app/recognition/__init__.py`, `requirements-facenet.txt`,
`scripts/download_recognition_model.py`, `scripts/reembed_cases.py`, `scripts/evaluate_recognition.py`,
`docs/recognition.md`, `tests/test_facenet_recognition.py`, `tests/test_recognition_scripts.py`, `tests/test_health.py`.
**Small edits to earlier files:** `app/core/config.py` (two bound constants, `allow_inf_nan=False`, comments),
`.env.example` (comments), `.gitignore`, `docs/security.md` (new section), `docs/architecture.md`, `docs/api.md`, `README.md`
(one sentence each), `tests/conftest.py` (+8 lines: pin recognition env, empty temp model dir), `tests/test_config.py`
(+2 tests), `tests/test_canary_logging.py` (+1 test: FaceNet never logs paths, library text or image bytes).
**Earlier tests changed:** only `test_health.py` (4 → 5 tests, same 4 behaviours plus the readiness one) — behaviour of
`/health` unchanged, assertions now use a real `FaceNetRecognitionService` with a mocked loader. All tests using
`tests/fake_recognition.py` are untouched.
**Migrations / HEAD constants:** none.

## 4. Commands (PowerShell, project root)
```powershell
# previous total (on the zip you uploaded, before applying this stage)
.\venv\Scripts\python.exe -m pytest --collect-only -q | Select-Object -Last 3

# apply: copy the files from this stage over the project, then
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# optional FaceNet install (three steps, see docs/recognition.md for why)
.\venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 requests==2.32.3 tqdm==4.67.1
.\venv\Scripts\python.exe -m pip install --no-deps facenet-pytorch==2.6.0
.\venv\Scripts\python.exe -c "import numpy, PIL, torch, torchvision, facenet_pytorch; print(numpy.__version__, PIL.__version__, torch.__version__, torchvision.__version__)"

# one-time weights download (prints size and SHA-256; --check also loads them into the network)
.\venv\Scripts\python.exe -m scripts.download_recognition_model --check

# .env: set  RECOGNITION_BACKEND=facenet   (RECOGNITION_MODEL_DIR=models/recognition is the default)
# no migration this stage; alembic current should still say 0008
.\venv\Scripts\python.exe -m alembic current

.\venv\Scripts\python.exe -m uvicorn app.main:app --reload

.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m pytest

# after switching backend / weights, on the server:
.\venv\Scripts\python.exe -m scripts.reembed_cases

# calibration on your own, licensed or consented dataset (never committed):
.\venv\Scripts\python.exe -m scripts.evaluate_recognition evaluation-datasets
```
Manual check with fake data (PowerShell 7 for `-Form`; replace the example credentials of YOUR test admin):
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health                           # recognition: unconfigured, then "ready" once weights are in place
$t = (Invoke-RestMethod -Method Post http://127.0.0.1:8000/auth/login -Body @{ username = "admin@example.test"; password = "<test password>" }).access_token
$h = @{ Authorization = "Bearer $t" }
.\venv\Scripts\python.exe -c "from PIL import Image; Image.new('RGB',(300,300),'white').save('blank.png')"
Invoke-RestMethod -Method Post http://127.0.0.1:8000/search -Headers $h -Form @{ file = Get-Item .\blank.png }
```

## 5. Expected output
- `ruff check .` → `All checks passed!`
- Previous total = N (from `--collect-only`). I added/replaced tests by static count (pytest was not available to me):
  files I touched held 36 tests before and hold 93 now, so **new collected total = N + 57**.
- Without torch: **`(N + 52) passed, 5 skipped`** (the 5 are `tests/test_facenet_torch.py`, each skipped individually).
  With torch installed: `(N + 57) passed`.
- Import check line: `2.1.3 11.3.0 2.6.0 0.21.0`.
- Download script prints `Source:`, `Saved:`, `Size: <bytes> bytes`, `SHA-256: <64 hex>`, the licence reminder, and with
  `--check` `Load check: ok (facenet-pytorch 2.6.0; the weights fit the network)`.
- `/health` with backend `none` or missing weights: `recognition = unconfigured`; with the weights and packages in place: `ready`.
- Search with the blank image while ready: HTTP 422 `no_face_detected` (proves the real pipeline ran). While not ready: 503.
- `reembed_cases`: one line such as `cases=3 cases_skipped=0 cases_failed=0 photos_done=7 photos_failed=1 photos_pending=0`.

## 6. What I could not run or verify
- **No PostgreSQL, pytest, ruff, torch, facenet-pytorch or network** in my environment. I could NOT run the full suite.
  I ran the DB-free tests (33 FaceNet + 20 scripts + 4 packaging; 5 torch tests correctly skip) with a small pytest shim
  and stubs against the real new modules: all passed; a deliberate mutation (gate `<`→`<=`, dropping the `logits.` filter)
  was caught. The database-backed tests I wrote (2 API tests in `test_facenet_recognition.py`, 5 re-embed tests,
  health/config/canary tests) were reviewed by hand against the existing fixtures but never executed.
- `ruff` was not run; I used a small AST checker for the selected rules (E402, E7xx, F541). Run `ruff check .` yourself.
- Real MTCNN/InceptionResnetV1 behaviour: the loading code was tested with fake torch modules only. Unverified: that
  facenet-pytorch 2.6.0 runs on numpy 2.1.3 / Pillow 11.3.0 / torch 2.6.0, that its wheel bundles MTCNN's `data/*.pt`,
  that torch 2.6's default `weights_only=True` accepts MTCNN's bundled weights, the exact 2.6.0 metadata bounds, the download
  URL, and that PyPI Windows wheels for torch 2.6.0 / torchvision 0.21.0 exist for cp311 (I believe all do).
- Timing (seconds per photo on your CPU) and memory use were not measured.
- Licence status of VGGFace2 / the checkpoint: not checked — you must.
- Conflicts with existing pins: none *in the files*. The only conflict is facenet-pytorch 2.6.0's own declared bounds
  (numpy<2, Pillow<10.3, torch<2.3, torchvision<0.18), bypassed with `--no-deps`; nothing was downgraded.

Stopping here: Stage 7 is not started.

---
# Files (verbatim)

## app/core/config.py

~~~~~~python
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

SECRET_KEY_MIN_LENGTH = 32

# Face-quality gate bounds. MTCNN cannot search for faces smaller than 12 px (its smallest pyramid level), and a
# gate above 1000 px would reject every realistic upload. The FaceNet service validates its own arguments with them.
RECOGNITION_MIN_FACE_SIZE_LOWER = 12
RECOGNITION_MIN_FACE_SIZE_UPPER = 1000


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a local .env file).
    There are deliberately no default values for database credentials or the JWT signing key.
    The storage and upload-limit settings are not secret, so they have defaults."""

    # hide_input_in_errors: a rejected value (for example a too-short SECRET_KEY) is never
    # echoed back in the validation error text.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    app_name: str = "Missing Person API"
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    postgres_user: str
    postgres_password: SecretStr
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str

    # Authentication. Only symmetric HMAC algorithms are allowed, so a configuration
    # mistake can never switch token verification to "none" or to an asymmetric scheme.
    secret_key: SecretStr
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=1)

    # Photo storage and upload limits.
    storage_backend: Literal["local"] = "local"
    storage_dir: str = Field(default="storage", min_length=1)
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    max_photos_per_case: int = Field(default=5, ge=1, le=20)   # the photo list is not paginated, so keep it small
    max_image_pixels: int = Field(default=25_000_000, ge=1)
    min_image_side: int = Field(default=100, ge=1)

    # HTTP hardening. CORS_ALLOWED_ORIGINS is a JSON list in the environment; [] disables CORS.
    cors_allowed_origins: list[str] = Field(default_factory=list)
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])
    enable_api_docs: bool = True
    max_request_body_bytes: int = Field(default=30 * 1024 * 1024, ge=1024)

    # Sliding-window limits (requests per window).
    login_ip_limit: int = Field(default=10, ge=1)
    login_email_limit: int = Field(default=5, ge=1)
    login_window_seconds: int = Field(default=60, ge=1)
    registration_ip_limit: int = Field(default=5, ge=1)
    registration_window_seconds: int = Field(default=3600, ge=1)
    face_search_admin_limit: int = Field(default=20, ge=1)
    embedding_admin_limit: int = Field(default=10, ge=1)
    admin_recognition_window_seconds: int = Field(default=60, ge=1)
    photo_upload_user_limit: int = Field(default=10, ge=1)
    photo_upload_window_seconds: int = Field(default=60, ge=1)

    # Face recognition and search. "none" stays the default; "facenet" loads local weights lazily (on first use,
    # never at start-up and never from the network). A missing weights file is NOT a configuration error: the API
    # starts and answers 503 for recognition until the file exists.
    recognition_backend: Literal["none", "facenet"] = "none"
    recognition_model_dir: str = Field(default="models/recognition", min_length=1)
    # Quality gates (image/detector quality, not identity thresholds): the smallest accepted face box side in
    # pixels of the analysed image, and the smallest accepted MTCNN detection probability.
    recognition_min_face_size: int = Field(
        default=40, ge=RECOGNITION_MIN_FACE_SIZE_LOWER, le=RECOGNITION_MIN_FACE_SIZE_UPPER
    )
    recognition_min_detection_probability: float = Field(default=0.90, gt=0.0, le=1.0, allow_inf_nan=False)
    search_default_limit: int = Field(default=5, ge=1, le=100)
    search_max_limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def _search_default_fits_the_maximum(self) -> "Settings":
        if self.secret_key.get_secret_value() == "CHANGE_ME" and self.app_env != "production":
            raise ValueError("SECRET_KEY must be replaced outside production configuration checks")
        if self.search_default_limit > self.search_max_limit:
            raise ValueError("SEARCH_DEFAULT_LIMIT must not be larger than SEARCH_MAX_LIMIT")
        if not self.recognition_model_dir.strip():
            raise ValueError("RECOGNITION_MODEL_DIR must not be empty")
        return self

    @field_validator("secret_key")
    @classmethod
    def _secret_key_is_long_enough(cls, value: SecretStr) -> SecretStr:
        if value.get_secret_value() == "CHANGE_ME":
            return value
        if len(value.get_secret_value()) < SECRET_KEY_MIN_LENGTH:
            raise ValueError(f"SECRET_KEY must be at least {SECRET_KEY_MIN_LENGTH} characters long")
        return value

    def validate_production(self) -> None:
        """Fail closed for unsafe production-only configuration."""
        if self.app_env != "production":
            return
        if self.secret_key.get_secret_value() == "CHANGE_ME":
            raise ValueError("SECRET_KEY must be changed before production startup")
        if self.postgres_password.get_secret_value() == "CHANGE_ME":
            raise ValueError("POSTGRES_PASSWORD must be changed before production startup")
        if "*" in self.cors_allowed_origins:
            raise ValueError("CORS_ALLOWED_ORIGINS cannot contain '*' in production")
        if self.enable_api_docs:
            raise ValueError("API documentation must be disabled in production")

    @property
    def database_url(self) -> URL:
        """Built from parts so special characters in the password never break the URL."""
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
~~~~~~

## app/recognition/__init__.py

~~~~~~python
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
~~~~~~

## app/recognition/facenet.py

~~~~~~python
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
~~~~~~

## requirements-facenet.txt

~~~~~~text
# Optional CPU-only FaceNet backend (RECOGNITION_BACKEND=facenet). The base requirements.txt stays light.
#
# Do NOT run "pip install -r requirements-facenet.txt" directly. facenet-pytorch 2.6.0's package metadata still
# demands numpy<2, Pillow<10.3, torch<2.3 and torchvision<0.18, which conflict with this project's pins
# (numpy 2.1.3, Pillow 11.3.0). pip would refuse (ResolutionImpossible) or, worse, downgrade them. Install in three
# steps instead (docs/recognition.md has the PowerShell commands):
#   1. pip install -r requirements.txt
#   2. pip install torch==2.6.0 torchvision==0.21.0 requests==2.32.3 tqdm==4.67.1     (normal install: pulls PyTorch's own dependencies)
#   3. pip install --no-deps facenet-pytorch==2.6.0                                   (skips only its stale version bounds)
# On Windows the PyPI wheels of torch / torchvision are CPU-only builds, so no extra index is needed.
-r requirements.txt

torch==2.6.0
torchvision==0.21.0
# facenet-pytorch imports requests and tqdm at run time:
requests==2.32.3
tqdm==4.67.1
# Pure-Python wheel. Its numpy / Pillow / torch bounds are deliberately bypassed with --no-deps (step 3).
facenet-pytorch==2.6.0
~~~~~~

## .gitignore

~~~~~~text
venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
.env
storage/
*.log
# Face recognition: model weights and evaluation datasets must never be committed
models/recognition/
evaluation-datasets/
*.pt
*.pth
*.part
~~~~~~

## .env.example

~~~~~~text
# Copy to .env and fill in. Never commit .env.
APP_ENV=development
LOG_LEVEL=INFO
LOG_JSON=false

POSTGRES_USER=missing_person_app
POSTGRES_PASSWORD=CHANGE_ME
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=missing_person

# Signing key for login tokens. At least 32 characters, random, never shared or committed.
# Generate one with:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
# Replace this placeholder before starting in development or test; production also has explicit startup checks.
SECRET_KEY=CHANGE_ME
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Photo storage (local disk only for now; the folder is git-ignored) and upload limits.
STORAGE_BACKEND=local
STORAGE_DIR=storage
MAX_UPLOAD_BYTES=5242880
MAX_PHOTOS_PER_CASE=5
MAX_IMAGE_PIXELS=25000000
MIN_IMAGE_SIDE=100

# Face recognition. "none" (default) = no model, recognition endpoints answer 503.
# "facenet" = CPU FaceNet; needs requirements-facenet.txt installed and the weights downloaded ONCE with
# python -m scripts.download_recognition_model (nothing is ever downloaded while the API runs).
# A missing weights file or PyTorch never stops start-up: the API starts and answers 503 for recognition.
RECOGNITION_BACKEND=none
# Folder that holds facenet-vggface2.pt. Relative paths are relative to the folder the API is started from;
# use an absolute path in production.
RECOGNITION_MODEL_DIR=models/recognition
# Quality gates, not identity thresholds: smallest face box side in pixels (12-1000) and smallest detector probability.
RECOGNITION_MIN_FACE_SIZE=40
RECOGNITION_MIN_DETECTION_PROBABILITY=0.90
SEARCH_DEFAULT_LIMIT=5
SEARCH_MAX_LIMIT=20

# Stage 8A HTTP hardening
ENABLE_API_DOCS=true
CORS_ALLOWED_ORIGINS=[]
TRUSTED_HOSTS=["localhost","127.0.0.1"]
MAX_REQUEST_BODY_BYTES=31457280

LOGIN_IP_LIMIT=10
LOGIN_EMAIL_LIMIT=5
LOGIN_WINDOW_SECONDS=60
REGISTRATION_IP_LIMIT=5
REGISTRATION_WINDOW_SECONDS=3600
FACE_SEARCH_ADMIN_LIMIT=20
EMBEDDING_ADMIN_LIMIT=10
ADMIN_RECOGNITION_WINDOW_SECONDS=60
PHOTO_UPLOAD_USER_LIMIT=10
PHOTO_UPLOAD_WINDOW_SECONDS=60
~~~~~~

## scripts/download_recognition_model.py

~~~~~~python
"""Explicit, one-time download of the FaceNet (InceptionResnetV1, VGGFace2) weights. YOU run it; the API never does.

    python -m scripts.download_recognition_model [--directory DIR] [--force] [--check | --check-only]

Where the file comes from: the release asset "20180402-114759-vggface2.pt" of the facenet-pytorch project on GitHub
(see WEIGHTS_URL). It is saved as facenet-vggface2.pt in RECOGNITION_MODEL_DIR (or --directory). The script prints
the size and the SHA-256 of the file it SAVED, computed from the file on disk. No checksum is hard-coded here: none
could be verified. Write the printed value down; compare it if you ever copy or re-download the file.

MTCNN's own (small) detector weights are not downloaded here: they ship inside the facenet-pytorch package.

LICENCE: these pretrained weights derive from the VGGFace2 dataset. The project's MIT licence covers its code, not
necessarily the dataset or the weights. Check the current terms of the dataset and of this checkpoint BEFORE any real
deployment (docs/recognition.md).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

WEIGHTS_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
)
WEIGHTS_FILENAME = "facenet-vggface2.pt"       # keep in sync with app/recognition/facenet.py
MIN_PLAUSIBLE_BYTES = 1_000_000                # a real checkpoint is about 100 MB; anything below is not one
CHUNK_BYTES = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(
    directory: Path, *, force: bool = False, timeout: float = 60.0, opener=None
) -> tuple[Path, int, str]:
    """Download into `directory`. Returns (path, size in bytes, SHA-256) of the SAVED file.
    Raises FileExistsError (nothing is replaced without force) or OSError. The file is written to a temporary name
    and moved into place only when complete, so an interrupted download never leaves a half file under the real name."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / WEIGHTS_FILENAME
    if destination.exists() and not force:
        raise FileExistsError("The weights file already exists; pass --force to replace it")

    descriptor, temporary_name = tempfile.mkstemp(prefix=".facenet-download-", suffix=".part", dir=directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        received = 0
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "missing-person-api-weights-download"})
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response, temporary.open("wb") as output:
            declared = response.headers.get("Content-Length") if getattr(response, "headers", None) else None
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if declared is not None and declared.isdigit() and int(declared) != received:
            raise OSError("The download was incomplete")
        if received < MIN_PLAUSIBLE_BYTES:
            raise OSError("The downloaded file is too small to be the weights")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination, destination.stat().st_size, sha256_of(destination)


def _directory(argument: str | None) -> Path:
    if argument is not None:
        return Path(argument)
    from app.core.config import get_settings          # imported late: --help and --directory need no .env

    return Path(get_settings().recognition_model_dir)


def _load_check(path: Path) -> str:
    """Loads the file into the embedding network (needs torch and facenet-pytorch). Never raises."""
    try:
        from app.recognition.facenet import verify_weights_file

        version = verify_weights_file(path)
    except Exception as exc:                          # RecognitionUnavailableError or an import problem
        return f"failed: {exc}"
    return f"ok (facenet-pytorch {version}; the weights fit the network)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.download_recognition_model",
        description="Download the FaceNet VGGFace2 weights once. Never run by the API.",
    )
    parser.add_argument("--directory", default=None, help="Destination folder (default: RECOGNITION_MODEL_DIR)")
    parser.add_argument("--force", action="store_true", help="Replace an existing weights file")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds (default 60)")
    check = parser.add_mutually_exclusive_group()
    check.add_argument("--check", action="store_true", help="After saving, load the weights with PyTorch to check them")
    check.add_argument("--check-only", action="store_true", help="Do not download: check the existing file")
    args = parser.parse_args(argv)

    try:
        directory = _directory(args.directory)
    except Exception:
        print("Error: could not read RECOGNITION_MODEL_DIR from the settings; fix .env or pass --directory.")
        return 2

    if args.check_only:
        path = directory / WEIGHTS_FILENAME
        if not path.is_file():
            print("Error: there is no weights file in that folder.")
            return 2
    else:
        try:
            path, _size, _digest = download_weights(directory, force=args.force, timeout=args.timeout)
        except FileExistsError as exc:
            print(f"Error: {exc}")
            return 2
        except Exception as exc:                      # no secrets are involved: the type and text help to diagnose
            print(f"Error: the weights could not be downloaded or saved ({type(exc).__name__}: {exc}).")
            return 1
        print(f"Source: {WEIGHTS_URL}")

    print(f"Saved: {WEIGHTS_FILENAME}" if not args.check_only else f"File: {WEIGHTS_FILENAME}")
    print(f"Size: {path.stat().st_size} bytes")
    print(f"SHA-256: {sha256_of(path)}")
    print("Record the SHA-256 above. Check the licence of the VGGFace2 weights before real use (docs/recognition.md).")
    if args.check or args.check_only:
        outcome = _load_check(path)
        print(f"Load check: {outcome}")
        if outcome.startswith("failed"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
~~~~~~

## scripts/reembed_cases.py

~~~~~~python
"""Re-enrol the photos of ALL APPROVED cases with the currently configured recognition model.

Run by an administrator, on the server, from the project root, after switching the recognition backend or its weights:

    python -m scripts.reembed_cases

It goes through the existing embedding service (app/services/embeddings.py), so the rules are the same as for
POST /admin/cases/{id}/embeddings: exactly one face per photo, one transaction per case, photos that already have an
embedding from the current model name and version are skipped (so running it twice is harmless), and photos with
embeddings from another model version get new ones that REPLACE the old ones.

Output is counts only, never a name, an id or a file name:

    cases=3 cases_skipped=0 cases_failed=0 photos_done=7 photos_failed=1 photos_pending=0

photos_* describe the state of the processed cases' photos AFTER the run (done = has an embedding from the current
model; failed = no usable single face or file missing; pending = not processed). Exit code 0 when no case failed,
1 when the model is unavailable or at least one case could not be processed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from sqlalchemy import select

from app.models.case import CaseStatus, MissingCase
from app.recognition import RecognitionUnavailableError, get_recognition_service
from app.services import cases as case_service
from app.services import embeddings as embedding_service
from app.storage import get_storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReembedCounts:
    cases: int = 0              # approved cases found
    cases_skipped: int = 0      # changed (or vanished) while the script ran
    cases_failed: int = 0       # could not be processed (storage problem or unexpected error); nothing saved for them
    photos_done: int = 0
    photos_failed: int = 0
    photos_pending: int = 0

    def line(self) -> str:
        return (
            f"cases={self.cases} cases_skipped={self.cases_skipped} cases_failed={self.cases_failed} "
            f"photos_done={self.photos_done} photos_failed={self.photos_failed} photos_pending={self.photos_pending}"
        )


def reembed_all_approved(db, storage, recognition) -> ReembedCounts:
    """Raises RecognitionUnavailableError when the model cannot be used (the whole run stops: nothing would work).
    A problem with ONE case is counted and the run carries on with the next."""
    case_ids = list(
        db.scalars(
            select(MissingCase.id)
            .where(MissingCase.status == CaseStatus.APPROVED.value)
            .order_by(MissingCase.created_at, MissingCase.id)
        )
    )
    skipped = failed = done = photos_failed = pending = 0
    for case_id in case_ids:
        try:
            result = embedding_service.enrol_case(db, storage, recognition, case_id)
        except RecognitionUnavailableError:
            raise
        except (case_service.CaseNotFoundError, embedding_service.CaseNotApprovedError):
            skipped += 1
            continue
        except Exception as exc:
            failed += 1
            logger.warning("reembed_case_failed", extra={"case_id": str(case_id), "reason": type(exc).__name__})
            continue
        done += result.summary.generated
        photos_failed += result.summary.failed
        pending += result.summary.pending
    return ReembedCounts(len(case_ids), skipped, failed, done, photos_failed, pending)


def _session_context():
    from app.db.session import SessionLocal

    return SessionLocal()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reembed_cases",
        description="Re-enrol all approved cases with the configured recognition model. Prints counts only.",
    )
    parser.parse_args(argv)
    try:
        recognition = get_recognition_service()
        recognition.model_info()                      # fail early and clearly when there is no usable model
    except RecognitionUnavailableError:
        print("Error: the configured recognition model is unavailable. Check RECOGNITION_BACKEND, the weights "
              "and the installed packages (docs/recognition.md).")
        return 1
    try:
        with _session_context() as db:
            counts = reembed_all_approved(db, get_storage(), recognition)
    except RecognitionUnavailableError:
        print("Error: the recognition model became unavailable during the run. Counts so far were not printed; "
              "run the script again (finished cases are skipped).")
        return 1
    except Exception:
        print("Error: re-embedding could not be completed.")
        return 1
    print(counts.line())
    return 1 if counts.cases_failed else 0


if __name__ == "__main__":
    sys.exit(main())
~~~~~~

## scripts/evaluate_recognition.py

~~~~~~python
"""Measure how the configured recognition model scores same-person and different-person image pairs on YOUR dataset.

    python -m scripts.evaluate_recognition DATASET_FOLDER [--thresholds 0.5,0.6] [--target-far 0.01,0.001]
                                                          [--max-images-per-person N]

DATASET_FOLDER holds one sub-folder per person with that person's images (LFW has exactly this layout):

    DATASET_FOLDER/<person>/<image>.jpg

Use a public dataset whose licence allows this, or photos taken with explicit informed consent. NEVER use real case
photos, and never commit the dataset (the folder name evaluation-datasets/ is git-ignored).

What it does: every image goes through the same one-face rule as the API (extract_single_face). Images with no usable
face, with several faces, or that cannot be read are counted and skipped. Then every pair of usable images is scored
with cosine similarity: pairs of the same person ("same") and of different people ("different"). It prints both
distributions and, for some threshold candidates, how many different-person pairs would have been accepted (false
accepts) and how many same-person pairs would have been rejected (false rejects) at that threshold ("accept when the
similarity is >= the threshold").

What it does NOT do: it makes no accuracy claim and chooses no threshold for you. The numbers describe this dataset,
this model and these quality gates only. Pairs share images, so they are not independent samples and the rates have
no confidence interval; a rate of 0 means "none seen", not "impossible". Check the result per group of people and
per photo condition on your own data before trusting any threshold.

Privacy: the output is counts and numbers only: no folder names, no file names, no paths.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.recognition import (
    MultipleFacesError,
    NoFaceDetectedError,
    RecognitionUnavailableError,
    extract_single_face,
    get_recognition_service,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
BIN_COUNT = 4000                               # histogram of scores from -1 to 1: bins 0.0005 wide
BIN_EDGES = np.linspace(-1.0, 1.0, BIN_COUNT + 1)
BLOCK_ROWS = 256                               # rows of the score matrix computed at once (bounds memory)
DEFAULT_TARGET_FARS = (0.1, 0.01, 0.001, 0.0001)
THRESHOLD_DECIMALS = 4


# ---------------------------------------------------------------- argument parsing
def _number_list(value: str, low: float, high: float, what: str, *, exclusive: bool = False) -> list[float]:
    try:
        numbers = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what} must be comma-separated numbers") from None
    def inside(x: float) -> bool:
        return low < x < high if exclusive else low <= x <= high

    if not numbers or any(not math.isfinite(x) or not inside(x) for x in numbers):
        raise argparse.ArgumentTypeError(f"{what} must be finite numbers from {low} to {high}")
    return sorted(set(numbers))


def _threshold_values(value: str) -> list[float]:
    return _number_list(value, -1.0, 1.0, "thresholds")


def _far_values(value: str) -> list[float]:
    return _number_list(value, 0.0, 1.0, "target false-accept rates", exclusive=True)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return number


# ---------------------------------------------------------------- collecting embeddings
def _image_files(person_dir: Path) -> list[Path]:
    return sorted(p for p in person_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _person_folders(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


@dataclass
class Collected:
    vectors: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    images: int = 0
    no_face: int = 0            # includes images that cannot be decoded: they have no usable face
    multiple_faces: int = 0
    unreadable: int = 0         # the file could not be read from disk
    people: int = 0             # folders with at least one image

    @property
    def usable(self) -> int:
        return len(self.vectors)

    @property
    def skipped(self) -> int:
        return self.no_face + self.multiple_faces + self.unreadable


def collect_embeddings(root: Path, recognition, max_images_per_person: int | None = None) -> Collected:
    """Raises RecognitionUnavailableError when the model fails: that must stop the run, not be counted as 'skipped'."""
    collected = Collected()
    label = 0
    for person_dir in _person_folders(root):
        files = _image_files(person_dir)
        if max_images_per_person is not None:
            files = files[:max_images_per_person]
        if not files:
            continue
        collected.people += 1
        for path in files:
            collected.images += 1
            try:
                data = path.read_bytes()
            except OSError:
                collected.unreadable += 1
                continue
            try:
                face = extract_single_face(recognition, data)
            except NoFaceDetectedError:
                collected.no_face += 1
                continue
            except MultipleFacesError:
                collected.multiple_faces += 1
                continue
            collected.vectors.append(face.embedding)
            collected.labels.append(label)
        label += 1
    return collected


# ---------------------------------------------------------------- scoring pairs
def iter_pair_scores(matrix: np.ndarray, labels: np.ndarray):
    """Yields (same_person_scores, different_person_scores) for blocks of rows; every unordered pair exactly once."""
    count = matrix.shape[0]
    columns = np.arange(count)[None, :]
    for start in range(0, count, BLOCK_ROWS):
        stop = min(count, start + BLOCK_ROWS)
        scores = np.clip(matrix[start:stop] @ matrix.T, -1.0, 1.0).astype(np.float64)
        later = columns > np.arange(start, stop)[:, None]
        same_label = labels[start:stop, None] == labels[None, :]
        yield scores[later & same_label], scores[later & ~same_label]


class Distribution:
    """Count, exact minimum / maximum / mean and a histogram (percentiles are exact to half a bin: 0.00025)."""

    def __init__(self) -> None:
        self.counts = np.zeros(BIN_COUNT, dtype=np.int64)
        self.n = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, scores: np.ndarray) -> None:
        if scores.size == 0:
            return
        self.counts += np.histogram(scores, bins=BIN_EDGES)[0]
        self.n += int(scores.size)
        self.total += float(scores.sum())
        self.minimum = min(self.minimum, float(scores.min()))
        self.maximum = max(self.maximum, float(scores.max()))

    def percentile(self, q: float) -> float:
        cumulative = np.cumsum(self.counts)
        index = min(int(np.searchsorted(cumulative, q / 100.0 * self.n, side="left")), BIN_COUNT - 1)
        return float((BIN_EDGES[index] + BIN_EDGES[index + 1]) / 2.0)

    def describe(self, label: str) -> str:
        if self.n == 0:
            return f"{label}: pairs=0 (not enough usable data)"
        return (
            f"{label}: pairs={self.n} min={self.minimum:.4f} p05={self.percentile(5):.4f} "
            f"median={self.percentile(50):.4f} mean={self.total / self.n:.4f} p95={self.percentile(95):.4f} "
            f"max={self.maximum:.4f}"
        )


def _scan(matrix: np.ndarray, labels: np.ndarray) -> tuple[Distribution, Distribution]:
    same, different = Distribution(), Distribution()
    for same_scores, different_scores in iter_pair_scores(matrix, labels):
        same.add(same_scores)
        different.add(different_scores)
    return same, different


def derive_candidates(same: Distribution, different: Distribution, target_fars) -> tuple[list, list]:
    """Threshold candidates from the histograms: (candidates, notes). A candidate is (description, threshold).
    - for each target false-accept rate f: the lowest threshold whose false-accept rate is <= f (only when the data
      can resolve it, i.e. there are at least 1/f different-person pairs);
    - the equal-error point: where the false-accept and false-reject rates are closest."""
    candidates: list[tuple[str, float]] = []
    notes: list[str] = []
    if same.n == 0 or different.n == 0:
        return candidates, notes
    # accepted_different[k] = different-person scores >= BIN_EDGES[k]; rejected_same[k] = same-person scores < BIN_EDGES[k]
    accepted_different = np.append(np.cumsum(different.counts[::-1])[::-1], 0)
    rejected_same = np.append(0, np.cumsum(same.counts))
    for far in target_fars:
        if different.n * far < 1:
            notes.append(
                f"target_far={far:g} not resolvable: {different.n} different-person pairs "
                f"(at least {math.ceil(1 / far)} are needed)"
            )
            continue
        index = int(np.argmax(accepted_different <= far * different.n))
        candidates.append((f"target_far={far:g}", float(BIN_EDGES[min(index, BIN_COUNT)])))
    far_curve = accepted_different / different.n
    frr_curve = rejected_same / same.n
    index = int(np.argmin(np.abs(far_curve - frr_curve)))
    candidates.append(("equal_error_point", float(BIN_EDGES[index])))
    return candidates, notes


def exact_counts(matrix: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> dict[float, tuple[int, int]]:
    """(false accepts, false rejects) at each threshold, counted exactly on the scores themselves."""
    false_accepts = {t: 0 for t in thresholds}
    false_rejects = {t: 0 for t in thresholds}
    for same_scores, different_scores in iter_pair_scores(matrix, labels):
        for t in thresholds:
            false_accepts[t] += int((different_scores >= t).sum())
            false_rejects[t] += int((same_scores < t).sum())
    return {t: (false_accepts[t], false_rejects[t]) for t in thresholds}


# ---------------------------------------------------------------- the report
def build_report(collected: Collected, explicit_thresholds, target_fars) -> list[str]:
    lines = [
        f"images={collected.images} usable={collected.usable} skipped={collected.skipped} "
        f"skipped_no_face={collected.no_face} skipped_multiple_faces={collected.multiple_faces} "
        f"skipped_unreadable={collected.unreadable}"
    ]
    labels = np.asarray(collected.labels, dtype=np.int64)
    per_person = np.bincount(labels) if labels.size else np.array([], dtype=np.int64)
    lines.append(
        f"people={collected.people} people_with_usable_images={int((per_person > 0).sum())} "
        f"people_with_2_or_more_usable_images={int((per_person > 1).sum())}"
    )
    if collected.usable < 2:
        lines.append("Not enough usable images for any pair.")
        return lines + _closing()
    matrix = np.stack(collected.vectors).astype(np.float32)
    same, different = _scan(matrix, labels)
    lines.append(f"same_person_pairs={same.n} different_person_pairs={different.n}")
    lines.append(same.describe("same_person"))
    lines.append(different.describe("different_person"))
    if same.n == 0 or different.n == 0:
        lines.append("Threshold trade-off unavailable: both same-person and different-person pairs are required.")
        return lines + _closing()

    candidates, notes = derive_candidates(same, different, target_fars)
    if explicit_thresholds:
        candidates = [("given", t) for t in explicit_thresholds]
    rounded = sorted({round(t, THRESHOLD_DECIMALS) for _, t in candidates})
    counts = exact_counts(matrix, labels, rounded)
    lines.append("threshold candidates (accept as the same person when similarity >= threshold):")
    for description, threshold in candidates:
        t = round(threshold, THRESHOLD_DECIMALS)
        fa, fr = counts[t]
        lines.append(
            f"  {description} threshold={t:.4f} false_accept={fa}/{different.n} ({fa / different.n:.4f}) "
            f"false_reject={fr}/{same.n} ({fr / same.n:.4f})"
        )
    lines.extend(f"  {note}" for note in notes)
    return lines + _closing()


def _closing() -> list[str]:
    return [
        "These numbers describe this dataset, this model and these quality gates only. They are not accuracy claims,",
        "not an identity probability and not a recommended threshold. A match is a lead for human review.",
    ]


# ---------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_recognition",
        description="Measure same-person / different-person similarity distributions; makes no accuracy claim.",
    )
    parser.add_argument("dataset", type=Path, help="Folder with one sub-folder of images per person")
    parser.add_argument("--thresholds", type=_threshold_values, default=None,
                        help="Your own threshold candidates, comma-separated, from -1 to 1 (replaces the derived ones)")
    parser.add_argument("--target-far", type=_far_values, default=list(DEFAULT_TARGET_FARS),
                        help="False-accept rates to derive thresholds for (default 0.1,0.01,0.001,0.0001)")
    parser.add_argument("--max-images-per-person", type=_positive_int, default=None,
                        help="Use at most this many images per person (the first ones, sorted): bounds the run time")
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print("Error: dataset folder does not exist.")
        return 2
    if not any(_image_files(folder) for folder in _person_folders(args.dataset)):
        print("No images found. images=0 usable=0 skipped=0")
        return 0
    try:
        recognition = get_recognition_service()
        recognition.model_info()
        collected = collect_embeddings(args.dataset, recognition, args.max_images_per_person)
    except RecognitionUnavailableError:
        print("Error: the configured recognition model is unavailable (is RECOGNITION_BACKEND=facenet and are the "
              "weights and packages installed? See docs/recognition.md).")
        return 1
    for line in build_report(collected, args.thresholds, args.target_far):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
~~~~~~

## docs/recognition.md

~~~~~~markdown
# Face recognition: FaceNet CPU backend (Stage 6B)

> **A face match is a lead, not proof of identity. A human must review every candidate before any action is taken.**
> Similarity scores are not identity probabilities. Do not make automated decisions from them.

## Architecture and contract

The application uses `RecognitionService` from `app/recognition/base.py`. The real implementation is
`FaceNetRecognitionService` in `app/recognition/facenet.py`; `UnconfiguredRecognitionService` (answers 503) stays the
default (`RECOGNITION_BACKEND=none`). A fake service exists only in `tests/fake_recognition.py`.

The FaceNet backend uses MTCNN (`facenet-pytorch`) for detection and InceptionResnetV1 with VGGFace2 weights for
512-dimensional embeddings, on the CPU. What happens to one image:

1. Pillow decodes it, applies the EXIF orientation and converts it to RGB (transparency is flattened onto white).
   Bytes that are not a usable image (corrupt, truncated, decompression bomb) have **no face**: `extract_faces()` returns
   `[]` and the existing code treats the photo as "no face detected" instead of failing the whole case.
2. MTCNN looks for faces. Images whose long side exceeds 2000 px are shrunk **for detection only**; the minimum face
   size handed to MTCNN is scaled the same way, boxes are mapped back, and the gates, the reported boxes and the 160x160
   crops all use the original image.
3. Two quality gates (below) drop faces. **All faces that pass are returned**, biggest first. The one-face rule for
   enrolment and search stays in `extract_single_face()`, so a later video stage can analyse frames with several faces.
4. The crops go through the network in batches of 16; each vector is returned as `float32`, L2-normalised
   (`app/recognition/vectors.py`), together with its box and detection probability.

Only one image is processed at a time per process (a lock): bounded CPU and memory, and no races on the detector.
Every worker process of a multi-worker deployment holds its own copy of the models (roughly 0.5 GB of RAM each).

| Setting | Default | Meaning |
| --- | ---: | --- |
| `RECOGNITION_BACKEND` | `none` | `none` or `facenet` |
| `RECOGNITION_MODEL_DIR` | `models/recognition` | Folder holding `facenet-vggface2.pt` (relative to where the API is started; use an absolute path in production) |
| `RECOGNITION_MIN_FACE_SIZE` | `40` | Smallest accepted face box side, in pixels of the original image (12 to 1000) |
| `RECOGNITION_MIN_DETECTION_PROBABILITY` | `0.90` | Smallest accepted MTCNN detection probability (above 0, at most 1) |

These are image-quality gates, not identity thresholds. Invalid values are refused when the settings load.

## Install (Windows, Python 3.11, CPU)

`requirements.txt` stays light. The optional file `requirements-facenet.txt` (it includes `requirements.txt`) pins
`torch==2.6.0`, `torchvision==0.21.0`, `requests==2.32.3`, `tqdm==4.67.1` and `facenet-pytorch==2.6.0`.

**Do not run `pip install -r requirements-facenet.txt` directly.** As far as I know, `facenet-pytorch==2.6.0`'s
metadata still asks for `numpy<2`, `Pillow<10.3`, `torch<2.3` and `torchvision<0.18`, which conflict with this project's
pins (numpy 2.1.3, Pillow 11.3.0): pip would refuse or downgrade them. Nothing is downgraded; install in three steps.
Step 2 installs PyTorch **with** its own dependencies; step 3 skips only facenet-pytorch's stale bounds:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 requests==2.32.3 tqdm==4.67.1
.\venv\Scripts\python.exe -m pip install --no-deps facenet-pytorch==2.6.0
.\venv\Scripts\python.exe -c "import numpy, PIL, torch, torchvision, facenet_pytorch; print(numpy.__version__, PIL.__version__, torch.__version__, torchvision.__version__)"
```

Expected: `2.1.3 11.3.0 2.6.0 0.21.0` (numpy and Pillow unchanged). On Windows the PyPI wheels of torch and torchvision
are CPU builds, so no extra index is needed; all packages have Windows wheels for Python 3.11, and facenet-pytorch is a
pure-Python wheel. `pip check` will keep reporting facenet-pytorch's stale bounds: that is the documented, deliberate
exception. If `import torch` fails with a DLL error, install the current Microsoft Visual C++ Redistributable (x64).

MTCNN's own small detector weights ship inside the facenet-pytorch package (check:
`python -c "import facenet_pytorch, pathlib; print(sorted(p.name for p in (pathlib.Path(facenet_pytorch.__file__).parent/'data').glob('*.pt')))"`
should list `onet.pt pnet.pt rnet.pt`).

## Download the weights once, explicitly

```powershell
.\venv\Scripts\python.exe -m scripts.download_recognition_model --check
```

It downloads the `20180402-114759-vggface2.pt` release asset of the facenet-pytorch project on GitHub, saves it as
`facenet-vggface2.pt` in `RECOGNITION_MODEL_DIR` (or `--directory`), and prints the size and the SHA-256 of the file it
saved, computed from the file on disk. **No checksum is hard-coded** (none could be verified): write the printed one down
and compare it when you copy the file between machines. It never replaces an existing file without `--force`, and an
interrupted or too-small download leaves nothing behind. `--check` additionally loads the weights into the network
(needs the install above); `--check-only` does that for a file already in place, without downloading.
The API itself never downloads anything and loads the weights with `weights_only=True`, so the file cannot run code.

**Licence.** The pretrained weights derive from the **VGGFace2** dataset. The project's MIT licence covers its code, not
necessarily the dataset or the weights, and dataset terms and availability have changed over time. Read the current terms
of VGGFace2 and of this checkpoint and obtain approval **before any real deployment** or use with real people's images.

## Enable it

```dotenv
RECOGNITION_BACKEND=facenet
RECOGNITION_MODEL_DIR=models/recognition
```

Start the API as usual. Behaviour you can rely on:

- Nothing heavy happens at start-up. The models load on first use, under a lock. A missing or unreadable weights file or a
  missing or broken PyTorch raises `RecognitionUnavailableError`: the API starts and answers **503** for recognition,
  with a generic message (paths and library text never reach the response or the logs; the log carries a reason code
  such as `weights_missing`, `packages_missing`, `weights_invalid`).
- `/health` reports recognition `ready` only when the models are loaded and usable, otherwise `unconfigured` (the API
  stays healthy). The first call that finds everything in place loads the models (a few seconds); later calls never
  reload them. A health check never waits for a load that another request has started: it says "not ready yet".
- A missing file or package is re-checked cheaply on every call, so **no restart is needed** after you download the
  weights or install the packages. A failed heavy load (corrupt file) is retried at most every 30 seconds.
  Replacing the weights file of a running, already loaded process needs a restart.
- `model_info()`: name `facenet-vggface2`, dimension 512, version `facenet-pytorch-<package version>-vgg2-<first 12 hex of
  the weights file's SHA-256>-p1` (at most 50 characters, the database column limit; `p1` is the preprocessing revision).
  A different weights file, package version or preprocessing therefore never shares embeddings with old ones.
- Approving a case enrols its photos inside that request: expect seconds, not milliseconds, per photo on a CPU.

## Switching backend or weights: re-embedding

Embeddings carry the model name and version; search skips embeddings of any other version, so after a switch **searches
find nothing until the approved cases are re-enrolled**. On the server, from the project root:

1. Back up the database and the weights file.
2. Install the packages, download the weights, set `RECOGNITION_BACKEND` and `RECOGNITION_MODEL_DIR`, restart the API.
3. Check `GET /health` shows `"recognition": "ready"` (the first call may take a few seconds).
4. Run once (during a quiet period: it holds the CPU):

   ```powershell
   .\venv\Scripts\python.exe -m scripts.reembed_cases
   ```

5. Read the counts, e.g. `cases=3 cases_skipped=0 cases_failed=0 photos_done=7 photos_failed=1 photos_pending=0`.
   They are counts only (no names, ids or file names). `photos_failed` are photos without exactly one usable face
   (or whose file is missing) with the new model; they are not searchable until that is fixed (their old vectors, if any,
   belong to the old model and are ignored by search). Exit code 1 means the model was unavailable or `cases_failed` is above zero; run it
   again (finished photos are skipped, so repeating is harmless).
6. Run a controlled search with consented fictional or test images and check the human-review workflow. Do not compare
   scores between models.

Old vectors of a photo are **replaced** when the new embedding succeeds, so switching back means re-embedding again.

## Evaluate on your own data

Use a public dataset whose licence allows it (for example LFW) or photos with explicit informed consent. Never real case
photos, never committed (`evaluation-datasets/` is git-ignored). One folder per person:

```powershell
.\venv\Scripts\python.exe -m scripts.evaluate_recognition evaluation-datasets
.\venv\Scripts\python.exe -m scripts.evaluate_recognition evaluation-datasets --target-far 0.01,0.001 --max-images-per-person 10
.\venv\Scripts\python.exe -m scripts.evaluate_recognition evaluation-datasets --thresholds 0.5,0.6,0.7
```

Every image goes through the same one-face rule as the API; images with no usable face, several faces or that cannot be
read are counted and skipped (a model failure stops the run instead). All pairs of usable images are scored with cosine
similarity. It prints counts, the same-person and different-person distributions (min, 5th percentile, median, mean,
95th percentile, max; percentiles are exact to 0.00025) and, for threshold candidates, the false accepts (different
people at or above the threshold) and false rejects (same person below it), exactly counted, e.g.
`threshold=0.6120 false_accept=12/4000 (0.0030) false_reject=45/900 (0.0500)`. Candidates are the lowest threshold for
each target false-accept rate (only when the data has at least 1/rate different-person pairs) and the equal-error point,
or your own `--thresholds`. It reports what it measured and makes **no accuracy claim** and chooses nothing for you.
Pairs share images and are not independent, so there are no confidence intervals.

## Known limitations

- Large age gaps (the typical missing-person case), hairstyle or appearance changes, pose, occlusion and low resolution
  reduce similarity; poor light, motion blur and camera quality reduce detection and embedding quality.
- Masks and other face coverings hide identity-relevant features: expect misses and misleading similarities.
- Accuracy can differ between groups of people (skin tone, age, sex, ethnicity) and between datasets; aggregate numbers hide
  this. Evaluate on the population and conditions you will actually see.
- The detector probability is not identity confidence; a cosine similarity is not a probability.
- **Thresholds are not universal and must be calibrated on your own representative, lawfully collected data.** No
  threshold in this project is validated.
- CPU inference is slow; search is an exact brute-force comparison in memory.
- **Results are leads for human review only.** Nothing here can establish identity.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Recognition endpoints 503, log reason `weights_missing` | Weights not downloaded, or `RECOGNITION_MODEL_DIR` points elsewhere (relative path resolved from another folder) |
| `packages_missing` | Install steps above not done in this venv |
| `packages_unusable` | `import torch` fails: missing Visual C++ Redistributable, or a broken install |
| `weights_invalid` | Truncated or wrong file: run `--check-only`; re-download with `--force` |
| `/health` stays `unconfigured` | Any of the above; read the warning in the API log |

## What was not verified when this stage was written

No PyTorch, facenet-pytorch, PostgreSQL or network was available in the authoring environment. The loading code was
tested with fake torch modules; real MTCNN and InceptionResnetV1 behaviour (including that the checkpoint holds
`logits.*` tensors that must be dropped, and that facenet-pytorch 2.6.0 runs on numpy 2.1.3 / Pillow 11.3.0 / torch 2.6.0)
is checked only by `tests/test_facenet_torch.py` once torch is installed, and by `--check`. The download URL and the
facenet-pytorch 2.6.0 metadata are recalled from the project, not fetched.
~~~~~~

## docs/security.md

~~~~~~markdown
# Security and privacy

## Threat model

The service handles sensitive reports and face images. The principal risks are account takeover, privilege escalation, cross-user case access, unauthorised photo access, oversized/malformed uploads, abuse of login/recognition endpoints, host/origin spoofing, leakage through logs, and accidental retention after a privacy deletion request.

## Access controls

- Passwords are hashed; plaintext passwords are never stored.
- Access tokens identify a user, but the API reloads that user from PostgreSQL on each request. `is_active` and `role` are checked in the database, so deactivation immediately invalidates existing tokens.
- Public registration cannot assign an admin role.
- Admin routes require an active database-backed admin account.
- Case-owner routes scope queries to the authenticated reporter; unknown and other-owner cases are intentionally indistinguishable.
- An admin cannot change their own role/deactivate themselves. The last active admin cannot be removed from the active-admin set.

## HTTP protections

Stage 8A adds a consistent error envelope, request IDs, validation errors that omit rejected values, generic 500 responses, security headers, trusted-host checks, explicit CORS allow-lists, a global streamed body cap, and rate limits. The limiter is per process; multiple workers require Redis-backed shared state. Production startup refuses placeholder secrets, wildcard CORS, and enabled API docs.

## Privacy decisions

- Original client filenames are discarded. Stored image keys are server-generated and are not returned by the API.
- Images are normalised to JPEG with metadata removed before storage/recognition.
- Face embeddings and query images are never returned or written to search logs.
- Recognition results are leads, not identity proof. Human review is mandatory.
- `audit_events.details` is restricted to non-negative counts, booleans, and short codes. Do not add names, emails, addresses, free-form notes, image data, embeddings, tokens, or credentials.
- The Stage 5 status-event table remains the detailed case-status history; Stage 8B adds only an audit action marker, avoiding duplication of the review reason.
- Hard deletion permanently removes the case, its photos, embeddings, detections, and status events. Search-log references are nulled by the existing FK. Audit records survive as actor/action/target UUID/count metadata; they do not retain case contents.

## What is logged

Application logs may contain request ID, HTTP method/path/status/duration, opaque database IDs, event codes, and aggregate counts. They must not contain names, email addresses, phone numbers, street/location details, review text, original filenames, storage keys/paths, photo bytes, embeddings, access tokens, or passwords. A hard-delete storage failure is logged with the photo UUID only.

## Retention

The application does not currently run automatic retention/purge jobs for audit events or search logs. Operators must define and document a lawful retention period, restrict database/log access, and implement a reviewed retention process before production. Do not describe indefinite retention as a legal requirement; it is only the current software behavior.

## Data requests and deletion

1. Verify the requester's authority and applicable legal basis outside the API.
2. An administrator may permanently delete a case using `DELETE /admin/cases/{case_id}?confirm=true`.
3. The response reports database row counts and any photo IDs whose stored files could not be deleted. A failed file deletion is not silently treated as success; investigate it with the storage operator.
4. The case and dependent database rows are not recoverable through the API after commit. Ensure backups and external storage replicas follow the organisation's separate retention/deletion policy.
5. Audit metadata remains for accountability, but the case target has no FK and the case contents are not copied into audit details.

## Face recognition backend (optional FaceNet, Stage 6B)

- **Supply chain.** The weights file is downloaded once, explicitly, by an operator (`scripts/download_recognition_model.py`),
  never by the API. The script prints the size and SHA-256 of what it saved and hard-codes no checksum: record the value
  and compare it when the file is copied. The API loads the file with `weights_only=True`, so a tampered file cannot run
  code through pickle; a wrong or corrupt file is refused (`weights_invalid`) and recognition answers 503.
- **Install.** `facenet-pytorch==2.6.0` is installed with `--no-deps` because its metadata conflicts with the project's
  numpy and Pillow pins (docs/recognition.md). That bypasses pip's resolver for one package: keep all five optional packages pinned, install
  only from PyPI, and re-test on the target machine. `pip check` will keep reporting the stale bounds.
- **File permissions.** Keep `RECOGNITION_MODEL_DIR` readable by the API account only and writable by nobody while it runs;
  keep it, and any evaluation dataset, out of source control (`models/recognition/`, `evaluation-datasets/`, `*.pt` are git-ignored).
- **Licence and consent.** The weights derive from the VGGFace2 dataset: review its terms before any real deployment.
  Evaluate only on public data whose licence allows it or on photos with explicit informed consent; never on real case photos.
- **Resource abuse.** Inference is CPU-heavy. Only admins reach enrolment and search, both are rate limited, and inference is
  serialised per process; images are bounded by the upload limits and shrunk before detection. Each worker process holds its
  own models.
- **Logs and errors.** Recognition logs contain reason codes (`weights_missing`, `weights_invalid`, `packages_missing`,
  `inference` error type names) and ids/counts only. Paths, third-party exception text, image bytes and embeddings never
  reach logs or API responses (`tests/test_canary_logging.py`). The 503 body is generic.
- **Scripts print counts only.** `reembed_cases` and `evaluate_recognition` never print names, ids, file names or paths.
- **Results are leads, not proof.** A score is not a probability, thresholds are not validated until calibrated on your own
  data, accuracy can differ between groups of people, and a human must review every candidate (docs/recognition.md).

## Reporting a security issue

Do not include real people's records, images, passwords, or tokens in a public issue. Report suspected vulnerabilities privately to the system owner/security contact designated by the deploying organisation. Preserve only the minimum technical evidence needed to reproduce the issue, redact personal data, and rotate any exposed credentials immediately.

## Known limitations

- Recognition is unconfigured by default. The optional FaceNet CPU backend requires separately installed dependencies,
  locally downloaded weights and a licence review before real deployment.
- The rate limiter is process-local; it is not suitable for horizontally scaled/multi-worker deployments without Redis.
- Local file storage is a single-host backend, not a replicated object store.
- Hard deletion can commit while one or more file deletions fail; those photo IDs are reported, but an operator-controlled storage cleanup procedure is required.
- No automatic audit/search-log retention scheduler or legal-basis/consent workflow is implemented.
- Face search is brute-force and is not calibrated as an identity probability. FaceNet results are leads only and
  require human review.
- Face-recognition performance may vary with age gaps, lighting, pose, masks, image quality and across demographic
  groups. Evaluate subgroup performance on representative, lawfully collected and consented data.
- Similarity thresholds are dataset- and use-case-dependent. Calibrate false-accept/false-reject trade-offs on the
  deploying organisation's own evaluation data; never use a score as proof of identity.
- FaceNet weights are downloaded explicitly by an operator and loaded from a local directory. The API does not
  download model files at runtime. Review the VGGFace2 dataset/model terms before any real deployment.
~~~~~~

## docs/architecture.md

~~~~~~markdown
# Architecture

## Runtime layers

```text
HTTP client
  -> FastAPI middleware (request ID, request-size cap, safe access log, security headers)
  -> CORS / trusted-host middleware
  -> API routers and dependencies (authentication, current-user / admin authorization)
  -> service layer (business rules, transaction boundaries, audit events)
  -> SQLAlchemy 2.0 models and PostgreSQL
  -> local storage interface for image bytes
  -> RecognitionService interface (unconfigured by default; optional CPU FaceNet backend, docs/recognition.md)
```

## Main modules

| Area | Files | Responsibility |
| --- | --- | --- |
| Configuration/security | `app/core/config.py`, `app/core/security.py` | Validated environment settings, password hashing, signed access tokens |
| HTTP controls | `app/api/middleware.py`, `app/api/errors.py`, `app/api/rate_limits.py`, `app/core/rate_limit.py` | Request IDs, standard errors, security headers, body cap, rate limits |
| Authentication | `app/api/deps.py`, `app/api/routes/auth.py` | DB-backed user identity, active-account and admin checks |
| Cases/review | `app/services/cases.py`, `app/services/review.py` | Case ownership, locking, status transitions, hard deletion |
| Photos | `app/services/photos.py`, `app/services/images.py`, `app/storage/` | Validate and normalise image files; private storage abstraction |
| Recognition | `app/recognition/`, `app/services/embeddings.py`, `app/services/search.py` | Model contract, optional FaceNet backend, enrolment, similarity search |
| Detections | `app/services/detections.py`, `app/api/routes/admin_detections.py` | Admin-recorded observation history and review |
| Audit | `app/models/audit_event.py`, `app/services/audit.py`, `app/api/routes/admin_audit.py` | Append-only admin actions and paginated audit/search history |

## Database relationships

- `users` owns `missing_cases` through `reporter_id` (`RESTRICT`).
- `missing_cases` owns `case_photos`, `face_embeddings`, `case_status_events`, and `detections` (the latter rows are explicitly deleted in FK-safe order by the privacy-delete service).
- `face_search_logs.top_case_id` uses `ON DELETE SET NULL`, so the aggregate search record remains without pointing to a deleted case.
- `audit_events.actor_id` references `users.id` with `ON DELETE RESTRICT`.
- `audit_events.target_id` intentionally has **no foreign key**, allowing the audit record to survive deletion of its target.

## Transaction boundaries

- Case status and its Stage 5 `case_status_events` row are committed together. The Stage 8B audit marker is added in the same transaction; the status event remains the detailed history source, including the review reason.
- User role/activity changes and their audit events commit together. Active administrator rows are locked before checking the last-admin invariant.
- Hard deletion locks the case, removes detections, embeddings, status events, photos, and the case in one database transaction, and commits an audit event with counts. Only after commit does it remove image files. A file-delete failure cannot restore the committed database rows; its photo ID is returned and logged.
- Search log rows contain only administrator ID, counts, top candidate reference/similarity, and model identifiers. Query images and query embeddings are not stored.

## Migrations

Alembic is the only schema-management path. Migration IDs are sequential and linear: `0001` through `0008`. Importing `app.models` registers model metadata for Alembic autogeneration and inspection; it does not create tables at runtime.
~~~~~~

## docs/api.md

~~~~~~markdown
# API reference

All errors use the Stage 8A envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "Case not found",
    "request_id": "..."
  }
}
```

Validation errors additionally include `fields`, each with `field` and a safe `message`. Rejected values are never echoed. `X-Request-ID` matches `error.request_id`. Common error statuses are `400`, `401`, `403`, `404`, `409`, `413`, `415`, `422`, `429`, `500`, and `503` as applicable. Authentication failures retain `WWW-Authenticate`; rate-limit errors retain `Retry-After`.

## Authentication and system

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `GET /health` | Public | `200` | `503` when database or storage is unavailable; recognition is reported as `ready` only when a model is loaded and usable, otherwise `unconfigured`, and that alone does not fail health |
| `POST /auth/register` | Public | `201` | `409` duplicate email; `422` invalid payload; `429` registration limit |
| `POST /auth/login` | Public | `200` | `401` bad credentials; `422` malformed form; `429` IP/email limit |
| `GET /auth/me` | Active user | `200` | `401` missing/invalid/expired token or deactivated account |

## Cases

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `POST /cases` | Active user | `201` | `401`, `422` |
| `GET /cases` | Active user (own cases) | `200` | `401`, `422` pagination/filter errors |
| `GET /cases/{case_id}` | Owner | `200` | `401`, `404` (also for another user's case) |
| `GET /cases/{case_id}/detections` | Owner | `200` | `401`, `404`; only confirmed detections are returned |

## Photos

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `POST /cases/{case_id}/photos` | Owner | `201` | `401`, `404`, `409`, `413`, `415`, `422`, `429`, `503` |
| `GET /cases/{case_id}/photos` | Owner | `200` | `401`, `404` |
| `GET /cases/{case_id}/photos/{photo_id}` | Owner | JPEG bytes `200` | `401`, `404`, `503` |
| `DELETE /cases/{case_id}/photos/{photo_id}` | Owner | `204` | `401`, `404`, `409` |

## Administrator case review and privacy deletion

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `GET /admin/cases` | Admin | `200` | `401`, `403`, `422` |
| `GET /admin/cases/{case_id}` | Admin | `200` | `401`, `403`, `404` |
| `GET /admin/cases/{case_id}/photos/{photo_id}` | Admin | JPEG bytes `200` | `401`, `403`, `404`, `503` |
| `POST /admin/cases/{case_id}/approve` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `POST /admin/cases/{case_id}/reject` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `PATCH /admin/cases/{case_id}/status` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `DELETE /admin/cases/{case_id}?confirm=true` | Admin | `200` with deletion counts and failed photo IDs | `401`, `403`, `404`, `422` if `confirm=true` is absent/false |
| `POST /admin/cases/{case_id}/embeddings` | Admin | `200` | `401`, `403`, `404`, `409`, `429`, `503` |

The permanent-delete operation commits row deletion before attempting file deletion. `failed_photo_ids` must be checked by the operator; a non-empty list is not a successful complete file purge.

## Administrator user management, audit and search history

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `GET /admin/users` | Admin | `200` | `401`, `403`, `422` |
| `PATCH /admin/users/{user_id}` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `GET /admin/audit-events` | Admin | `200` paginated | `401`, `403`, `422` |
| `GET /admin/search-logs` | Admin | `200` paginated | `401`, `403`, `422` |

`PATCH /admin/users/{user_id}` accepts `{ "role": "user" | "admin", "is_active": true | false }`, with either field optional. Unknown fields and an empty update are rejected. An administrator cannot change their own role or deactivate themselves; the last active admin cannot be demoted/deactivated.

`GET /admin/audit-events` supports `action`, `actor_id`, `created_from`, `created_to`, `limit` (1..100), and `offset` (0+). `GET /admin/search-logs` supports `admin_id`, `searched_from`, `searched_to`, `limit`, and `offset`. Date filters must include a timezone and the start cannot be later than the end.

## Face search and detections

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `POST /search` | Admin | `200` ranked candidate leads | `401`, `403`, `413`, `415`, `422`, `429`, `503` |
| `POST /admin/cases/{case_id}/detections` | Admin | `201` | `401`, `403`, `404`, `409`, `422` |
| `GET /admin/cases/{case_id}/detections` | Admin | `200` paginated | `401`, `403`, `404`, `422` |
| `GET /admin/detections` | Admin | `200` paginated | `401`, `403`, `422` |
| `GET /admin/detections/{detection_id}` | Admin | `200` | `401`, `403`, `404` |
| `PATCH /admin/detections/{detection_id}/review` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |

Face search and embedding generation are rate-limited per admin; photo uploads per user. A detection is an observation/lead, not a confirmed identity.

## Query defaults

Paginated routes default to `limit=20&offset=0`. Detection lists and audit/search history cap `limit` at 100. Face search defaults to `SEARCH_DEFAULT_LIMIT` and is capped by `SEARCH_MAX_LIMIT`.
~~~~~~

## README.md

~~~~~~markdown
# Missing Person API

A Python 3.11 / FastAPI backend for case registration, administrator review, photo storage, face-enrolment/search interfaces, and detection history. The recognition backend is deliberately unconfigured by default: no embeddings, matches, or identities are fabricated. An optional CPU FaceNet backend (local weights, explicit one-time download) is described in [docs/recognition.md](docs/recognition.md).

> **Privacy and safety:** face-search results are leads, not proof of identity. An administrator must review a result before taking action. Use only with a lawful basis, clear notice, strict access control, and an approved retention policy.

## Requirements

- Windows 10/11 or another supported OS
- Python **3.11** (use the 64-bit installer)
- PostgreSQL 14 or newer
- Git (optional)

## Windows PowerShell setup

```powershell
# From the project folder
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation for the current terminal, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Edit `.env`. Set `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, and `POSTGRES_DB` to your local PostgreSQL values. Generate a signing key:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Put the generated value in `SECRET_KEY`. Never commit `.env`. The provided `.env.example` is a template, not a deployable secret configuration.

## PostgreSQL

Create a database and a least-privilege application user using pgAdmin or `psql`. Example (run as a PostgreSQL administrator and replace the example password):

```sql
CREATE ROLE missing_person_app LOGIN PASSWORD 'replace-with-a-strong-password';
CREATE DATABASE missing_person OWNER missing_person_app;
```

For tests, create a separate database whose name ends in `_test`, for example `missing_person_test`. The test fixture refuses to run against a database without that suffix and uses a temporary storage directory.

## Migrations

All schema changes are Alembic migrations. Do **not** use `Base.metadata.create_all()`.

```powershell
alembic upgrade head
alembic current
alembic history
```

Stage 8B adds migration `0008` after `0007`; it creates `audit_events`. Existing migrations are not rewritten.

## Run the API

```powershell
uvicorn app.main:app --reload
```

- Health: `http://127.0.0.1:8000/health`
- API docs in development: `http://127.0.0.1:8000/docs`
- Register: `POST /auth/register`; login: `POST /auth/login`

Create the first administrator using the controlled script (not public registration):

```powershell
python -m scripts.create_admin --email admin@example.com --full-name "System Administrator"
```

Follow the prompts and protect the resulting credentials. Public registration always creates a normal `user`.

## Tests and coverage

Use a separate PostgreSQL test database and set `POSTGRES_DB=missing_person_test` in the environment before running tests. `tests/conftest.py` selects a throwaway storage directory and refuses a non-test database.

```powershell
pytest
pytest --cov=app --cov-report=term-missing
```

The migration tests use the real Alembic chain and exercise upgrade/downgrade. Never point tests at a development or production database.

## Lint

```powershell
ruff check .
```

The Ruff configuration is in `pyproject.toml`; the pinned developer dependency is in `requirements-dev.txt`.

## Deployment notes

- `APP_ENV=production` requires real `SECRET_KEY` and `POSTGRES_PASSWORD`, explicit CORS origins (no `*`), and `ENABLE_API_DOCS=false`.
- Set `TRUSTED_HOSTS` to the actual host names and `CORS_ALLOWED_ORIGINS` to the exact browser origins that need access.
- The Stage 8A rate limiter is in-memory and **per process**. Use one worker or replace it with a shared Redis limiter before running multiple workers.
- Configure TLS at the reverse proxy. HSTS is emitted only in production mode.
- See [architecture](docs/architecture.md), [API reference](docs/api.md), [security](docs/security.md), [recognition](docs/recognition.md), and [hardening](docs/hardening.md).
~~~~~~

## tests/conftest.py

~~~~~~python
import os
import tempfile

from dotenv import dotenv_values

# Must happen BEFORE the app is imported: tests always use a separate database, a throwaway
# signing key that is worthless outside the test run, and a temporary photo storage folder.
STORAGE_ROOT = tempfile.mkdtemp(prefix="missing_person_test_storage_")
MODEL_ROOT = tempfile.mkdtemp(prefix="missing_person_test_models_")      # stays EMPTY: tests never load real weights
_LOCAL_ENV = dotenv_values(".env")
os.environ["APP_ENV"] = "test"
os.environ["LOG_JSON"] = "false"
os.environ["POSTGRES_DB"] = os.environ.get("TEST_POSTGRES_DB", "missing_person_test")
os.environ.setdefault(
    "POSTGRES_USER", os.environ.get("TEST_POSTGRES_USER") or _LOCAL_ENV.get("POSTGRES_USER") or "missing_person_test_user"
)
os.environ.setdefault(
    "POSTGRES_PASSWORD",
    os.environ.get("TEST_POSTGRES_PASSWORD") or _LOCAL_ENV.get("POSTGRES_PASSWORD") or "test-only-password-not-for-production",
)
os.environ["SECRET_KEY"] = "test-only-secret-key-never-use-outside-the-test-suite"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "30"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["STORAGE_DIR"] = STORAGE_ROOT
os.environ["MAX_UPLOAD_BYTES"] = str(5 * 1024 * 1024)
os.environ["MAX_PHOTOS_PER_CASE"] = "5"
os.environ["MAX_IMAGE_PIXELS"] = "25000000"
os.environ["MIN_IMAGE_SIDE"] = "100"
# Whatever a developer's .env says, tests start with NO real recognition backend (the unconfigured service), an empty
# model folder and the documented quality gates. Tests that need FaceNet settings set them themselves.
os.environ["RECOGNITION_BACKEND"] = "none"
os.environ["RECOGNITION_MODEL_DIR"] = MODEL_ROOT
os.environ["RECOGNITION_MIN_FACE_SIZE"] = "40"
os.environ["RECOGNITION_MIN_DETECTION_PROBABILITY"] = "0.90"
os.environ["MAX_REQUEST_BODY_BYTES"] = str(30 * 1024 * 1024)
os.environ["CORS_ALLOWED_ORIGINS"] = "[]"
os.environ["TRUSTED_HOSTS"] = '["*"]'
os.environ["ENABLE_API_DOCS"] = "true"
os.environ["LOGIN_IP_LIMIT"] = "10"
os.environ["LOGIN_EMAIL_LIMIT"] = "5"
os.environ["LOGIN_WINDOW_SECONDS"] = "60"
os.environ["REGISTRATION_IP_LIMIT"] = "5"
os.environ["REGISTRATION_WINDOW_SECONDS"] = "3600"
os.environ["FACE_SEARCH_ADMIN_LIMIT"] = "20"
os.environ["EMBEDDING_ADMIN_LIMIT"] = "10"
os.environ["ADMIN_RECOGNITION_WINDOW_SECONDS"] = "60"
os.environ["PHOTO_UPLOAD_USER_LIMIT"] = "10"
os.environ["PHOTO_UPLOAD_WINDOW_SECONDS"] = "60"

import shutil  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from io import BytesIO  # noqa: E402
from pathlib import Path  # noqa: E402

import image_factory  # noqa: E402  (tests/image_factory.py)
import pytest  # noqa: E402
from fake_recognition import FakeRecognitionService  # noqa: E402  (tests/fake_recognition.py: a FAKE, tests only)
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.security import create_access_token  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models.user import ROLE_USER  # noqa: E402
from app.recognition import get_recognition_service  # noqa: E402
from app.schemas.case import CaseCreate  # noqa: E402
from app.services import cases as case_service  # noqa: E402
from app.services import images as image_service  # noqa: E402
from app.services import photos as photo_service  # noqa: E402
from app.services import users as user_service  # noqa: E402
from app.storage import get_storage  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PASSWORD = "correct-horse-battery"

# Tables emptied before every test that touches the database. Add new tables here in later stages.
TABLES_TO_CLEAN = [
    "audit_events", "detections", "face_search_logs", "face_embeddings", "case_status_events", "case_photos", "missing_cases", "users",
]


@pytest.fixture(scope="session", autouse=True)
def _require_test_database():
    name = get_settings().postgres_db
    if not name.endswith("_test"):
        pytest.exit(f"Refusing to run: database '{name}' does not end with '_test'.", returncode=2)


@pytest.fixture(scope="session", autouse=True)
def _require_temporary_storage():
    """The storage folder is emptied before every test, so refuse to run against any other folder."""
    if Path(get_settings().storage_dir).resolve() != Path(STORAGE_ROOT).resolve():
        pytest.exit("Refusing to run: STORAGE_DIR is not the temporary test folder.", returncode=2)
    yield
    shutil.rmtree(STORAGE_ROOT, ignore_errors=True)
    shutil.rmtree(MODEL_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _empty_storage(_require_temporary_storage):
    """Every test starts with an empty photo storage."""
    for child in Path(STORAGE_ROOT).iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()


@pytest.fixture(scope="session")
def alembic_cfg() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    # Absolute path, so the tests work no matter which folder pytest is started from.
    script_location = (PROJECT_ROOT / "alembic").as_posix().replace("%", "%%")
    config.set_main_option("script_location", script_location)
    return config


@pytest.fixture(scope="session", autouse=True)
def _migrated_database(_require_test_database, alembic_cfg):
    """Build the schema with the real migrations (not metadata.create_all), so they are tested too."""
    command.upgrade(alembic_cfg, "head")


@pytest.fixture()
def clean_db():
    """Every test that uses the database starts with empty tables."""
    with engine.begin() as connection:
        for table in TABLES_TO_CLEAN:
            connection.execute(text(f'TRUNCATE TABLE "{table}" CASCADE'))


@pytest.fixture()
def db(clean_db):
    with SessionLocal() as session:
        yield session


@pytest.fixture()
def client(clean_db):
    app.state.rate_limiter.reset()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def make_user(db):
    """Create a user directly through the service layer (fast, no HTTP)."""

    def _make(
        email="user@example.com",
        password=TEST_PASSWORD,
        full_name="Test User",
        role=ROLE_USER,
        is_active=True,
    ):
        user = user_service.create_user(db, email=email, full_name=full_name, password=password, role=role)
        if not is_active:
            user.is_active = False
            db.commit()
        return user

    return _make


@pytest.fixture()
def auth_headers():
    """Authorization header for a user, built from a freshly signed token."""

    def _headers(user):
        return {"Authorization": f"Bearer {create_access_token(user.id)}"}

    return _headers


@pytest.fixture()
def case_payload():
    """A valid case request body made of entirely fictional data.
    Override any field: case_payload(age=30, city="Elsewhere")."""

    def _payload(**overrides):
        payload = {
            "name": "Test Person",
            "age": 12,
            "gender": "female",
            "contact_phone": "+1 555 010 0199",
            "contact_email": "family@example.com",
            "city": "Sampletown",
            "state": "Samplestate",
            "address": "1 Sample Street, Example Colony",
            "last_seen_location": "Sample Market",
            "last_seen_date": (date.today() - timedelta(days=3)).isoformat(),
        }
        payload.update(overrides)
        return payload

    return _payload


@pytest.fixture()
def make_case(db, case_payload):
    """Create a case directly through the service layer for the given owner."""

    def _make(owner, **overrides):
        return case_service.create_case(db, owner, CaseCreate(**case_payload(**overrides)))

    return _make


# ------------------------------------------------------------------ photos and storage
@pytest.fixture()
def storage():
    """The real LocalStorage the app uses in tests (rooted in the temporary folder)."""
    return get_storage()


@pytest.fixture()
def override_storage():
    """Replace the storage the API uses for one test: override_storage(my_fake_storage)."""

    def _override(replacement):
        app.dependency_overrides[get_storage] = lambda: replacement

    yield _override
    app.dependency_overrides.pop(get_storage, None)


@pytest.fixture()
def make_image_bytes():
    """Builds a fictional image with Pillow: make_image_bytes("PNG", size=(300, 200), seed=3)."""
    return image_factory.make_image_bytes


@pytest.fixture()
def make_png_with_declared_size():
    """A tiny PNG whose header claims a huge size (decompression-bomb style)."""
    return image_factory.make_png_with_declared_size


@pytest.fixture()
def make_noisy_png():
    """A PNG that does not compress, so it is big: for 'too many bytes' tests."""
    return image_factory.make_noisy_png


@pytest.fixture()
def upload_photos(client, auth_headers):
    """POST /cases/{id}/photos for `user`. Each item is raw bytes or a (file name, bytes, content type) tuple.
    Returns the response without asserting anything."""

    def _upload(user, case_id, *items):
        files = []
        for index, item in enumerate(items):
            name, data, content_type = item if isinstance(item, tuple) else (f"upload-{index}.jpg", item, "image/jpeg")
            files.append(("files", (name, data, content_type)))
        return client.post(f"/cases/{case_id}/photos", files=files, headers=auth_headers(user))

    return _upload


@pytest.fixture()
def make_photos(db, storage, make_image_bytes):
    """Add photos to a case through the service layer (real storage, no HTTP): make_photos(owner, case, 3)."""

    def _make(owner, case, count=1, first_seed=0):
        streams = [BytesIO(make_image_bytes("JPEG", seed=first_seed + index)) for index in range(count)]
        return photo_service.add_photos(db, storage, owner, case.id, streams)

    return _make


# ------------------------------------------------------------------ face recognition (a FAKE, tests only)
@pytest.fixture()
def fake_recognition():
    """Makes the API use a FakeRecognitionService for one test. Register what it 'finds' per image:
    fake_recognition.set_faces(photo.sha256, [1, 0, 0, 0]). Without this fixture the real, unconfigured
    service is used, which answers 503."""
    fake = FakeRecognitionService()
    app.dependency_overrides[get_recognition_service] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_recognition_service, None)


@pytest.fixture()
def final_sha():
    """The SHA-256 of the JPEG the application will give to the recognition service for an uploaded image
    (the normalised result of the Stage 4 pipeline): final_sha(raw_image_bytes)."""

    def _sha(raw: bytes) -> str:
        return image_service.process_image_bytes(raw, photo_service._limits()).sha256

    return _sha
~~~~~~

## tests/test_config.py

~~~~~~python
import pytest
from pydantic import ValidationError

from app.core.config import Settings

DB = {"postgres_user": "u", "postgres_password": "s3cr3t-pw", "postgres_db": "d"}
GOOD_KEY = "k" * 32


def test_database_credentials_are_required(monkeypatch):
    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, secret_key=GOOD_KEY)   # no .env, no DB env vars -> must fail, no defaults


def test_password_is_not_exposed_in_repr():
    s = Settings(_env_file=None, secret_key=GOOD_KEY, **DB)
    assert "s3cr3t-pw" not in repr(s)
    assert "s3cr3t-pw" not in s.database_url.render_as_string(hide_password=True)


def test_special_characters_in_password_are_encoded():
    s = Settings(_env_file=None, secret_key=GOOD_KEY, **{**DB, "postgres_password": "p@ss:w/rd#1"})
    url = s.database_url
    assert url.password == "p@ss:w/rd#1"
    assert "p%40ss%3Aw%2Frd%231" in url.render_as_string(hide_password=False)


def test_secret_key_is_required_and_has_no_default(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, **DB)
    assert "secret_key" in str(exc_info.value)


def test_secret_key_must_be_at_least_32_characters_and_is_not_echoed():
    too_short = "too-short-marker-0123456789"          # 27 characters
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, secret_key=too_short, **DB)
    assert "32" in str(exc_info.value)
    assert too_short not in str(exc_info.value)
    assert Settings(_env_file=None, secret_key="k" * 32, **DB).secret_key.get_secret_value() == "k" * 32


def test_secret_key_is_not_exposed_in_repr():
    key = "visible-only-through-get-secret-value-0123456789"
    s = Settings(_env_file=None, secret_key=key, **DB)
    assert key not in repr(s)
    assert key not in str(s)


def test_jwt_settings_have_sensible_defaults(monkeypatch):
    monkeypatch.delenv("JWT_ALGORITHM", raising=False)
    monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES", raising=False)
    s = Settings(_env_file=None, secret_key=GOOD_KEY, **DB)
    assert s.jwt_algorithm == "HS256"
    assert s.access_token_expire_minutes == 30


def test_storage_and_upload_settings_have_documented_defaults(monkeypatch):
    for name in (
        "STORAGE_BACKEND", "STORAGE_DIR", "MAX_UPLOAD_BYTES", "MAX_PHOTOS_PER_CASE", "MAX_IMAGE_PIXELS", "MIN_IMAGE_SIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None, secret_key=GOOD_KEY, **DB)
    assert s.storage_backend == "local"
    assert s.storage_dir == "storage"
    assert s.max_upload_bytes == 5 * 1024 * 1024
    assert s.max_photos_per_case == 5
    assert s.max_image_pixels == 25_000_000
    assert s.min_image_side == 100


def test_only_the_local_storage_backend_is_allowed():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, secret_key=GOOD_KEY, storage_backend="s3", **DB)


def test_upload_limits_must_be_sensible():
    for bad in ({"max_upload_bytes": 0}, {"max_photos_per_case": 0}, {"max_photos_per_case": 21},
                {"max_image_pixels": 0}, {"min_image_side": 0}, {"storage_dir": ""}):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, secret_key=GOOD_KEY, **DB, **bad)


def test_recognition_and_search_settings_have_documented_defaults(monkeypatch):
    for name in ("RECOGNITION_BACKEND", "SEARCH_DEFAULT_LIMIT", "SEARCH_MAX_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None, secret_key=GOOD_KEY, **DB)
    assert s.recognition_backend == "none"
    assert (s.search_default_limit, s.search_max_limit) == (5, 20)


def test_only_supported_recognition_backends_are_allowed():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, secret_key=GOOD_KEY, recognition_backend="real-model", **DB)


def test_search_limits_must_be_sensible():
    for bad in ({"search_default_limit": 0}, {"search_max_limit": 0}, {"search_max_limit": 101},
                {"search_default_limit": 21, "search_max_limit": 20}):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, secret_key=GOOD_KEY, **DB, **bad)



def test_facenet_settings_have_safe_defaults_and_accept_only_known_backends(monkeypatch):
    for name in (
        "RECOGNITION_BACKEND", "RECOGNITION_MODEL_DIR",
        "RECOGNITION_MIN_FACE_SIZE", "RECOGNITION_MIN_DETECTION_PROBABILITY",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, secret_key=GOOD_KEY, **DB)
    assert settings.recognition_backend == "none"
    assert settings.recognition_model_dir == "models/recognition"
    assert settings.recognition_min_face_size == 40
    assert settings.recognition_min_detection_probability == 0.90
    facenet = Settings(
        _env_file=None, secret_key=GOOD_KEY, recognition_backend="facenet",
        recognition_model_dir="D:/models/facenet", recognition_min_face_size=60,
        recognition_min_detection_probability=0.95, **DB,
    )
    assert facenet.recognition_backend == "facenet"
    assert facenet.recognition_min_face_size == 60
    assert facenet.recognition_min_detection_probability == 0.95


def test_facenet_quality_settings_and_model_directory_are_validated():
    invalid = (
        {"recognition_backend": "other"},
        {"recognition_model_dir": ""},
        {"recognition_min_face_size": 0},
        {"recognition_min_face_size": 1001},
        {"recognition_min_detection_probability": 0},
        {"recognition_min_detection_probability": 1.01},
    )
    for values in invalid:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, secret_key=GOOD_KEY, **DB, **values)


def test_facenet_gate_bounds_are_inclusive_and_bad_values_are_refused():
    for good in ({"recognition_min_face_size": 12}, {"recognition_min_face_size": 1000},
                 {"recognition_min_detection_probability": 1.0}, {"recognition_min_detection_probability": 0.01}):
        Settings(_env_file=None, secret_key=GOOD_KEY, recognition_backend="facenet", **DB, **good)
    for bad in ({"recognition_min_face_size": 11}, {"recognition_min_face_size": "big"},
                {"recognition_min_detection_probability": float("nan")},
                {"recognition_min_detection_probability": float("inf")},
                {"recognition_min_detection_probability": -0.5}, {"recognition_model_dir": "   "}):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, secret_key=GOOD_KEY, **DB, **bad)


def test_a_missing_weights_folder_is_not_a_configuration_error(tmp_path):
    # Selecting facenet must never fail settings validation (and so never start-up): the folder is only used lazily.
    settings = Settings(
        _env_file=None, secret_key=GOOD_KEY, recognition_backend="facenet",
        recognition_model_dir=str(tmp_path / "does-not-exist"), **DB,
    )
    assert settings.recognition_backend == "facenet"
    assert not (tmp_path / "does-not-exist").exists()
~~~~~~

## tests/test_health.py

~~~~~~python
import numpy as np
from fake_recognition import FakeRecognitionService

from app.api.routes import health as health_route
from app.recognition import FaceNetRecognitionService
from app.recognition import facenet
from app.recognition.base import ModelInfo
from app.recognition.facenet import WEIGHTS_FILENAME, _Loaded


def test_health_ok_when_database_is_reachable(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["storage"] == "ok"
    assert body["recognition"] == "unconfigured"
    assert body["version"]
    assert response.headers["x-request-id"]


def test_health_returns_503_when_database_is_down(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.health.check_database", lambda: False)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded", "version": "0.1.0", "database": "unavailable",
        "storage": "ok", "recognition": "unconfigured",
    }


def test_health_reports_recognition_unconfigured_when_facenet_weights_are_missing(client, monkeypatch, tmp_path):
    service = FaceNetRecognitionService(model_dir=tmp_path)
    monkeypatch.setattr(health_route, "get_recognition_service", lambda: service)
    response = client.get("/health")
    assert response.status_code == 200                       # a missing model does not make the API unhealthy
    assert response.json()["recognition"] == "unconfigured"
    assert service.is_configured is False


def test_health_reports_recognition_ready_for_a_loaded_service(client, monkeypatch):
    monkeypatch.setattr(health_route, "get_recognition_service", FakeRecognitionService)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["recognition"] == "ready"


def test_health_is_ready_only_when_facenet_can_really_be_used_and_never_reloads_it(client, monkeypatch, tmp_path):
    service = FaceNetRecognitionService(model_dir=tmp_path)
    monkeypatch.setattr(health_route, "get_recognition_service", lambda: service)
    loads = []

    def load():
        loads.append(1)
        return _Loaded(object(), object(), object(), ModelInfo("facenet-vggface2", "v", 512))

    monkeypatch.setattr(service, "_load_models", load)
    monkeypatch.setattr(facenet, "_package_available", lambda name: True)
    assert client.get("/health").json()["recognition"] == "unconfigured"          # no weights file yet
    assert loads == []                                                              # and nothing heavy was tried
    (tmp_path / WEIGHTS_FILENAME).write_bytes(np.zeros(facenet.MIN_PLAUSIBLE_WEIGHTS_BYTES, dtype=np.uint8).tobytes())
    for _ in range(3):                                                              # a polling load balancer
        assert client.get("/health").json()["recognition"] == "ready"
    assert len(loads) == 1
~~~~~~

## tests/test_canary_logging.py

~~~~~~python
import logging
import uuid
from datetime import datetime, timezone

from app.models.user import User


def test_full_user_journey_does_not_log_personal_data_credentials_or_tokens(
    client, db, make_user, make_image_bytes, final_sha, fake_recognition, auth_headers, caplog
):
    personal = {
        "admin_email": "canary_admin_zorpax@example.test",
        "full_name": "CANARY_FULLNAME_ZORPAX",
        "email": "canary_email_zorpax@example.test",
        "case_name": "CANARY_CASE_NAME_ZORPAX",
        "phone": "+120255501987654",
        "case_email": "canary_contact_zorpax@example.test",
        "city": "CANARY_CITY_ZORPAX",
        "state": "CANARY_STATE_ZORPAX",
        "address": "CANARY_ADDRESS_ZORPAX_77",
        "last_seen_location": "CANARY_LOCATION_ZORPAX",
        "password": "CANARY_PASSWORD_ZORPAX_123!",
        "detection_location": "CANARY_DETECTION_PLACE_ZORPAX",
        "filename": "CANARY_FILENAME_ZORPAX.png",
    }
    admin = make_user(email=personal["admin_email"], role="admin")
    image = make_image_bytes("PNG", size=(240, 240), seed=67)
    fake_recognition.set_faces(final_sha(image), [1, 0, 0, 0])

    with caplog.at_level(logging.INFO):
        registered = client.post("/auth/register", json={
            "email": personal["email"], "full_name": personal["full_name"], "password": personal["password"],
        })
        assert registered.status_code == 201, registered.text
        login = client.post("/auth/login", data={"username": personal["email"], "password": personal["password"]})
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
        owner = db.get(User, uuid.UUID(registered.json()["id"]))
        owner_headers = auth_headers(owner)
        created = client.post("/cases", headers=owner_headers, json={
            "name": personal["case_name"], "age": 19, "gender": "female",
            "contact_phone": personal["phone"], "contact_email": personal["case_email"],
            "city": personal["city"], "state": personal["state"], "address": personal["address"],
            "last_seen_location": personal["last_seen_location"],
            "last_seen_date": "2026-10-20",
        })
        assert created.status_code == 201, created.text
        case_id = created.json()["id"]
        uploaded = client.post(
            f"/cases/{case_id}/photos", headers=owner_headers,
            files={"files": (personal["filename"], image, "image/png")},
        )
        assert uploaded.status_code == 201, uploaded.text
        approved = client.post(f"/admin/cases/{case_id}/approve", headers=auth_headers(admin))
        assert approved.status_code == 200, approved.text
        enrolled = client.post(f"/admin/cases/{case_id}/embeddings", headers=auth_headers(admin))
        assert enrolled.status_code == 200, enrolled.text
        searched = client.post(
            "/search", headers=auth_headers(admin),
            files={"file": (personal["filename"], image, "image/png")},
        )
        assert searched.status_code == 200, searched.text
        detected = client.post(
            f"/admin/cases/{case_id}/detections", headers=auth_headers(admin),
            json={
                "detected_at": datetime.now(timezone.utc).isoformat(), "source": "manual",
                "location_text": personal["detection_location"], "confidence": 0.88,
            },
        )
        assert detected.status_code == 201, detected.text

    log_text = caplog.text + "\n" + "\n".join(str(record.__dict__) for record in caplog.records)
    for value in personal.values():
        assert value not in log_text, value
    assert token not in log_text
    assert personal["password"] not in log_text
    assert "case_status_changed" in log_text or "case_status_change" in log_text
    assert "face_search" in log_text
    assert "detection_recorded" in log_text


def test_the_facenet_service_never_logs_paths_exception_text_or_image_bytes(tmp_path, caplog, monkeypatch):
    """FaceNet touches files and third-party libraries, which can put paths and messages into exceptions. Neither the
    model folder, the weights path, the library's text nor the image bytes may reach a log or an exception message."""
    from io import BytesIO

    from PIL import Image

    from app.recognition import FaceNetRecognitionService, RecognitionUnavailableError
    from app.recognition import facenet

    secret_dir = tmp_path / "CANARY_MODEL_FOLDER_ZORPAX"
    service = FaceNetRecognitionService(model_dir=secret_dir)
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            try:
                service.model_info()                                   # weights missing
            except RecognitionUnavailableError as exc:
                assert "CANARY" not in str(exc)
        secret_dir.mkdir()
        (secret_dir / facenet.WEIGHTS_FILENAME).write_bytes(b"\0" * facenet.MIN_PLAUSIBLE_WEIGHTS_BYTES)
        monkeypatch.setattr(facenet, "_package_available", lambda name: True)

        def boom():
            raise RuntimeError(f"CANARY_LIBRARY_TEXT {secret_dir}")

        monkeypatch.setattr(service, "_load_models", boom)
        try:
            service.model_info()                                       # heavy load fails with a revealing message
        except RecognitionUnavailableError as exc:
            assert "CANARY" not in str(exc) and str(secret_dir) not in str(exc)

        buffer = BytesIO()
        Image.new("RGB", (10, 10), "white").save(buffer, format="PNG")
        ready = FaceNetRecognitionService(model_dir=secret_dir)
        monkeypatch.setattr(ready, "_preflight", lambda: None)
        monkeypatch.setattr(ready, "_load_models", lambda: (_ for _ in ()).throw(ValueError("CANARY_X")))
        try:
            ready.extract_faces(buffer.getvalue() + b"CANARY_IMAGE_BYTES")
        except RecognitionUnavailableError:
            pass

    log_text = caplog.text + "\n" + "\n".join(str(record.__dict__) for record in caplog.records)
    for canary in ("CANARY", "ZORPAX", str(secret_dir), "CANARY_IMAGE_BYTES"):
        assert canary not in log_text, canary
    assert "weights_missing" in log_text
~~~~~~

## tests/test_facenet_recognition.py

~~~~~~python
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
~~~~~~

## tests/test_facenet_torch.py

~~~~~~python
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
~~~~~~

## tests/test_recognition_scripts.py

~~~~~~python
"""The three Stage 6B scripts. None of these tests needs torch, a real model, a real dataset or the network."""
import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from fake_recognition import FakeRecognitionService
from sqlalchemy import select

from app.models.face_embedding import FaceEmbedding
from app.recognition import UnconfiguredRecognitionService
from scripts import download_recognition_model, evaluate_recognition, reembed_cases

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


# ------------------------------------------------------------------ argument handling
def test_download_script_help_does_not_download():
    with pytest.raises(SystemExit) as exc:
        download_recognition_model.main(["--help"])
    assert exc.value.code == 0


def test_reembed_script_help_does_not_load_a_model():
    with pytest.raises(SystemExit) as exc:
        reembed_cases.main(["--help"])
    assert exc.value.code == 0


def test_evaluation_script_help_does_not_load_a_model():
    with pytest.raises(SystemExit) as exc:
        evaluate_recognition.main(["--help"])
    assert exc.value.code == 0


def test_the_scripts_refuse_unknown_arguments():
    for script in (download_recognition_model, reembed_cases, evaluate_recognition):
        with pytest.raises(SystemExit) as exc:
            script.main(["--no-such-option"])
        assert exc.value.code == 2


def test_threshold_and_rate_parsers_accept_valid_values_and_reject_the_rest():
    assert evaluate_recognition._threshold_values("-0.5, 0.5,0.5") == [-0.5, 0.5]
    assert evaluate_recognition._far_values("0.01,0.001") == [0.001, 0.01]
    for parser, bad in (
        (evaluate_recognition._threshold_values, "0.5,1.2"), (evaluate_recognition._threshold_values, "abc"),
        (evaluate_recognition._threshold_values, ""), (evaluate_recognition._threshold_values, "nan"),
        (evaluate_recognition._far_values, "0"), (evaluate_recognition._far_values, "1"),
        (evaluate_recognition._positive_int, "0"), (evaluate_recognition._positive_int, "x"),
    ):
        with pytest.raises(Exception):
            parser(bad)
    assert evaluate_recognition._positive_int("3") == 3


# ------------------------------------------------------------------ evaluation script
def _dataset(root, make_image_bytes, fake):
    """Two fictional people with two images each, plus one image without a face and one with two faces.
    Folder and file names are canaries: they must never be printed."""
    layout = {
        "CANARY_PERSON_A": [[1, 0, 0, 0], [0.9, 0.1, 0, 0]],
        "CANARY_PERSON_B": [[0, 1, 0, 0], [0.1, 0.9, 0, 0]],
    }
    seed = 0
    for person, vectors in layout.items():
        (root / person).mkdir()
        for index, vector in enumerate(vectors):
            seed += 1
            data = make_image_bytes("PNG", seed=seed)
            (root / person / f"CANARY_FILE_{index}.png").write_bytes(data)
            fake.set_faces(data, vector)
    (root / "CANARY_PERSON_C").mkdir()
    no_face = make_image_bytes("PNG", seed=90)
    (root / "CANARY_PERSON_C" / "CANARY_NOFACE.png").write_bytes(no_face)
    two_faces = make_image_bytes("PNG", seed=91)
    (root / "CANARY_PERSON_C" / "CANARY_TWO.png").write_bytes(two_faces)
    fake.set_faces(two_faces, [1, 0, 0, 0], [0, 1, 0, 0])
    (root / "CANARY_PERSON_C" / "notes.txt").write_text("not an image")


def test_evaluation_empty_folder_prints_counts_only_and_never_loads_a_model(tmp_path, capsys, monkeypatch):
    def must_not_be_called():
        pytest.fail("the model must not be loaded for an empty dataset")

    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", must_not_be_called)
    (tmp_path / "CANARY_EMPTY_PERSON").mkdir()
    assert evaluate_recognition.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "No images found. images=0 usable=0 skipped=0" in output
    assert "CANARY" not in output


def test_evaluation_refuses_a_missing_folder_without_loading_a_model(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: pytest.fail("no model"))
    assert evaluate_recognition.main([str(tmp_path / "missing")]) == 2
    assert "dataset folder does not exist" in capsys.readouterr().out


def test_evaluation_reports_counts_and_measured_rates_and_prints_no_names(tmp_path, capsys, monkeypatch, make_image_bytes):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path), "--thresholds", "0.0,0.15,0.5,0.999"]) == 0
    output = capsys.readouterr().out
    assert "images=6 usable=4 skipped=2 skipped_no_face=1 skipped_multiple_faces=1 skipped_unreadable=0" in output
    assert "people=3 people_with_usable_images=2 people_with_2_or_more_usable_images=2" in output
    assert "same_person_pairs=2 different_person_pairs=4" in output
    # Scores: same-person 0.9939 (x2); different-person 0.0, 0.1104, 0.1104, 0.2195.
    assert re.search(r"threshold=0\.0000 false_accept=4/4 \(1\.0000\) false_reject=0/2 \(0\.0000\)", output)
    assert re.search(r"threshold=0\.1500 false_accept=1/4 \(0\.2500\) false_reject=0/2 \(0\.0000\)", output)
    assert re.search(r"threshold=0\.5000 false_accept=0/4 \(0\.0000\) false_reject=0/2 \(0\.0000\)", output)
    assert re.search(r"threshold=0\.9990 false_accept=0/4 \(0\.0000\) false_reject=2/2 \(1\.0000\)", output)
    assert "not accuracy claims" in output
    assert "CANARY" not in output and "notes" not in output and str(tmp_path) not in output


def test_evaluation_derives_candidates_and_says_when_a_rate_cannot_be_resolved(tmp_path, capsys, monkeypatch, make_image_bytes):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path), "--target-far", "0.5,0.0001"]) == 0
    output = capsys.readouterr().out
    assert "target_far=0.5 threshold=" in output
    assert "equal_error_point threshold=" in output
    assert "target_far=0.0001 not resolvable: 4 different-person pairs (at least 10000 are needed)" in output


def test_evaluation_limits_images_per_person(tmp_path, capsys, monkeypatch, make_image_bytes):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path), "--max-images-per-person", "1"]) == 0
    output = capsys.readouterr().out
    assert "same_person_pairs=0 different_person_pairs=1" in output
    assert "Threshold trade-off unavailable" in output


def test_a_model_that_breaks_during_the_run_stops_it_instead_of_counting_skipped_images(
    tmp_path, capsys, monkeypatch, make_image_bytes
):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    fake.fail_on_call = 2
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "recognition model is unavailable" in output
    assert "usable=" not in output and "CANARY" not in output


def test_evaluation_with_no_backend_configured_says_so(tmp_path, capsys, monkeypatch, make_image_bytes):
    _dataset(tmp_path, make_image_bytes, FakeRecognitionService())
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", UnconfiguredRecognitionService)
    assert evaluate_recognition.main([str(tmp_path)]) == 1
    assert "RECOGNITION_BACKEND=facenet" in capsys.readouterr().out


def test_pair_scores_cover_every_pair_once_even_across_blocks(monkeypatch):
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(7, 5)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, 1, 2, 3])
    monkeypatch.setattr(evaluate_recognition, "BLOCK_ROWS", 3)
    same, different = [], []
    for same_scores, different_scores in evaluate_recognition.iter_pair_scores(matrix, labels):
        same.extend(same_scores.tolist())
        different.extend(different_scores.tolist())
    expected_same, expected_different = [], []
    for i in range(7):
        for j in range(i + 1, 7):
            (expected_same if labels[i] == labels[j] else expected_different).append(float(matrix[i] @ matrix[j]))
    assert sorted(same) == pytest.approx(sorted(expected_same), abs=1e-6)
    assert sorted(different) == pytest.approx(sorted(expected_different), abs=1e-6)
    assert len(same) + len(different) == 21


def test_derived_thresholds_match_the_scores_they_were_derived_from():
    rng = np.random.default_rng(0)
    same_scores = np.clip(rng.normal(0.8, 0.05, 2000), -1, 1)
    different_scores = np.clip(rng.normal(0.1, 0.1, 20000), -1, 1)
    same, different = evaluate_recognition.Distribution(), evaluate_recognition.Distribution()
    same.add(same_scores)
    different.add(different_scores)
    candidates, notes = evaluate_recognition.derive_candidates(same, different, [0.01, 0.001])
    assert notes == []
    by_name = dict(candidates)
    step = 0.0005
    for far, name in ((0.01, "target_far=0.01"), (0.001, "target_far=0.001")):
        threshold = by_name[name]
        assert (different_scores >= threshold).mean() <= far                       # meets the target ...
        assert (different_scores >= threshold - step).mean() > far                 # ... and is the lowest edge that does
    eer = by_name["equal_error_point"]
    far_rate, frr_rate = (different_scores >= eer).mean(), (same_scores < eer).mean()
    assert abs(far_rate - frr_rate) < 0.01
    assert different.percentile(50) == pytest.approx(float(np.median(different_scores)), abs=0.001)
    assert same.minimum == float(same_scores.min()) and different.maximum == float(different_scores.max())


# ------------------------------------------------------------------ download script (no network)
class _FakeResponse:
    def __init__(self, data, declared=None, fail_after=None):
        self._data, self._pos, self._fail_after = data, 0, fail_after
        self.headers = {} if declared is None else {"Content-Length": str(declared)}

    def read(self, size):
        if self._fail_after is not None and self._pos >= self._fail_after:
            raise OSError("connection dropped")
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


PAYLOAD = b"fictional weights " * 100_000          # 1.8 MB: above the plausibility floor


def _serve(monkeypatch, response):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: response)


def test_download_saves_the_file_and_prints_its_size_and_sha256(tmp_path, capsys, monkeypatch):
    _serve(monkeypatch, _FakeResponse(PAYLOAD, declared=len(PAYLOAD)))
    assert download_recognition_model.main(["--directory", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    saved = tmp_path / download_recognition_model.WEIGHTS_FILENAME
    assert saved.read_bytes() == PAYLOAD
    assert f"Size: {len(PAYLOAD)} bytes" in output
    assert f"SHA-256: {hashlib.sha256(PAYLOAD).hexdigest()}" in output
    assert "github.com/timesler/facenet-pytorch" in output and "licence" in output
    assert [p.name for p in tmp_path.iterdir()] == [saved.name]                    # no temporary file left behind


def test_download_never_replaces_an_existing_file_without_force(tmp_path, capsys, monkeypatch):
    saved = tmp_path / download_recognition_model.WEIGHTS_FILENAME
    saved.write_bytes(b"original")
    _serve(monkeypatch, _FakeResponse(PAYLOAD))
    assert download_recognition_model.main(["--directory", str(tmp_path)]) == 2
    assert saved.read_bytes() == b"original"
    assert download_recognition_model.main(["--directory", str(tmp_path), "--force"]) == 0
    assert saved.read_bytes() == PAYLOAD
    capsys.readouterr()


def test_a_failed_incomplete_or_tiny_download_saves_nothing(tmp_path, capsys, monkeypatch):
    for response in (
        _FakeResponse(PAYLOAD, fail_after=500_000),             # connection drops half way
        _FakeResponse(PAYLOAD, declared=len(PAYLOAD) + 10),     # fewer bytes than announced
        _FakeResponse(b"<html>not the weights</html>"),         # far too small
    ):
        _serve(monkeypatch, response)
        assert download_recognition_model.main(["--directory", str(tmp_path)]) == 1
        assert "could not be downloaded or saved" in capsys.readouterr().out
        assert list(tmp_path.iterdir()) == []


def test_check_only_needs_an_existing_file_and_reports_an_unloadable_one(tmp_path, capsys):
    assert download_recognition_model.main(["--directory", str(tmp_path), "--check-only"]) == 2
    (tmp_path / download_recognition_model.WEIGHTS_FILENAME).write_bytes(PAYLOAD)
    assert download_recognition_model.main(["--directory", str(tmp_path), "--check-only"]) == 1    # not a checkpoint
    output = capsys.readouterr().out
    assert f"SHA-256: {hashlib.sha256(PAYLOAD).hexdigest()}" in output and "Load check: failed" in output


def test_no_checksum_is_hard_coded_anywhere_in_the_scripts():
    for script in SCRIPTS_DIR.glob("*.py"):
        assert not re.search(r"\b[0-9a-f]{64}\b", script.read_text(encoding="utf-8")), script.name


# ------------------------------------------------------------------ re-embedding script (needs the test database)
class _SessionContext:
    def __init__(self, db):
        self.db = db

    def __call__(self):
        return self

    def __enter__(self):
        return self.db

    def __exit__(self, *args):
        return False


@pytest.fixture()
def reembed_env(db, storage, monkeypatch, fake_recognition):
    monkeypatch.setattr(reembed_cases, "_session_context", _SessionContext(db))
    monkeypatch.setattr(reembed_cases, "get_storage", lambda: storage)
    monkeypatch.setattr(reembed_cases, "get_recognition_service", lambda: fake_recognition)
    return fake_recognition


def _approved_case(db, make_user, make_case, make_photos, fake, email, name, seed, vector=(1, 0, 0, 0)):
    owner = make_user(email=email)
    case = make_case(owner, name=name)
    photos = make_photos(owner, case, 1, first_seed=seed)
    fake.set_faces(photos[0].sha256, list(vector))
    case.status = "approved"                                  # set directly: no approval hook runs
    db.commit()
    return case, photos


def test_reembed_prints_counts_only_and_a_second_run_changes_nothing(
    db, make_user, make_case, make_photos, reembed_env, capsys
):
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "canary-owner-1@example.test", "CANARY Case One", 11)
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "canary-owner-2@example.test", "CANARY Case Two", 22)
    assert reembed_cases.main([]) == 0
    output = capsys.readouterr().out
    assert output.strip() == "cases=2 cases_skipped=0 cases_failed=0 photos_done=2 photos_failed=0 photos_pending=0"
    calls_after_first = len(reembed_env.calls)
    assert reembed_cases.main([]) == 0
    assert len(reembed_env.calls) == calls_after_first                           # nothing was processed again
    assert "CANARY" not in output and "canary" not in output


def test_only_approved_cases_are_processed(db, make_user, make_case, make_photos, reembed_env, capsys):
    case, _ = _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 31)
    pending_owner = make_user(email="o2@example.test")
    pending_case = make_case(pending_owner, name="Fictional Two")
    make_photos(pending_owner, pending_case, 1, first_seed=32)
    assert reembed_cases.main([]) == 0
    assert capsys.readouterr().out.startswith("cases=1 ")
    assert db.scalar(select(FaceEmbedding.id).where(FaceEmbedding.case_id == pending_case.id)) is None


def test_a_new_model_version_replaces_the_old_embeddings(db, make_user, make_case, make_photos, reembed_env, capsys):
    case, photos = _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 41)
    assert reembed_cases.main([]) == 0
    reembed_env.version = "2"                                                  # the backend / weights were switched
    reembed_env.set_faces(photos[0].sha256, [0, 1, 0, 0])
    assert reembed_cases.main([]) == 0
    capsys.readouterr()
    db.expire_all()
    rows = list(db.scalars(select(FaceEmbedding).where(FaceEmbedding.case_id == case.id)))
    assert [row.model_version for row in rows] == ["2"]                         # replaced, not duplicated


def test_a_problem_with_one_case_is_counted_and_the_others_are_still_processed(
    db, make_user, make_case, make_photos, reembed_env, capsys
):
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 51)
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o2@example.test", "Fictional Two", 52)
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o3@example.test", "Fictional Three", 53)
    reembed_env.crash_on_call = 2                                               # the second photo blows up
    assert reembed_cases.main([]) == 1
    assert capsys.readouterr().out.strip() == (
        "cases=3 cases_skipped=0 cases_failed=1 photos_done=2 photos_failed=0 photos_pending=0"
    )


def test_an_unavailable_model_stops_the_run_with_a_generic_message(
    db, make_user, make_case, make_photos, reembed_env, capsys
):
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 61)
    reembed_env.unavailable = True
    assert reembed_cases.main([]) == 1
    output = capsys.readouterr().out
    assert "recognition model is unavailable" in output and "cases=" not in output
    assert db.scalar(select(FaceEmbedding.id)) is None


def test_no_backend_configured_is_reported_not_crashed(monkeypatch, capsys):
    monkeypatch.setattr(reembed_cases, "get_recognition_service", UnconfiguredRecognitionService)
    assert reembed_cases.main([]) == 1
    assert "recognition model is unavailable" in capsys.readouterr().out
~~~~~~

## tests/test_recognition_packaging.py

~~~~~~python
"""Files that decide what can leak or break outside the code: .gitignore, the requirements files, .env.example."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _lines(name: str) -> list[str]:
    return (ROOT / name).read_text(encoding="utf-8").splitlines()


def test_model_weights_and_evaluation_datasets_are_git_ignored_on_their_own_lines():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "\\n" not in text                                    # a literal backslash-n once made these entries useless
    ignored = {line.strip() for line in text.splitlines()}
    assert {"models/recognition/", "evaluation-datasets/", ".env", "storage/"} <= ignored


def test_the_base_requirements_stay_light_and_keep_their_pins():
    base = _lines("requirements.txt")
    assert "numpy==2.1.3" in base and "Pillow==11.3.0" in base
    for line in base:
        assert not re.match(r"(?i)(torch|torchvision|facenet)", line), line


def test_the_optional_requirements_include_the_base_and_do_not_repin_numpy_or_pillow():
    optional = [line.strip() for line in _lines("requirements-facenet.txt") if line.strip() and not line.startswith("#")]
    assert optional[0] == "-r requirements.txt"
    names = {re.split(r"[=<>!~\[ ]", line, maxsplit=1)[0].lower() for line in optional[1:]}
    assert {"torch", "torchvision", "facenet-pytorch"} <= names
    assert not names & {"numpy", "pillow"}                      # never downgrade a base pin silently
    assert all("==" in line for line in optional[1:])           # everything is pinned exactly


def test_env_example_documents_the_recognition_settings_with_safe_values():
    values = dict(line.split("=", 1) for line in _lines(".env.example") if "=" in line and not line.startswith("#"))
    assert values["RECOGNITION_BACKEND"] == "none"
    assert values["RECOGNITION_MODEL_DIR"]
    assert int(values["RECOGNITION_MIN_FACE_SIZE"]) >= 12
    assert 0 < float(values["RECOGNITION_MIN_DETECTION_PROBABILITY"]) <= 1
    assert values["SECRET_KEY"] == "CHANGE_ME" and values["POSTGRES_PASSWORD"] == "CHANGE_ME"
~~~~~~
