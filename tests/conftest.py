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
    "found_person_reports", "audit_events", "detections", "face_search_logs", "face_embeddings",
    "case_status_events", "case_photos", "missing_cases", "users",
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
