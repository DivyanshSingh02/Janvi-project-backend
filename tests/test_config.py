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
