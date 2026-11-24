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
