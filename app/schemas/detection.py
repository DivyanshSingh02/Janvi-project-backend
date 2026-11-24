import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator


DetectionSource = Literal["camera", "video", "manual"]
DetectionReviewStatus = Literal["unreviewed", "confirmed", "dismissed"]
ReviewDecision = Literal["confirmed", "dismissed"]
_FRAME_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


def _trimmed_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("must not contain control characters")
    return value


def _optional_text(value):
    if isinstance(value, str) and not value.strip():
        return None
    if value is None:
        return None
    return _trimmed_text(value)


class DetectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected_at: datetime
    source: DetectionSource
    camera_id: Annotated[str | None, BeforeValidator(_optional_text), Field(max_length=64)] = None
    location_text: Annotated[str | None, BeforeValidator(_optional_text), Field(max_length=200)] = None
    latitude: float | None = None
    longitude: float | None = None
    confidence: float
    frame_reference: str | None = Field(default=None, max_length=255)

    @field_validator("detected_at")
    @classmethod
    def _validate_detected_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("detected_at must include a timezone")
        value = value.astimezone(timezone.utc)
        if value < datetime(2000, 1, 1, tzinfo=timezone.utc):
            raise ValueError("detected_at must not be before 2000-01-01")
        if value > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("detected_at must not be more than 5 minutes in the future")
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        return value

    @field_validator("latitude")
    @classmethod
    def _validate_latitude(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not -90 <= value <= 90):
            raise ValueError("latitude must be finite and between -90 and 90")
        return value

    @field_validator("longitude")
    @classmethod
    def _validate_longitude(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not -180 <= value <= 180):
            raise ValueError("longitude must be finite and between -180 and 180")
        return value

    @field_validator("frame_reference")
    @classmethod
    def _validate_frame_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _FRAME_REFERENCE.fullmatch(value) or ".." in value:
            raise ValueError("frame_reference must be an opaque reference, not a path")
        # The grammar already excludes slash-rooted paths and backslashes; explicitly
        # reject Windows drive-rooted paths, which otherwise fit the opaque grammar.
        if re.match(r"^[A-Za-z]:/", value):
            raise ValueError("frame_reference must not be an absolute filesystem path")
        return value

    @model_validator(mode="after")
    def _location_and_coordinates(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if not (self.camera_id or self.location_text or self.latitude is not None):
            raise ValueError("provide camera_id, location_text or coordinates")
        if self.source == "camera" and not self.camera_id:
            raise ValueError("camera_id is required when source is camera")
        return self


class DetectionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_status: ReviewDecision
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("note")
    @classmethod
    def _validate_note(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return _trimmed_text(value)


class DetectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    case_id: UUID
    detected_at: datetime
    source: DetectionSource
    camera_id: str | None
    location_text: str | None
    latitude: float | None
    longitude: float | None
    confidence: float
    frame_reference: str | None
    review_status: DetectionReviewStatus
    reviewed_by_id: UUID | None
    reviewed_at: datetime | None
    review_note: str | None
    recorded_by_id: UUID
    created_at: datetime


class OwnerDetectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    detected_at: datetime
    location_text: str | None
