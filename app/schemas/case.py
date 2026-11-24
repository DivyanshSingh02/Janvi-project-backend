import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.models.case import CaseStatus, Gender

EARLIEST_LAST_SEEN_DATE = date(1900, 1, 1)
_PHONE_SEPARATORS = re.compile(r"[\s\-()]")
_PHONE_PATTERN = re.compile(r"[+]?[0-9]{7,15}")


def _reject_control_characters(value: str) -> str:
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("must not contain control characters")
    return value


def _text(min_length: int, max_length: int):
    """A string that is trimmed, length-checked, and free of control characters."""
    return Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=min_length, max_length=max_length),
        AfterValidator(_reject_control_characters),
    ]


def _blank_to_none(value):
    """An empty form field arrives as "" - treat it as 'not provided'."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _normalise_choice(value):
    return value.strip().lower() if isinstance(value, str) else value


def _normalise_phone(value: str) -> str:
    cleaned = _PHONE_SEPARATORS.sub("", value)
    if not _PHONE_PATTERN.fullmatch(cleaned):
        raise ValueError("must be a phone number of 7 to 15 digits, optionally starting with +")
    return cleaned


def _lowercase(value: str) -> str:
    return value.lower()


PersonName = _text(2, 120)
CityName = _text(2, 80)
StateName = _text(2, 80)
Address = _text(5, 300)
LastSeenLocation = _text(3, 200)
Age = Annotated[int, Field(strict=True, ge=0, le=120)]  # strict: "12", 12.5 and true are rejected
GenderInput = Annotated[Gender, BeforeValidator(_normalise_choice)]
ContactPhone = Annotated[str, AfterValidator(_normalise_phone)]
ContactEmail = Annotated[EmailStr, AfterValidator(_lowercase)]


class CaseCreate(BaseModel):
    """Body of POST /cases. extra="forbid": a client cannot send status, reporter_id, id or
    timestamps - they are rejected with 422 instead of being silently ignored."""

    model_config = ConfigDict(extra="forbid")

    name: PersonName
    age: Age
    gender: GenderInput
    contact_phone: Annotated[ContactPhone | None, BeforeValidator(_blank_to_none)] = None
    contact_email: Annotated[ContactEmail | None, BeforeValidator(_blank_to_none)] = None
    city: CityName
    state: StateName
    address: Address
    last_seen_location: LastSeenLocation
    last_seen_date: date

    @field_validator("last_seen_date")
    @classmethod
    def _last_seen_date_is_plausible(cls, value: date) -> date:
        # One day of slack: the caller may be in a time zone that is already "tomorrow" in UTC.
        latest_allowed = datetime.now(timezone.utc).date() + timedelta(days=1)
        if value < EARLIEST_LAST_SEEN_DATE:
            raise ValueError(f"must not be before {EARLIEST_LAST_SEEN_DATE.isoformat()}")
        if value > latest_allowed:
            raise ValueError("must not be in the future")
        return value

    @model_validator(mode="after")
    def _at_least_one_contact(self) -> "CaseCreate":
        if self.contact_phone is None and self.contact_email is None:
            raise ValueError("provide at least one of contact_phone or contact_email")
        return self


class CaseRead(BaseModel):
    """What the API returns about a case to its owner. It deliberately has no reporter_id, and no reviewed_by_id
    either: the owner sees WHY and WHEN a case was reviewed, never WHO reviewed it."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    age: int
    gender: Gender
    contact_phone: str | None
    contact_email: str | None
    city: str
    state: str
    address: str
    last_seen_location: str
    last_seen_date: date
    status: CaseStatus
    created_at: datetime
    updated_at: datetime
    review_reason: str | None
    reviewed_at: datetime | None
