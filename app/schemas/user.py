from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints, model_validator

from app.core.security import PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH

FullName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class UserCreate(BaseModel):
    """Public registration payload. There is NO role field: any extra field a client sends
    (including "role") is ignored, so registration can only ever create role "user"."""

    email: EmailStr
    full_name: FullName
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH, repr=False)


class UserRead(BaseModel):
    """What the API returns about a user. hashed_password is not a field, so it cannot leak."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    full_name: str
    role: Literal["user", "admin"]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class UserUpdate(BaseModel):
    """Admin-only partial update. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "admin"] | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def _at_least_one_change(self) -> "UserUpdate":
        if self.role is None and self.is_active is None:
            raise ValueError("At least one field must be provided")
        return self
