from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from app.models.photo import CasePhoto


class PhotoRead(BaseModel):
    """What the API returns about a photo. There is deliberately no storage key, no sha256 and no
    file name: they are internal."""

    id: UUID
    content_type: Literal["image/jpeg"]
    size_bytes: int
    width: int
    height: int
    created_at: datetime
    url: str          # relative API path of the image endpoint (requires the same bearer token)

    @classmethod
    def from_photo(cls, photo: CasePhoto, *, url: str | None = None) -> "PhotoRead":
        """`url` is only passed by the admin endpoints: the owner's URL answers 404 for an admin."""
        return cls(
            id=photo.id,
            content_type=photo.content_type,
            size_bytes=photo.size_bytes,
            width=photo.width,
            height=photo.height,
            created_at=photo.created_at,
            url=url if url is not None else f"/cases/{photo.case_id}/photos/{photo.id}",
        )
