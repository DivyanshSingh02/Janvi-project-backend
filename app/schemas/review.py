from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict, field_validator, model_validator

from app.models.case import CaseStatus, Gender, MissingCase
from app.models.case_status_event import CaseStatusEvent
from app.models.photo import CasePhoto
from app.schemas.case import CaseRead, _blank_to_none, _text
from app.schemas.embeddings import EmbeddingSummaryRead
from app.schemas.photo import PhotoRead
from app.services.embeddings import EmbeddingSummary
from app.services.review import (
    REASON_MAX_LENGTH,
    REASON_MIN_LENGTH,
    REASON_REQUIRED_FOR,
    REVIEW_DECISIONS,
    CaseDetail,
)

# Trimmed, 5..1000 characters, no control characters (so, like every other text field since Stage 3, a reason is
# one line of plain text). An optional reason that is empty or only blanks counts as "not given".
Reason = _text(REASON_MIN_LENGTH, REASON_MAX_LENGTH)
OptionalReason = Annotated[Reason | None, BeforeValidator(_blank_to_none)]


# ---------------------------------------------------------------- request bodies
class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: OptionalReason = None


class RejectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Reason


class StatusChangeRequest(BaseModel):
    """Body of PATCH /admin/cases/{id}/status. approved and rejected are review decisions with their own
    endpoints, so they are refused here."""

    model_config = ConfigDict(extra="forbid")

    status: CaseStatus
    reason: OptionalReason = None

    @field_validator("status")
    @classmethod
    def _not_a_review_decision(cls, value: CaseStatus) -> CaseStatus:
        if value in REVIEW_DECISIONS:
            raise ValueError("use the approve/reject endpoint")
        return value

    @model_validator(mode="after")
    def _reason_when_required(self) -> "StatusChangeRequest":
        if self.status in REASON_REQUIRED_FOR and self.reason is None:
            raise ValueError(f"a reason is required when the new status is {self.status.value}")
        return self


# ---------------------------------------------------------------- responses (admin only)
class AdminCaseSummary(BaseModel):
    """One row of the review queue: enough to pick the next case, nothing sensitive beyond the name."""

    id: UUID
    name: str
    age: int
    gender: Gender
    city: str
    state: str
    status: CaseStatus
    created_at: datetime
    photo_count: int

    @classmethod
    def from_case(cls, case: MissingCase, photo_count: int) -> "AdminCaseSummary":
        return cls(
            id=case.id,
            name=case.name,
            age=case.age,
            gender=case.gender,
            city=case.city,
            state=case.state,
            status=case.status,
            created_at=case.created_at,
            photo_count=photo_count,
        )


class ReporterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    full_name: str
    email: str


class StatusEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    from_status: CaseStatus
    to_status: CaseStatus
    changed_by_id: UUID
    reason: str | None
    created_at: datetime


def _admin_photo_url(photo: CasePhoto) -> str:
    return f"/admin/cases/{photo.case_id}/photos/{photo.id}"


class AdminCaseDetail(CaseRead):
    """Every case field (CaseRead, including review_reason and reviewed_at) plus what only an admin sees: who
    reported it, who reviewed it, the photo list (with admin URLs) and the status history, oldest first."""

    reviewed_by_id: UUID | None
    reporter: ReporterRead
    photos: list[PhotoRead]
    history: list[StatusEventRead]

    @classmethod
    def from_detail(cls, detail: CaseDetail) -> "AdminCaseDetail":
        return cls(
            **CaseRead.model_validate(detail.case).model_dump(),
            reviewed_by_id=detail.case.reviewed_by_id,
            reporter=ReporterRead.model_validate(detail.reporter),
            photos=[PhotoRead.from_photo(photo, url=_admin_photo_url(photo)) for photo in detail.photos],
            history=[StatusEventRead.model_validate(event) for event in detail.history],
        )


class ApprovedCaseDetail(AdminCaseDetail):
    """The response of POST /admin/cases/{id}/approve: the case detail plus how the face embeddings went.
    An approval never fails because of them: with no recognition model the photos are simply all pending."""

    embeddings: EmbeddingSummaryRead

    @classmethod
    def from_approval(cls, detail: CaseDetail, summary: EmbeddingSummary) -> "ApprovedCaseDetail":
        return cls(
            **AdminCaseDetail.from_detail(detail).model_dump(), embeddings=EmbeddingSummaryRead.from_summary(summary)
        )


class CaseHardDeleteRead(BaseModel):
    """Counts and file-cleanup status only; never returns case or reporter data."""

    case_id: UUID
    deleted: bool = True
    photos_deleted: int
    embeddings_deleted: int
    detections_deleted: int
    status_events_deleted: int
    files_deleted: int
    failed_photo_ids: list[UUID]
