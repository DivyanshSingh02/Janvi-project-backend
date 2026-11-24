from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

FoundPersonReportStatus = Literal["pending", "reviewed", "contacted", "dismissed"]
FoundPersonReportDecision = Literal["reviewed", "contacted", "dismissed"]


class FoundPersonReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    review_status: FoundPersonReportStatus
    created_at: datetime


class AdminFoundPersonReportRead(FoundPersonReportRead):
    reporter_id: UUID
    reporter_name: str
    reporter_email: str
    reviewed_by_id: UUID | None
    reviewed_at: datetime | None
    review_note: str | None

    @classmethod
    def from_report(cls, report, reporter) -> "AdminFoundPersonReportRead":
        return cls(
            id=report.id,
            review_status=report.review_status,
            created_at=report.created_at,
            reporter_id=reporter.id,
            reporter_name=reporter.full_name,
            reporter_email=reporter.email,
            reviewed_by_id=report.reviewed_by_id,
            reviewed_at=report.reviewed_at,
            review_note=report.review_note,
        )


class FoundPersonReportReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_status: FoundPersonReportDecision
    note: str | None = Field(default=None, max_length=1000)


class FoundPersonReportDeleteRead(BaseModel):
    report_id: UUID
    photo_deleted: bool
