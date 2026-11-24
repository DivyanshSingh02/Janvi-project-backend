import logging
import uuid
from datetime import datetime, timezone
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.found_person_report import FoundPersonReport
from app.models.user import User
from app.services import images
from app.services.photos import _limits
from app.storage import Storage, StorageError, StorageObjectNotFoundError

logger = logging.getLogger(__name__)


class FoundPersonReportNotFoundError(Exception):
    """The report does not exist or is not owned by the caller."""


class FoundPersonReportFileMissingError(Exception):
    """The report exists but its stored photo is missing."""


class FoundPersonReportStorageError(Exception):
    """The storage backend failed while handling a report photo."""


class InvalidFoundPersonReportReviewError(Exception):
    """The requested review status transition is not allowed."""


def submit_report(db: Session, storage: Storage, reporter: User, stream: BinaryIO) -> FoundPersonReport:
    processed = images.process_upload(stream, _limits())
    report_id = uuid.uuid4()
    storage_key = f"found-person-reports/{report_id.hex}.jpg"
    try:
        storage.save(storage_key, processed.data, "image/jpeg")
    except StorageError:
        raise FoundPersonReportStorageError from None

    report = FoundPersonReport(
        id=report_id,
        reporter_id=reporter.id,
        storage_key=storage_key,
        content_type="image/jpeg",
        size_bytes=processed.size_bytes,
        width=processed.width,
        height=processed.height,
        sha256=processed.sha256,
    )
    try:
        db.add(report)
        db.commit()
        db.refresh(report)
    except Exception:
        db.rollback()
        _delete_uploaded_file(storage, storage_key)
        raise
    return report


def _delete_uploaded_file(storage: Storage, storage_key: str) -> None:
    try:
        storage.delete(storage_key)
    except StorageError:
        logger.error("found_person_report_cleanup_failed", extra={"file_count": 1})


def list_reporter_reports(
    db: Session, reporter: User, *, limit: int, offset: int
) -> tuple[list[FoundPersonReport], int]:
    condition = FoundPersonReport.reporter_id == reporter.id
    total = db.scalar(select(func.count()).select_from(FoundPersonReport).where(condition)) or 0
    rows = db.scalars(
        select(FoundPersonReport)
        .where(condition)
        .order_by(FoundPersonReport.created_at.desc(), FoundPersonReport.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), total


def list_admin_reports(
    db: Session, *, review_status: str | None, limit: int, offset: int
) -> tuple[list[tuple[FoundPersonReport, User]], int]:
    conditions = []
    if review_status is not None:
        conditions.append(FoundPersonReport.review_status == review_status)
    total = db.scalar(
        select(func.count()).select_from(FoundPersonReport).where(*conditions)
    ) or 0
    rows = db.execute(
        select(FoundPersonReport, User)
        .join(User, User.id == FoundPersonReport.reporter_id)
        .where(*conditions)
        .order_by(FoundPersonReport.created_at, FoundPersonReport.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return [(report, reporter) for report, reporter in rows], total


def get_report_for_admin(db: Session, report_id: uuid.UUID) -> FoundPersonReport:
    report = db.get(FoundPersonReport, report_id)
    if report is None:
        raise FoundPersonReportNotFoundError
    return report


def get_reporter_for_admin(db: Session, reporter_id: uuid.UUID) -> User:
    reporter = db.get(User, reporter_id)
    if reporter is None:
        raise FoundPersonReportNotFoundError
    return reporter


def get_report_photo(report: FoundPersonReport, storage: Storage) -> bytes:
    try:
        return storage.read(report.storage_key)
    except StorageObjectNotFoundError:
        logger.error("found_person_report_file_missing", extra={"report_id": str(report.id)})
        raise FoundPersonReportFileMissingError from None
    except StorageError:
        raise FoundPersonReportStorageError from None


def delete_report(db: Session, storage: Storage, report_id: uuid.UUID) -> bool:
    report = db.get(FoundPersonReport, report_id)
    if report is None:
        raise FoundPersonReportNotFoundError
    storage_key = report.storage_key
    db.delete(report)
    db.commit()
    try:
        storage.delete(storage_key)
    except StorageError:
        logger.error("found_person_report_file_delete_failed", extra={"report_id": str(report_id)})
        return False
    return True


def review_report(
    db: Session,
    admin: User,
    report_id: uuid.UUID,
    review_status: str,
    note: str | None,
) -> FoundPersonReport:
    report = db.scalar(
        select(FoundPersonReport)
        .where(FoundPersonReport.id == report_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if report is None:
        raise FoundPersonReportNotFoundError

    allowed_transitions = {
        "pending": {"reviewed", "contacted", "dismissed"},
        "reviewed": {"contacted", "dismissed"},
    }
    if review_status not in allowed_transitions.get(report.review_status, set()):
        db.rollback()
        raise InvalidFoundPersonReportReviewError

    report.review_status = review_status
    report.reviewed_by_id = admin.id
    report.reviewed_at = datetime.now(timezone.utc)
    report.review_note = note
    db.commit()
    db.refresh(report)
    return report
