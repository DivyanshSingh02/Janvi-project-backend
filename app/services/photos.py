import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.case import CaseStatus, MissingCase
from app.models.photo import JPEG_CONTENT_TYPE, CasePhoto
from app.models.user import User
from app.services import cases as case_service
from app.services import images
from app.storage import Storage, StorageError, StorageObjectNotFoundError

logger = logging.getLogger(__name__)

_UNIQUE_VIOLATION = "23505"  # PostgreSQL SQLSTATE
UPLOAD_ALLOWED_STATUSES = frozenset({CaseStatus.PENDING.value, CaseStatus.APPROVED.value})
DELETE_ALLOWED_STATUSES = frozenset({CaseStatus.PENDING.value})


class PhotoNotFoundError(Exception):
    """No such photo on this case (or it is on another case): indistinguishable on purpose."""


class PhotoFileMissingError(Exception):
    """The database row exists but the stored file is gone."""


class UploadNotAllowedError(Exception):
    """The case status does not allow adding photos."""


class DeleteNotAllowedError(Exception):
    """The case status does not allow deleting photos."""


class InvalidFileCountError(Exception):
    def __init__(self, maximum: int):
        super().__init__("Wrong number of files")
        self.maximum = maximum


class PhotoLimitReachedError(Exception):
    """The case would end up with more photos than allowed."""


class DuplicatePhotoError(Exception):
    """The same image is already on the case (or twice in the same request)."""


class PhotoStorageError(Exception):
    """The storage backend failed."""


def _limits() -> images.ImageLimits:
    settings = get_settings()
    return images.ImageLimits(
        max_bytes=settings.max_upload_bytes,
        max_pixels=settings.max_image_pixels,
        min_side=settings.min_image_side,
    )


def _lock_case(db: Session, case_id: uuid.UUID, owner: User) -> MissingCase:
    """Owner-only lookup (CaseNotFoundError otherwise), then the case row is locked until the end of
    the transaction. That serialises concurrent uploads / deletes on one case, so the per-case limit
    and the duplicate check cannot be raced, and the status cannot change underneath us."""
    case = case_service.get_case_for_owner(db, case_id, owner)
    db.refresh(case, with_for_update=True)
    return case


def _find_photo(db: Session, case_id: uuid.UUID, photo_id: uuid.UUID) -> CasePhoto:
    photo = db.scalar(select(CasePhoto).where(CasePhoto.id == photo_id, CasePhoto.case_id == case_id))
    if photo is None:
        raise PhotoNotFoundError
    return photo


def add_photos(
    db: Session, storage: Storage, owner: User, case_id: uuid.UUID, streams: list[BinaryIO]
) -> list[CasePhoto]:
    """Validate EVERY file first; only then write the files, and only then commit the rows.
    Any failure removes the files already written, so nothing is left behind."""
    case = _lock_case(db, case_id, owner)
    try:
        if case.status not in UPLOAD_ALLOWED_STATUSES:
            raise UploadNotAllowedError
        maximum = get_settings().max_photos_per_case
        if not 1 <= len(streams) <= maximum:
            raise InvalidFileCountError(maximum)
        existing_hashes = list(db.scalars(select(CasePhoto.sha256).where(CasePhoto.case_id == case.id)))
        if len(existing_hashes) + len(streams) > maximum:
            raise PhotoLimitReachedError

        limits = _limits()
        processed = [images.process_upload(stream, limits) for stream in streams]   # raises on the first bad file

        seen = set(existing_hashes)
        for item in processed:
            if item.sha256 in seen:
                raise DuplicatePhotoError
            seen.add(item.sha256)
    except Exception:
        db.rollback()                                    # nothing was written yet; release the row lock
        raise

    written_keys: list[str] = []
    uploaded_at = datetime.now(timezone.utc)
    try:
        rows: list[CasePhoto] = []
        for index, item in enumerate(processed):
            key = f"cases/{case.id}/{uuid.uuid4().hex}.jpg"   # always server-generated
            storage.save(key, item.data, JPEG_CONTENT_TYPE)
            written_keys.append(key)
            rows.append(
                CasePhoto(
                    case_id=case.id,
                    storage_key=key,
                    content_type=JPEG_CONTENT_TYPE,
                    size_bytes=item.size_bytes,
                    width=item.width,
                    height=item.height,
                    sha256=item.sha256,
                    # One microsecond apart: photos from one request keep their order even on a coarse clock.
                    created_at=uploaded_at + timedelta(microseconds=index),
                )
            )
        db.add_all(rows)
        db.commit()
    except Exception as exc:
        db.rollback()
        _delete_written_files(storage, written_keys)
        if isinstance(exc, StorageError):
            raise PhotoStorageError from None
        if isinstance(exc, IntegrityError) and getattr(exc.orig, "sqlstate", None) == _UNIQUE_VIOLATION:
            raise DuplicatePhotoError from None
        raise
    return rows


def _delete_written_files(storage: Storage, keys: list[str]) -> None:
    failed = 0
    for key in keys:
        try:
            storage.delete(key)
        except StorageError:
            failed += 1
    if failed:
        logger.error("photo_cleanup_failed", extra={"file_count": failed})   # a count only: never a key or path


def _photos_of_case(db: Session, case_id: uuid.UUID) -> list[CasePhoto]:
    return list(
        db.scalars(
            select(CasePhoto).where(CasePhoto.case_id == case_id).order_by(CasePhoto.created_at, CasePhoto.id)
        )
    )


def list_photos(db: Session, owner: User, case_id: uuid.UUID) -> list[CasePhoto]:
    """All photos of one case, oldest first. At most MAX_PHOTOS_PER_CASE of them, so no pagination."""
    case = case_service.get_case_for_owner(db, case_id, owner)
    return _photos_of_case(db, case.id)


def list_photos_for_admin(db: Session, case_id: uuid.UUID) -> list[CasePhoto]:
    """Admin code only (no ownership check). Raises CaseNotFoundError for an unknown case."""
    case = case_service.get_case(db, case_id)
    return _photos_of_case(db, case.id)


def get_photo(db: Session, owner: User, case_id: uuid.UUID, photo_id: uuid.UUID) -> CasePhoto:
    case = case_service.get_case_for_owner(db, case_id, owner)
    return _find_photo(db, case.id, photo_id)


def get_photo_for_admin(db: Session, case_id: uuid.UUID, photo_id: uuid.UUID) -> CasePhoto:
    """Admin code only (no ownership check). A photo that belongs to another case is PhotoNotFoundError."""
    case = case_service.get_case(db, case_id)
    return _find_photo(db, case.id, photo_id)


def read_photo(storage: Storage, photo: CasePhoto) -> bytes:
    try:
        return storage.read(photo.storage_key)
    except StorageObjectNotFoundError:
        logger.error("photo_file_missing", extra={"photo_id": str(photo.id), "case_id": str(photo.case_id)})
        raise PhotoFileMissingError from None
    except StorageError:
        raise PhotoStorageError from None


def delete_photo(
    db: Session, storage: Storage, owner: User, case_id: uuid.UUID, photo_id: uuid.UUID
) -> None:
    """Owner only, and only while the case is pending. The row is removed first, then the file."""
    case = _lock_case(db, case_id, owner)
    try:
        photo = _find_photo(db, case.id, photo_id)
        if case.status not in DELETE_ALLOWED_STATUSES:
            raise DeleteNotAllowedError
    except Exception:
        db.rollback()
        raise
    storage_key = photo.storage_key
    db.delete(photo)
    db.commit()
    try:
        storage.delete(storage_key)
    except StorageError:
        # The row is already gone. A leftover file is harmless; log the ids and carry on.
        logger.warning("photo_file_delete_failed", extra={"photo_id": str(photo_id), "case_id": str(case.id)})
