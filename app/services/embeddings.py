"""Face enrolment: turning the photos of an APPROVED case into embeddings.

Only approved cases ever have embeddings. A photo is processed when it is pending, failed, or has embeddings only
from another model version (those old rows are replaced). Exactly one face is required per photo: a photo with no
face or several faces is marked failed with a short code while the other photos carry on. If the recognition
service is unavailable nothing is written or changed. Everything is saved in ONE transaction at the end.
"""
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.case import CaseStatus
from app.models.face_embedding import FaceEmbedding
from app.models.photo import EMBEDDING_DONE, EMBEDDING_FAILED, EMBEDDING_PENDING, CasePhoto
from app.recognition import (
    MultipleFacesError,
    NoFaceDetectedError,
    RecognitionService,
    RecognitionUnavailableError,
    extract_single_face,
)
from app.recognition import vectors
from app.services import cases as case_service
from app.services import photos as photo_service
from app.storage import Storage

logger = logging.getLogger(__name__)

ERROR_NO_FACE = "no_face"
ERROR_MULTIPLE_FACES = "multiple_faces"
ERROR_FILE_MISSING = "file_missing"


class CaseNotApprovedError(Exception):
    """Embeddings exist only for approved cases."""


@dataclass(frozen=True)
class EmbeddingSummary:
    generated: int      # photos with an embedding
    pending: int        # photos not processed yet
    failed: int         # photos processed without a usable single face


@dataclass(frozen=True)
class PhotoEmbeddingState:
    photo_id: uuid.UUID
    status: str
    error: str | None


@dataclass(frozen=True)
class EnrolmentResult:
    case_id: uuid.UUID
    model_name: str
    model_version: str
    summary: EmbeddingSummary
    photos: list[PhotoEmbeddingState]


def summarise(db: Session, case_id: uuid.UUID) -> EmbeddingSummary:
    """Counts of the case's photos by embedding status. Works without any recognition service."""
    counts = dict(
        db.execute(
            select(CasePhoto.embedding_status, func.count())
            .where(CasePhoto.case_id == case_id)
            .group_by(CasePhoto.embedding_status)
        ).all()
    )
    return EmbeddingSummary(
        generated=counts.get(EMBEDDING_DONE, 0),
        pending=counts.get(EMBEDDING_PENDING, 0),
        failed=counts.get(EMBEDDING_FAILED, 0),
    )


def _states(photos: list[CasePhoto]) -> list[PhotoEmbeddingState]:
    return [PhotoEmbeddingState(photo.id, photo.embedding_status, photo.embedding_error) for photo in photos]


def enrol_case(db: Session, storage: Storage, recognition: RecognitionService, case_id: uuid.UUID) -> EnrolmentResult:
    """Raises CaseNotFoundError, CaseNotApprovedError, RecognitionUnavailableError (nothing changed) or
    photo_service.PhotoStorageError (a storage failure; nothing changed)."""
    try:
        case = case_service.get_case_for_update(db, case_id)
        if case.status != CaseStatus.APPROVED.value:
            raise CaseNotApprovedError
        info = recognition.model_info()                           # raises RecognitionUnavailableError when unconfigured

        photos = photo_service.list_photos_for_admin(db, case.id)
        current = set(
            db.scalars(
                select(FaceEmbedding.photo_id).where(
                    FaceEmbedding.case_id == case.id,
                    FaceEmbedding.model_name == info.name,
                    FaceEmbedding.model_version == info.version,
                )
            )
        )
        for photo in photos:
            if photo.embedding_status == EMBEDDING_DONE and photo.id in current:
                continue                                          # already done with this model: idempotent
            _enrol_photo(db, storage, recognition, case.id, photo, info)
        db.commit()
    except Exception:
        db.rollback()                                             # nothing is saved; releases the row lock
        raise
    return EnrolmentResult(
        case_id=case_id,
        model_name=info.name,
        model_version=info.version,
        summary=summarise(db, case_id),
        photos=_states(photos),
    )


def _enrol_photo(db, storage, recognition, case_id, photo: CasePhoto, info) -> None:
    try:
        jpeg = photo_service.read_photo(storage, photo)
    except photo_service.PhotoFileMissingError:
        _mark_failed(photo, ERROR_FILE_MISSING)
        return
    try:
        face = extract_single_face(recognition, jpeg)
    except NoFaceDetectedError:
        _mark_failed(photo, ERROR_NO_FACE)
        return
    except MultipleFacesError:
        _mark_failed(photo, ERROR_MULTIPLE_FACES)
        return
    # A real result: replace whatever this photo had from any model (an older version is regenerated).
    db.execute(delete(FaceEmbedding).where(FaceEmbedding.photo_id == photo.id))
    db.add(
        FaceEmbedding(
            photo_id=photo.id,
            case_id=case_id,
            model_name=info.name,
            model_version=info.version,
            dimension=info.dimension,
            vector=vectors.pack(face.embedding),
        )
    )
    photo.embedding_status = EMBEDDING_DONE
    photo.embedding_error = None
    db.flush()


def _mark_failed(photo: CasePhoto, code: str) -> None:
    photo.embedding_status = EMBEDDING_FAILED
    photo.embedding_error = code


def after_approval_enrolment(
    db: Session, case_id: uuid.UUID, *, recognition: RecognitionService, storage: Storage
) -> EmbeddingSummary:
    """The body of the Stage 5 hook. Runs AFTER the approval was committed, as a separate step. Whatever goes
    wrong here, the approval stays: the case simply keeps its photos pending (a warning with the case id is
    logged). Returns the photo counts by embedding status."""
    try:
        return enrol_case(db, storage, recognition, case_id).summary
    except RecognitionUnavailableError:
        logger.warning(
            "embeddings_pending_after_approval", extra={"case_id": str(case_id), "reason": "recognition_unavailable"}
        )
    except Exception as exc:
        logger.warning(
            "embeddings_pending_after_approval", extra={"case_id": str(case_id), "reason": type(exc).__name__}
        )
    return summarise(db, case_id)
