import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status
from fastapi.routing import APIRoute

from app.api.deps import CurrentUser, DbSession
from app.api.rate_limits import limit_photo_upload
from app.core.config import get_settings
from app.schemas.photo import PhotoRead
from app.services import cases as case_service
from app.services import images
from app.services import photos as photo_service
from app.storage import Storage, get_storage

logger = logging.getLogger(__name__)

StorageDep = Annotated[Storage, Depends(get_storage)]

_ONE_MIB = 1024 * 1024
_NOT_ALLOWED_TO_ADD = "Photos can only be added to a pending or approved case"
_NOT_ALLOWED_TO_DELETE = "Photos can only be deleted while the case is pending"

# These exceptions are re-raised for the central handler in app/api/errors.py.
_HANDLED = (
    case_service.CaseNotFoundError,
    photo_service.PhotoNotFoundError,
    photo_service.PhotoFileMissingError,
    photo_service.UploadNotAllowedError,
    photo_service.DeleteNotAllowedError,
    photo_service.InvalidFileCountError,
    photo_service.PhotoLimitReachedError,
    photo_service.DuplicatePhotoError,
    photo_service.PhotoStorageError,
    images.InvalidImageError,
    images.UnsupportedImageFormatError,
    images.ImageTooLargeError,
    images.ImageTooSmallError,
)


def image_response(data: bytes, content_type: str) -> Response:
    """The one place that decides how a stored photo is sent (also used by the admin photo endpoint)."""
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": 'inline; filename="photo.jpg"',     # generic: the original name is never kept
        },
    )


class _BodyLimitedRoute(APIRoute):
    """Rejects an upload with 413 as soon as its Content-Length header is over the limit, before a single
    byte of the body is read (FastAPI would otherwise parse the whole multipart body first).
    The global request middleware also enforces MAX_REQUEST_BODY_BYTES while streaming the body."""

    max_upload_count: int | None = None

    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def handler(request: Request) -> Response:
            if request.method == "POST":
                declared = request.headers.get("content-length")
                if declared is not None:
                    try:
                        declared_bytes = int(declared)
                    except ValueError:
                        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Length") from None
                    settings = get_settings()
                    upload_count = self.max_upload_count or settings.max_photos_per_case
                    limit = upload_count * settings.max_upload_bytes + _ONE_MIB
                    if declared_bytes > limit:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="The upload is too large"
                        )
            return await original_handler(request)

        return handler


router = APIRouter(prefix="/cases/{case_id}/photos", tags=["Photos"], route_class=_BodyLimitedRoute)


@router.post("", response_model=list[PhotoRead], status_code=status.HTTP_201_CREATED)
def upload_photos(
    case_id: UUID,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    files: list[UploadFile],
    request: Request,
):
    """Owner only. Multipart form with the field name "files" repeated once per image. The client's file
    names and Content-Types are ignored: the real format is detected, and every image is stored as JPEG.
    All files are checked before anything is stored; one bad file rejects the whole request."""
    limit_photo_upload(request, user.id)
    try:
        photos = photo_service.add_photos(db, storage, user, case_id, [upload.file for upload in files])
    except _HANDLED as exc:
        logger.info(
            "photo_upload_rejected",
            extra={"case_id": str(case_id), "user_id": str(user.id), "reason": type(exc).__name__},
        )
        raise
    # Ids, counts and sizes only. File names, storage keys and paths must never reach a log.
    logger.info(
        "photos_uploaded",
        extra={
            "case_id": str(case_id),
            "user_id": str(user.id),
            "photo_count": len(photos),
            "total_bytes": sum(photo.size_bytes for photo in photos),
        },
    )
    return [PhotoRead.from_photo(photo) for photo in photos]


@router.get("", response_model=list[PhotoRead])
def list_photos(case_id: UUID, user: CurrentUser, db: DbSession):
    """A plain list, not a Page: a case has at most MAX_PHOTOS_PER_CASE (default 5) photos, so there is
    nothing to paginate."""
    try:
        photos = photo_service.list_photos(db, user, case_id)
    except _HANDLED as exc:
        raise
    return [PhotoRead.from_photo(photo) for photo in photos]


@router.get("/{photo_id}", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
def download_photo(case_id: UUID, photo_id: UUID, user: CurrentUser, db: DbSession, storage: StorageDep):
    """The image itself (the stored, metadata-free JPEG), for the case owner only."""
    try:
        photo = photo_service.get_photo(db, user, case_id, photo_id)
        data = photo_service.read_photo(storage, photo)
    except _HANDLED as exc:
        raise
    return image_response(data, photo.content_type)


@router.delete("/{photo_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_photo(case_id: UUID, photo_id: UUID, user: CurrentUser, db: DbSession, storage: StorageDep):
    try:
        photo_service.delete_photo(db, storage, user, case_id, photo_id)
    except _HANDLED as exc:
        raise
    logger.info(
        "photo_deleted", extra={"case_id": str(case_id), "photo_id": str(photo_id), "user_id": str(user.id)}
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
