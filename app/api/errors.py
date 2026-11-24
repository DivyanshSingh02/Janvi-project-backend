"""Central API error mapping. Public messages are fixed and never include rejected values."""
import logging

from fastapi import Request
from starlette.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.recognition.base import InvalidEmbeddingError, MultipleFacesError, NoFaceDetectedError, RecognitionUnavailableError
from app.recognition.unconfigured import NOT_CONFIGURED_MESSAGE
from app.services import cases, detections, embeddings, found_person_reports, images, photos, review, users
from app.storage.base import StorageError

logger = logging.getLogger("app.errors")

# Exception class -> (HTTP status, stable code, safe public message).
DOMAIN_ERRORS = {
    cases.CaseNotFoundError: (404, "case_not_found", "Case not found"),
    detections.DetectionNotFoundError: (404, "detection_not_found", "Detection not found"),
    found_person_reports.FoundPersonReportNotFoundError: (404, "found_person_report_not_found", "Found-person report not found"),
    found_person_reports.FoundPersonReportFileMissingError: (404, "found_person_report_photo_not_found", "Found-person report photo not found"),
    photos.PhotoNotFoundError: (404, "photo_not_found", "Photo not found"),
    photos.PhotoFileMissingError: (404, "photo_not_found", "Photo not found"),
    users.EmailAlreadyRegisteredError: (409, "email_already_registered", "This email is already registered"),
    users.UserNotFoundError: (404, "user_not_found", "User not found"),
    users.UserManagementConflictError: (409, "user_management_conflict", "This user change is not allowed"),
    detections.CaseNotApprovedError: (409, "case_not_approved", "The case must be approved for this operation"),
    embeddings.CaseNotApprovedError: (409, "case_not_approved", "Face embeddings can only be generated for an approved case"),
    detections.DetectionAlreadyReviewedError: (409, "detection_already_reviewed", "This detection has already been reviewed"),
    found_person_reports.InvalidFoundPersonReportReviewError: (409, "found_person_report_already_reviewed", "This report cannot be changed to that review status"),
    review.InvalidTransitionError: (409, "invalid_transition", "This case status transition is not allowed"),
    review.ReasonRequiredError: (422, "reason_required", "A reason is required for this status change"),
    review.InvalidReasonError: (422, "invalid_reason", "The supplied reason is not allowed"),
    review.CaseHasNoPhotosError: (409, "case_has_no_photos", "A case needs at least one photo before it can be approved"),
    photos.UploadNotAllowedError: (409, "upload_not_allowed", "Photos can only be added to a pending or approved case"),
    photos.DeleteNotAllowedError: (409, "delete_not_allowed", "Photos can only be deleted while the case is pending"),
    photos.PhotoLimitReachedError: (409, "photo_limit_reached", "This case already has the maximum number of photos"),
    photos.DuplicatePhotoError: (409, "duplicate_photo", "This image has already been added to the case"),
    photos.PhotoStorageError: (503, "storage_unavailable", "Photo storage is temporarily unavailable"),
    found_person_reports.FoundPersonReportStorageError: (503, "storage_unavailable", "Photo storage is temporarily unavailable"),
    images.InvalidImageError: (400, "invalid_image", "The file is not a valid image"),
    images.UnsupportedImageFormatError: (415, "unsupported_image_format", "Only static JPEG, PNG and WebP images are accepted"),
    images.ImageTooLargeError: (413, "image_too_large", "The image is too large"),
    images.ImageTooSmallError: (422, "image_too_small", "The image is too small"),
    RecognitionUnavailableError: (503, "recognition_unavailable", "The face recognition service is temporarily unavailable"),
    InvalidEmbeddingError: (503, "recognition_unavailable", "The face recognition service is temporarily unavailable"),
    NoFaceDetectedError: (422, "no_face_detected", "No face was found in the image"),
    MultipleFacesError: (422, "multiple_faces_detected", "More than one face was found; use an image with exactly one face"),
    StorageError: (503, "storage_unavailable", "Photo storage is temporarily unavailable"),
}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def error_response(request: Request, status_code: int, code: str, message: str, *,
                   fields: list[dict] | None = None, headers: dict | None = None) -> JSONResponse:
    error = {"code": code, "message": message, "request_id": _request_id(request)}
    if fields is not None:
        error["fields"] = fields
    response = JSONResponse(status_code=status_code, content={"error": error}, headers=headers)
    response.headers["X-Request-ID"] = _request_id(request)
    return response


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = exc.status_code
    code = {
        400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
        405: "method_not_allowed", 409: "conflict", 413: "request_too_large",
        415: "unsupported_media_type", 422: "validation_error", 429: "rate_limited",
        503: "service_unavailable",
    }.get(status_code, f"http_{status_code}")
    detail = exc.detail
    # Do not expose arbitrary exception details or non-string objects.
    message = detail if isinstance(detail, str) else "The request could not be completed"
    return error_response(request, status_code, code, message, headers=exc.headers)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = []
    for item in exc.errors():
        loc = [str(part) for part in item.get("loc", ()) if part not in ("body", "query", "path", "header", "cookie")]
        field = ".".join(loc) or str(item.get("loc", ["request"])[0])
        error_type = str(item.get("type", ""))
        safe_messages = {
            "missing": "Field is required",
            "string_too_short": "Value is too short",
            "string_too_long": "Value is too long",
            "greater_than_equal": "Value is below the allowed minimum",
            "less_than_equal": "Value is above the allowed maximum",
            "greater_than": "Value must be greater than the allowed minimum",
            "less_than": "Value must be less than the allowed maximum",
            "email": "Enter a valid email address",
            "enum": "Select a valid option",
            "literal_error": "Select a valid option",
            "extra_forbidden": "Unexpected field",
            "int_parsing": "Enter a valid integer",
            "float_parsing": "Enter a valid number",
            "bool_parsing": "Enter a valid boolean",
        }
        message = safe_messages.get(error_type, "Invalid value")
        fields.append({"field": field, "message": message})
    return error_response(request, 422, "validation_error", "Request validation failed", fields=fields)


async def domain_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, InvalidEmbeddingError):
        return error_response(request, 503, "recognition_unavailable", "The face recognition model returned an unusable result")
    if isinstance(exc, RecognitionUnavailableError):
        message = NOT_CONFIGURED_MESSAGE if str(exc) == NOT_CONFIGURED_MESSAGE else "The face recognition service is temporarily unavailable"
        return error_response(request, 503, "recognition_unavailable", message)
    if isinstance(exc, review.InvalidTransitionError):
        message = f"A case that is {exc.from_status.value} cannot be changed to {exc.to_status.value}"
        return error_response(request, 409, "invalid_transition", message)
    for exception_type, (status_code, code, message) in DOMAIN_ERRORS.items():
        if isinstance(exc, exception_type):
            return error_response(request, status_code, code, message)
    # A dynamic count message is safe: it contains only a configured integer.
    if isinstance(exc, photos.InvalidFileCountError):
        return error_response(request, 422, "invalid_file_count", f"Send between 1 and {exc.maximum} image files")
    from app.core.rate_limit import RateLimitExceeded
    if isinstance(exc, RateLimitExceeded):
        return error_response(request, 429, "rate_limited", "Too many requests; try again later",
                              headers={"Retry-After": str(exc.retry_after)})
    logger.error(
        "unhandled_application_exception",
        extra={"request_id": _request_id(request)},
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return error_response(request, 500, "internal_server_error", "An unexpected error occurred")
