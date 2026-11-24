import logging
from typing import Annotated

from fastapi import APIRouter, Query, Request, UploadFile, status

from app.api.deps import CurrentUser, DbSession
from app.api.rate_limits import limit_photo_upload
from app.api.routes.photos import StorageDep, _BodyLimitedRoute
from app.schemas.common import Page
from app.schemas.found_person_report import FoundPersonReportRead
from app.services import found_person_reports as report_service

logger = logging.getLogger(__name__)


class _OneImageBodyLimitedRoute(_BodyLimitedRoute):
    max_upload_count = 1


router = APIRouter(
    prefix="/found-person-reports",
    tags=["Found person reports"],
    route_class=_OneImageBodyLimitedRoute,
)


@router.post("", response_model=FoundPersonReportRead, status_code=status.HTTP_201_CREATED)
def submit_found_person_report(
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    request: Request,
    file: UploadFile,
):
    """Submit one found-person photo for private administrator review. No search results are returned."""
    limit_photo_upload(request, user.id)
    report = report_service.submit_report(db, storage, user, file.file)
    logger.info(
        "found_person_report_submitted",
        extra={"report_id": str(report.id), "reporter_id": str(user.id)},
    )
    return report


@router.get("", response_model=Page[FoundPersonReportRead])
def list_my_found_person_reports(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    reports, total = report_service.list_reporter_reports(db, user, limit=limit, offset=offset)
    return Page[FoundPersonReportRead](
        items=reports,
        total=total,
        limit=limit,
        offset=offset,
    )
