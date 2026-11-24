from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import CurrentUser, DbSession
from app.schemas.common import Page
from app.schemas.detection import OwnerDetectionRead
from app.services import cases as case_service
from app.services import detections as detection_service

router = APIRouter(prefix="/cases", tags=["Detections"])


@router.get("/{case_id}/detections", response_model=Page[OwnerDetectionRead])
def my_case_detections(
    case_id: UUID, user: CurrentUser, db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    items, total = detection_service.list_owner_confirmed_detections(
        db, case_id, user, limit=limit, offset=offset
    )
    return Page[OwnerDetectionRead](items=items, total=total, limit=limit, offset=offset)
