from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.exceptions import RequestValidationError
from sqlalchemy import func, select

from app.api.deps import AdminUser, DbSession
from app.models.face_search_log import FaceSearchLog
from app.schemas.audit import AuditEventRead, SearchLogRead
from app.schemas.common import Page
from app.services import audit

router = APIRouter(prefix="/admin", tags=["Audit"])


def _check_date_range(start: datetime | None, end: datetime | None, field: str) -> None:
    for name, value in ((f"{field}_from", start), (f"{field}_to", end)):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise RequestValidationError([{
                "type": "value_error", "loc": ("query", name), "msg": "Value error, timezone required"
            }])
    if start is not None and end is not None and start > end:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", f"{field}_from"),
            "msg": "Value error, start must not be after end",
        }])


@router.get("/audit-events", response_model=Page[AuditEventRead])
def list_audit_events(
    admin: AdminUser,
    db: DbSession,
    action: Annotated[str | None, Query(max_length=64)] = None,
    actor_id: UUID | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    _check_date_range(created_from, created_to, "created")
    items, total = audit.list_events(
        db, action=action, actor_id=actor_id, created_from=created_from,
        created_to=created_to, limit=limit, offset=offset,
    )
    return Page[AuditEventRead](
        items=[AuditEventRead.model_validate(item) for item in items],
        total=total, limit=limit, offset=offset,
    )


@router.get("/search-logs", response_model=Page[SearchLogRead])
def list_search_logs(
    admin: AdminUser,
    db: DbSession,
    admin_id: UUID | None = None,
    searched_from: datetime | None = None,
    searched_to: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    _check_date_range(searched_from, searched_to, "searched")
    filters = []
    if admin_id:
        filters.append(FaceSearchLog.admin_id == admin_id)
    if searched_from:
        filters.append(FaceSearchLog.searched_at >= searched_from)
    if searched_to:
        filters.append(FaceSearchLog.searched_at <= searched_to)
    total = db.scalar(select(func.count()).select_from(FaceSearchLog).where(*filters)) or 0
    rows = db.scalars(
        select(FaceSearchLog).where(*filters)
        .order_by(FaceSearchLog.searched_at.desc(), FaceSearchLog.id.desc())
        .limit(limit).offset(offset)
    ).all()
    return Page[SearchLogRead](
        items=[SearchLogRead.model_validate(item) for item in rows],
        total=total, limit=limit, offset=offset,
    )
