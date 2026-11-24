"""Audit-event creation and read queries. Events are added to the caller's transaction."""
import re
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.audit_event import AuditEvent

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_ALLOWED_ACTIONS = frozenset({
    "user_role_changed", "user_deactivated", "case_review_approved", "case_review_rejected",
    "case_status_changed", "case_hard_deleted",
})
_ALLOWED_TARGET_TYPES = frozenset({"user", "case"})
_ALLOWED_VALUE_CODES = frozenset({
    "user", "admin", "pending", "approved", "rejected", "found", "closed",
    "camera", "video", "manual", "unreviewed", "confirmed", "dismissed",
})


class InvalidAuditDetailsError(ValueError):
    """Audit details contained something other than a count, flag or short code."""


def _validate_details(details: dict[str, Any] | None) -> dict[str, Any]:
    clean = details or {}
    if not isinstance(clean, dict):
        raise InvalidAuditDetailsError("Audit details must be an object")
    for key, value in clean.items():
        if not isinstance(key, str) or not _CODE.fullmatch(key):
            raise InvalidAuditDetailsError("Audit detail keys must be short codes")
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            continue
        if isinstance(value, str) and value in _ALLOWED_VALUE_CODES:
            continue
        raise InvalidAuditDetailsError("Audit detail values must be counts, flags or short codes")
    return clean


def record_event(
    db: Session,
    *,
    actor_id: uuid.UUID,
    action: str,
    target_type: str,
    target_id: uuid.UUID | None,
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    """Add an event but do not flush or commit; caller owns transaction boundaries."""
    if action not in _ALLOWED_ACTIONS or target_type not in _ALLOWED_TARGET_TYPES:
        raise InvalidAuditDetailsError("Audit action and target type must be short codes")
    event = AuditEvent(
        actor_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=_validate_details(details),
    )
    db.add(event)
    return event


def list_events(
    db: Session,
    *,
    action: str | None,
    actor_id: uuid.UUID | None,
    created_from: datetime | None,
    created_to: datetime | None,
    limit: int,
    offset: int,
) -> tuple[list[AuditEvent], int]:
    filters = []
    if action:
        filters.append(AuditEvent.action == action)
    if actor_id:
        filters.append(AuditEvent.actor_id == actor_id)
    if created_from:
        filters.append(AuditEvent.created_at >= created_from)
    if created_to:
        filters.append(AuditEvent.created_at <= created_to)
    total = db.scalar(select(func.count()).select_from(AuditEvent).where(*filters)) or 0
    rows = db.scalars(
        select(AuditEvent).where(*filters)
        .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
        .limit(limit).offset(offset)
    ).all()
    return list(rows), total
