import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import CurrentUser, DbSession
from app.models.case import CaseStatus
from app.schemas.case import CaseCreate, CaseRead
from app.schemas.common import Page
from app.services import cases as case_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cases", tags=["Cases"])


@router.post("", response_model=CaseRead, status_code=status.HTTP_201_CREATED)
def create_case(payload: CaseCreate, user: CurrentUser, db: DbSession):
    case = case_service.create_case(db, user, payload)
    # Ids only. Names, phone numbers, emails and addresses must never reach a log.
    logger.info("case_created", extra={"case_id": str(case.id), "user_id": str(user.id)})
    return case


@router.get("", response_model=Page[CaseRead])
def list_my_cases(
    user: CurrentUser,
    db: DbSession,
    status_filter: Annotated[CaseStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Only the caller's own cases, newest first."""
    items, total = case_service.list_cases_for_owner(
        db, user, status=status_filter, limit=limit, offset=offset
    )
    return Page[CaseRead](
        items=[CaseRead.model_validate(case) for case in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{case_id}", response_model=CaseRead)
def get_my_case(case_id: UUID, user: CurrentUser, db: DbSession):
    """Owner only. Someone else's case answers 404, exactly like an id that does not exist."""
    return case_service.get_case_for_owner(db, case_id, user)
