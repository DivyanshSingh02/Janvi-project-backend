from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from app.api.deps import AdminUser, DbSession
from app.schemas.common import Page
from app.schemas.user import UserRead, UserUpdate
from app.services import users as user_service

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/users", response_model=Page[UserRead])
def list_users(
    admin: AdminUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    items, total = user_service.list_users(db, limit=limit, offset=offset)
    return Page[UserRead](
        items=[UserRead.model_validate(user) for user in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.patch("/users/{user_id}", response_model=UserRead)
def update_user(user_id: UUID, payload: UserUpdate, admin: AdminUser, db: DbSession):
    user = user_service.update_user(
        db, actor=admin, user_id=user_id, role=payload.role, is_active=payload.is_active
    )
    return UserRead.model_validate(user)
