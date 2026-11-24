from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import TokenError, decode_access_token
from app.db.session import get_db
from app.models.user import ROLE_ADMIN, User
from app.services import users as user_service

# auto_error=False: a missing header is handled below, so a missing, malformed, expired or
# otherwise invalid token all produce exactly the same 401 response.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(token: Annotated[str | None, Depends(oauth2_scheme)], db: DbSession) -> User:
    """The token only says WHO the caller claims to be. The user is always loaded from the
    database, and is_active and role are checked there, never taken from the token."""
    if token is None:
        raise _unauthorized()
    try:
        user_id = decode_access_token(token)
    except TokenError:
        raise _unauthorized() from None
    user = user_service.get_by_id(db, user_id)
    if user is None or not user.is_active:
        raise _unauthorized()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    if user.role != ROLE_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
