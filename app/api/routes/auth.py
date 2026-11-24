import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import CurrentUser, DbSession
from app.api.rate_limits import limit_login, limit_registration
from app.core.security import create_access_token
from app.schemas.user import Token, UserCreate, UserRead
from app.services import users as user_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: DbSession, request: Request):
    limit_registration(request)
    try:
        user = user_service.create_user(
            db, email=payload.email, full_name=payload.full_name, password=payload.password
        )
    except user_service.EmailAlreadyRegisteredError:
        raise
    logger.info("user_registered", extra={"user_id": str(user.id)})
    return user


@router.post("/login", response_model=Token)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession, request: Request):
    limit_login(request, form.username)
    """OAuth2 password form: the `username` field carries the email address."""
    user = user_service.authenticate(db, form.username, form.password)
    if user is None:
        # One message for unknown email, wrong password and inactive account. Nothing identifying is logged.
        logger.info("login_rejected")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info("login_succeeded", extra={"user_id": str(user.id)})
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserRead)
def me(user: CurrentUser):
    return user
