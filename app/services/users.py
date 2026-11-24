import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import DUMMY_PASSWORD_HASH, hash_password, verify_password
from app.models.user import ROLE_ADMIN, ROLE_USER, ROLES, User
from app.services import audit

_UNIQUE_VIOLATION = "23505"  # PostgreSQL SQLSTATE


class EmailAlreadyRegisteredError(Exception):
    """A user with this email (compared case-insensitively) already exists."""


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == normalize_email(email)))


def get_by_id(db: Session, user_id: uuid.UUID) -> User | None:
    return db.get(User, user_id)


def create_user(
    db: Session, *, email: str, full_name: str, password: str, role: str = ROLE_USER
) -> User:
    """Create a user. The public registration route never passes `role`, so it is always "user";
    only trusted code (the create_admin script) passes "admin".
    Raises EmailAlreadyRegisteredError, or PasswordPolicyError for a bad password."""
    if role not in ROLES:
        raise ValueError("Unknown role")
    normalized = normalize_email(email)
    if get_by_email(db, normalized) is not None:
        raise EmailAlreadyRegisteredError
    user = User(
        email=normalized,
        full_name=full_name.strip(),
        hashed_password=hash_password(password),
        role=role,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        # Two requests registering the same email at the same moment: the database decides.
        if getattr(exc.orig, "sqlstate", None) == _UNIQUE_VIOLATION:
            raise EmailAlreadyRegisteredError from None
        raise
    return user


def authenticate(db: Session, email: str, password: str) -> User | None:
    """Return the user for a correct email + password on an active account, otherwise None.
    A password hash is verified on every path (a dummy one when the email is unknown), so the
    response time does not reveal whether an email is registered."""
    user = get_by_email(db, email)
    if user is None:
        verify_password(password, DUMMY_PASSWORD_HASH)
        return None
    password_ok = verify_password(password, user.hashed_password)
    if not password_ok or not user.is_active:
        return None
    return user


def list_users(db: Session, *, limit: int, offset: int) -> tuple[list[User], int]:
    """Newest first. Returns (users on this page, total number of users)."""
    total = db.scalar(select(func.count()).select_from(User)) or 0
    items = db.scalars(
        select(User).order_by(User.created_at.desc(), User.id.desc()).limit(limit).offset(offset)
    ).all()
    return list(items), total


class UserNotFoundError(Exception):
    """The requested user does not exist."""


class UserManagementConflictError(Exception):
    """A requested admin account change would violate an admin safety rule."""


def update_user(
    db: Session,
    *,
    actor: User,
    user_id: uuid.UUID,
    role: str | None,
    is_active: bool | None,
) -> User:
    """Update role/activity with serialized last-active-admin protection and audit events."""
    if role is not None and role not in ROLES:
        raise ValueError("Unknown role")
    try:
        # Lock the active-admin set first. This serializes concurrent demotions/deactivations,
        # so two administrators cannot both pass the last-admin check at the same time.
        active_admins = list(db.scalars(
            select(User).where(User.role == ROLE_ADMIN, User.is_active.is_(True))
            .order_by(User.id).with_for_update()
        ))
        target = db.scalar(select(User).where(User.id == user_id).with_for_update())
        if target is None:
            raise UserNotFoundError

        if target.id == actor.id and (
            (role is not None and role != target.role) or is_active is False
        ):
            raise UserManagementConflictError("An administrator cannot change their own role or deactivate themselves")

        role_changed = role is not None and role != target.role
        deactivating = is_active is False and target.is_active is True
        removes_admin = target.role == ROLE_ADMIN and target.is_active and (role == ROLE_USER or deactivating)
        if removes_admin and len(active_admins) <= 1:
            raise UserManagementConflictError("The last active administrator cannot be demoted or deactivated")

        old_role = target.role
        if role is not None:
            target.role = role
        if is_active is not None:
            target.is_active = is_active
        if role_changed:
            audit.record_event(
                db, actor_id=actor.id, action="user_role_changed", target_type="user", target_id=target.id,
                details={"from_role": old_role, "to_role": target.role},
            )
        if deactivating:
            audit.record_event(
                db, actor_id=actor.id, action="user_deactivated", target_type="user", target_id=target.id,
                details={"is_active": False},
            )
        db.commit()
        db.refresh(target)
        return target
    except Exception:
        db.rollback()
        raise
