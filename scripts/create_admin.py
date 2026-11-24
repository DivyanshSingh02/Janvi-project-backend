"""Create an admin account, or promote an existing user to admin.

Run from the project root:

    python -m scripts.create_admin --email EMAIL --full-name NAME

The password is asked for interactively (twice, hidden). It is never a command-line
argument or an environment variable. If the email already exists, that user is promoted to
admin and no password is asked for.
"""
import argparse
import sys
from dataclasses import dataclass
from getpass import getpass
from typing import Callable

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import PasswordPolicyError, validate_password_policy
from app.db.session import SessionLocal
from app.models.user import ROLE_ADMIN, User
from app.services import users as user_service

_EMAIL_ADAPTER = TypeAdapter(EmailStr)
_MAX_PASSWORD_ATTEMPTS = 3


@dataclass(frozen=True)
class AdminResult:
    user: User
    created: bool         # True: a new account was created. False: an existing account was found.
    already_admin: bool   # True: the existing account was already an admin; nothing was changed.


class PasswordPromptAborted(Exception):
    """The operator did not provide a valid, matching password."""


def create_or_promote_admin(
    db: Session, *, email: str, full_name: str, get_password: Callable[[], str]
) -> AdminResult:
    """The whole decision logic. `get_password` is only called when a NEW account is created,
    so tests can pass a plain function and never touch getpass."""
    existing = user_service.get_by_email(db, email)
    if existing is None:
        user = user_service.create_user(
            db, email=email, full_name=full_name, password=get_password(), role=ROLE_ADMIN
        )
        return AdminResult(user=user, created=True, already_admin=False)
    if existing.role == ROLE_ADMIN:
        return AdminResult(user=existing, created=False, already_admin=True)
    existing.role = ROLE_ADMIN
    db.commit()
    return AdminResult(user=existing, created=False, already_admin=False)


def _prompt_new_password() -> str:
    for _ in range(_MAX_PASSWORD_ATTEMPTS):
        password = getpass("Password: ")
        try:
            validate_password_policy(password)
        except PasswordPolicyError as exc:
            print(f"{exc} Try again.")
            continue
        if getpass("Repeat password: ") != password:
            print("The passwords do not match. Try again.")
            continue
        return password
    raise PasswordPromptAborted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.create_admin",
        description="Create an admin account, or promote an existing user to admin.",
    )
    parser.add_argument("--email", required=True)
    parser.add_argument("--full-name", required=True)
    args = parser.parse_args(argv)

    try:
        email = _EMAIL_ADAPTER.validate_python(args.email)
    except ValidationError:
        print("Error: that is not a valid email address.")
        return 1
    full_name = args.full_name.strip()
    if not 1 <= len(full_name) <= 120:
        print("Error: the full name must be between 1 and 120 characters.")
        return 1

    try:
        with SessionLocal() as db:
            result = create_or_promote_admin(
                db, email=email, full_name=full_name, get_password=_prompt_new_password
            )
    except PasswordPromptAborted:
        print("No valid password was entered. Nothing was changed.")
        return 1
    except user_service.EmailAlreadyRegisteredError:
        print("Error: this email was registered while the script was running. Run it again to promote it.")
        return 1
    except SQLAlchemyError:
        # Not printed in detail: database errors can contain the values that were being written.
        print("Error: the database request failed. Check the POSTGRES_* settings and run 'alembic upgrade head'.")
        return 1

    if result.created:
        print(f"Created admin account (id={result.user.id}). It can now log in at POST /auth/login.")
    elif result.already_admin:
        print(f"This account is already an admin (id={result.user.id}). Nothing was changed.")
    else:
        print(
            f"An account with this email already exists (id={result.user.id}). It has been promoted to admin. "
            "Its password and name were not changed."
        )
    if not result.user.is_active:
        print("Warning: this account is inactive, so it still cannot log in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
