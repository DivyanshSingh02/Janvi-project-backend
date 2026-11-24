"""Password hashing (scrypt from the standard library) and JWT access tokens (PyJWT).

Nothing in this module logs anything: passwords, hashes and tokens must never reach a log.
"""
import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import get_settings

# ---------------------------------------------------------------- password policy
PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128

# ---------------------------------------------------------------- scrypt parameters
# N=2**15, r=8, p=1 uses about 32 MiB and roughly 50-100 ms per hash on a normal PC.
# The parameters are stored inside every hash, so they can be raised later without
# breaking existing passwords.
_SCHEME = "scrypt"
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


class PasswordPolicyError(ValueError):
    """The password is shorter or longer than the allowed range."""


class TokenError(Exception):
    """The access token is missing required data, malformed, expired or not validly signed."""


def validate_password_policy(password: str) -> None:
    if not (PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH):
        raise PasswordPolicyError(
            f"Password must be between {PASSWORD_MIN_LENGTH} and {PASSWORD_MAX_LENGTH} characters long."
        )


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    # OpenSSL refuses to run when the memory needed exceeds `maxmem` (default 32 MiB),
    # so the limit is derived from the parameters, plus 1 MiB of headroom.
    maxmem = 128 * r * (n + p + 2) + 1024 * 1024
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=maxmem, dklen=dklen)


def hash_password(password: str) -> str:
    """Return 'scrypt$N$r$p$salt$hash' (salt and hash are base64). A new random salt every time."""
    validate_password_policy(password)
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _scrypt(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_DKLEN)
    return "$".join(
        [
            _SCHEME,
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time check. A malformed stored hash simply does not verify; it never raises."""
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored_hash.split("$")
        if scheme != _SCHEME:
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(digest_b64, validate=True)
        actual = _scrypt(password, salt, int(n), int(r), int(p), len(expected))
    except (ValueError, OverflowError):  # includes binascii.Error and bad scrypt parameters
        return False
    return hmac.compare_digest(actual, expected)


# Verified against when a login names an unknown email, so "unknown email" and "wrong
# password" cost the same time. Computed once at import so no request pays for it.
DUMMY_PASSWORD_HASH = hash_password("dummy-password-used-only-for-timing")


# ---------------------------------------------------------------- JWT access tokens
def create_access_token(user_id: uuid.UUID, *, now: datetime | None = None) -> str:
    """Claims: sub (user id), iat, exp. There is deliberately no role claim; the role is
    always read from the database."""
    settings = get_settings()
    issued_at = now or datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": str(user_id), "iat": issued_at, "exp": expires_at}
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> uuid.UUID:
    """Return the user id from a valid token, or raise TokenError."""
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],  # fixed list: the token cannot pick its own algorithm
            options={"require": ["sub", "exp", "iat"]},
        )
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError, KeyError, TypeError, AttributeError):
        raise TokenError("invalid access token") from None
