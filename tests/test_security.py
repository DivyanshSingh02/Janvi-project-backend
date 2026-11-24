import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.core import security
from app.core.config import get_settings

PASSWORD = "correct-horse-battery"


def _key() -> str:
    return get_settings().secret_key.get_secret_value()


def _claims(**overrides) -> dict:
    now = int(datetime.now(timezone.utc).timestamp())
    claims = {"sub": str(uuid.uuid4()), "iat": now, "exp": now + 600}
    claims.update(overrides)
    return claims


# ------------------------------------------------------------------ passwords
def test_password_hash_is_not_the_plaintext_and_verifies():
    hashed = security.hash_password(PASSWORD)
    assert hashed != PASSWORD
    assert PASSWORD not in hashed
    assert hashed.startswith("scrypt$")
    assert security.verify_password(PASSWORD, hashed) is True


def test_same_password_gets_a_different_salt_every_time():
    first = security.hash_password(PASSWORD)
    second = security.hash_password(PASSWORD)
    assert first != second
    assert security.verify_password(PASSWORD, first)
    assert security.verify_password(PASSWORD, second)


def test_wrong_password_does_not_verify():
    hashed = security.hash_password(PASSWORD)
    assert security.verify_password("another-password-123", hashed) is False
    assert security.verify_password(PASSWORD + " ", hashed) is False


def test_malformed_stored_hashes_never_verify_and_never_raise():
    junk = [
        "",
        "plain-text",
        "scrypt$1$2$3",
        "bcrypt$32768$8$1$AAAA$AAAA",
        "scrypt$notanumber$8$1$AAAA$AAAA",
        "scrypt$1000$8$1$AAAA$AAAA",          # N is not a power of two
        "scrypt$32768$8$1$!!!!$!!!!",         # not valid base64
    ]
    for stored in junk:
        assert security.verify_password(PASSWORD, stored) is False


def test_password_policy_is_10_to_128_characters():
    for bad in ("", "a" * 9, "a" * 129):
        with pytest.raises(security.PasswordPolicyError):
            security.hash_password(bad)
    for good in ("a" * 10, "a" * 128):
        assert security.verify_password(good, security.hash_password(good))


# ------------------------------------------------------------------ tokens
def test_access_token_round_trip_and_claims():
    user_id = uuid.uuid4()
    token = security.create_access_token(user_id)
    assert security.decode_access_token(token) == user_id
    claims = jwt.decode(token, _key(), algorithms=["HS256"])
    assert set(claims) == {"sub", "iat", "exp"}          # in particular: no role claim
    assert claims["sub"] == str(user_id)
    assert claims["exp"] - claims["iat"] == get_settings().access_token_expire_minutes * 60


def test_expired_token_is_rejected():
    long_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    token = security.create_access_token(uuid.uuid4(), now=long_ago)
    with pytest.raises(security.TokenError):
        security.decode_access_token(token)


def test_token_signed_with_another_key_is_rejected():
    forged = jwt.encode(_claims(), "another-key-another-key-another-key-0123", algorithm="HS256")
    with pytest.raises(security.TokenError):
        security.decode_access_token(forged)


def test_unsigned_alg_none_token_is_rejected():
    unsigned = jwt.encode(_claims(), None, algorithm="none")
    with pytest.raises(security.TokenError):
        security.decode_access_token(unsigned)


def test_token_missing_required_claims_is_rejected():
    claims = _claims()
    del claims["exp"]
    with pytest.raises(security.TokenError):
        security.decode_access_token(jwt.encode(claims, _key(), algorithm="HS256"))


def test_token_with_a_subject_that_is_not_a_uuid_is_rejected():
    with pytest.raises(security.TokenError):
        security.decode_access_token(jwt.encode(_claims(sub="not-a-uuid"), _key(), algorithm="HS256"))
