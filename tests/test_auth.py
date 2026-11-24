import base64
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import create_access_token
from app.models.user import User

PASSWORD = "correct-horse-battery"


def _payload(**overrides) -> dict:
    payload = {"email": "alice@example.com", "full_name": "Alice Example", "password": PASSWORD}
    payload.update(overrides)
    return payload


def _login(client, email, password):
    return client.post("/auth/login", data={"username": email, "password": password})


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _with_swapped_subject(token: str, new_subject: uuid.UUID) -> str:
    """Change the user id inside a token but keep the old signature."""
    header, payload, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["sub"] = str(new_subject)
    new_payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return ".".join([header, new_payload, signature])


# ------------------------------------------------------------------ register
def test_register_creates_a_user_with_role_user_and_hides_the_hash(client, db):
    response = client.post("/auth/register", json=_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "user"
    assert body["is_active"] is True
    assert body["email"] == "alice@example.com"
    assert body["full_name"] == "Alice Example"
    assert uuid.UUID(body["id"])
    assert "hashed_password" not in body
    assert "password" not in body
    assert "scrypt" not in response.text
    stored = db.scalar(select(User).where(User.email == "alice@example.com"))
    assert stored.hashed_password != PASSWORD
    assert stored.role == "user"


def test_register_ignores_a_role_field_in_the_request_body(client, db):
    response = client.post("/auth/register", json=_payload(role="admin", is_active=False))
    assert response.status_code == 201
    assert response.json()["role"] == "user"
    stored = db.scalar(select(User).where(User.email == "alice@example.com"))
    assert stored.role == "user"
    assert stored.is_active is True
    token = _login(client, "alice@example.com", PASSWORD).json()["access_token"]
    assert client.get("/admin/users", headers=_bearer(token)).status_code == 403


def test_register_stores_the_email_in_lowercase(client):
    response = client.post("/auth/register", json=_payload(email="Alice.Example@Example.COM"))
    assert response.status_code == 201
    assert response.json()["email"] == "alice.example@example.com"


def test_register_duplicate_email_returns_409(client):
    assert client.post("/auth/register", json=_payload()).status_code == 201
    assert client.post("/auth/register", json=_payload()).status_code == 409


def test_register_duplicate_email_with_different_case_returns_409(client):
    assert client.post("/auth/register", json=_payload(email="alice@example.com")).status_code == 201
    assert client.post("/auth/register", json=_payload(email="ALICE@Example.com")).status_code == 409


def test_register_short_password_returns_422_and_does_not_echo_it(client):
    response = client.post("/auth/register", json=_payload(password="tiny-pw-9"))
    assert response.status_code == 422
    assert any("password" in error["field"] for error in response.json()["error"]["fields"])
    assert "tiny-pw-9" not in response.text


def test_register_too_long_password_returns_422(client):
    assert client.post("/auth/register", json=_payload(password="a" * 129)).status_code == 422


def test_register_invalid_email_returns_422(client):
    assert client.post("/auth/register", json=_payload(email="not-an-email")).status_code == 422


# ------------------------------------------------------------------ login
def test_login_returns_a_bearer_token_that_works(client, make_user):
    make_user(email="alice@example.com")
    response = _login(client, "alice@example.com", PASSWORD)
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert client.get("/auth/me", headers=_bearer(body["access_token"])).status_code == 200


def test_wrong_password_and_unknown_email_give_identical_401_responses(client, make_user):
    make_user(email="alice@example.com")
    wrong_password = _login(client, "alice@example.com", "not-the-right-password")
    unknown_email = _login(client, "nobody@example.com", PASSWORD)
    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json()["error"]["code"] == unknown_email.json()["error"]["code"]
    assert wrong_password.json()["error"]["message"] == unknown_email.json()["error"]["message"]
    assert wrong_password.headers["www-authenticate"] == unknown_email.headers["www-authenticate"] == "Bearer"


def test_login_email_is_case_insensitive(client, make_user):
    make_user(email="alice@example.com")
    assert _login(client, "  ALICE@Example.com ", PASSWORD).status_code == 200


def test_inactive_user_cannot_log_in(client, make_user):
    make_user(email="alice@example.com", is_active=False)
    inactive = _login(client, "alice@example.com", PASSWORD)
    wrong = _login(client, "alice@example.com", "not-the-right-password")
    assert inactive.status_code == 401
    assert inactive.json()["error"]["message"] == wrong.json()["error"]["message"]          # does not reveal account state


# ------------------------------------------------------------------ existing tokens
def test_existing_token_of_a_deactivated_user_is_rejected(client, db, make_user, auth_headers):
    user = make_user()
    headers = auth_headers(user)
    assert client.get("/auth/me", headers=headers).status_code == 200
    user.is_active = False
    db.commit()
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_token_of_a_deleted_user_is_rejected(client, db, make_user, auth_headers):
    user = make_user()
    headers = auth_headers(user)
    db.delete(user)
    db.commit()
    assert client.get("/auth/me", headers=headers).status_code == 401


# ------------------------------------------------------------------ /auth/me
def test_me_without_token_returns_401_with_www_authenticate(client):
    response = client.get("/auth/me")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_with_a_garbage_token_returns_401(client):
    response = client.get("/auth/me", headers=_bearer("this.is.not-a-jwt"))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_with_a_tampered_token_returns_401(client, make_user):
    victim = make_user(email="victim@example.com")
    attacker = make_user(email="attacker@example.com")
    attacker_token = create_access_token(attacker.id)
    forged = _with_swapped_subject(attacker_token, victim.id)
    response = client.get("/auth/me", headers=_bearer(forged))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_with_a_token_signed_by_another_key_returns_401(client, make_user):
    user = make_user()
    now = int(datetime.now(timezone.utc).timestamp())
    forged = jwt.encode(
        {"sub": str(user.id), "iat": now, "exp": now + 600},
        "another-key-another-key-another-key-0123",
        algorithm=get_settings().jwt_algorithm,
    )
    assert client.get("/auth/me", headers=_bearer(forged)).status_code == 401


def test_me_with_an_expired_token_returns_401(client, make_user):
    user = make_user()
    expired = create_access_token(user.id, now=datetime.now(timezone.utc) - timedelta(hours=2))
    response = client.get("/auth/me", headers=_bearer(expired))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_returns_the_current_user(client, make_user, auth_headers):
    user = make_user(email="alice@example.com", full_name="Alice Example")
    response = client.get("/auth/me", headers=auth_headers(user))
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user.id)
    assert body["email"] == "alice@example.com"
    assert body["role"] == "user"
    assert "hashed_password" not in body


# ------------------------------------------------------------------ logging
def test_logs_never_contain_passwords_tokens_or_email_addresses(client, caplog):
    caplog.set_level(logging.INFO, logger="app")
    email = "private.person@example.com"
    password = "a-very-private-passphrase"
    assert client.post("/auth/register", json=_payload(email=email, password=password)).status_code == 201
    assert _login(client, email, "the-wrong-passphrase").status_code == 401
    token = _login(client, email, password).json()["access_token"]
    assert client.get("/auth/me", headers=_bearer(token)).status_code == 200
    assert "user_registered" in caplog.text and "login_succeeded" in caplog.text   # logging was captured
    for secret in (email, "private.person", password, "the-wrong-passphrase", token):
        assert secret not in caplog.text
