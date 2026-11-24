import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from app.models.case import MissingCase

UTC_TODAY = datetime.now(timezone.utc).date()


def _case_count(db) -> int:
    return db.scalar(select(func.count()).select_from(MissingCase))


# ------------------------------------------------------------------ the happy path
def test_create_case_returns_201_pending_with_normalised_fields(client, make_user, auth_headers, case_payload):
    user = make_user(email="reporter@example.com")
    payload = case_payload(
        name="  Test Person  ",
        gender="  FEMALE ",
        contact_email="Family@Example.COM",
        contact_phone=" +1 (555) 010-0199 ",
        city="  Sampletown ",
    )
    response = client.post("/cases", json=payload, headers=auth_headers(user))
    assert response.status_code == 201
    body = response.json()
    assert uuid.UUID(body["id"])
    assert body["status"] == "pending"
    assert body["name"] == "Test Person"
    assert body["gender"] == "female"
    assert body["contact_email"] == "family@example.com"
    assert body["contact_phone"] == "+15550100199"
    assert body["city"] == "Sampletown"
    assert body["age"] == 12
    assert body["last_seen_date"] == payload["last_seen_date"]
    assert "reporter_id" not in body


def test_the_case_is_stored_for_the_logged_in_reporter(client, db, make_user, auth_headers, case_payload):
    user = make_user(email="reporter@example.com")
    response = client.post("/cases", json=case_payload(), headers=auth_headers(user))
    stored = db.scalar(select(MissingCase).where(MissingCase.id == uuid.UUID(response.json()["id"])))
    assert stored.reporter_id == user.id
    assert stored.status == "pending"


def test_boundary_values_are_accepted(client, make_user, auth_headers, case_payload):
    headers = auth_headers(make_user(email="reporter@example.com"))
    boundaries = [
        {"age": 0},
        {"age": 120},
        {"name": "Al"},
        {"name": "A" * 120},
        {"last_seen_date": "1900-01-01"},
        {"last_seen_date": UTC_TODAY.isoformat()},
        {"contact_phone": "1234567"},
        {"contact_phone": "+" + "1" * 15},
    ]
    for overrides in boundaries:
        response = client.post("/cases", json=case_payload(**overrides), headers=headers)
        assert response.status_code == 201, overrides


# ------------------------------------------------------------------ contact details
def test_a_phone_number_alone_is_enough(client, make_user, auth_headers, case_payload):
    payload = case_payload()
    del payload["contact_email"]
    response = client.post("/cases", json=payload, headers=auth_headers(make_user()))
    assert response.status_code == 201
    assert response.json()["contact_email"] is None
    assert response.json()["contact_phone"] == "+15550100199"


def test_an_email_alone_is_enough(client, make_user, auth_headers, case_payload):
    payload = case_payload()
    del payload["contact_phone"]
    response = client.post("/cases", json=payload, headers=auth_headers(make_user()))
    assert response.status_code == 201
    assert response.json()["contact_phone"] is None


def test_blank_contact_values_count_as_not_provided(client, db, make_user, auth_headers, case_payload):
    headers = auth_headers(make_user())
    accepted = client.post("/cases", json=case_payload(contact_phone="   "), headers=headers)
    assert accepted.status_code == 201
    assert accepted.json()["contact_phone"] is None
    rejected = client.post("/cases", json=case_payload(contact_phone="", contact_email="  "), headers=headers)
    assert rejected.status_code == 422
    assert _case_count(db) == 1


def test_at_least_one_contact_is_required(client, db, make_user, auth_headers, case_payload):
    headers = auth_headers(make_user())
    neither_sent = case_payload()
    del neither_sent["contact_phone"]
    del neither_sent["contact_email"]
    assert client.post("/cases", json=neither_sent, headers=headers).status_code == 422
    both_null = case_payload(contact_phone=None, contact_email=None)
    assert client.post("/cases", json=both_null, headers=headers).status_code == 422
    assert _case_count(db) == 0


# ------------------------------------------------------------------ authentication
def test_creating_a_case_requires_a_valid_token(client, db, case_payload):
    missing = client.post("/cases", json=case_payload())
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    garbage = client.post("/cases", json=case_payload(), headers={"Authorization": "Bearer not-a-real-token"})
    assert garbage.status_code == 401
    assert _case_count(db) == 0


# ------------------------------------------------------------------ mass assignment
@pytest.mark.parametrize(
    "field, value",
    [
        ("status", "approved"),
        ("reporter_id", str(uuid.uuid4())),
        ("id", str(uuid.uuid4())),
        ("created_at", "2020-01-01T00:00:00Z"),
        ("updated_at", "2020-01-01T00:00:00Z"),
        ("role", "admin"),
    ],
)
def test_a_client_cannot_set_server_controlled_fields(
    client, db, make_user, auth_headers, case_payload, field, value
):
    headers = auth_headers(make_user(email="reporter@example.com"))
    response = client.post("/cases", json=case_payload(**{field: value}), headers=headers)
    assert response.status_code == 422
    assert _case_count(db) == 0


# ------------------------------------------------------------------ validation
INVALID_PAYLOADS = {
    "empty name": {"name": ""},
    "blank name": {"name": "   "},
    "one character name": {"name": "A"},
    "name over 120 characters": {"name": "A" * 121},
    "null name": {"name": None},
    "age below 0": {"age": -1},
    "age above 120": {"age": 121},
    "age that is not a number": {"age": "abc"},
    "age sent as a string": {"age": "12"},
    "fractional age": {"age": 12.5},
    "boolean age": {"age": True},
    "null age": {"age": None},
    "unknown gender": {"gender": "robot"},
    "null gender": {"gender": None},
    "phone with letters": {"contact_phone": "call-me-maybe"},
    "phone too short": {"contact_phone": "12345"},
    "phone too long": {"contact_phone": "1" * 16},
    "invalid email": {"contact_email": "not-an-email"},
    "date far in the future": {"last_seen_date": (UTC_TODAY + timedelta(days=30)).isoformat()},
    "date two days ahead": {"last_seen_date": (UTC_TODAY + timedelta(days=2)).isoformat()},
    "date before 1900": {"last_seen_date": "1899-12-31"},
    "date that is not a date": {"last_seen_date": "yesterday"},
    "control character in name": {"name": "Test\x00Person"},
    "line break in address": {"address": "line one\nline two"},
    "city too short": {"city": "A"},
    "state too short": {"state": "A"},
    "address too short": {"address": "abcd"},
    "last seen location too short": {"last_seen_location": "ab"},
}


@pytest.mark.parametrize("overrides", INVALID_PAYLOADS.values(), ids=INVALID_PAYLOADS.keys())
def test_invalid_case_data_is_rejected_with_422(client, db, make_user, auth_headers, case_payload, overrides):
    headers = auth_headers(make_user(email="reporter@example.com"))
    response = client.post("/cases", json=case_payload(**overrides), headers=headers)
    assert response.status_code == 422
    assert _case_count(db) == 0


@pytest.mark.parametrize(
    "field",
    ["name", "age", "gender", "city", "state", "address", "last_seen_location", "last_seen_date"],
)
def test_every_required_field_must_be_present(client, db, make_user, auth_headers, case_payload, field):
    payload = case_payload()
    del payload[field]
    response = client.post("/cases", json=payload, headers=auth_headers(make_user()))
    assert response.status_code == 422
    assert _case_count(db) == 0


def test_validation_errors_do_not_echo_personal_data_back(client, make_user, auth_headers, case_payload):
    payload = case_payload(name="Distinctive Personname", contact_phone="+15550123456", age=-1)
    response = client.post("/cases", json=payload, headers=auth_headers(make_user()))
    assert response.status_code == 422
    assert "Distinctive" not in response.text
    assert "5550123456" not in response.text


# ------------------------------------------------------------------ logging
def test_creating_a_case_does_not_write_personal_data_to_the_logs(
    client, make_user, auth_headers, case_payload, caplog
):
    user = make_user(email="reporter@example.com")
    payload = case_payload(
        name="Zyxwv Qponml",
        contact_phone="+15550123456",
        contact_email="zyxwv.contact@example.com",
        address="987 Distinctive Lane",
        last_seen_location="Distinctive Bazaar",
    )
    with caplog.at_level(logging.INFO):
        response = client.post("/cases", json=payload, headers=auth_headers(user))
    assert response.status_code == 201

    record = next(r for r in caplog.records if r.getMessage() == "case_created")
    assert record.case_id == response.json()["id"]
    assert record.user_id == str(user.id)
    for secret in ("Zyxwv", "5550123456", "zyxwv.contact", "Distinctive"):
        assert secret not in caplog.text
        assert all(secret not in str(r.__dict__) for r in caplog.records)
