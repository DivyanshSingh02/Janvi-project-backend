import uuid
from datetime import datetime, timedelta, timezone

from app.models.case import MissingCase

CASE_KEYS = {
    "id", "name", "age", "gender", "contact_phone", "contact_email", "city", "state", "address",
    "last_seen_location", "last_seen_date", "status", "created_at", "updated_at",
    "review_reason", "reviewed_at",       # added in Stage 5; there is deliberately NO reviewed_by_id
}


# ------------------------------------------------------------------ list
def test_listing_requires_a_token(client):
    response = client.get("/cases")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_a_user_sees_only_their_own_cases(client, make_user, make_case, auth_headers):
    alice = make_user(email="alice@example.com")
    bob = make_user(email="bob@example.com")
    make_case(alice, name="Alice Case One")
    make_case(alice, name="Alice Case Two")
    make_case(bob, name="Bob Case")

    alice_view = client.get("/cases", headers=auth_headers(alice)).json()
    assert alice_view["total"] == 2
    assert {item["name"] for item in alice_view["items"]} == {"Alice Case One", "Alice Case Two"}

    bob_view = client.get("/cases", headers=auth_headers(bob)).json()
    assert bob_view["total"] == 1
    assert [item["name"] for item in bob_view["items"]] == ["Bob Case"]


def test_an_admin_also_sees_only_their_own_cases_here(client, make_user, make_case, auth_headers):
    admin = make_user(email="boss@example.com", role="admin")
    make_case(make_user(email="alice@example.com"))
    body = client.get("/cases", headers=auth_headers(admin)).json()
    assert body["total"] == 0
    assert body["items"] == []


def test_the_list_response_has_the_expected_shape_and_no_reporter_data(client, make_user, make_case, auth_headers):
    user = make_user(email="alice@example.com")
    make_case(user)
    response = client.get("/cases", headers=auth_headers(user))
    body = response.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert (body["limit"], body["offset"]) == (20, 0)
    assert set(body["items"][0]) == CASE_KEYS
    assert "reporter" not in response.text
    assert "alice@example.com" not in response.text


def test_cases_are_listed_newest_first_and_paginated(client, db, make_user, make_case, auth_headers):
    user = make_user(email="alice@example.com")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(5):                                   # fixed timestamps: Number 4 is the newest
        case = make_case(user, name=f"Case Number {index}")
        db.get(MissingCase, case.id).created_at = base + timedelta(days=index)
    db.commit()

    headers = auth_headers(user)
    first = client.get("/cases?limit=2&offset=0", headers=headers).json()
    second = client.get("/cases?limit=2&offset=2", headers=headers).json()
    last = client.get("/cases?limit=2&offset=4", headers=headers).json()

    assert [c["name"] for c in first["items"]] == ["Case Number 4", "Case Number 3"]
    assert [c["name"] for c in second["items"]] == ["Case Number 2", "Case Number 1"]
    assert [c["name"] for c in last["items"]] == ["Case Number 0"]
    assert first["total"] == second["total"] == last["total"] == 5
    assert (second["limit"], second["offset"]) == (2, 2)


def test_invalid_pagination_is_rejected_with_422(client, make_user, auth_headers):
    headers = auth_headers(make_user(email="alice@example.com"))
    for query in ("limit=101", "limit=0", "offset=-1"):
        assert client.get(f"/cases?{query}", headers=headers).status_code == 422, query
    assert client.get("/cases?limit=100", headers=headers).status_code == 200


def test_the_status_filter_returns_only_matching_cases(client, db, make_user, make_case, auth_headers):
    user = make_user(email="alice@example.com")
    approved = make_case(user, name="Approved Case")
    make_case(user, name="Pending Case")
    db.get(MissingCase, approved.id).status = "approved"
    db.commit()

    headers = auth_headers(user)
    only_approved = client.get("/cases?status=approved", headers=headers).json()
    only_pending = client.get("/cases?status=pending", headers=headers).json()
    nothing_closed = client.get("/cases?status=closed", headers=headers).json()
    assert [c["name"] for c in only_approved["items"]] == ["Approved Case"]
    assert only_approved["total"] == 1
    assert [c["name"] for c in only_pending["items"]] == ["Pending Case"]
    assert nothing_closed["total"] == 0


def test_an_invalid_status_filter_is_rejected_with_422(client, make_user, auth_headers):
    headers = auth_headers(make_user(email="alice@example.com"))
    assert client.get("/cases?status=archived", headers=headers).status_code == 422


# ------------------------------------------------------------------ detail
def test_the_owner_can_read_their_case(client, make_user, make_case, auth_headers):
    user = make_user(email="alice@example.com")
    case = make_case(user, name="Alice Case")
    response = client.get(f"/cases/{case.id}", headers=auth_headers(user))
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(case.id)
    assert body["name"] == "Alice Case"
    assert set(body) == CASE_KEYS


def test_someone_elses_case_looks_exactly_like_a_case_that_does_not_exist(client, make_user, make_case, auth_headers):
    alice = make_user(email="alice@example.com")
    bob = make_user(email="bob@example.com")
    alices_case = make_case(alice)

    someone_elses = client.get(f"/cases/{alices_case.id}", headers=auth_headers(bob))
    nonexistent = client.get(f"/cases/{uuid.uuid4()}", headers=auth_headers(bob))
    assert someone_elses.status_code == 404
    assert nonexistent.status_code == 404
    assert someone_elses.json()["error"]["code"] == nonexistent.json()["error"]["code"]
    assert someone_elses.json()["error"]["message"] == nonexistent.json()["error"]["message"]


def test_an_admin_gets_no_special_access_on_this_endpoint(client, make_user, make_case, auth_headers):
    admin = make_user(email="boss@example.com", role="admin")
    case = make_case(make_user(email="alice@example.com"))
    assert client.get(f"/cases/{case.id}", headers=auth_headers(admin)).status_code == 404


def test_a_malformed_case_id_is_rejected_with_422(client, make_user, auth_headers):
    headers = auth_headers(make_user(email="alice@example.com"))
    assert client.get("/cases/not-a-uuid", headers=headers).status_code == 422


def test_reading_a_case_requires_a_token(client, make_user, make_case):
    case = make_case(make_user(email="alice@example.com"))
    assert client.get(f"/cases/{case.id}").status_code == 401
