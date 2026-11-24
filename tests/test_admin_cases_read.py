"""The admin review queue, the admin case detail and the admin photo endpoint."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import get_settings
from app.models.case import MissingCase
from app.models.photo import CasePhoto

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SUMMARY_KEYS = {"id", "name", "age", "gender", "city", "state", "status", "created_at", "photo_count"}
CASE_KEYS = {
    "id", "name", "age", "gender", "contact_phone", "contact_email", "city", "state", "address",
    "last_seen_location", "last_seen_date", "status", "created_at", "updated_at", "review_reason", "reviewed_at",
}
DETAIL_KEYS = CASE_KEYS | {"reviewed_by_id", "reporter", "photos", "history"}
PHOTO_KEYS = {"id", "content_type", "size_bytes", "width", "height", "created_at", "url"}


@pytest.fixture()
def reporter(make_user):
    return make_user(email="reporter@example.com", full_name="Fictional Reporter")


@pytest.fixture()
def admin(make_user):
    return make_user(email="boss@example.com", full_name="Fictional Admin", role="admin")


def _stamp(db, case, days: int) -> None:
    """Fixed creation time: day 0 is the oldest."""
    db.get(MissingCase, case.id).created_at = BASE + timedelta(days=days)
    db.commit()


def _force_status(db, case, status: str) -> None:
    case.status = status
    if status == "rejected":
        case.review_reason = "Fictional rejection reason"
    db.commit()


def _names(response) -> list[str]:
    return [item["name"] for item in response.json()["items"]]


# ------------------------------------------------------------------ the queue
def test_the_queue_defaults_to_pending_cases_oldest_first(client, db, admin, reporter, make_case, auth_headers):
    newest = make_case(reporter, name="Queue Newest")
    oldest = make_case(reporter, name="Queue Oldest")
    middle = make_case(reporter, name="Queue Middle")
    approved = make_case(reporter, name="Queue Approved")
    for days, case in enumerate((oldest, middle, newest, approved)):
        _stamp(db, case, days)
    _force_status(db, approved, "approved")

    response = client.get("/admin/cases", headers=auth_headers(admin))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert (body["total"], body["limit"], body["offset"]) == (3, 20, 0)
    assert _names(response) == ["Queue Oldest", "Queue Middle", "Queue Newest"]


def test_the_queue_includes_cases_of_every_reporter(client, make_user, admin, make_case, auth_headers):
    make_case(make_user(email="first@example.com"), name="First Reporter Case")
    make_case(make_user(email="second@example.com"), name="Second Reporter Case")
    assert client.get("/admin/cases", headers=auth_headers(admin)).json()["total"] == 2


def test_sort_newest_reverses_the_order(client, db, admin, reporter, make_case, auth_headers):
    cases = [make_case(reporter, name=f"Queue Case {index}") for index in range(3)]
    for days, case in enumerate(cases):
        _stamp(db, case, days)
    headers = auth_headers(admin)
    assert _names(client.get("/admin/cases?sort=oldest", headers=headers)) == [
        "Queue Case 0", "Queue Case 1", "Queue Case 2",
    ]
    assert _names(client.get("/admin/cases?sort=newest", headers=headers)) == [
        "Queue Case 2", "Queue Case 1", "Queue Case 0",
    ]


@pytest.mark.parametrize("status", ["pending", "approved", "rejected", "found", "closed"])
def test_the_status_filter_returns_only_cases_with_that_status(client, db, admin, reporter, make_case, auth_headers, status):
    statuses = ["pending", "approved", "rejected", "found", "closed"]
    for index, case_status in enumerate(statuses):
        case = make_case(reporter, name=f"Status {case_status}")
        _stamp(db, case, index)
        _force_status(db, case, case_status)
    response = client.get(f"/admin/cases?status={status}", headers=auth_headers(admin))
    assert response.status_code == 200
    assert _names(response) == [f"Status {status}"]
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["status"] == status


def test_an_empty_queue_is_a_valid_answer(client, admin, auth_headers):
    body = client.get("/admin/cases", headers=auth_headers(admin)).json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_the_queue_is_paginated(client, db, admin, reporter, make_case, auth_headers):
    for index in range(5):
        _stamp(db, make_case(reporter, name=f"Queue Case {index}"), index)
    headers = auth_headers(admin)
    first = client.get("/admin/cases?limit=2&offset=0", headers=headers).json()
    second = client.get("/admin/cases?limit=2&offset=2", headers=headers).json()
    last = client.get("/admin/cases?limit=2&offset=4", headers=headers).json()
    assert [c["name"] for c in first["items"]] == ["Queue Case 0", "Queue Case 1"]
    assert [c["name"] for c in second["items"]] == ["Queue Case 2", "Queue Case 3"]
    assert [c["name"] for c in last["items"]] == ["Queue Case 4"]
    assert first["total"] == second["total"] == last["total"] == 5
    assert (second["limit"], second["offset"]) == (2, 2)
    newest_first = client.get("/admin/cases?sort=newest&limit=2&offset=1", headers=headers).json()
    assert [c["name"] for c in newest_first["items"]] == ["Queue Case 3", "Queue Case 2"]


def test_invalid_query_parameters_are_rejected_with_422(client, admin, auth_headers):
    headers = auth_headers(admin)
    for query in ("limit=101", "limit=0", "offset=-1", "status=archived", "sort=sideways", "limit=abc"):
        assert client.get(f"/admin/cases?{query}", headers=headers).status_code == 422, query
    assert client.get("/admin/cases?limit=100", headers=headers).status_code == 200


def test_the_photo_count_is_correct_for_every_row(client, db, admin, reporter, make_case, make_photos, auth_headers):
    none, one, three = (make_case(reporter, name=f"Photos {count}") for count in (0, 1, 3))
    for days, case in enumerate((none, one, three)):
        _stamp(db, case, days)
    make_photos(reporter, one, 1, first_seed=10)
    make_photos(reporter, three, 3, first_seed=20)
    items = client.get("/admin/cases", headers=auth_headers(admin)).json()["items"]
    assert {item["name"]: item["photo_count"] for item in items} == {"Photos 0": 0, "Photos 1": 1, "Photos 3": 3}


def test_a_queue_row_shows_only_the_summary_fields(client, admin, reporter, make_case, auth_headers):
    make_case(reporter, name="Summary Case", address="7 Hidden Street, Example Colony", contact_phone="+15550123456")
    response = client.get("/admin/cases", headers=auth_headers(admin))
    assert set(response.json()["items"][0]) == SUMMARY_KEYS
    for private in ("7 Hidden Street", "5550123456", "family@example.com", "reporter@example.com", "Sample Market"):
        assert private not in response.text


# ------------------------------------------------------------------ the detail
def test_the_detail_has_every_case_field_the_reporter_the_review_fields_photos_and_history(
    client, admin, reporter, make_case, make_photos, auth_headers
):
    case = make_case(reporter, name="Detail Case", address="9 Detail Road, Example Colony")
    photos = make_photos(reporter, case, 2)
    response = client.get(f"/admin/cases/{case.id}", headers=auth_headers(admin))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == DETAIL_KEYS
    assert body["id"] == str(case.id)
    assert (body["name"], body["age"], body["gender"]) == ("Detail Case", 12, "female")
    assert body["address"] == "9 Detail Road, Example Colony"
    assert body["contact_phone"] == "+15550100199" and body["contact_email"] == "family@example.com"
    assert body["last_seen_location"] == "Sample Market" and body["status"] == "pending"
    assert body["reporter"] == {"id": str(reporter.id), "full_name": "Fictional Reporter", "email": "reporter@example.com"}
    assert (body["review_reason"], body["reviewed_by_id"], body["reviewed_at"]) == (None, None, None)
    assert body["history"] == []
    assert [photo["id"] for photo in body["photos"]] == [str(photo.id) for photo in photos]
    assert set(body["photos"][0]) == PHOTO_KEYS
    for secret in (photos[0].storage_key, photos[0].sha256, "storage_key", "sha256", "hashed_password"):
        assert secret not in response.text


def test_the_detail_of_a_case_without_photos_has_an_empty_photo_list(client, admin, reporter, make_case, auth_headers):
    case = make_case(reporter)
    assert client.get(f"/admin/cases/{case.id}", headers=auth_headers(admin)).json()["photos"] == []


def test_an_unknown_case_is_a_404_and_a_malformed_id_is_a_422(client, admin, auth_headers):
    headers = auth_headers(admin)
    assert client.get(f"/admin/cases/{uuid.uuid4()}", headers=headers).status_code == 404
    assert client.get("/admin/cases/not-a-uuid", headers=headers).status_code == 422


def test_the_photo_urls_in_the_detail_are_admin_urls_that_work(client, admin, reporter, make_case, make_photos, auth_headers):
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    headers = auth_headers(admin)
    url = client.get(f"/admin/cases/{case.id}", headers=headers).json()["photos"][0]["url"]
    assert url == f"/admin/cases/{case.id}/photos/{photo.id}"
    assert client.get(url, headers=headers).status_code == 200
    # The owner's URL (/cases/...) stays owner-only, so it must not be the one an admin is told to use.
    assert client.get(f"/cases/{case.id}/photos/{photo.id}", headers=headers).status_code == 404


# ------------------------------------------------------------------ the admin photo endpoint
def test_the_admin_photo_endpoint_returns_the_stored_bytes_with_the_same_security_headers(
    client, db, admin, reporter, make_case, make_photos, storage, auth_headers
):
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    response = client.get(f"/admin/cases/{case.id}/photos/{photo.id}", headers=auth_headers(admin))
    assert response.status_code == 200
    assert response.content == storage.read(photo.storage_key)
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"] == 'inline; filename="photo.jpg"'
    assert photo.storage_key not in str(response.headers)


def test_a_photo_of_another_case_is_a_404(client, admin, reporter, make_case, make_photos, auth_headers):
    case, other_case = make_case(reporter), make_case(reporter)
    photo = make_photos(reporter, case, 1, first_seed=1)[0]
    other_photo = make_photos(reporter, other_case, 1, first_seed=2)[0]
    headers = auth_headers(admin)
    assert client.get(f"/admin/cases/{case.id}/photos/{photo.id}", headers=headers).status_code == 200
    mismatched = client.get(f"/admin/cases/{case.id}/photos/{other_photo.id}", headers=headers)
    assert mismatched.status_code == 404
    assert client.get(f"/admin/cases/{other_case.id}/photos/{photo.id}", headers=headers).status_code == 404
    unknown = client.get(f"/admin/cases/{case.id}/photos/{uuid.uuid4()}", headers=headers)
    assert unknown.status_code == 404
    assert mismatched.json()["error"]["code"] == unknown.json()["error"]["code"]
    assert mismatched.json()["error"]["message"] == unknown.json()["error"]["message"]  # no way to tell the two apart


def test_an_unknown_case_or_a_malformed_id_on_the_admin_photo_endpoint(client, admin, reporter, make_case, make_photos, auth_headers):
    photo = make_photos(reporter, make_case(reporter), 1)[0]
    headers = auth_headers(admin)
    assert client.get(f"/admin/cases/{uuid.uuid4()}/photos/{photo.id}", headers=headers).status_code == 404
    assert client.get(f"/admin/cases/{photo.case_id}/photos/not-a-uuid", headers=headers).status_code == 422


def test_a_photo_whose_file_is_gone_is_a_404(client, admin, reporter, make_case, make_photos, storage, auth_headers):
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    storage.delete(photo.storage_key)
    assert client.get(f"/admin/cases/{case.id}/photos/{photo.id}", headers=auth_headers(admin)).status_code == 404


def test_the_owner_endpoints_still_refuse_an_admin(client, admin, reporter, make_case, make_photos, auth_headers):
    """Stage 4 behaviour is unchanged: admins read photos through /admin/..., not through the owner's URLs."""
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    headers = auth_headers(admin)
    assert client.get(f"/cases/{case.id}/photos", headers=headers).status_code == 404
    assert client.get(f"/cases/{case.id}/photos/{photo.id}", headers=headers).status_code == 404
