import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.models.photo import CasePhoto
from app.storage import LocalStorage, StorageError


@pytest.fixture()
def owner(make_user):
    return make_user(email="owner@example.com")


@pytest.fixture()
def case(make_case, owner):
    return make_case(owner)


def _photo_count(db) -> int:
    return db.scalar(select(func.count()).select_from(CasePhoto))


def _files_on_disk() -> list[Path]:
    return [path for path in Path(get_settings().storage_dir).rglob("*") if path.is_file()]


def _url(case_id, photo_id=None) -> str:
    return f"/cases/{case_id}/photos" + (f"/{photo_id}" if photo_id else "")


class DeleteFailsStorageForTests(LocalStorage):
    """A fake storage whose delete() always fails. Used only by tests."""

    def delete(self, key):
        raise StorageError("simulated delete failure")


# ------------------------------------------------------------------ list
def test_the_list_is_oldest_first_and_exposes_no_internals(client, db, owner, case, make_photos, auth_headers):
    photos = make_photos(owner, case, 3)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, photo in enumerate(reversed(photos)):                  # fixed timestamps: photos[2] is the oldest
        db.get(CasePhoto, photo.id).created_at = base + timedelta(days=index)
    db.commit()

    response = client.get(_url(case.id), headers=auth_headers(owner))
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)                                      # a plain list, not a Page
    assert [item["id"] for item in body] == [str(photo.id) for photo in reversed(photos)]
    assert set(body[0]) == {"id", "content_type", "size_bytes", "width", "height", "created_at", "url"}
    for secret in (photos[0].storage_key, photos[0].sha256, "storage_key", "sha256", str(get_settings().storage_dir)):
        assert secret not in response.text


def test_a_case_without_photos_has_an_empty_list(client, owner, case, auth_headers):
    response = client.get(_url(case.id), headers=auth_headers(owner))
    assert response.status_code == 200
    assert response.json() == []


# ------------------------------------------------------------------ download
def test_the_download_returns_the_exact_stored_bytes_with_the_specified_headers(
    client, db, owner, case, upload_photos, make_image_bytes, storage, auth_headers
):
    uploaded = upload_photos(owner, case.id, ("secret-original-name.jpg", make_image_bytes("PNG"), "image/png"))
    assert uploaded.status_code == 201
    photo_id = uploaded.json()[0]["id"]
    row = db.scalar(select(CasePhoto).where(CasePhoto.id == uuid.UUID(photo_id)))

    response = client.get(uploaded.json()[0]["url"], headers=auth_headers(owner))
    assert response.status_code == 200
    assert response.content == storage.read(row.storage_key)
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("inline")
    assert "secret-original-name" not in disposition
    assert row.storage_key not in str(response.headers)


def test_a_missing_file_gives_404_and_is_logged_by_id(client, owner, case, make_photos, storage, auth_headers, caplog):
    photo = make_photos(owner, case, 1)[0]
    storage.delete(photo.storage_key)                                  # the row stays, the file disappears
    with caplog.at_level(logging.INFO):
        response = client.get(_url(case.id, photo.id), headers=auth_headers(owner))
    assert response.status_code == 404
    record = next(r for r in caplog.records if r.getMessage() == "photo_file_missing")
    assert record.photo_id == str(photo.id)
    assert photo.storage_key not in caplog.text


# ------------------------------------------------------------------ access
def test_every_photo_endpoint_requires_a_token(client, owner, case, make_photos):
    photo = make_photos(owner, case, 1)[0]
    for method, url in (("get", _url(case.id)), ("get", _url(case.id, photo.id)), ("delete", _url(case.id, photo.id))):
        response = getattr(client, method)(url)
        assert response.status_code == 401, (method, url)
        assert response.headers["www-authenticate"] == "Bearer"


def test_someone_elses_case_and_photos_look_exactly_like_missing_ones(
    client, make_user, make_case, owner, case, make_photos, auth_headers
):
    photo = make_photos(owner, case, 1)[0]
    intruder = make_user(email="intruder@example.com")
    admin = make_user(email="boss@example.com", role="admin")
    nowhere = uuid.uuid4()
    for user in (intruder, admin):                                     # admin access to photos arrives in Stage 5
        headers = auth_headers(user)
        assert client.get(_url(case.id), headers=headers).status_code == 404
        assert client.get(_url(case.id, photo.id), headers=headers).status_code == 404
        assert client.delete(_url(case.id, photo.id), headers=headers).status_code == 404
    foreign = client.get(_url(case.id, photo.id), headers=auth_headers(intruder))
    missing = client.get(_url(nowhere, nowhere), headers=auth_headers(intruder))
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json()["error"]["code"] == missing.json()["error"]["code"]
    assert foreign.json()["error"]["message"] == missing.json()["error"]["message"]


def test_a_photo_id_used_with_the_wrong_case_id_is_a_404(client, owner, case, make_case, make_photos, auth_headers):
    other_case = make_case(owner)                                      # same owner, different case
    photo = make_photos(owner, case, 1)[0]
    headers = auth_headers(owner)
    assert client.get(_url(case.id, photo.id), headers=headers).status_code == 200
    assert client.get(_url(other_case.id, photo.id), headers=headers).status_code == 404
    assert client.delete(_url(other_case.id, photo.id), headers=headers).status_code == 404
    assert client.get(_url(case.id, uuid.uuid4()), headers=headers).status_code == 404


def test_malformed_ids_are_rejected_with_422(client, owner, case, auth_headers):
    headers = auth_headers(owner)
    assert client.get(_url(case.id, "not-a-uuid"), headers=headers).status_code == 422
    assert client.get("/cases/not-a-uuid/photos", headers=headers).status_code == 422


# ------------------------------------------------------------------ delete
def test_deleting_a_photo_removes_the_row_and_the_file(client, db, owner, case, make_photos, storage, auth_headers):
    keep, remove = make_photos(owner, case, 2)
    response = client.delete(_url(case.id, remove.id), headers=auth_headers(owner))
    assert response.status_code == 204
    assert response.content == b""
    assert _photo_count(db) == 1
    assert storage.exists(remove.storage_key) is False
    assert storage.exists(keep.storage_key) is True
    assert client.get(_url(case.id, remove.id), headers=auth_headers(owner)).status_code == 404
    assert client.get(_url(case.id, keep.id), headers=auth_headers(owner)).status_code == 200


def test_photos_can_only_be_deleted_while_the_case_is_pending(client, db, owner, case, make_photos, storage, auth_headers):
    photo = make_photos(owner, case, 1)[0]
    for status in ("approved", "rejected", "found", "closed"):
        case.status = status
        case.review_reason = "Fictional rejection reason" if status == "rejected" else None   # CHECK from migration 0005
        db.commit()
        response = client.delete(_url(case.id, photo.id), headers=auth_headers(owner))
        assert response.status_code == 409, status
    assert _photo_count(db) == 1
    assert storage.exists(photo.storage_key)


def test_another_users_photo_cannot_be_deleted(client, db, make_user, owner, case, make_photos, storage, auth_headers):
    photo = make_photos(owner, case, 1)[0]
    intruder = make_user(email="intruder@example.com")
    assert client.delete(_url(case.id, photo.id), headers=auth_headers(intruder)).status_code == 404
    assert _photo_count(db) == 1
    assert storage.exists(photo.storage_key)


def test_a_failing_file_delete_still_succeeds_and_logs_the_photo_id(
    client, db, owner, case, make_photos, auth_headers, override_storage, caplog
):
    photo = make_photos(owner, case, 1)[0]
    override_storage(DeleteFailsStorageForTests(get_settings().storage_dir))
    with caplog.at_level(logging.INFO):
        response = client.delete(_url(case.id, photo.id), headers=auth_headers(owner))
    assert response.status_code == 204
    assert _photo_count(db) == 0                                       # the row is gone
    record = next(r for r in caplog.records if r.getMessage() == "photo_file_delete_failed")
    assert record.photo_id == str(photo.id)
    assert photo.storage_key not in caplog.text


def test_after_a_delete_the_same_image_and_the_freed_slot_can_be_used_again(
    client, db, owner, case, upload_photos, make_image_bytes, auth_headers
):
    images = [make_image_bytes("JPEG", seed=index) for index in range(5)]
    created = upload_photos(owner, case.id, *images)
    assert created.status_code == 201
    assert upload_photos(owner, case.id, make_image_bytes("JPEG", seed=99)).status_code == 409      # full
    assert client.delete(created.json()[0]["url"], headers=auth_headers(owner)).status_code == 204
    assert upload_photos(owner, case.id, images[0]).status_code == 201                              # same image, new slot
    assert _photo_count(db) == 5


def test_delete_logs_ids_only(client, owner, case, make_photos, auth_headers, caplog):
    photo = make_photos(owner, case, 1)[0]
    with caplog.at_level(logging.INFO):
        assert client.delete(_url(case.id, photo.id), headers=auth_headers(owner)).status_code == 204
    record = next(r for r in caplog.records if r.getMessage() == "photo_deleted")
    assert (record.case_id, record.photo_id, record.user_id) == (str(case.id), str(photo.id), str(owner.id))
    assert photo.storage_key not in caplog.text
