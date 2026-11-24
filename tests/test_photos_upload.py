import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.case import MissingCase
from app.models.photo import CasePhoto
from app.services import photos as photo_service
from app.storage import LocalStorage, StorageWriteError

PHOTO_KEYS = {"id", "content_type", "size_bytes", "width", "height", "created_at", "url"}


@pytest.fixture()
def owner(make_user):
    return make_user(email="owner@example.com")


@pytest.fixture()
def case(make_case, owner):
    return make_case(owner)


def _photo_rows(db, case_id):
    return list(db.scalars(select(CasePhoto).where(CasePhoto.case_id == case_id)))


def _photo_count(db) -> int:
    return db.scalar(select(func.count()).select_from(CasePhoto))


def _files_on_disk() -> list[Path]:
    return [path for path in Path(get_settings().storage_dir).rglob("*") if path.is_file()]


def _set_status(db, case, status: str) -> None:
    case.status = status
    if status == "rejected":
        case.review_reason = "Fictional rejection reason"     # the database requires one since migration 0005
    db.commit()


# ------------------------------------------------------------------ accepted images
def test_jpeg_png_and_webp_are_accepted_and_stored_as_jpeg(owner, case, upload_photos, make_image_bytes, db, storage):
    response = upload_photos(
        owner,
        case.id,
        ("a.jpg", make_image_bytes("JPEG", seed=1), "image/jpeg"),
        ("b.png", make_image_bytes("PNG", size=(220, 140), seed=2), "image/png"),
        ("c.webp", make_image_bytes("WEBP", size=(150, 300), seed=3), "image/webp"),
    )
    assert response.status_code == 201
    body = response.json()
    assert [(p["width"], p["height"]) for p in body] == [(200, 200), (220, 140), (150, 300)]

    rows = {str(row.id): row for row in _photo_rows(db, case.id)}
    assert set(rows) == {item["id"] for item in body}
    for item in body:
        row = rows[item["id"]]
        stored = storage.read(row.storage_key)                       # the file really exists in storage
        assert storage.exists(row.storage_key)
        opened = Image.open(BytesIO(stored))
        assert opened.format == "JPEG"
        assert opened.size == (item["width"], item["height"])
        assert item["size_bytes"] == len(stored) == row.size_bytes
        assert item["content_type"] == "image/jpeg"
        assert item["url"] == f"/cases/{case.id}/photos/{item['id']}"
        assert row.sha256 == hashlib.sha256(stored).hexdigest()
    assert len(_files_on_disk()) == 3


def test_the_response_never_exposes_the_storage_key_the_hash_or_a_path(owner, case, upload_photos, make_image_bytes, db):
    response = upload_photos(owner, case.id, make_image_bytes("PNG"))
    assert response.status_code == 201
    assert set(response.json()[0]) == PHOTO_KEYS
    row = _photo_rows(db, case.id)[0]
    storage_dir = str(get_settings().storage_dir)
    for secret in (row.storage_key, row.sha256, storage_dir, Path(storage_dir).as_posix(), "storage_key", "sha256"):
        assert secret not in response.text


def test_the_clients_file_name_and_content_type_are_ignored(owner, case, upload_photos, make_image_bytes):
    response = upload_photos(
        owner,
        case.id,
        ("holiday.txt", make_image_bytes("PNG", seed=1), "text/plain"),
        ("scan.pdf", make_image_bytes("JPEG", seed=2), "image/gif"),
    )
    assert response.status_code == 201
    assert len(response.json()) == 2


def test_a_traversal_file_name_is_stored_under_a_generated_key_and_never_logged(
    owner, case, upload_photos, make_image_bytes, db, caplog
):
    with caplog.at_level(logging.INFO):
        response = upload_photos(owner, case.id, ("../../evil.jpg", make_image_bytes("JPEG"), "image/jpeg"))
    assert response.status_code == 201

    key = _photo_rows(db, case.id)[0].storage_key
    assert re.fullmatch(rf"cases/{case.id}/[0-9a-f]{{32}}\.jpg", key)
    assert [path.relative_to(get_settings().storage_dir).as_posix() for path in _files_on_disk()] == [key]
    assert not (Path(get_settings().storage_dir).parent / "evil.jpg").exists()
    assert "evil" not in response.text
    assert "evil" not in caplog.text
    assert all("evil" not in str(record.__dict__) for record in caplog.records)


def test_metadata_including_gps_is_gone_from_the_stored_file(owner, case, upload_photos, make_image_bytes, db, storage):
    source = make_image_bytes("JPEG", with_metadata=True)
    assert Image.open(BytesIO(source)).getexif().get_ifd(0x8825)      # the fictional GPS tags are in the upload
    assert upload_photos(owner, case.id, source).status_code == 201

    stored = storage.read(_photo_rows(db, case.id)[0].storage_key)
    exif = Image.open(BytesIO(stored)).getexif()
    assert len(exif) == 0
    assert not exif.get_ifd(0x8825)
    assert b"FakeCam" not in stored
    assert b"Exif" not in stored


# ------------------------------------------------------------------ rejected files
def test_a_text_file_renamed_jpg_is_rejected_with_400(owner, case, upload_photos, db):
    response = upload_photos(owner, case.id, ("photo.jpg", b"just some text, not a picture\n" * 10, "image/jpeg"))
    assert response.status_code == 400
    assert _photo_count(db) == 0 and _files_on_disk() == []


def test_gif_and_bmp_are_rejected_with_415(owner, case, upload_photos, make_image_bytes, db):
    for fmt in ("GIF", "BMP"):
        response = upload_photos(owner, case.id, (f"photo.{fmt.lower()}", make_image_bytes(fmt), "image/jpeg"))
        assert response.status_code == 415, fmt
    assert _photo_count(db) == 0 and _files_on_disk() == []


def test_animated_images_are_rejected_with_415(owner, case, upload_photos, make_image_bytes, db):
    for fmt in ("PNG", "WEBP"):
        assert upload_photos(owner, case.id, make_image_bytes(fmt, frames=3)).status_code == 415, fmt
    assert _photo_count(db) == 0


def test_corrupt_image_data_is_rejected_with_400(owner, case, upload_photos, make_image_bytes, db):
    jpeg = make_image_bytes("JPEG")
    assert upload_photos(owner, case.id, jpeg[: len(jpeg) // 2]).status_code == 400            # cut in half
    assert upload_photos(owner, case.id, jpeg[:2] + bytes(range(256)) * 20).status_code == 400  # valid start, garbage after
    assert _photo_count(db) == 0 and _files_on_disk() == []


def test_a_file_over_max_upload_bytes_is_rejected_with_413(owner, case, upload_photos, make_noisy_png, db, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_bytes", 20_000)
    assert upload_photos(owner, case.id, make_noisy_png((200, 200))).status_code == 413     # a real image, about 120 KB
    assert upload_photos(owner, case.id, b"x" * 20_001).status_code == 413                  # one byte over the limit
    assert _photo_count(db) == 0 and _files_on_disk() == []


def test_too_many_pixels_and_decompression_bombs_are_rejected_with_413(
    owner, case, upload_photos, make_png_with_declared_size, make_image_bytes, db, monkeypatch
):
    assert upload_photos(owner, case.id, make_png_with_declared_size(6000, 6000)).status_code == 413     # 36 million pixels
    assert upload_photos(owner, case.id, make_png_with_declared_size(20000, 20000)).status_code == 413   # 400 million pixels
    monkeypatch.setattr(get_settings(), "max_image_pixels", 10_000)
    assert upload_photos(owner, case.id, make_image_bytes("PNG", size=(200, 200))).status_code == 413     # a real image over a lowered limit
    assert _photo_count(db) == 0


def test_an_image_smaller_than_min_image_side_is_rejected_with_422(owner, case, upload_photos, make_image_bytes, db):
    assert upload_photos(owner, case.id, make_image_bytes("PNG", size=(99, 200))).status_code == 422
    assert upload_photos(owner, case.id, make_image_bytes("PNG", size=(200, 99))).status_code == 422
    assert _photo_count(db) == 0


def test_an_image_exactly_min_image_side_is_accepted(owner, case, upload_photos, make_image_bytes):
    assert upload_photos(owner, case.id, make_image_bytes("PNG", size=(100, 100))).status_code == 201


# ------------------------------------------------------------------ file counts
def test_a_request_without_files_is_rejected_with_422(client, owner, case, auth_headers, db):
    url = f"/cases/{case.id}/photos"
    assert client.post(url, headers=auth_headers(owner)).status_code == 422
    assert client.post(url, data={"note": "no file here"}, headers=auth_headers(owner)).status_code == 422
    assert _photo_count(db) == 0


def test_more_files_than_allowed_in_one_request_are_rejected_with_422(owner, case, upload_photos, make_image_bytes, db):
    response = upload_photos(owner, case.id, *[make_image_bytes("JPEG", seed=index) for index in range(6)])
    assert response.status_code == 422
    assert "between 1 and 5" in response.json()["error"]["message"]
    assert _photo_count(db) == 0 and _files_on_disk() == []


def test_a_case_can_hold_at_most_five_photos_in_total(owner, case, upload_photos, make_image_bytes, db):
    three = [make_image_bytes("JPEG", seed=index) for index in range(3)]
    assert upload_photos(owner, case.id, *three).status_code == 201
    over = upload_photos(owner, case.id, *[make_image_bytes("JPEG", seed=10 + index) for index in range(3)])
    assert over.status_code == 409
    assert _photo_count(db) == 3                                      # the rejected request stored nothing
    assert len(_files_on_disk()) == 3

    assert upload_photos(owner, case.id, *[make_image_bytes("JPEG", seed=20 + index) for index in range(2)]).status_code == 201
    assert _photo_count(db) == 5
    assert upload_photos(owner, case.id, make_image_bytes("JPEG", seed=30)).status_code == 409


def test_an_upload_over_the_content_length_limit_is_rejected_before_it_is_processed(
    owner, case, upload_photos, db, monkeypatch
):
    monkeypatch.setattr(get_settings(), "max_upload_bytes", 1000)      # request limit: 5 x 1000 + 1 MiB

    def must_not_be_reached(*args, **kwargs):
        raise AssertionError("the body must be rejected before the service is called")

    monkeypatch.setattr(photo_service, "add_photos", must_not_be_reached)
    response = upload_photos(owner, case.id, ("big.jpg", b"0" * 1_100_000, "image/jpeg"))
    assert response.status_code == 413
    assert _photo_count(db) == 0 and _files_on_disk() == []


# ------------------------------------------------------------------ atomicity
def test_one_bad_file_in_a_batch_stores_nothing(owner, case, upload_photos, make_image_bytes, db):
    response = upload_photos(
        owner,
        case.id,
        make_image_bytes("JPEG", seed=1),
        make_image_bytes("PNG", seed=2),
        ("notes.jpg", b"definitely not an image", "image/jpeg"),
    )
    assert response.status_code == 400
    assert _photo_count(db) == 0
    assert _files_on_disk() == []


def test_a_database_commit_failure_leaves_no_file_behind(owner, case, upload_photos, make_image_bytes, db, monkeypatch):
    def failing_commit(self):
        raise RuntimeError("simulated commit failure")

    monkeypatch.setattr(Session, "commit", failing_commit)
    with pytest.raises(Exception):                                     # the unhandled error reaches the test client
        upload_photos(owner, case.id, make_image_bytes("JPEG", seed=1), make_image_bytes("JPEG", seed=2))
    monkeypatch.undo()

    assert _photo_count(db) == 0
    assert _files_on_disk() == []


class FlakyStorageForTests(LocalStorage):
    """A fake storage that fails on a chosen save() call. Used only by tests."""

    def __init__(self, root, fail_on_save: int):
        super().__init__(root)
        self.fail_on_save = fail_on_save
        self.saves = 0

    def save(self, key, data, content_type):
        self.saves += 1
        if self.saves == self.fail_on_save:
            raise StorageWriteError("simulated storage failure")
        super().save(key, data, content_type)


def test_a_storage_failure_halfway_removes_the_files_already_written(
    owner, case, upload_photos, make_image_bytes, db, override_storage
):
    flaky = FlakyStorageForTests(get_settings().storage_dir, fail_on_save=2)
    override_storage(flaky)
    response = upload_photos(owner, case.id, *[make_image_bytes("JPEG", seed=index) for index in range(3)])
    assert response.status_code == 503
    assert flaky.saves == 2                                            # the first file was written, the second failed
    assert _files_on_disk() == []                                      # ... and the first one was removed again
    assert _photo_count(db) == 0


def test_the_same_image_twice_in_one_request_is_a_409_and_stores_nothing(owner, case, upload_photos, make_image_bytes, db):
    same = make_image_bytes("JPEG", seed=5)
    response = upload_photos(owner, case.id, ("first.jpg", same, "image/jpeg"), ("second.jpg", same, "image/jpeg"))
    assert response.status_code == 409
    assert _photo_count(db) == 0 and _files_on_disk() == []


def test_an_image_already_on_the_case_is_a_409(owner, case, upload_photos, make_image_bytes, db):
    same = make_image_bytes("JPEG", seed=5)
    assert upload_photos(owner, case.id, same).status_code == 201
    again = upload_photos(owner, case.id, make_image_bytes("JPEG", seed=6), same)      # a new image plus the old one
    assert again.status_code == 409
    assert _photo_count(db) == 1                                       # the new image was not stored either
    assert len(_files_on_disk()) == 1


def test_the_same_image_may_be_used_on_different_cases(owner, make_case, upload_photos, make_image_bytes):
    same = make_image_bytes("JPEG", seed=5)
    assert upload_photos(owner, make_case(owner).id, same).status_code == 201
    assert upload_photos(owner, make_case(owner).id, same).status_code == 201


# ------------------------------------------------------------------ case status
def test_pending_and_approved_cases_accept_photos(owner, case, upload_photos, make_image_bytes, db):
    for index, status in enumerate(("pending", "approved")):
        _set_status(db, case, status)
        assert upload_photos(owner, case.id, make_image_bytes("JPEG", seed=index)).status_code == 201, status


def test_rejected_found_and_closed_cases_do_not_accept_photos(owner, case, upload_photos, make_image_bytes, db):
    for status in ("rejected", "found", "closed"):
        _set_status(db, case, status)
        assert upload_photos(owner, case.id, make_image_bytes("JPEG")).status_code == 409, status
    assert _photo_count(db) == 0 and _files_on_disk() == []


# ------------------------------------------------------------------ access
def test_uploading_requires_a_token(client, case, make_image_bytes):
    response = client.post(
        f"/cases/{case.id}/photos", files=[("files", ("a.jpg", make_image_bytes("JPEG"), "image/jpeg"))]
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_someone_elses_case_looks_exactly_like_a_case_that_does_not_exist(
    make_user, case, upload_photos, make_image_bytes, db
):
    intruder = make_user(email="intruder@example.com")
    admin = make_user(email="boss@example.com", role="admin")
    image = make_image_bytes("JPEG")
    foreign = upload_photos(intruder, case.id, image)
    missing = upload_photos(intruder, "00000000-0000-4000-8000-000000000000", image)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json()["error"]["code"] == missing.json()["error"]["code"]
    assert foreign.json()["error"]["message"] == missing.json()["error"]["message"]
    assert upload_photos(admin, case.id, image).status_code == 404      # admin access to photos arrives in Stage 5
    assert _photo_count(db) == 0 and _files_on_disk() == []


def test_a_malformed_case_id_is_rejected_with_422(owner, upload_photos, make_image_bytes):
    assert upload_photos(owner, "not-a-uuid", make_image_bytes("JPEG")).status_code == 422


# ------------------------------------------------------------------ logging
def test_logs_contain_ids_and_sizes_but_never_file_names_or_keys(
    owner, case, upload_photos, make_image_bytes, db, caplog
):
    with caplog.at_level(logging.INFO):
        rejected = upload_photos(owner, case.id, ("private-name.jpg", b"not an image", "image/jpeg"))
        accepted = upload_photos(owner, case.id, ("secret-holiday-photo.jpg", make_image_bytes("JPEG"), "image/jpeg"))
    assert rejected.status_code == 400 and accepted.status_code == 201

    uploaded = next(r for r in caplog.records if r.getMessage() == "photos_uploaded")
    assert uploaded.case_id == str(case.id) and uploaded.user_id == str(owner.id)
    assert uploaded.photo_count == 1 and uploaded.total_bytes == accepted.json()[0]["size_bytes"]
    refused = next(r for r in caplog.records if r.getMessage() == "photo_upload_rejected")
    assert refused.reason == "InvalidImageError"

    key = _photo_rows(db, case.id)[0].storage_key
    for secret in ("private-name", "secret-holiday", key, key.rsplit("/", 1)[-1], str(get_settings().storage_dir)):
        assert secret not in caplog.text
        assert all(secret not in str(record.__dict__) for record in caplog.records)
