import uuid

from sqlalchemy import select

from app.models.found_person_report import FoundPersonReport
from app.services import embeddings as embedding_service
from app.services import found_person_reports as report_service


def _submit(client, user, auth_headers, image):
    return client.post(
        "/found-person-reports",
        files={"file": ("untrusted-original-name.png", image, "image/png")},
        headers=auth_headers(user),
    )


def test_reporter_can_submit_and_only_sees_their_own_report(
    client, reporter, make_user, auth_headers, make_image_bytes
):
    first = _submit(client, reporter, auth_headers, make_image_bytes("PNG", seed=3))
    assert first.status_code == 201, first.text
    assert set(first.json()) == {"id", "review_status", "created_at"}
    assert first.json()["review_status"] == "pending"

    other_user = make_user(email="other@example.com")
    other = _submit(client, other_user, auth_headers, make_image_bytes("PNG", seed=4))
    assert other.status_code == 201

    mine = client.get("/found-person-reports", headers=auth_headers(reporter))
    assert mine.status_code == 200
    assert [item["id"] for item in mine.json()["items"]] == [first.json()["id"]]
    assert "reporter_email" not in mine.text
    assert "case" not in mine.text

    assert client.get("/found-person-reports").status_code == 401
    assert client.post(
        "/found-person-reports",
        files={"file": ("x.png", make_image_bytes("PNG"), "image/png")},
    ).status_code == 401


def test_admin_can_review_submission_and_download_photo(
    client, reporter, admin, auth_headers, make_image_bytes
):
    raw_image = make_image_bytes("PNG", seed=6)
    submitted = _submit(client, reporter, auth_headers, raw_image)
    report_id = submitted.json()["id"]

    pending = client.get("/admin/found-person-reports", headers=auth_headers(admin))
    assert pending.status_code == 200
    row = pending.json()["items"][0]
    assert row["id"] == report_id
    assert row["reporter_email"] == reporter.email
    assert row["review_status"] == "pending"

    photo = client.get(f"/admin/found-person-reports/{report_id}/photo", headers=auth_headers(admin))
    assert photo.status_code == 200
    assert photo.headers["content-type"] == "image/jpeg"
    assert photo.content != raw_image

    reviewed = client.patch(
        f"/admin/found-person-reports/{report_id}/review",
        headers=auth_headers(admin),
        json={"review_status": "reviewed", "note": "Reviewed privately"},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["review_status"] == "reviewed"
    assert reviewed.json()["review_note"] == "Reviewed privately"

    contacted = client.patch(
        f"/admin/found-person-reports/{report_id}/review",
        headers=auth_headers(admin),
        json={"review_status": "contacted"},
    )
    assert contacted.status_code == 200
    assert contacted.json()["review_status"] == "contacted"

    invalid_transition = client.patch(
        f"/admin/found-person-reports/{report_id}/review",
        headers=auth_headers(admin),
        json={"review_status": "dismissed"},
    )
    assert invalid_transition.status_code == 409
    assert "Reviewed privately" not in client.get(
        "/found-person-reports", headers=auth_headers(reporter)
    ).text
    assert client.get("/admin/found-person-reports").status_code == 401


def test_admin_can_permanently_delete_a_report_and_its_photo(
    client, db, storage, reporter, admin, auth_headers, make_image_bytes
):
    submitted = _submit(client, reporter, auth_headers, make_image_bytes("PNG", seed=9))
    report_id = uuid.UUID(submitted.json()["id"])
    report = db.get(FoundPersonReport, report_id)
    assert report is not None
    storage_key = report.storage_key
    assert storage.exists(storage_key)

    url = f"/admin/found-person-reports/{report_id}"
    assert client.delete(url, headers=auth_headers(admin)).status_code == 422
    deleted = client.delete(url + "?confirm=true", headers=auth_headers(admin))
    assert deleted.status_code == 200
    assert deleted.json() == {"report_id": str(report_id), "photo_deleted": True}
    assert not storage.exists(storage_key)
    db.expire_all()
    assert db.get(FoundPersonReport, report_id) is None


def test_admin_searches_a_submitted_photo_but_reporter_never_gets_matches(
    client,
    db,
    storage,
    fake_recognition,
    reporter,
    admin,
    auth_headers,
    make_case,
    make_photos,
    make_image_bytes,
    final_sha,
):
    case = make_case(reporter, name="Private Match Person")
    photo = make_photos(reporter, case, 1, first_seed=20)[0]
    fake_recognition.set_faces(photo.sha256, [1, 0, 0, 0])
    case.status = "approved"
    db.commit()
    embedding_service.enrol_case(db, storage, fake_recognition, case.id)

    found_image = make_image_bytes("PNG", size=(220, 220), seed=21)
    submitted = _submit(client, reporter, auth_headers, found_image)
    report_id = submitted.json()["id"]
    stored_image = report_service.get_report_photo(
        report_service.get_report_for_admin(db, uuid.UUID(report_id)), storage
    )
    fake_recognition.set_faces(final_sha(stored_image), [1, 0, 0, 0])

    response = client.post(
        f"/admin/found-person-reports/{report_id}/search",
        headers=auth_headers(admin),
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["case"]["name"] == "Private Match Person"
    assert reporter.email not in response.text
    assert "contact_phone" not in response.text

    reporter_view = client.get("/found-person-reports", headers=auth_headers(reporter))
    assert reporter_view.status_code == 200
    assert "results" not in reporter_view.text
    assert "Private Match Person" not in reporter_view.text

    report = db.scalar(select(FoundPersonReport).where(FoundPersonReport.id == uuid.UUID(report_id)))
    assert report is not None
    denied = client.post(
        f"/admin/found-person-reports/{report_id}/search",
        headers=auth_headers(reporter),
    )
    assert denied.status_code == 403
