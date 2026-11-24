"""Face enrolment: POST /admin/cases/{id}/embeddings and the step that runs after an approval."""
import logging
import uuid

import pytest
from sqlalchemy import func, select

from app.models.case import MissingCase
from app.models.face_embedding import FaceEmbedding
from app.models.photo import CasePhoto
from app.recognition import vectors

UNAVAILABLE = "No face recognition model is configured"
ONE_FACE = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]


@pytest.fixture()
def reporter(make_user):
    return make_user(email="reporter@example.com", full_name="Fictional Reporter")


@pytest.fixture()
def admin(make_user):
    return make_user(email="boss@example.com", full_name="Fictional Admin", role="admin")


def _force_status(db, case, status: str) -> None:
    case.status = status
    if status == "rejected":
        case.review_reason = "Fictional earlier rejection"
    db.commit()


def _register(fake, photos, *vector_list) -> None:
    for photo, vector in zip(photos, vector_list):
        fake.set_faces(photo.sha256, vector)


@pytest.fixture()
def approved_case(db, make_case, make_photos, reporter):
    """An approved case with two photos (status set directly, so no approval hook has run)."""
    case = make_case(reporter, name="Fictional Enrolment Case")
    photos = make_photos(reporter, case, 2)
    _force_status(db, case, "approved")
    return case, photos


def _enrol(client, auth_headers, admin, case_id):
    return client.post(f"/admin/cases/{case_id}/embeddings", headers=auth_headers(admin))


def _rows(db, case_id) -> list[FaceEmbedding]:
    db.expire_all()
    return list(db.scalars(select(FaceEmbedding).where(FaceEmbedding.case_id == case_id).order_by(FaceEmbedding.created_at)))


def _status_now(db, case) -> str:
    db.expire_all()
    return db.get(MissingCase, case.id).status


def _photo_states(db, case_id) -> dict:
    db.expire_all()
    rows = db.scalars(select(CasePhoto).where(CasePhoto.case_id == case_id))
    return {row.id: (row.embedding_status, row.embedding_error) for row in rows}


# ------------------------------------------------------------------ no model
def test_enrolment_without_a_model_answers_503_and_changes_nothing(client, db, auth_headers, admin, approved_case):
    case, photos = approved_case
    response = _enrol(client, auth_headers, admin, case.id)                    # the real, unconfigured service
    assert response.status_code == 503
    assert response.json()["error"]["message"] == UNAVAILABLE
    assert _rows(db, case.id) == []
    assert set(_photo_states(db, case.id).values()) == {("pending", None)}
    assert db.get(MissingCase, case.id).status == "approved"


# ------------------------------------------------------------------ access and status
def test_enrolment_is_admin_only(client, auth_headers, reporter, approved_case, fake_recognition):
    case, _ = approved_case
    assert client.post(f"/admin/cases/{case.id}/embeddings").status_code == 401
    assert _enrol(client, auth_headers, reporter, case.id).status_code == 403
    assert fake_recognition.calls == []


def test_an_unknown_case_is_a_404(client, auth_headers, admin, fake_recognition):
    assert _enrol(client, auth_headers, admin, uuid.uuid4()).status_code == 404


def test_only_approved_cases_can_be_enrolled(client, db, auth_headers, admin, approved_case, fake_recognition):
    case, photos = approved_case
    _register(fake_recognition, photos, *ONE_FACE)
    for status in ("pending", "rejected", "found", "closed"):
        _force_status(db, case, status)
        response = _enrol(client, auth_headers, admin, case.id)
        assert response.status_code == 409, status
        assert "approved" in response.json()["error"]["message"]
    assert _rows(db, case.id) == []
    assert fake_recognition.calls == []
    assert set(_photo_states(db, case.id).values()) == {("pending", None)}


# ------------------------------------------------------------------ the happy path
def test_enrolment_stores_one_normalised_vector_per_photo_with_the_model_and_dimension(
    client, db, auth_headers, admin, approved_case, fake_recognition
):
    case, photos = approved_case
    _register(fake_recognition, photos, [3, 4, 0, 0], [0, 0, 5, 12])
    response = _enrol(client, auth_headers, admin, case.id)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"generated", "pending", "failed", "case_id", "model", "photos"}
    assert (body["generated"], body["pending"], body["failed"]) == (2, 0, 0)
    assert body["model"] == {"name": "fake-face-model", "version": "1"}
    assert {item["photo_id"] for item in body["photos"]} == {str(photo.id) for photo in photos}
    assert all(item["status"] == "done" and item["error"] is None for item in body["photos"])
    assert "vector" not in response.text and '"embedding"' not in response.text     # vectors are never returned

    rows = {row.photo_id: row for row in _rows(db, case.id)}
    assert set(rows) == {photo.id for photo in photos}
    for photo, expected in zip(photos, ([0.6, 0.8, 0, 0], [0, 0, 5 / 13, 12 / 13])):
        row = rows[photo.id]
        assert (row.model_name, row.model_version, row.dimension, row.case_id) == ("fake-face-model", "1", 4, case.id)
        assert len(row.vector) == 4 * 4
        assert vectors.unpack(row.vector, 4).tolist() == pytest.approx(expected, abs=1e-6)     # stored L2-normalised
    assert set(_photo_states(db, case.id).values()) == {("done", None)}


def test_enrolment_is_idempotent(client, db, auth_headers, admin, approved_case, fake_recognition):
    case, photos = approved_case
    _register(fake_recognition, photos, *ONE_FACE)
    first = _enrol(client, auth_headers, admin, case.id).json()
    calls_after_first = len(fake_recognition.calls)
    second = _enrol(client, auth_headers, admin, case.id)
    assert second.status_code == 200
    assert (second.json()["generated"], second.json()["pending"], second.json()["failed"]) == (2, 0, 0)
    assert second.json()["photos"] == first["photos"]
    assert len(_rows(db, case.id)) == 2                                          # no duplicate rows
    assert len(fake_recognition.calls) == calls_after_first == 2                 # nothing was processed again


# ------------------------------------------------------------------ failures per photo
def test_a_photo_without_exactly_one_face_is_failed_while_the_others_succeed(
    client, db, auth_headers, admin, reporter, make_case, make_photos, fake_recognition
):
    case = make_case(reporter)
    photos = make_photos(reporter, case, 3)
    _force_status(db, case, "approved")
    fake_recognition.set_faces(photos[0].sha256, [1, 0, 0, 0])                   # one face
    # photos[1] is not registered: no face
    fake_recognition.set_faces(photos[2].sha256, [0, 1, 0, 0], [0, 0, 1, 0])     # two faces
    response = _enrol(client, auth_headers, admin, case.id)
    assert response.status_code == 200
    body = response.json()
    assert (body["generated"], body["pending"], body["failed"]) == (1, 0, 2)
    by_photo = {item["photo_id"]: (item["status"], item["error"]) for item in body["photos"]}
    assert by_photo == {
        str(photos[0].id): ("done", None),
        str(photos[1].id): ("failed", "no_face"),
        str(photos[2].id): ("failed", "multiple_faces"),
    }
    assert [row.photo_id for row in _rows(db, case.id)] == [photos[0].id]
    assert db.get(MissingCase, case.id).status == "approved"


def test_a_retry_fixes_a_failed_photo_and_only_that_photo_is_processed(
    client, db, auth_headers, admin, approved_case, fake_recognition
):
    case, photos = approved_case
    fake_recognition.set_faces(photos[0].sha256, [1, 0, 0, 0])                   # photos[1]: no face yet
    first = _enrol(client, auth_headers, admin, case.id).json()
    assert (first["generated"], first["failed"]) == (1, 1)
    calls_before = len(fake_recognition.calls)

    fake_recognition.set_faces(photos[1].sha256, [0, 1, 0, 0])                   # now the 'model' finds a face
    second = _enrol(client, auth_headers, admin, case.id).json()
    assert (second["generated"], second["pending"], second["failed"]) == (2, 0, 0)
    assert len(fake_recognition.calls) == calls_before + 1                       # only the failed photo was retried
    assert set(_photo_states(db, case.id).values()) == {("done", None)}           # the error code is cleared


def test_a_photo_whose_file_is_gone_is_failed_and_the_rest_carries_on(
    client, db, auth_headers, admin, approved_case, fake_recognition, storage
):
    case, photos = approved_case
    _register(fake_recognition, photos, *ONE_FACE)
    storage.delete(photos[0].storage_key)
    body = _enrol(client, auth_headers, admin, case.id).json()
    assert (body["generated"], body["failed"]) == (1, 1)
    assert _photo_states(db, case.id)[photos[0].id] == ("failed", "file_missing")


# ------------------------------------------------------------------ model versions
def test_embeddings_from_an_older_model_version_are_regenerated(
    client, db, auth_headers, admin, approved_case, fake_recognition
):
    case, photos = approved_case
    _register(fake_recognition, photos, [1, 0, 0, 0], [0, 1, 0, 0])
    _enrol(client, auth_headers, admin, case.id)
    assert {row.model_version for row in _rows(db, case.id)} == {"1"}

    fake_recognition.version = "2"                                               # a new model version
    _register(fake_recognition, photos, [0, 0, 1, 0], [0, 0, 0, 1])
    calls_before = len(fake_recognition.calls)
    body = _enrol(client, auth_headers, admin, case.id).json()
    assert body["model"] == {"name": "fake-face-model", "version": "2"}
    assert (body["generated"], body["failed"]) == (2, 0)
    assert len(fake_recognition.calls) == calls_before + 2                       # both photos were processed again
    rows = _rows(db, case.id)
    assert len(rows) == 2 and {row.model_version for row in rows} == {"2"}       # the old rows were replaced
    assert vectors.unpack(rows[0].vector, 4).tolist() in ([0, 0, 1, 0], [0, 0, 0, 1])


# ------------------------------------------------------------------ a faulty or failing model
def test_a_model_that_returns_an_unusable_embedding_is_a_503_and_marks_nothing(
    client, db, auth_headers, admin, approved_case, fake_recognition
):
    case, photos = approved_case
    fake_recognition.set_faces(photos[0].sha256, [1, 0, 0, 0])
    fake_recognition.set_faces(photos[1].sha256, [float("nan"), 0, 0, 0])
    response = _enrol(client, auth_headers, admin, case.id)
    assert response.status_code == 503
    assert "unusable" in response.json()["error"]["message"]
    assert _rows(db, case.id) == []                                              # the good photo was not saved either
    assert set(_photo_states(db, case.id).values()) == {("pending", None)}       # and nothing is called "failed"


def test_a_service_that_fails_halfway_saves_nothing_and_a_retry_works(
    client, db, auth_headers, admin, approved_case, fake_recognition
):
    case, photos = approved_case
    _register(fake_recognition, photos, *ONE_FACE)
    fake_recognition.fail_on_call = 2
    response = _enrol(client, auth_headers, admin, case.id)
    assert response.status_code == 503
    assert response.json()["error"]["message"] == "The face recognition service is temporarily unavailable"
    assert _rows(db, case.id) == []
    assert set(_photo_states(db, case.id).values()) == {("pending", None)}
    fake_recognition.fail_on_call = None
    assert _enrol(client, auth_headers, admin, case.id).json()["generated"] == 2


def test_photos_added_after_approval_wait_until_the_next_enrolment(
    client, db, auth_headers, admin, reporter, approved_case, make_photos, fake_recognition
):
    case, photos = approved_case
    _register(fake_recognition, photos, *ONE_FACE)
    _enrol(client, auth_headers, admin, case.id)
    newcomer = make_photos(reporter, case, 1, first_seed=50)[0]                  # an approved case still accepts photos
    assert _photo_states(db, case.id)[newcomer.id] == ("pending", None)
    fake_recognition.set_faces(newcomer.sha256, [0, 0, 0, 1])
    calls_before = len(fake_recognition.calls)
    body = _enrol(client, auth_headers, admin, case.id).json()
    assert (body["generated"], body["pending"]) == (3, 0)
    assert len(fake_recognition.calls) == calls_before + 1


def test_enrolment_logs_ids_and_counts_only(client, auth_headers, admin, approved_case, fake_recognition, caplog):
    case, photos = approved_case
    _register(fake_recognition, photos, [0.123456, 0.5, 0, 0], [0, 0.654321, 0, 1])
    with caplog.at_level(logging.INFO):
        assert _enrol(client, auth_headers, admin, case.id).status_code == 200
    record = next(r for r in caplog.records if r.getMessage() == "embeddings_generated")
    assert (record.case_id, record.admin_id) == (str(case.id), str(admin.id))
    assert (record.generated, record.pending, record.failed) == (2, 0, 0)
    for secret in ("Fictional Enrolment Case", "reporter@example.com", "0.123456", "0.654321", "Sample"):
        assert secret not in caplog.text, secret
        assert all(secret not in str(r.__dict__) for r in caplog.records), secret


# ------------------------------------------------------------------ the step after an approval
@pytest.fixture()
def pending_case(make_case, make_photos, reporter):
    case = make_case(reporter, name="Fictional Approval Case")
    return case, make_photos(reporter, case, 2)


def _approve(client, auth_headers, admin, case_id):
    return client.post(f"/admin/cases/{case_id}/approve", json={}, headers=auth_headers(admin))


def test_approving_without_a_model_keeps_the_case_approved_with_pending_photos(
    client, db, auth_headers, admin, pending_case, caplog
):
    case, _ = pending_case
    with caplog.at_level(logging.INFO):
        response = _approve(client, auth_headers, admin, case.id)                # the real, unconfigured service
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["embeddings"] == {"generated": 0, "pending": 2, "failed": 0}
    assert _rows(db, case.id) == []
    assert set(_photo_states(db, case.id).values()) == {("pending", None)}
    warning = next(r for r in caplog.records if r.getMessage() == "embeddings_pending_after_approval")
    assert warning.levelno == logging.WARNING and warning.case_id == str(case.id)


def test_approving_with_a_model_generates_the_embeddings_after_the_approval(
    client, db, auth_headers, admin, pending_case, fake_recognition
):
    case, photos = pending_case
    _register(fake_recognition, photos, *ONE_FACE)
    response = _approve(client, auth_headers, admin, case.id)
    assert response.status_code == 200
    assert response.json()["embeddings"] == {"generated": 2, "pending": 0, "failed": 0}
    assert response.json()["history"][-1]["to_status"] == "approved"
    assert len(_rows(db, case.id)) == 2
    assert set(_photo_states(db, case.id).values()) == {("done", None)}


def test_a_photo_without_a_face_shows_up_as_failed_in_the_approval_summary(
    client, db, auth_headers, admin, pending_case, fake_recognition
):
    case, photos = pending_case
    fake_recognition.set_faces(photos[0].sha256, [1, 0, 0, 0])                   # photos[1] has no face
    body = _approve(client, auth_headers, admin, case.id).json()
    assert body["status"] == "approved"
    assert body["embeddings"] == {"generated": 1, "pending": 0, "failed": 1}


def test_an_approval_is_never_undone_by_the_recognition_step(
    client, db, auth_headers, admin, make_case, make_photos, reporter, fake_recognition, caplog
):
    for index, (attribute, expected_reason) in enumerate(
        (("fail_on_call", "recognition_unavailable"), ("crash_on_call", "RuntimeError"))
    ):
        case = make_case(reporter)
        photos = make_photos(reporter, case, 2, first_seed=70 + index * 10)
        _register(fake_recognition, photos, *ONE_FACE)
        setattr(fake_recognition, attribute, len(fake_recognition.calls) + 1)    # the very next call breaks
        caplog.clear()
        with caplog.at_level(logging.INFO):
            response = _approve(client, auth_headers, admin, case.id)
        assert response.status_code == 200, attribute
        assert response.json()["status"] == "approved"
        assert response.json()["embeddings"] == {"generated": 0, "pending": 2, "failed": 0}
        assert _status_now(db, case) == "approved"
        assert _rows(db, case.id) == []
        warning = next(r for r in caplog.records if r.getMessage() == "embeddings_pending_after_approval")
        assert (warning.case_id, warning.reason) == (str(case.id), expected_reason)


def test_only_an_approval_reports_embeddings(client, auth_headers, admin, reporter, make_case, make_photos, fake_recognition):
    rejected = make_case(reporter)
    reject = client.post(
        f"/admin/cases/{rejected.id}/reject", json={"reason": "Fictional rejection"}, headers=auth_headers(admin)
    )
    assert reject.status_code == 200 and "embeddings" not in reject.json()

    approved = make_case(reporter)
    make_photos(reporter, approved, 1, first_seed=90)
    assert "embeddings" in _approve(client, auth_headers, admin, approved.id).json()
    detail = client.get(f"/admin/cases/{approved.id}", headers=auth_headers(admin))
    assert "embeddings" not in detail.json()
    found = client.patch(f"/admin/cases/{approved.id}/status", json={"status": "found"}, headers=auth_headers(admin))
    assert found.status_code == 200 and "embeddings" not in found.json()
