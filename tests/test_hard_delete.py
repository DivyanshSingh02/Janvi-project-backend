import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.models.audit_event import AuditEvent
from app.models.case import MissingCase
from app.models.case_status_event import CaseStatusEvent
from app.models.detection import Detection
from app.models.face_embedding import FaceEmbedding
from app.models.photo import CasePhoto
from app.models.face_search_log import FaceSearchLog
from app.recognition import vectors
from app.storage import Storage, StorageError


class DeleteFailingStorage(Storage):
    def __init__(self, wrapped, failing_key):
        self.wrapped = wrapped
        self.failing_key = failing_key

    def save(self, key, data, content_type):
        self.wrapped.save(key, data, content_type)

    def read(self, key):
        return self.wrapped.read(key)

    def exists(self, key):
        return self.wrapped.exists(key)

    def delete(self, key):
        if key == self.failing_key:
            raise StorageError("simulated storage failure")
        self.wrapped.delete(key)


def _delete(client, admin, headers, case_id, **params):
    return client.delete(f"/admin/cases/{case_id}", headers=headers(admin), params=params)


def _add_detection(db, case, admin):
    row = Detection(
        case_id=case.id, detected_at=datetime.now(timezone.utc),
        source="manual", location_text="Fictional place", confidence=0.7, recorded_by_id=admin.id,
    )
    db.add(row)
    db.commit()
    return row


def _add_embedding(db, photo, case):
    row = FaceEmbedding(
        photo_id=photo.id, case_id=case.id, model_name="fake-face-model", model_version="1", dimension=4,
        vector=vectors.pack([1, 0, 0, 0]),
    )
    db.add(row)
    db.commit()
    return row


def test_hard_delete_requires_confirmation_and_removes_rows_and_files(
    client, db, make_user, make_case, make_photos, storage, auth_headers, fake_recognition, make_image_bytes, final_sha
):
    admin = make_user(email="admin@example.test", role="admin")
    reporter = make_user(email="reporter@example.test")
    case = make_case(reporter, name="CANARY_PRIVATE_NAME", address="CANARY_PRIVATE_ADDRESS")
    photo = make_photos(reporter, case, 1)[0]
    _add_embedding(db, photo, case)
    _add_detection(db, case, admin)
    db.add(CaseStatusEvent(
        case_id=case.id, from_status="pending", to_status="approved", changed_by_id=admin.id,
        reason=None,
    ))
    case.status = "approved"
    db.commit()
    query = make_image_bytes("PNG", size=(220, 220), seed=88)
    fake_recognition.set_faces(final_sha(query), [1, 0, 0, 0])
    before_delete = client.post(
        "/search", headers=auth_headers(admin), files={"file": ("query.png", query, "image/png")}
    )
    assert before_delete.status_code == 200, before_delete.text
    assert [item["case"]["id"] for item in before_delete.json()["results"]] == [str(case.id)]
    key = photo.storage_key

    assert _delete(client, admin, auth_headers, case.id).status_code == 422
    assert _delete(client, admin, auth_headers, case.id, confirm="false").status_code == 422
    response = _delete(client, admin, auth_headers, case.id, confirm="true")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted"] is True
    assert body["photos_deleted"] == body["embeddings_deleted"] == 1
    assert body["detections_deleted"] == body["status_events_deleted"] == 1
    assert body["files_deleted"] == 1 and body["failed_photo_ids"] == []
    assert not storage.exists(key)
    db.expire_all()
    assert db.get(MissingCase, case.id) is None
    assert db.scalar(select(func.count()).select_from(CasePhoto).where(CasePhoto.case_id == case.id)) == 0
    assert db.scalar(select(func.count()).select_from(FaceEmbedding).where(FaceEmbedding.case_id == case.id)) == 0
    assert db.scalar(select(func.count()).select_from(Detection).where(Detection.case_id == case.id)) == 0
    assert db.scalar(select(func.count()).select_from(CaseStatusEvent).where(CaseStatusEvent.case_id == case.id)) == 0
    event = db.scalar(select(AuditEvent).where(AuditEvent.action == "case_hard_deleted"))
    assert event is not None and event.target_id == case.id
    assert event.details == {
        "cases_deleted": 1, "photos_deleted": 1, "embeddings_deleted": 1,
        "detections_deleted": 1, "status_events_deleted": 1,
    }
    assert "CANARY_PRIVATE" not in str(event.details)
    assert client.get(f"/cases/{case.id}", headers=auth_headers(reporter)).status_code == 404
    assert client.get(f"/admin/cases/{case.id}", headers=auth_headers(admin)).status_code == 404
    after_delete = client.post(
        "/search", headers=auth_headers(admin), files={"file": ("query.png", query, "image/png")}
    )
    assert after_delete.status_code == 200, after_delete.text
    assert after_delete.json()["results"] == []
    db.expire_all()
    previous_search = db.scalar(select(FaceSearchLog).order_by(FaceSearchLog.searched_at).limit(1))
    assert previous_search.top_case_id is None


def test_hard_delete_reports_and_logs_storage_failure_by_photo_id(
    client, db, make_user, make_case, make_photos, storage, auth_headers, override_storage, caplog
):
    admin = make_user(email="admin@example.test", role="admin")
    reporter = make_user(email="reporter@example.test")
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    override_storage(DeleteFailingStorage(storage, photo.storage_key))

    with caplog.at_level("ERROR"):
        response = _delete(client, admin, auth_headers, case.id, confirm="true")
    assert response.status_code == 200, response.text
    assert response.json()["failed_photo_ids"] == [str(photo.id)]
    failure_record = next(record for record in caplog.records if record.getMessage() == "hard_delete_photo_file_failed")
    assert failure_record.photo_id == str(photo.id)
    assert photo.storage_key not in str(failure_record.__dict__)
    db.expire_all()
    assert db.get(MissingCase, case.id) is None


def test_hard_delete_unknown_case_and_authorization(client, make_user, auth_headers):
    admin = make_user(email="admin@example.test", role="admin")
    normal = make_user(email="normal@example.test")
    missing = uuid.uuid4()
    assert client.delete(f"/admin/cases/{missing}?confirm=true").status_code == 401
    assert client.delete(
        f"/admin/cases/{missing}?confirm=true", headers=auth_headers(normal)
    ).status_code == 403
    assert client.delete(
        f"/admin/cases/{missing}?confirm=true", headers=auth_headers(admin)
    ).status_code == 404
