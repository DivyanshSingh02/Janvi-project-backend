from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.models.audit_event import AuditEvent
from app.models.face_search_log import FaceSearchLog
from app.services import audit


def test_audit_events_pagination_and_filters(client, db, make_user, auth_headers):
    admin = make_user(email="admin@example.test", role="admin")
    other = make_user(email="other-admin@example.test", role="admin")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, (actor, action) in enumerate([
        (admin, "user_role_changed"), (other, "user_deactivated"), (admin, "case_review_approved")
    ]):
        db.add(AuditEvent(
            actor_id=actor.id, action=action, target_type="user" if action.startswith("user_") else "case",
            target_id=None, details={"record_count": index}, created_at=base + timedelta(days=index),
        ))
    db.commit()
    headers = auth_headers(admin)
    first = client.get("/admin/audit-events?limit=2&offset=0", headers=headers).json()
    second = client.get("/admin/audit-events?limit=2&offset=2", headers=headers).json()
    assert first["total"] == 3 and len(first["items"]) == 2
    assert len(second["items"]) == 1 and second["offset"] == 2
    filtered = client.get(
        f"/admin/audit-events?action=user_role_changed&actor_id={admin.id}", headers=headers
    ).json()
    assert filtered["total"] == 1
    ranged = client.get(
        "/admin/audit-events?created_from=2026-01-02T00:00:00Z&created_to=2026-01-03T00:00:00Z",
        headers=headers,
    ).json()
    assert ranged["total"] == 2
    assert all("record_count" in item["details"] for item in first["items"])


def test_search_logs_pagination_and_filters(client, db, make_user, auth_headers):
    admin = make_user(email="admin@example.test", role="admin")
    other = make_user(email="other-admin@example.test", role="admin")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, actor in enumerate((admin, other, admin)):
        db.add(FaceSearchLog(
            admin_id=actor.id, result_count=index, top_case_id=None, top_similarity=None,
            model_name="fictional-model", model_version="1", searched_at=base + timedelta(days=index),
        ))
    db.commit()
    headers = auth_headers(admin)
    first = client.get("/admin/search-logs?limit=2", headers=headers).json()
    assert first["total"] == 3 and len(first["items"]) == 2
    filtered = client.get(f"/admin/search-logs?admin_id={other.id}", headers=headers).json()
    assert filtered["total"] == 1
    ranged = client.get(
        "/admin/search-logs?searched_from=2026-01-02T00:00:00Z&searched_to=2026-01-03T00:00:00Z",
        headers=headers,
    ).json()
    assert ranged["total"] == 2


def test_audit_and_search_log_endpoints_are_admin_only(client, make_user, auth_headers):
    normal = make_user(email="normal@example.test")
    admin = make_user(email="admin@example.test", role="admin")
    assert client.get("/admin/audit-events").status_code == 401
    assert client.get("/admin/search-logs").status_code == 401
    assert client.get("/admin/audit-events", headers=auth_headers(normal)).status_code == 403
    assert client.get("/admin/search-logs", headers=auth_headers(normal)).status_code == 403
    assert client.get("/admin/audit-events", headers=auth_headers(admin)).status_code == 200
    assert client.get("/admin/search-logs", headers=auth_headers(admin)).status_code == 200


def test_audit_helper_rejects_free_text_personal_data(db, make_user):
    admin = make_user(email="admin@example.test", role="admin")
    try:
        audit.record_event(
            db, actor_id=admin.id, action="case_review_approved", target_type="case", target_id=None,
            details={"person_name": "Fictional Person"},
        )
    except audit.InvalidAuditDetailsError:
        pass
    else:
        raise AssertionError("free-form audit details should be rejected")
    try:
        audit.record_event(
            db, actor_id=admin.id, action="case_review_approved", target_type="case", target_id=None,
            details={"operator": "alice"},
        )
    except audit.InvalidAuditDetailsError:
        pass
    else:
        raise AssertionError("arbitrary string values should not be accepted as audit codes")


def test_case_review_and_status_changes_add_audit_markers(client, db, make_user, make_case, make_photos, auth_headers):
    admin = make_user(email="admin@example.test", role="admin")
    owner = make_user(email="owner@example.test")
    case = make_case(owner)
    make_photos(owner, case, 1)
    headers = auth_headers(admin)
    approved = client.post(f"/admin/cases/{case.id}/approve", headers=headers)
    assert approved.status_code == 200, approved.text
    found = client.patch(
        f"/admin/cases/{case.id}/status", headers=headers, json={"status": "found"}
    )
    assert found.status_code == 200, found.text
    actions = list(db.scalars(select(AuditEvent.action).where(AuditEvent.target_id == case.id)))
    assert sorted(actions) == ["case_review_approved", "case_status_changed"]
