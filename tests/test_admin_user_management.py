from app.models.audit_event import AuditEvent
from app.models.user import User
from app.services import users as user_service
import pytest
from sqlalchemy import select


def _patch(client, admin, headers, user_id, **payload):
    return client.patch(f"/admin/users/{user_id}", json=payload, headers=headers(admin))


def test_admin_can_promote_and_deactivate_users_and_existing_tokens_stop_working(
    client, db, make_user, auth_headers
):
    admin = make_user(email="root@example.test", role="admin")
    target = make_user(email="promote@example.test")
    target_headers = auth_headers(target)

    promoted = _patch(client, admin, auth_headers, target.id, role="admin")
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["role"] == "admin"

    # Keep another active admin so deactivation is permitted.
    deactivated = _patch(client, admin, auth_headers, target.id, is_active=False)
    assert deactivated.status_code == 200, deactivated.text
    assert deactivated.json()["is_active"] is False
    assert client.get("/auth/me", headers=target_headers).status_code == 401

    actions = list(db.scalars(select(AuditEvent.action).where(AuditEvent.target_id == target.id)))
    assert sorted(actions) == ["user_deactivated", "user_role_changed"]


def test_admin_cannot_change_own_role_or_deactivate_self(client, make_user, auth_headers):
    admin = make_user(email="self@example.test", role="admin")
    headers = auth_headers(admin)
    assert _patch(client, admin, auth_headers, admin.id, role="user").status_code == 409
    assert _patch(client, admin, auth_headers, admin.id, is_active=False).status_code == 409
    assert client.get("/admin/users", headers=headers).status_code == 200


def test_last_active_admin_cannot_be_demoted_or_deactivated(client, make_user, auth_headers):
    admin = make_user(email="last-admin@example.test", role="admin")
    headers = auth_headers(admin)
    assert _patch(client, admin, auth_headers, admin.id, role="user").status_code == 409
    # A different admin cannot remove the last active admin either; this setup always retains one admin.
    second = make_user(email="second-admin@example.test", role="admin")
    # Deactivate first, leaving second; second is now the last one.
    assert _patch(client, second, auth_headers, admin.id, is_active=False).status_code == 200
    assert _patch(client, second, auth_headers, second.id, is_active=False).status_code == 409


def test_admin_user_patch_authorization_and_unknown_user(client, make_user, auth_headers):
    normal = make_user(email="normal@example.test")
    admin = make_user(email="admin@example.test", role="admin")
    assert client.patch(f"/admin/users/{normal.id}", json={"is_active": False}).status_code == 401
    assert client.patch(
        f"/admin/users/{normal.id}", json={"is_active": False}, headers=auth_headers(normal)
    ).status_code == 403
    assert client.patch(
        "/admin/users/00000000-0000-0000-0000-000000000000", json={"is_active": False},
        headers=auth_headers(admin),
    ).status_code == 404


def test_user_patch_requires_at_least_one_known_field(client, make_user, auth_headers):
    admin = make_user(email="admin@example.test", role="admin")
    headers = auth_headers(admin)
    assert client.patch("/admin/users/00000000-0000-0000-0000-000000000000", json={}, headers=headers).status_code == 422
    assert client.patch(
        "/admin/users/00000000-0000-0000-0000-000000000000", json={"email": "x@example.test"}, headers=headers
    ).status_code == 422


def test_service_rejects_removing_the_last_active_admin(db, make_user):
    actor = make_user(email="operator@example.test")
    last_admin = make_user(email="sole-admin@example.test", role="admin")
    with pytest.raises(user_service.UserManagementConflictError):
        user_service.update_user(
            db, actor=actor, user_id=last_admin.id, role="user", is_active=None
        )
    db.expire_all()
    assert db.get(User, last_admin.id).role == "admin"
    assert db.get(User, last_admin.id).is_active is True
