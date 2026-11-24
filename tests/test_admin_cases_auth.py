"""Every admin case endpoint: no token -> 401, a normal user -> 403, an admin -> success."""
import pytest
from sqlalchemy import func, select

from app.models.case import MissingCase
from app.models.case_status_event import CaseStatusEvent

REASON = "Fictional review note"

# name -> (method, path template, JSON body, status the case must be in first)
ENDPOINTS = {
    "queue": ("GET", "/admin/cases", None, "pending"),
    "detail": ("GET", "/admin/cases/{case}", None, "pending"),
    "photo": ("GET", "/admin/cases/{case}/photos/{photo}", None, "pending"),
    "approve": ("POST", "/admin/cases/{case}/approve", {"reason": REASON}, "pending"),
    "reject": ("POST", "/admin/cases/{case}/reject", {"reason": REASON}, "pending"),
    "status": ("PATCH", "/admin/cases/{case}/status", {"status": "found"}, "approved"),
}


@pytest.fixture()
def reporter(make_user):
    return make_user(email="reporter@example.com")


@pytest.fixture()
def case_with_photo(make_case, make_photos, reporter):
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    return case, photo


def _call(client, name, case, photo, headers=None):
    method, path, body, _ = ENDPOINTS[name]
    return client.request(method, path.format(case=case.id, photo=photo.id), json=body, headers=headers or {})


def _state(db, case):
    db.expire_all()
    stored = db.get(MissingCase, case.id)
    events = db.scalar(select(func.count()).select_from(CaseStatusEvent))
    return stored.status, stored.reviewed_by_id, events


@pytest.mark.parametrize("name", ENDPOINTS)
def test_the_endpoint_requires_a_token_a_normal_user_is_refused_and_an_admin_succeeds(
    name, client, db, make_user, auth_headers, reporter, case_with_photo
):
    case, photo = case_with_photo
    case.status = ENDPOINTS[name][3]
    db.commit()
    before = _state(db, case)

    missing = _call(client, name, case, photo)
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    garbage = _call(client, name, case, photo, {"Authorization": "Bearer not-a-real-token"})
    assert garbage.status_code == 401

    stranger = make_user(email="stranger@example.com")
    for user in (stranger, reporter):                                    # the case's own reporter is a normal user too
        refused = _call(client, name, case, photo, auth_headers(user))
        assert refused.status_code == 403, user.email
        assert refused.json()["error"]["message"] == "Administrator access required"
    assert _state(db, case) == before                                    # the refused calls changed nothing

    admin = make_user(email="boss@example.com", role="admin")
    assert _call(client, name, case, photo, auth_headers(admin)).status_code == 200


@pytest.mark.parametrize("name", ENDPOINTS)
def test_a_deactivated_admin_is_refused_with_401(name, client, db, make_user, auth_headers, case_with_photo):
    case, photo = case_with_photo
    case.status = ENDPOINTS[name][3]
    db.commit()
    admin = make_user(email="boss@example.com", role="admin")
    headers = auth_headers(admin)
    admin.is_active = False
    db.commit()
    before = _state(db, case)
    assert _call(client, name, case, photo, headers).status_code == 401
    assert _state(db, case) == before


@pytest.mark.parametrize("name", ENDPOINTS)
def test_a_user_demoted_after_receiving_a_token_is_refused(name, client, db, make_user, auth_headers, case_with_photo):
    """The role is read from the database on every request, never from the token."""
    case, photo = case_with_photo
    case.status = ENDPOINTS[name][3]
    db.commit()
    admin = make_user(email="boss@example.com", role="admin")
    headers = auth_headers(admin)                                        # token issued while the user was an admin
    admin.role = "user"                                                  # demoted afterwards
    db.commit()
    before = _state(db, case)
    assert _call(client, name, case, photo, headers).status_code == 403
    assert _state(db, case) == before
