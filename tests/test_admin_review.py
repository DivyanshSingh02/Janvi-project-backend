"""Approving, rejecting and changing the status of a case."""
import itertools
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models.case import CaseStatus, MissingCase
from app.models.case_status_event import CaseStatusEvent
from app.models.user import User
from app.recognition import UnconfiguredRecognitionService
from app.services import cases as case_service
from app.services import embeddings as embedding_service
from app.services import photos as photo_service
from app.services import review as review_service

ALL = [status.value for status in CaseStatus]
# Written out independently of app/services/review.py.
ALLOWED = {
    ("pending", "approved"),
    ("pending", "rejected"),
    ("approved", "found"),
    ("approved", "closed"),
    ("found", "closed"),
    ("rejected", "closed"),
}
PAIRS = list(itertools.product(ALL, ALL))
REASON = "Fictional review note"


@pytest.fixture()
def reporter(make_user):
    return make_user(email="reporter@example.com", full_name="Fictional Reporter")


@pytest.fixture()
def admin(make_user):
    return make_user(email="boss@example.com", full_name="Fictional Admin", role="admin")


@pytest.fixture()
def case_with_photo(make_case, make_photos, reporter):
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    return case, photo


@pytest.fixture()
def case(case_with_photo):
    return case_with_photo[0]


def _force_status(db, case, status: str) -> None:
    case.status = status
    if status == "rejected":
        case.review_reason = "Fictional earlier rejection"
    db.commit()


def _events(db, case_id) -> list[CaseStatusEvent]:
    db.expire_all()
    return list(
        db.scalars(
            select(CaseStatusEvent).where(CaseStatusEvent.case_id == case_id).order_by(CaseStatusEvent.created_at)
        )
    )


def _stored(db, case_id) -> MissingCase:
    db.expire_all()
    return db.get(MissingCase, case_id)


def _approve(client, headers, case_id, **body):
    return client.post(f"/admin/cases/{case_id}/approve", json=body, headers=headers)


def _reject(client, headers, case_id, **body):
    return client.post(f"/admin/cases/{case_id}/reject", json=body, headers=headers)


def _patch(client, headers, case_id, **body):
    return client.patch(f"/admin/cases/{case_id}/status", json=body, headers=headers)


def _request_change(client, headers, case_id, to_status: str, reason: str | None = REASON):
    """Use the endpoint that is meant for this target status."""
    if to_status == "approved":
        return _approve(client, headers, case_id, reason=reason)
    if to_status == "rejected":
        return _reject(client, headers, case_id, reason=reason)
    return _patch(client, headers, case_id, status=to_status, reason=reason)


def _assert_untouched(db, case_id, status: str = "pending") -> None:
    assert _stored(db, case_id).status == status
    assert _events(db, case_id) == []


# ------------------------------------------------------------------ the transition matrix
@pytest.mark.parametrize("from_status, to_status", PAIRS, ids=[f"{f}->{t}" for f, t in PAIRS])
def test_the_full_transition_matrix(client, db, auth_headers, admin, case, from_status, to_status):
    _force_status(db, case, from_status)
    response = _request_change(client, auth_headers(admin), case.id, to_status)
    stored = _stored(db, case.id)
    events = _events(db, case.id)
    if (from_status, to_status) in ALLOWED:
        assert response.status_code == 200, response.text
        assert response.json()["status"] == to_status
        assert stored.status == to_status
        assert [(e.from_status, e.to_status, e.changed_by_id) for e in events] == [(from_status, to_status, admin.id)]
    else:
        assert response.status_code == 409, response.text
        assert stored.status == from_status                               # nothing changed ...
        assert events == []                                               # ... and nothing was recorded


def test_closed_is_terminal(client, db, auth_headers, admin, case):
    _force_status(db, case, "closed")
    headers = auth_headers(admin)
    for target in ALL:
        assert _request_change(client, headers, case.id, target).status_code == 409, target
    assert _stored(db, case.id).status == "closed"
    assert _events(db, case.id) == []


def test_a_refused_transition_names_the_two_statuses_and_nothing_else(client, db, auth_headers, admin, case):
    _force_status(db, case, "found")
    response = _patch(client, auth_headers(admin), case.id, status="found", reason=REASON)
    assert response.status_code == 409
    assert response.json()["error"]["message"] == "A case that is found cannot be changed to found"
    assert REASON not in response.text


def test_the_whole_life_of_a_case(client, db, auth_headers, make_user, admin, case):
    second_admin = make_user(email="second.boss@example.com", role="admin")
    first, second = auth_headers(admin), auth_headers(second_admin)
    assert _approve(client, first, case.id).status_code == 200
    assert _patch(client, second, case.id, status="found").status_code == 200
    closed = _patch(client, first, case.id, status="closed", reason="Fictional closing note")
    assert closed.status_code == 200 and closed.json()["status"] == "closed"
    assert _patch(client, first, case.id, status="found").status_code == 409


# ------------------------------------------------------------------ reasons
def test_rejecting_needs_a_reason_of_5_to_1000_characters_after_trimming(client, db, auth_headers, admin, case):
    headers = auth_headers(admin)
    url = f"/admin/cases/{case.id}/reject"
    for body in (None, {}, {"reason": None}, {"reason": ""}, {"reason": "     "}, {"reason": "abcd"},
                 {"reason": "  abcd  "}, {"reason": "a" * 1001}, {"reason": "bad\x00reason"}, {"reason": 12345}):
        response = client.post(url, json=body, headers=headers)
        assert response.status_code == 422, body
    _assert_untouched(db, case.id)

    assert _reject(client, headers, case.id, reason="  abcde  ").status_code == 200       # 5 after trimming
    assert _stored(db, case.id).review_reason == "abcde"                                  # stored trimmed


def test_a_reason_of_exactly_1000_characters_is_accepted(client, db, auth_headers, admin, case):
    response = _reject(client, auth_headers(admin), case.id, reason="a" * 1000)
    assert response.status_code == 200
    assert _stored(db, case.id).review_reason == "a" * 1000


def test_closing_needs_a_reason(client, db, auth_headers, admin, case):
    _force_status(db, case, "approved")
    headers = auth_headers(admin)
    for body in ({"status": "closed"}, {"status": "closed", "reason": None}, {"status": "closed", "reason": ""},
                 {"status": "closed", "reason": "   "}, {"status": "closed", "reason": "abcd"},
                 {"status": "closed", "reason": "a" * 1001}):
        assert client.patch(f"/admin/cases/{case.id}/status", json=body, headers=headers).status_code == 422, body
    assert _stored(db, case.id).status == "approved"
    assert _events(db, case.id) == []
    assert _patch(client, headers, case.id, status="closed", reason="abcde").status_code == 200


def test_approving_without_a_reason_is_fine_in_every_form(client, db, auth_headers, admin, make_case, make_photos, reporter):
    headers = auth_headers(admin)
    bodies = [None, {}, {"reason": None}, {"reason": ""}, {"reason": "   "}]
    for index, body in enumerate(bodies):
        case = make_case(reporter)
        make_photos(reporter, case, 1, first_seed=100 + index)
        response = client.post(f"/admin/cases/{case.id}/approve", json=body, headers=headers) if body is not None \
            else client.post(f"/admin/cases/{case.id}/approve", headers=headers)               # no body at all
        assert response.status_code == 200, body
        assert response.json()["review_reason"] is None
        assert _events(db, case.id)[0].reason is None


def test_approving_and_found_accept_a_reason_but_apply_the_same_length_rules(client, db, auth_headers, admin, case):
    headers = auth_headers(admin)
    for too_short_or_long in ("ab", "a" * 1001):
        assert _approve(client, headers, case.id, reason=too_short_or_long).status_code == 422
    _assert_untouched(db, case.id)
    assert _approve(client, headers, case.id, reason="  Photos look clear  ").status_code == 200
    assert _stored(db, case.id).review_reason == "Photos look clear"

    for too_short_or_long in ("ab", "a" * 1001):
        assert _patch(client, headers, case.id, status="found", reason=too_short_or_long).status_code == 422
    assert _stored(db, case.id).status == "approved"
    assert _patch(client, headers, case.id, status="found").status_code == 200            # a reason is optional here
    assert [e.to_status for e in _events(db, case.id)] == ["approved", "found"]


def test_unknown_fields_in_the_body_are_rejected(client, db, auth_headers, admin, case):
    headers = auth_headers(admin)
    assert _approve(client, headers, case.id, reviewed_by_id=str(uuid.uuid4())).status_code == 422
    assert _reject(client, headers, case.id, reason=REASON, status="approved").status_code == 422
    _force_status(db, case, "approved")
    assert _patch(client, headers, case.id, status="found", reviewed_at="2020-01-01T00:00:00Z").status_code == 422
    assert _stored(db, case.id).status == "approved"


def test_approved_and_rejected_are_not_accepted_by_the_patch_endpoint(client, db, auth_headers, admin, case):
    headers = auth_headers(admin)
    for target in ("approved", "rejected"):
        response = _patch(client, headers, case.id, status=target, reason=REASON)
        assert response.status_code == 422
        assert "use the approve/reject endpoint" in response.text
    _assert_untouched(db, case.id)


def test_an_unknown_or_missing_status_is_rejected_by_the_patch_endpoint(client, db, auth_headers, admin, case):
    headers = auth_headers(admin)
    for body in ({"status": "archived"}, {"status": None}, {"reason": REASON}, {}):
        assert client.patch(f"/admin/cases/{case.id}/status", json=body, headers=headers).status_code == 422, body
    _assert_untouched(db, case.id)


def test_patching_to_pending_is_a_409_not_a_422(client, db, auth_headers, admin, case):
    """pending is a real status, just not a reachable one."""
    response = _patch(client, auth_headers(admin), case.id, status="pending")
    assert response.status_code == 409


def test_validation_errors_do_not_echo_the_reason_back(client, auth_headers, admin, case):
    response = _reject(client, auth_headers(admin), case.id, reason="Zyxw")             # 4 characters: too short
    assert response.status_code == 422
    assert "Zyxw" not in response.text


def test_unknown_cases_are_404_and_malformed_ids_are_422(client, auth_headers, admin):
    headers = auth_headers(admin)
    unknown = uuid.uuid4()
    assert _approve(client, headers, unknown, reason=REASON).status_code == 404
    assert _reject(client, headers, unknown, reason=REASON).status_code == 404
    assert _patch(client, headers, unknown, status="closed", reason=REASON).status_code == 404
    for path in ("approve", "reject", "status"):
        method = client.patch if path == "status" else client.post
        assert method(f"/admin/cases/not-a-uuid/{path}", json={"reason": REASON, "status": "closed"},
                      headers=headers).status_code == 422


# ------------------------------------------------------------------ preconditions
def test_a_case_without_photos_cannot_be_approved(client, db, auth_headers, admin, reporter, make_case):
    case = make_case(reporter)
    response = _approve(client, auth_headers(admin), case.id, reason=REASON)
    assert response.status_code == 409
    assert "photo" in response.json()["error"]["message"]
    _assert_untouched(db, case.id)


def test_a_case_without_photos_can_still_be_rejected(client, db, auth_headers, admin, reporter, make_case):
    case = make_case(reporter)
    assert _reject(client, auth_headers(admin), case.id, reason=REASON).status_code == 200


def test_the_photo_check_counts_the_photos_at_the_moment_of_approval(
    client, db, auth_headers, admin, reporter, make_case, make_photos, storage
):
    case = make_case(reporter)
    photo = make_photos(reporter, case, 1)[0]
    photo_service.delete_photo(db, storage, reporter, case.id, photo.id)               # the owner removes the last photo
    assert _approve(client, auth_headers(admin), case.id).status_code == 409
    make_photos(reporter, case, 1, first_seed=50)                                       # ... and adds another one
    assert _approve(client, auth_headers(admin), case.id).status_code == 200


def test_approving_twice_is_a_409_and_records_one_event(client, db, auth_headers, admin, case):
    headers = auth_headers(admin)
    assert _approve(client, headers, case.id, reason=REASON).status_code == 200
    again = _approve(client, headers, case.id, reason=REASON)
    assert again.status_code == 409
    assert len(_events(db, case.id)) == 1


def test_rejecting_an_approved_case_and_approving_a_rejected_one_are_409(client, db, auth_headers, admin, case):
    headers = auth_headers(admin)
    _force_status(db, case, "approved")
    assert _reject(client, headers, case.id, reason=REASON).status_code == 409
    _force_status(db, case, "rejected")
    assert _approve(client, headers, case.id).status_code == 409


# ------------------------------------------------------------------ history and review fields
def test_approving_records_the_event_and_the_review_fields_together(client, db, auth_headers, admin, case):
    before = datetime.now().astimezone()
    response = _approve(client, auth_headers(admin), case.id, reason="Photos are clear")
    body = response.json()
    assert response.status_code == 200
    assert (body["status"], body["review_reason"], body["reviewed_by_id"]) == ("approved", "Photos are clear", str(admin.id))
    assert body["reviewed_at"] is not None
    stored = _stored(db, case.id)
    assert stored.reviewed_by_id == admin.id
    assert stored.reviewed_at >= before
    (event,) = _events(db, case.id)
    assert (event.from_status, event.to_status, event.changed_by_id, event.reason) == (
        "pending", "approved", admin.id, "Photos are clear",
    )
    assert event.created_at == stored.reviewed_at                                       # one moment for both
    assert [h["to_status"] for h in body["history"]] == ["approved"]


def test_rejecting_records_the_reason_the_reviewer_and_the_time(client, db, auth_headers, admin, case):
    response = _reject(client, auth_headers(admin), case.id, reason="Not enough information")
    body = response.json()
    assert response.status_code == 200
    assert (body["status"], body["review_reason"], body["reviewed_by_id"]) == (
        "rejected", "Not enough information", str(admin.id),
    )
    assert body["reviewed_at"] is not None
    (event,) = _events(db, case.id)
    assert (event.from_status, event.to_status, event.reason) == ("pending", "rejected", "Not enough information")


def test_the_history_is_oldest_first_with_the_right_admin_for_each_step(client, db, auth_headers, make_user, admin, case):
    other_admin = make_user(email="second.boss@example.com", role="admin")
    assert _approve(client, auth_headers(admin), case.id).status_code == 200
    assert _patch(client, auth_headers(other_admin), case.id, status="found", reason="Found safe").status_code == 200
    assert _patch(client, auth_headers(admin), case.id, status="closed", reason="Family confirmed").status_code == 200

    detail = client.get(f"/admin/cases/{case.id}", headers=auth_headers(admin)).json()
    history = detail["history"]
    assert [(h["from_status"], h["to_status"], h["changed_by_id"], h["reason"]) for h in history] == [
        ("pending", "approved", str(admin.id), None),
        ("approved", "found", str(other_admin.id), "Found safe"),
        ("found", "closed", str(admin.id), "Family confirmed"),
    ]
    stamps = [datetime.fromisoformat(h["created_at"]) for h in history]
    assert stamps == sorted(stamps) and len(set(stamps)) == 3                           # strictly in order
    assert set(history[0]) == {"id", "from_status", "to_status", "changed_by_id", "reason", "created_at"}


def test_only_approve_and_reject_set_the_review_fields(client, db, auth_headers, make_user, admin, case):
    other_admin = make_user(email="second.boss@example.com", role="admin")
    _approve(client, auth_headers(admin), case.id, reason="Photos are clear")
    approved = _stored(db, case.id)
    reviewed_at = approved.reviewed_at
    _patch(client, auth_headers(other_admin), case.id, status="found", reason="Found safe")
    _patch(client, auth_headers(other_admin), case.id, status="closed", reason="Family confirmed")
    closed = _stored(db, case.id)
    assert closed.reviewed_by_id == admin.id                    # still the admin who approved, not the one who closed
    assert closed.reviewed_at == reviewed_at
    assert closed.review_reason == "Photos are clear"           # found and closed keep their reasons in the history


def test_a_rejected_case_keeps_its_rejection_reason_when_it_is_closed(client, db, auth_headers, make_user, admin, case):
    other_admin = make_user(email="second.boss@example.com", role="admin")
    _reject(client, auth_headers(admin), case.id, reason="Duplicate report")
    assert _patch(client, auth_headers(other_admin), case.id, status="closed", reason="Archived").status_code == 200
    stored = _stored(db, case.id)
    assert (stored.status, stored.review_reason, stored.reviewed_by_id) == ("closed", "Duplicate report", admin.id)


def test_changing_the_status_leaves_the_rest_of_the_case_alone(client, db, auth_headers, admin, case):
    before = client.get(f"/admin/cases/{case.id}", headers=auth_headers(admin)).json()
    after = _approve(client, auth_headers(admin), case.id).json()
    for field in ("name", "age", "gender", "contact_phone", "contact_email", "city", "state", "address",
                  "last_seen_location", "last_seen_date", "created_at", "reporter", "photos"):
        assert after[field] == before[field], field


# ------------------------------------------------------------------ what the owner sees
def test_the_owner_sees_why_a_case_was_rejected_but_never_who_rejected_it(
    client, db, auth_headers, make_user, reporter, make_case
):
    admin = make_user(email="distinctive.admin@example.com", full_name="Distinctive Adminname", role="admin")
    case = make_case(reporter)
    assert _reject(client, auth_headers(admin), case.id, reason="Photo does not show a face").status_code == 200

    response = client.get(f"/cases/{case.id}", headers=auth_headers(reporter))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["review_reason"] == "Photo does not show a face"
    assert body["reviewed_at"] is not None
    assert "reviewed_by_id" not in body and "reviewed_by" not in body
    for identity in (str(admin.id), admin.email, "Distinctive", "Adminname"):
        assert identity not in response.text

    listing = client.get("/cases", headers=auth_headers(reporter))
    assert listing.json()["items"][0]["review_reason"] == "Photo does not show a face"
    for identity in (str(admin.id), admin.email, "Distinctive"):
        assert identity not in listing.text


def test_a_case_that_has_not_been_reviewed_shows_empty_review_fields(client, auth_headers, reporter, case):
    body = client.get(f"/cases/{case.id}", headers=auth_headers(reporter)).json()
    assert (body["review_reason"], body["reviewed_at"]) == (None, None)


def test_the_owner_cannot_use_the_review_endpoints_on_their_own_case(client, db, auth_headers, reporter, case):
    headers = auth_headers(reporter)
    assert _approve(client, headers, case.id).status_code == 403
    assert _reject(client, headers, case.id, reason=REASON).status_code == 403
    assert _patch(client, headers, case.id, status="closed", reason=REASON).status_code == 403
    _assert_untouched(db, case.id)


# ------------------------------------------------------------------ logging
def test_logs_contain_ids_and_statuses_but_never_the_reason_or_personal_data(
    client, db, auth_headers, make_user, make_case, make_photos, caplog
):
    reporter = make_user(email="zyxwv.reporter@example.com", full_name="Zyxwv Reportername")
    admin = make_user(email="distinctive.admin@example.com", full_name="Distinctive Adminname", role="admin")
    approved_case = make_case(
        reporter, name="Zyxwv Qponml", contact_phone="+15550123456", contact_email="zyxwv.contact@example.com",
        address="987 Distinctive Lane", last_seen_location="Distinctive Bazaar",
    )
    make_photos(reporter, approved_case, 1)
    rejected_case = make_case(reporter, name="Another Fictional")
    headers = auth_headers(admin)

    with caplog.at_level(logging.INFO):
        approved = _approve(client, headers, approved_case.id, reason="Reasonmarker approval text")
        refused = _approve(client, headers, approved_case.id, reason="Reasonmarker second attempt")
        rejected = _reject(client, headers, rejected_case.id, reason="Reasonmarker rejection text")
        invalid = _reject(client, headers, rejected_case.id, reason="Rea")             # 422: never reaches the service
    assert (approved.status_code, refused.status_code, rejected.status_code, invalid.status_code) == (200, 409, 200, 422)

    changed = [r for r in caplog.records if r.getMessage() == "case_status_changed"]
    assert [(r.case_id, r.admin_id, r.from_status, r.to_status) for r in changed] == [
        (str(approved_case.id), str(admin.id), "pending", "approved"),
        (str(rejected_case.id), str(admin.id), "pending", "rejected"),
    ]
    (turned_down,) = [r for r in caplog.records if r.getMessage() == "case_status_change_rejected"]
    assert (turned_down.case_id, turned_down.admin_id) == (str(approved_case.id), str(admin.id))
    assert (turned_down.from_status, turned_down.to_status, turned_down.error) == (
        "approved", "approved", "InvalidTransitionError",
    )

    secrets = ("Reasonmarker", "Zyxwv", "5550123456", "zyxwv.contact", "Distinctive", "Another Fictional",
               "reporter@example.com", "Adminname")
    for secret in secrets:
        assert secret not in caplog.text, secret
        assert all(secret not in str(record.__dict__) for record in caplog.records), secret
    for record in changed + [turned_down]:
        assert not hasattr(record, "reason")


# ------------------------------------------------------------------ the step after an approval (Stage 6)
def test_after_approval_reports_pending_photos_when_no_model_is_configured(db, case_with_photo, storage):
    case, _ = case_with_photo
    _force_status(db, case, "approved")
    summary = review_service.after_approval(
        db, case, recognition=UnconfiguredRecognitionService(), storage=storage
    )
    assert (summary.generated, summary.pending, summary.failed) == (0, 1, 0)


def test_after_approval_is_called_once_per_approval_and_only_for_approvals(
    client, db, auth_headers, admin, case, monkeypatch
):
    calls = []

    def spy(session, approved, **services):
        calls.append(approved.id)
        return embedding_service.EmbeddingSummary(generated=0, pending=1, failed=0)

    monkeypatch.setattr(review_service, "after_approval", spy)
    headers = auth_headers(admin)
    assert _approve(client, headers, case.id).status_code == 200
    assert calls == [case.id]
    assert _approve(client, headers, case.id).status_code == 409                       # a refused approval does not call it
    _patch(client, headers, case.id, status="found")
    _patch(client, headers, case.id, status="closed", reason=REASON)
    assert calls == [case.id]


def test_the_hook_runs_after_the_approval_has_been_committed(client, db, auth_headers, admin, case, monkeypatch):
    seen = []

    def spy(session, approved, **services):
        with SessionLocal() as other:                                                  # a second, independent connection
            stored = other.get(MissingCase, approved.id)
            events = other.scalar(select(func.count()).select_from(CaseStatusEvent))
            seen.append((stored.status, events))
        return embedding_service.EmbeddingSummary(generated=0, pending=1, failed=0)

    monkeypatch.setattr(review_service, "after_approval", spy)
    assert _approve(client, auth_headers(admin), case.id).status_code == 200
    assert seen == [("approved", 1)]                                                   # already visible to everybody else


def test_a_hook_that_blows_up_cannot_undo_the_approval(client, db, auth_headers, admin, case, monkeypatch):
    def failing_hook(session, approved, **services):
        raise RuntimeError("simulated failure of the recognition step")

    monkeypatch.setattr(review_service, "after_approval", failing_hook)
    response = _approve(client, auth_headers(admin), case.id)
    monkeypatch.undo()
    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert response.json()["embeddings"] == {"generated": 0, "pending": 1, "failed": 0}
    assert _stored(db, case.id).status == "approved"
    assert len(_events(db, case.id)) == 1
    assert _stored(db, case.id).reviewed_by_id == admin.id


# ------------------------------------------------------------------ the service on its own
def test_the_service_refuses_bad_input_before_touching_the_case(db, admin, case):
    with pytest.raises(review_service.ReasonRequiredError):
        review_service.change_status(db, admin, case.id, CaseStatus.REJECTED, None)
    with pytest.raises(review_service.InvalidReasonError):
        review_service.change_status(db, admin, case.id, CaseStatus.APPROVED, "abcd")
    with pytest.raises(review_service.InvalidTransitionError):
        review_service.change_status(db, admin, case.id, CaseStatus.FOUND, None)
    _assert_untouched(db, case.id)


def test_the_service_refuses_to_approve_a_case_without_photos(db, admin, reporter, make_case):
    case = make_case(reporter)
    with pytest.raises(review_service.CaseHasNoPhotosError):
        review_service.change_status(db, admin, case.id, CaseStatus.APPROVED, None)
    _assert_untouched(db, case.id)


def test_the_service_reports_an_unknown_case(db, admin):
    with pytest.raises(case_service.CaseNotFoundError):
        review_service.change_status(db, admin, uuid.uuid4(), CaseStatus.REJECTED, REASON)


def test_the_service_reports_the_change_it_made(db, admin, case):
    change = review_service.change_status(db, admin, case.id, CaseStatus.REJECTED, REASON)
    assert (change.case_id, change.from_status, change.to_status) == (case.id, CaseStatus.PENDING, CaseStatus.REJECTED)


# ------------------------------------------------------------------ two admins at once
def test_two_admins_approving_the_same_case_at_the_same_moment_only_one_wins(make_user, admin, case):
    second_admin = make_user(email="second.boss@example.com", role="admin")
    barrier = threading.Barrier(2)

    def attempt(admin_id) -> str:
        with SessionLocal() as session:                                               # each admin has their own session
            who = session.get(User, admin_id)
            barrier.wait(timeout=10)
            try:
                review_service.change_status(session, who, case.id, CaseStatus.APPROVED, None)
            except review_service.InvalidTransitionError:
                return "refused"
            return "approved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, [admin.id, second_admin.id]))
    assert sorted(outcomes) == ["approved", "refused"]
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(CaseStatusEvent)) == 1
        assert session.get(MissingCase, case.id).status == "approved"
