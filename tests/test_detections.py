import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.detection import Detection


@pytest.fixture()
def reporter(make_user):
    return make_user(email="reporter@example.com", full_name="Fictional Reporter")


@pytest.fixture()
def admin(make_user):
    return make_user(email="admin@example.com", full_name="Fictional Admin", role="admin")


@pytest.fixture()
def approved_case(db, reporter, make_case):
    case = make_case(reporter)
    case.status = "approved"
    db.commit()
    return case


@pytest.fixture()
def detection_payload():
    return {
        "detected_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        "source": "manual",
        "location_text": "Fictional Sample Plaza",
        "confidence": 0.82,
        "frame_reference": "manual:sample-frame-001",
    }


@pytest.fixture()
def make_detection(db, admin, approved_case):
    def create(case=None, *, detected_at=None, camera_id="camera-a", location_text="Fictional Sample Plaza",
               latitude=None, longitude=None, confidence=0.8, source="manual", frame_reference="manual:frame-001",
               review_status="unreviewed"):
        reviewed = review_status != "unreviewed"
        row = Detection(
            case_id=(case or approved_case).id,
            detected_at=detected_at or datetime.now(timezone.utc),
            source=source,
            camera_id=camera_id,
            location_text=location_text,
            latitude=latitude,
            longitude=longitude,
            confidence=confidence,
            frame_reference=frame_reference,
            review_status=review_status,
            reviewed_by_id=admin.id if reviewed else None,
            reviewed_at=datetime.now(timezone.utc) if reviewed else None,
            recorded_by_id=admin.id,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    return create


_ADMIN_ENDPOINTS = (
    ("POST", "/admin/cases/{case_id}/detections"),
    ("GET", "/admin/cases/{case_id}/detections"),
    ("GET", "/admin/detections"),
    ("GET", "/admin/detections/{detection_id}"),
    ("PATCH", "/admin/detections/{detection_id}/review"),
)


def _path(template, case_id, detection_id):
    return template.replace("{case_id}", str(case_id)).replace("{detection_id}", str(detection_id))


def _request(client, method, path, payload=None, headers=None):
    if method == "GET":
        return client.get(path, headers=headers)
    return client.request(method, path, headers=headers, json=payload)


@pytest.mark.parametrize("method,template", _ADMIN_ENDPOINTS)
def test_every_admin_endpoint_requires_authentication(
    client, approved_case, make_detection, detection_payload, method, template
):
    detection = make_detection()
    path = _path(template, approved_case.id, detection.id)
    payload = detection_payload if method == "POST" else ({"review_status": "confirmed"} if method == "PATCH" else None)
    assert _request(client, method, path, payload).status_code == 401


@pytest.mark.parametrize("method,template", _ADMIN_ENDPOINTS)
def test_every_admin_endpoint_rejects_non_admin_user(
    client, reporter, auth_headers, approved_case, make_detection, detection_payload, method, template
):
    detection = make_detection()
    path = _path(template, approved_case.id, detection.id)
    payload = detection_payload if method == "POST" else ({"review_status": "confirmed"} if method == "PATCH" else None)
    assert _request(client, method, path, payload, auth_headers(reporter)).status_code == 403


def test_admin_can_record_detection(client, admin, auth_headers, approved_case, detection_payload):
    response = client.post(
        f"/admin/cases/{approved_case.id}/detections", json=detection_payload, headers=auth_headers(admin)
    )
    assert response.status_code == 201, response.text
    assert response.json()["case_id"] == str(approved_case.id)
    assert response.json()["recorded_by_id"] == str(admin.id)
    assert response.json()["review_status"] == "unreviewed"


@pytest.mark.parametrize("case_status", ["pending", "rejected", "found", "closed"])
def test_only_approved_cases_accept_detections(
    client, db, admin, auth_headers, reporter, make_case, detection_payload, case_status
):
    case = make_case(reporter)
    case.status = case_status
    if case_status == "rejected":
        case.review_reason = "Fictional rejection reason"
    db.commit()
    response = client.post(
        f"/admin/cases/{case.id}/detections", json=detection_payload, headers=auth_headers(admin)
    )
    assert response.status_code == 409


def test_unknown_case_returns_404(client, admin, auth_headers, detection_payload):
    response = client.post(
        f"/admin/cases/{uuid.uuid4()}/detections", json=detection_payload, headers=auth_headers(admin)
    )
    assert response.status_code == 404


@pytest.mark.parametrize("invalid", [
    "naive_datetime", "future_datetime", "old_datetime", "confidence_low", "confidence_high",
    "confidence_nan", "confidence_inf", "latitude_range", "longitude_range", "latitude_only",
    "longitude_only", "no_location", "camera_without_id", "frame_parent", "frame_absolute",
    "frame_windows_absolute", "frame_backslash", "frame_spaces", "frame_too_long", "camera_too_long",
    "location_control", "unknown_field",
])
def test_detection_create_validation(invalid, client, admin, auth_headers, approved_case, detection_payload):
    body = dict(detection_payload)
    if invalid == "naive_datetime":
        body["detected_at"] = "2026-11-03T12:00:00"
    elif invalid == "future_datetime":
        body["detected_at"] = (datetime.now(timezone.utc) + timedelta(minutes=6)).isoformat()
    elif invalid == "old_datetime":
        body["detected_at"] = "1999-12-31T23:59:59+00:00"
    elif invalid == "confidence_low":
        body["confidence"] = -0.1
    elif invalid == "confidence_high":
        body["confidence"] = 1.1
    elif invalid == "confidence_nan":
        body["confidence"] = float("nan")
    elif invalid == "confidence_inf":
        body["confidence"] = float("inf")
    elif invalid == "latitude_range":
        body.update(latitude=91, longitude=10)
    elif invalid == "longitude_range":
        body.update(latitude=10, longitude=181)
    elif invalid == "latitude_only":
        body["latitude"] = 10
    elif invalid == "longitude_only":
        body["longitude"] = 10
    elif invalid == "no_location":
        body.pop("location_text")
    elif invalid == "camera_without_id":
        body["source"] = "camera"
    elif invalid == "frame_parent":
        body["frame_reference"] = "frames/../private.jpg"
    elif invalid == "frame_absolute":
        body["frame_reference"] = "/tmp/frame.jpg"
    elif invalid == "frame_windows_absolute":
        body["frame_reference"] = "C:/frames/frame.jpg"
    elif invalid == "frame_backslash":
        body["frame_reference"] = "frames\\frame.jpg"
    elif invalid == "frame_spaces":
        body["frame_reference"] = "frame 01.jpg"
    elif invalid == "frame_too_long":
        body["frame_reference"] = "a" * 256
    elif invalid == "camera_too_long":
        body["camera_id"] = "c" * 65
    elif invalid == "location_control":
        body["location_text"] = "Bad\nLocation"
    elif invalid == "unknown_field":
        body["surprise"] = "not allowed"

    headers = auth_headers(admin)
    headers["Content-Type"] = "application/json"
    response = client.post(
        f"/admin/cases/{approved_case.id}/detections",
        headers=headers,
        content=json.dumps(body, allow_nan=True),
    )
    assert response.status_code == 422, (invalid, response.text)


def test_detection_text_is_trimmed(client, admin, auth_headers, approved_case, detection_payload):
    detection_payload["location_text"] = "  Fictional Plaza  "
    detection_payload["camera_id"] = "  cam-01  "
    response = client.post(
        f"/admin/cases/{approved_case.id}/detections", json=detection_payload, headers=auth_headers(admin)
    )
    assert response.status_code == 201
    assert response.json()["location_text"] == "Fictional Plaza"
    assert response.json()["camera_id"] == "cam-01"


def test_case_history_is_newest_first_and_paginated(client, admin, auth_headers, approved_case, make_detection):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(5):
        make_detection(detected_at=base + timedelta(days=day))
    url = f"/admin/cases/{approved_case.id}/detections"
    first = client.get(url + "?limit=2&offset=0", headers=auth_headers(admin)).json()
    second = client.get(url + "?limit=2&offset=2", headers=auth_headers(admin)).json()
    assert len(first["items"]) == len(second["items"]) == 2
    assert first["total"] == second["total"] == 5
    assert first["items"][0]["detected_at"] > first["items"][1]["detected_at"]


def test_all_admin_history_filters(client, admin, auth_headers, approved_case, make_detection):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = make_detection(detected_at=start, camera_id="camera-a", confidence=0.4)
    b = make_detection(detected_at=start + timedelta(days=1), camera_id="camera-b", confidence=0.8,
                       review_status="confirmed")
    c = make_detection(detected_at=start + timedelta(days=2), camera_id="camera-a", confidence=0.9,
                       review_status="dismissed")
    base = f"/admin/cases/{approved_case.id}/detections"
    headers = auth_headers(admin)
    def ids(query):
        response = client.get(f"{base}?{query}", headers=headers)
        assert response.status_code == 200, response.text
        return {item["id"] for item in response.json()["items"]}
    assert ids("camera_id=camera-a") == {str(a.id), str(c.id)}
    assert ids("review_status=confirmed") == {str(b.id)}
    assert ids("min_confidence=0.8") == {str(b.id), str(c.id)}
    assert ids("detected_from=2026-01-02T00:00:00%2B00:00") == {str(b.id), str(c.id)}
    assert ids("detected_to=2026-01-02T00:00:00%2B00:00") == {str(a.id), str(b.id)}


def test_global_history_spans_cases_and_filters_by_case_id(
    client, db, admin, auth_headers, reporter, make_case, approved_case, make_detection
):
    first = make_detection()
    second_case = make_case(reporter)
    second_case.status = "approved"
    db.commit()
    second = make_detection(case=second_case)
    response = client.get("/admin/detections", headers=auth_headers(admin))
    assert response.status_code == 200
    assert response.json()["total"] == 2
    filtered = client.get(f"/admin/detections?case_id={second_case.id}", headers=auth_headers(admin))
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["id"] == str(second.id)


def test_pagination_and_invalid_time_filters(client, admin, auth_headers, approved_case):
    headers = auth_headers(admin)
    assert client.get(f"/admin/cases/{approved_case.id}/detections?limit=101", headers=headers).status_code == 422
    assert client.get(f"/admin/cases/{approved_case.id}/detections?detected_from=2026-01-01T00:00:00", headers=headers).status_code == 422
    assert client.get(
        f"/admin/cases/{approved_case.id}/detections?detected_from=2026-02-01T00:00:00Z&detected_to=2026-01-01T00:00:00Z",
        headers=headers,
    ).status_code == 422


def test_admin_can_read_detection_detail_and_missing_detection_is_404(client, admin, auth_headers, make_detection):
    detection = make_detection()
    headers = auth_headers(admin)
    response = client.get(f"/admin/detections/{detection.id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == str(detection.id)
    assert client.get(f"/admin/detections/{uuid.uuid4()}", headers=headers).status_code == 404


@pytest.mark.parametrize("decision", ["confirmed", "dismissed"])
def test_admin_can_review_an_unreviewed_detection(client, admin, auth_headers, make_detection, decision):
    detection = make_detection()
    response = client.patch(
        f"/admin/detections/{detection.id}/review", headers=auth_headers(admin),
        json={"review_status": decision, "note": "  Fictional review note  "},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_status"] == decision
    assert body["reviewed_by_id"] == str(admin.id)
    assert body["reviewed_at"] is not None
    assert body["review_note"] == "Fictional review note"


def test_detection_can_only_be_reviewed_once(client, admin, auth_headers, make_detection):
    detection = make_detection()
    url = f"/admin/detections/{detection.id}/review"
    headers = auth_headers(admin)
    assert client.patch(url, headers=headers, json={"review_status": "confirmed"}).status_code == 200
    assert client.patch(url, headers=headers, json={"review_status": "dismissed"}).status_code == 409


def test_invalid_review_status_and_unknown_review_fields_are_422(client, admin, auth_headers, make_detection):
    detection = make_detection()
    url = f"/admin/detections/{detection.id}/review"
    headers = auth_headers(admin)
    assert client.patch(url, headers=headers, json={"review_status": "unreviewed"}).status_code == 422
    assert client.patch(url, headers=headers, json={"review_status": "confirmed", "other": 1}).status_code == 422


def test_owner_sees_only_confirmed_detections_and_only_limited_fields(
    client, reporter, auth_headers, make_detection
):
    confirmed = make_detection(review_status="confirmed")
    make_detection(review_status="dismissed")
    make_detection(review_status="unreviewed")
    response = client.get(f"/cases/{confirmed.case_id}/detections", headers=auth_headers(reporter))
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert set(body["items"][0]) == {"id", "detected_at", "location_text"}
    assert body["items"][0]["id"] == str(confirmed.id)


def test_another_user_and_an_admin_who_is_not_owner_get_404(
    client, make_user, admin, auth_headers, approved_case
):
    other = make_user(email="other@example.com")
    assert client.get(f"/cases/{approved_case.id}/detections", headers=auth_headers(other)).status_code == 404
    assert client.get(f"/cases/{approved_case.id}/detections", headers=auth_headers(admin)).status_code == 404


def test_owner_history_requires_authentication(client, approved_case):
    assert client.get(f"/cases/{approved_case.id}/detections").status_code == 401


def test_detection_logs_do_not_contain_location_or_camera_values(
    client, admin, auth_headers, approved_case, detection_payload, caplog
):
    caplog.set_level("INFO")
    detection_payload.update(
        location_text="PRIVATE_LOCATION_MARKER", camera_id="PRIVATE_CAMERA_MARKER", latitude=12.345, longitude=67.89
    )
    response = client.post(
        f"/admin/cases/{approved_case.id}/detections", json=detection_payload, headers=auth_headers(admin)
    )
    assert response.status_code == 201
    assert "PRIVATE_LOCATION_MARKER" not in caplog.text
    assert "PRIVATE_CAMERA_MARKER" not in caplog.text
    assert "12.345" not in caplog.text
    assert "67.89" not in caplog.text
    record = next(record for record in caplog.records if record.getMessage() == "detection_recorded")
    assert record.detection_id == response.json()["id"]
    assert record.case_id == str(approved_case.id)
    assert record.admin_id == str(admin.id)
    assert not hasattr(record, "location_text")
    assert not hasattr(record, "camera_id")


def test_owner_history_is_paginated(client, reporter, auth_headers, make_detection):
    for index in range(3):
        make_detection(review_status="confirmed", detected_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index))
    case_id = make_detection().case_id
    # The additional unreviewed row is not visible and therefore does not affect total.
    first = client.get(f"/cases/{case_id}/detections?limit=2&offset=0", headers=auth_headers(reporter)).json()
    second = client.get(f"/cases/{case_id}/detections?limit=2&offset=2", headers=auth_headers(reporter)).json()
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert first["total"] == second["total"] == 3
