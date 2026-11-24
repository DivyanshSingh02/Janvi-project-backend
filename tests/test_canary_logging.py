import logging
import uuid
from datetime import datetime, timezone

from app.models.user import User


def test_full_user_journey_does_not_log_personal_data_credentials_or_tokens(
    client, db, make_user, make_image_bytes, final_sha, fake_recognition, auth_headers, caplog
):
    personal = {
        "admin_email": "canary_admin_zorpax@example.test",
        "full_name": "CANARY_FULLNAME_ZORPAX",
        "email": "canary_email_zorpax@example.test",
        "case_name": "CANARY_CASE_NAME_ZORPAX",
        "phone": "+120255501987654",
        "case_email": "canary_contact_zorpax@example.test",
        "city": "CANARY_CITY_ZORPAX",
        "state": "CANARY_STATE_ZORPAX",
        "address": "CANARY_ADDRESS_ZORPAX_77",
        "last_seen_location": "CANARY_LOCATION_ZORPAX",
        "password": "CANARY_PASSWORD_ZORPAX_123!",
        "detection_location": "CANARY_DETECTION_PLACE_ZORPAX",
        "filename": "CANARY_FILENAME_ZORPAX.png",
    }
    admin = make_user(email=personal["admin_email"], role="admin")
    image = make_image_bytes("PNG", size=(240, 240), seed=67)
    fake_recognition.set_faces(final_sha(image), [1, 0, 0, 0])

    with caplog.at_level(logging.INFO):
        registered = client.post("/auth/register", json={
            "email": personal["email"], "full_name": personal["full_name"], "password": personal["password"],
        })
        assert registered.status_code == 201, registered.text
        login = client.post("/auth/login", data={"username": personal["email"], "password": personal["password"]})
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
        owner = db.get(User, uuid.UUID(registered.json()["id"]))
        owner_headers = auth_headers(owner)
        created = client.post("/cases", headers=owner_headers, json={
            "name": personal["case_name"], "age": 19, "gender": "female",
            "contact_phone": personal["phone"], "contact_email": personal["case_email"],
            "city": personal["city"], "state": personal["state"], "address": personal["address"],
            "last_seen_location": personal["last_seen_location"],
            "last_seen_date": "2026-10-20",
        })
        assert created.status_code == 201, created.text
        case_id = created.json()["id"]
        uploaded = client.post(
            f"/cases/{case_id}/photos", headers=owner_headers,
            files={"files": (personal["filename"], image, "image/png")},
        )
        assert uploaded.status_code == 201, uploaded.text
        approved = client.post(f"/admin/cases/{case_id}/approve", headers=auth_headers(admin))
        assert approved.status_code == 200, approved.text
        enrolled = client.post(f"/admin/cases/{case_id}/embeddings", headers=auth_headers(admin))
        assert enrolled.status_code == 200, enrolled.text
        searched = client.post(
            "/search", headers=auth_headers(admin),
            files={"file": (personal["filename"], image, "image/png")},
        )
        assert searched.status_code == 200, searched.text
        detected = client.post(
            f"/admin/cases/{case_id}/detections", headers=auth_headers(admin),
            json={
                "detected_at": datetime.now(timezone.utc).isoformat(), "source": "manual",
                "location_text": personal["detection_location"], "confidence": 0.88,
            },
        )
        assert detected.status_code == 201, detected.text

    log_text = caplog.text + "\n" + "\n".join(str(record.__dict__) for record in caplog.records)
    for value in personal.values():
        assert value not in log_text, value
    assert token not in log_text
    assert personal["password"] not in log_text
    assert "case_status_changed" in log_text or "case_status_change" in log_text
    assert "face_search" in log_text
    assert "detection_recorded" in log_text


def test_the_facenet_service_never_logs_paths_exception_text_or_image_bytes(tmp_path, caplog, monkeypatch):
    """FaceNet touches files and third-party libraries, which can put paths and messages into exceptions. Neither the
    model folder, the weights path, the library's text nor the image bytes may reach a log or an exception message."""
    from io import BytesIO

    from PIL import Image

    from app.recognition import FaceNetRecognitionService, RecognitionUnavailableError
    from app.recognition import facenet

    secret_dir = tmp_path / "CANARY_MODEL_FOLDER_ZORPAX"
    service = FaceNetRecognitionService(model_dir=secret_dir)
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            try:
                service.model_info()                                   # weights missing
            except RecognitionUnavailableError as exc:
                assert "CANARY" not in str(exc)
        secret_dir.mkdir()
        (secret_dir / facenet.WEIGHTS_FILENAME).write_bytes(b"\0" * facenet.MIN_PLAUSIBLE_WEIGHTS_BYTES)
        monkeypatch.setattr(facenet, "_package_available", lambda name: True)

        def boom():
            raise RuntimeError(f"CANARY_LIBRARY_TEXT {secret_dir}")

        monkeypatch.setattr(service, "_load_models", boom)
        try:
            service.model_info()                                       # heavy load fails with a revealing message
        except RecognitionUnavailableError as exc:
            assert "CANARY" not in str(exc) and str(secret_dir) not in str(exc)

        buffer = BytesIO()
        Image.new("RGB", (10, 10), "white").save(buffer, format="PNG")
        ready = FaceNetRecognitionService(model_dir=secret_dir)
        monkeypatch.setattr(ready, "_preflight", lambda: None)
        monkeypatch.setattr(ready, "_load_models", lambda: (_ for _ in ()).throw(ValueError("CANARY_X")))
        try:
            ready.extract_faces(buffer.getvalue() + b"CANARY_IMAGE_BYTES")
        except RecognitionUnavailableError:
            pass

    log_text = caplog.text + "\n" + "\n".join(str(record.__dict__) for record in caplog.records)
    for canary in ("CANARY", "ZORPAX", str(secret_dir), "CANARY_IMAGE_BYTES"):
        assert canary not in log_text, canary
    assert "weights_missing" in log_text
