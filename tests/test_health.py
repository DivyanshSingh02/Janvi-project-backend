import numpy as np
from fake_recognition import FakeRecognitionService

from app.api.routes import health as health_route
from app.recognition import FaceNetRecognitionService
from app.recognition import facenet
from app.recognition.base import ModelInfo
from app.recognition.facenet import WEIGHTS_FILENAME, _Loaded


def test_health_ok_when_database_is_reachable(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["storage"] == "ok"
    assert body["recognition"] == "unconfigured"
    assert body["version"]
    assert response.headers["x-request-id"]


def test_health_returns_503_when_database_is_down(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.health.check_database", lambda: False)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded", "version": "0.1.0", "database": "unavailable",
        "storage": "ok", "recognition": "unconfigured",
    }


def test_health_reports_recognition_unconfigured_when_facenet_weights_are_missing(client, monkeypatch, tmp_path):
    service = FaceNetRecognitionService(model_dir=tmp_path)
    monkeypatch.setattr(health_route, "get_recognition_service", lambda: service)
    response = client.get("/health")
    assert response.status_code == 200                       # a missing model does not make the API unhealthy
    assert response.json()["recognition"] == "unconfigured"
    assert service.is_configured is False


def test_health_reports_recognition_ready_for_a_loaded_service(client, monkeypatch):
    monkeypatch.setattr(health_route, "get_recognition_service", FakeRecognitionService)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["recognition"] == "ready"


def test_health_is_ready_only_when_facenet_can_really_be_used_and_never_reloads_it(client, monkeypatch, tmp_path):
    service = FaceNetRecognitionService(model_dir=tmp_path)
    monkeypatch.setattr(health_route, "get_recognition_service", lambda: service)
    loads = []

    def load():
        loads.append(1)
        return _Loaded(object(), object(), object(), ModelInfo("facenet-vggface2", "v", 512))

    monkeypatch.setattr(service, "_load_models", load)
    monkeypatch.setattr(facenet, "_package_available", lambda name: True)
    assert client.get("/health").json()["recognition"] == "unconfigured"          # no weights file yet
    assert loads == []                                                              # and nothing heavy was tried
    (tmp_path / WEIGHTS_FILENAME).write_bytes(np.zeros(facenet.MIN_PLAUSIBLE_WEIGHTS_BYTES, dtype=np.uint8).tobytes())
    for _ in range(3):                                                              # a polling load balancer
        assert client.get("/health").json()["recognition"] == "ready"
    assert len(loads) == 1
