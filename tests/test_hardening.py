import logging
from types import SimpleNamespace

from starlette.requests import Request

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.rate_limit import RateLimit, RateLimitExceeded, SlidingWindowRateLimiter
from app.api.rate_limits import limit_admin, limit_login, limit_photo_upload, limit_registration
from app.main import app, create_app


def assert_standard_error(response, *, fields=None):
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert {"code", "message", "request_id"} <= set(error)
    assert error["request_id"] == response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    if fields is True:
        assert isinstance(error.get("fields"), list)
    elif fields is False:
        assert "fields" not in error
    elif isinstance(fields, list):
        assert error["fields"] == fields
    return error


def test_http_validation_and_not_found_errors_share_shape_and_request_id(client):
    unauthenticated = client.get("/auth/me")
    assert unauthenticated.status_code == 401
    assert_standard_error(unauthenticated, fields=False)
    assert unauthenticated.headers["www-authenticate"] == "Bearer"

    rejected = "NO_ECHO"
    invalid = client.post("/auth/register", json={
        "email": "not-an-email", "full_name": "Test User", "password": rejected,
    })
    assert invalid.status_code == 422
    assert_standard_error(invalid, fields=True)
    assert "fields" in invalid.json()["error"]
    assert rejected not in invalid.text

    missing = client.get("/not-a-real-route")
    assert missing.status_code == 404
    assert_standard_error(missing, fields=False)


def test_forced_internal_error_is_generic_and_logged_with_request_id(client, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="app")

    def explode():
        raise RuntimeError("INTERNAL_SECRET_MARKER")

    monkeypatch.setattr("app.api.routes.health.check_database", explode)
    response = client.get("/health")
    assert response.status_code == 500
    error = assert_standard_error(response, fields=False)
    assert error["code"] == "internal_server_error"
    assert error["message"] == "An unexpected error occurred"
    assert "INTERNAL_SECRET_MARKER" not in response.text
    assert "Traceback" not in response.text
    assert "INTERNAL_SECRET_MARKER" in caplog.text
    assert any(getattr(record, "request_id", None) == error["request_id"] for record in caplog.records)


def test_rate_limiter_sliding_window_and_independent_keys():
    now = [100.0]
    limiter = SlidingWindowRateLimiter(clock=lambda: now[0])
    limit = RateLimit(2, 10)
    limiter.check("login_ip", "10.0.0.1", limit)
    limiter.check("login_ip", "10.0.0.1", limit)
    with pytest.raises(RateLimitExceeded) as caught:
        limiter.check("login_ip", "10.0.0.1", limit)
    assert caught.value.retry_after == 10

    # A second client IP and two distinct admins have independent buckets.
    limiter.check("login_ip", "10.0.0.2", limit)
    limiter.check("face_search_admin", "admin-a", limit)
    limiter.check("face_search_admin", "admin-b", limit)

    now[0] += 10
    limiter.check("login_ip", "10.0.0.1", limit)  # window recovered without sleeping


def test_login_rate_limit_returns_429_retry_after_and_standard_error(client, make_user):
    settings = get_settings()
    old_email_limit, old_ip_limit = settings.login_email_limit, settings.login_ip_limit
    settings.login_email_limit = 2
    settings.login_ip_limit = 20
    try:
        make_user(email="limit@example.com")
        for _ in range(2):
            response = client.post("/auth/login", data={
                "username": "limit@example.com", "password": "wrong-password",
            })
            assert response.status_code == 401
        limited = client.post("/auth/login", data={
            "username": "limit@example.com", "password": "wrong-password",
        })
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) >= 1
        assert_standard_error(limited, fields=False)
    finally:
        settings.login_email_limit, settings.login_ip_limit = old_email_limit, old_ip_limit


def test_security_headers_are_present_and_json_is_not_cached(client, monkeypatch):
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert response.headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in response.headers

    settings = get_settings()
    old_env = settings.app_env
    settings.app_env = "production"
    try:
        production = client.get("/health")
        assert production.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"
    finally:
        settings.app_env = old_env


def test_photo_response_keeps_its_cache_control_header(client, make_user, make_case, upload_photos, make_image_bytes, auth_headers):
    user = make_user()
    case = make_case(user)
    uploaded = upload_photos(user, case.id, make_image_bytes("JPEG"))
    assert uploaded.status_code == 201
    photo_id = uploaded.json()[0]["id"]
    response = client.get(f"/cases/{case.id}/photos/{photo_id}", headers=auth_headers(user))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_cors_allows_only_configured_origin(monkeypatch):
    settings = Settings(
        _env_file=None, app_env="development", secret_key="s" * 32,
        postgres_user="u", postgres_password="p", postgres_db="d",
        cors_allowed_origins=["https://allowed.example"], trusted_hosts=["*"],
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.configure_logging", lambda *args, **kwargs: None)
    cors_app = create_app()
    with TestClient(cors_app) as cors_client:
        allowed = cors_client.options("/auth/login", headers={
            "Origin": "https://allowed.example", "Access-Control-Request-Method": "POST",
        })
        blocked = cors_client.options("/auth/login", headers={
            "Origin": "https://blocked.example", "Access-Control-Request-Method": "POST",
        })
    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"
    assert "access-control-allow-origin" not in blocked.headers
    assert_standard_error(blocked, fields=False)


def test_empty_cors_allow_list_disables_cross_origin_access(monkeypatch):
    settings = Settings(
        _env_file=None, app_env="development", secret_key="s" * 32,
        postgres_user="u", postgres_password="p", postgres_db="d",
        cors_allowed_origins=[], trusted_hosts=["*"],
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.configure_logging", lambda *args, **kwargs: None)
    no_cors_app = create_app()
    with TestClient(no_cors_app) as no_cors_client:
        response = no_cors_client.options("/auth/login", headers={
            "Origin": "https://any.example", "Access-Control-Request-Method": "POST",
        })
    assert "access-control-allow-origin" not in response.headers


def _settings_for_production(**overrides):
    values = dict(
        _env_file=None, app_env="production", secret_key="s" * 32,
        postgres_user="u", postgres_password="real-database-password", postgres_db="d",
        cors_allowed_origins=[], enable_api_docs=False,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize("overrides, message", [
    ({"secret_key": "CHANGE_ME"}, "SECRET_KEY"),
    ({"postgres_password": "CHANGE_ME"}, "POSTGRES_PASSWORD"),
    ({"cors_allowed_origins": ["*"]}, "CORS_ALLOWED_ORIGINS"),
    ({"enable_api_docs": True}, "documentation"),
])
def test_production_configuration_is_refused(monkeypatch, overrides, message):
    settings = _settings_for_production(**overrides)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.configure_logging", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match=message):
        create_app()


@pytest.mark.parametrize("environment", ["development", "test"])
def test_development_and_test_configuration_still_start(monkeypatch, environment):
    settings = Settings(
        _env_file=None, app_env=environment, secret_key="s" * 32,
        postgres_user="u", postgres_password="p", postgres_db="d",
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.configure_logging", lambda *args, **kwargs: None)
    created = create_app()
    assert created.state.rate_limiter is not None


def test_global_request_body_limit_returns_standard_413(client, monkeypatch):
    settings = get_settings()
    old_limit = settings.max_request_body_bytes
    settings.max_request_body_bytes = 512
    try:
        response = client.post("/auth/register", content=b"x" * 513,
                               headers={"content-type": "application/octet-stream"})
    finally:
        settings.max_request_body_bytes = old_limit
    assert response.status_code == 413
    assert_standard_error(response, fields=False)


def test_health_reports_storage_failure_and_unconfigured_recognition(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.health.check_database", lambda: True)
    monkeypatch.setattr("app.api.routes.health.check_storage_writable", lambda: False)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["storage"] == "unavailable"
    assert response.json()["recognition"] == "unconfigured"


def _request_for_key(app_state, ip="10.0.0.1"):
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/", "raw_path": b"/",
        "query_string": b"", "root_path": "", "headers": [],
        "client": (ip, 12345), "server": ("testserver", 80),
        "app": SimpleNamespace(state=app_state), "state": {},
    }
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    return Request(scope, receive)


def test_login_limits_are_per_ip_and_normalized_email_hash():
    settings = get_settings()
    old_ip, old_email = settings.login_ip_limit, settings.login_email_limit
    settings.login_ip_limit, settings.login_email_limit = 1, 2
    limiter = SlidingWindowRateLimiter(clock=lambda: 100.0)
    state = SimpleNamespace(rate_limiter=limiter)
    try:
        first_ip = _request_for_key(state, "10.0.0.1")
        second_ip = _request_for_key(state, "10.0.0.2")
        third_ip = _request_for_key(state, "10.0.0.3")
        limit_login(first_ip, "ALICE@example.com")
        with pytest.raises(RateLimitExceeded):
            limit_login(first_ip, "other@example.com")  # IP limit is independent of email
        limit_login(second_ip, "alice@example.com")  # case-folded email shares the same hash bucket
        with pytest.raises(RateLimitExceeded):
            limit_login(third_ip, "Alice@Example.com")  # email limit spans client IPs
    finally:
        settings.login_ip_limit, settings.login_email_limit = old_ip, old_email


def test_registration_admin_and_photo_limits_are_scoped_to_the_right_keys(monkeypatch):
    settings = get_settings()
    old = (
        settings.registration_ip_limit, settings.face_search_admin_limit,
        settings.embedding_admin_limit, settings.photo_upload_user_limit,
    )
    settings.registration_ip_limit = 1
    settings.face_search_admin_limit = 1
    settings.embedding_admin_limit = 1
    settings.photo_upload_user_limit = 1
    now = [50.0]
    limiter = SlidingWindowRateLimiter(clock=lambda: now[0])
    state = SimpleNamespace(rate_limiter=limiter)
    try:
        request_a = _request_for_key(state, "10.0.0.1")
        request_b = _request_for_key(state, "10.0.0.2")
        limit_registration(request_a)
        with pytest.raises(RateLimitExceeded):
            limit_registration(request_a)
        limit_registration(request_b)  # a different registration IP has its own bucket

        limit_admin(request_a, "admin-a")
        with pytest.raises(RateLimitExceeded):
            limit_admin(request_a, "admin-a")
        limit_admin(request_a, "admin-b")  # independent admin key

        limit_admin(request_a, "admin-a", embeddings=True)
        with pytest.raises(RateLimitExceeded):
            limit_admin(request_a, "admin-a", embeddings=True)

        limit_photo_upload(request_a, "user-a")
        with pytest.raises(RateLimitExceeded):
            limit_photo_upload(request_a, "user-a")
        limit_photo_upload(request_a, "user-b")  # independent user key
        now[0] += 61
        limit_photo_upload(request_a, "user-a")  # expired window recovers
    finally:
        (settings.registration_ip_limit, settings.face_search_admin_limit,
         settings.embedding_admin_limit, settings.photo_upload_user_limit) = old


def test_streaming_request_body_limit_without_content_length(client):
    settings = get_settings()
    old_limit = settings.max_request_body_bytes
    settings.max_request_body_bytes = 512
    try:
        response = client.post(
            "/auth/register",
            content=iter([b"a" * 400, b"b" * 200]),
            headers={"content-type": "application/octet-stream"},
        )
    finally:
        settings.max_request_body_bytes = old_limit
    assert response.status_code == 413
    assert_standard_error(response, fields=False)


def test_trusted_hosts_rejects_unlisted_host_with_standard_error(monkeypatch):
    settings = Settings(
        _env_file=None, app_env="development", secret_key="s" * 32,
        postgres_user="u", postgres_password="p", postgres_db="d",
        trusted_hosts=["allowed.example"],
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.configure_logging", lambda *args, **kwargs: None)
    restricted_app = create_app()
    with TestClient(restricted_app) as restricted_client:
        response = restricted_client.get("/health", headers={"Host": "blocked.example"})
    assert response.status_code == 400
    assert_standard_error(response, fields=False)
    assert response.headers["x-frame-options"] == "DENY"
