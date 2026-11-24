from datetime import datetime, timedelta, timezone

from app.models.user import User


def test_admin_users_without_token_returns_401(client):
    response = client.get("/admin/users")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_admin_users_returns_403_for_a_normal_user(client, make_user, auth_headers):
    user = make_user(email="normal@example.com")
    assert client.get("/admin/users", headers=auth_headers(user)).status_code == 403


def test_admin_users_returns_200_for_an_admin(client, make_user, auth_headers):
    admin = make_user(email="boss@example.com", role="admin")
    make_user(email="normal@example.com")
    response = client.get("/admin/users", headers=auth_headers(admin))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert body["total"] == 2
    assert body["limit"] == 20 and body["offset"] == 0
    assert {item["email"] for item in body["items"]} == {"boss@example.com", "normal@example.com"}
    assert "hashed_password" not in response.text
    assert "scrypt" not in response.text


def test_admin_users_are_listed_newest_first_and_paginated(client, db, make_user, auth_headers):
    admin = make_user(email="boss@example.com", role="admin")
    created = [make_user(email=f"person{i}@example.com") for i in range(5)]
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, user in enumerate([admin, *created]):        # fixed timestamps: admin oldest, person4 newest
        db.get(User, user.id).created_at = base + timedelta(days=index)
    db.commit()

    headers = auth_headers(admin)
    first = client.get("/admin/users?limit=2&offset=0", headers=headers).json()
    second = client.get("/admin/users?limit=2&offset=2", headers=headers).json()
    last = client.get("/admin/users?limit=2&offset=4", headers=headers).json()

    assert [u["email"] for u in first["items"]] == ["person4@example.com", "person3@example.com"]
    assert [u["email"] for u in second["items"]] == ["person2@example.com", "person1@example.com"]
    assert [u["email"] for u in last["items"]] == ["person0@example.com", "boss@example.com"]
    assert first["total"] == second["total"] == last["total"] == 6
    assert (second["limit"], second["offset"]) == (2, 2)


def test_admin_users_rejects_invalid_pagination_with_422(client, make_user, auth_headers):
    headers = auth_headers(make_user(email="boss@example.com", role="admin"))
    for query in ("limit=101", "limit=0", "offset=-1"):
        assert client.get(f"/admin/users?{query}", headers=headers).status_code == 422, query
    assert client.get("/admin/users?limit=100", headers=headers).status_code == 200


def test_the_role_is_read_from_the_database_not_from_the_token(client, db, make_user, auth_headers):
    admin = make_user(email="boss@example.com", role="admin")
    headers = auth_headers(admin)                            # token issued while the user was an admin
    assert client.get("/admin/users", headers=headers).status_code == 200
    admin.role = "user"                                      # demoted afterwards
    db.commit()
    assert client.get("/admin/users", headers=headers).status_code == 403
