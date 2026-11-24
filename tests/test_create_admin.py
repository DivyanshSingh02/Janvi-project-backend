import pytest
from sqlalchemy import select

from app.core.security import PasswordPolicyError
from app.models.user import User
from app.services import users as user_service
from scripts import create_admin

PASSWORD = "correct-horse-battery"


def _fail_if_asked() -> str:
    raise AssertionError("the password must not be requested here")


def test_creates_an_admin_when_the_email_is_new(db):
    result = create_admin.create_or_promote_admin(
        db, email="Boss@Example.com", full_name="Boss Person", get_password=lambda: PASSWORD
    )
    assert result.created is True and result.already_admin is False
    assert result.user.role == "admin"
    assert result.user.email == "boss@example.com"
    assert user_service.authenticate(db, "boss@example.com", PASSWORD) is not None


def test_promotes_an_existing_user_without_asking_for_a_password(db, make_user):
    make_user(email="alice@example.com", full_name="Alice Example", password=PASSWORD)
    result = create_admin.create_or_promote_admin(
        db, email="ALICE@example.com", full_name="Some Other Name", get_password=_fail_if_asked
    )
    assert result.created is False and result.already_admin is False
    db.expire_all()
    stored = db.scalar(select(User).where(User.email == "alice@example.com"))
    assert stored.role == "admin"
    assert stored.full_name == "Alice Example"                        # name not overwritten
    assert user_service.authenticate(db, "alice@example.com", PASSWORD) is not None   # password unchanged


def test_an_account_that_is_already_admin_is_left_alone(db, make_user):
    make_user(email="boss@example.com", role="admin")
    result = create_admin.create_or_promote_admin(
        db, email="boss@example.com", full_name="Boss Person", get_password=_fail_if_asked
    )
    assert result.created is False and result.already_admin is True


def test_a_weak_password_creates_nothing(db):
    with pytest.raises(PasswordPolicyError):
        create_admin.create_or_promote_admin(
            db, email="boss@example.com", full_name="Boss Person", get_password=lambda: "short"
        )
    assert user_service.get_by_email(db, "boss@example.com") is None


def test_main_creates_an_admin_from_the_prompts_and_never_prints_the_password(db, monkeypatch, capsys):
    answers = iter([PASSWORD, PASSWORD])
    monkeypatch.setattr(create_admin, "getpass", lambda prompt="": next(answers))
    code = create_admin.main(["--email", "boss@example.com", "--full-name", "Boss Person"])
    output = capsys.readouterr().out
    assert code == 0
    assert "Created admin account" in output
    assert PASSWORD not in output
    db.expire_all()
    assert user_service.get_by_email(db, "boss@example.com").role == "admin"


def test_main_gives_up_after_mismatching_passwords_and_creates_nothing(db, monkeypatch, capsys):
    answers = iter(["first-password-123", "different-password-456"] * 3)
    monkeypatch.setattr(create_admin, "getpass", lambda prompt="": next(answers))
    code = create_admin.main(["--email", "boss@example.com", "--full-name", "Boss Person"])
    assert code == 1
    assert "Nothing was changed" in capsys.readouterr().out
    assert user_service.get_by_email(db, "boss@example.com") is None
