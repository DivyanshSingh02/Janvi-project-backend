import uuid
from datetime import date

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import delete, inspect, text
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal, engine
from app.models.case import MissingCase
from app.models.photo import CasePhoto
from app.models.user import User

HEAD = "0009"


def _current_revision() -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def test_head_revision_is_0009_and_the_chain_is_unbroken(alembic_cfg):
    script = ScriptDirectory.from_config(alembic_cfg)
    assert script.get_heads() == [HEAD]
    assert script.get_revision("0009").down_revision == "0008"
    assert script.get_revision("0008").down_revision == "0007"
    assert script.get_revision("0007").down_revision == "0006"
    assert script.get_revision("0006").down_revision == "0005"
    assert script.get_revision("0005").down_revision == "0004"
    assert script.get_revision("0004").down_revision == "0003"
    assert script.get_revision("0003").down_revision == "0002"
    assert script.get_revision("0002").down_revision == "0001"
    assert _current_revision() == HEAD


def test_found_person_reports_table_has_private_report_and_review_fields():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("found_person_reports")}
    assert set(columns) == {
        "id", "reporter_id", "storage_key", "content_type", "size_bytes", "width", "height", "sha256",
        "review_status", "reviewed_by_id", "reviewed_at", "review_note", "created_at",
    }
    assert columns["reporter_id"]["nullable"] is False
    assert columns["reviewed_by_id"]["nullable"] is True
    assert columns["reviewed_at"]["nullable"] is True
    assert columns["review_note"]["nullable"] is True
    assert columns["created_at"]["type"].timezone is True
    assert {constraint["name"] for constraint in inspector.get_check_constraints("found_person_reports")} == {
        "ck_found_person_reports_content_type_valid",
        "ck_found_person_reports_size_positive",
        "ck_found_person_reports_width_positive",
        "ck_found_person_reports_height_positive",
        "ck_found_person_reports_sha256_format",
        "ck_found_person_reports_review_status_valid",
        "ck_found_person_reports_review_fields_match_status",
    }


def test_users_table_has_the_expected_columns_and_explicit_constraint_names():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("users")}
    assert set(columns) == {
        "id", "email", "full_name", "hashed_password", "role", "is_active", "created_at", "updated_at"
    }
    assert columns["created_at"]["type"].timezone is True
    assert columns["updated_at"]["type"].timezone is True
    assert inspector.get_pk_constraint("users")["name"] == "pk_users"
    assert [c["name"] for c in inspector.get_unique_constraints("users")] == ["uq_users_email"]
    assert {c["name"] for c in inspector.get_check_constraints("users")} == {
        "ck_users_role_valid",
        "ck_users_email_lowercase",
    }


def test_missing_cases_table_has_the_expected_columns_constraints_and_indexes():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("missing_cases")}
    assert set(columns) == {
        "id", "reporter_id", "name", "age", "gender", "contact_phone", "contact_email", "city",
        "state", "address", "last_seen_location", "last_seen_date", "status", "created_at", "updated_at",
        "review_reason", "reviewed_by_id", "reviewed_at",                      # added by migration 0005
    }
    assert columns["reporter_id"]["nullable"] is False
    assert columns["contact_phone"]["nullable"] is True
    assert columns["contact_email"]["nullable"] is True
    assert isinstance(columns["last_seen_date"]["type"], sa.Date)
    assert columns["created_at"]["type"].timezone is True
    assert columns["updated_at"]["type"].timezone is True

    assert inspector.get_pk_constraint("missing_cases")["name"] == "pk_missing_cases"
    assert {c["name"] for c in inspector.get_check_constraints("missing_cases")} == {
        "ck_missing_cases_status_valid",
        "ck_missing_cases_gender_valid",
        "ck_missing_cases_age_range",
        "ck_missing_cases_contact_present",
        "ck_missing_cases_rejected_requires_reason",
    }
    foreign_keys = {fk["name"]: fk for fk in inspector.get_foreign_keys("missing_cases")}
    assert set(foreign_keys) == {"fk_missing_cases_reporter_id_users", "fk_missing_cases_reviewed_by_id_users"}
    for foreign_key in foreign_keys.values():
        assert foreign_key["referred_table"] == "users"
        assert foreign_key["options"].get("ondelete") == "RESTRICT"
    assert {index["name"] for index in inspector.get_indexes("missing_cases")} == {
        "ix_missing_cases_reporter_id_created_at",
        "ix_missing_cases_status",
    }
    with engine.connect() as connection:
        index_definition = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_missing_cases_reporter_id_created_at'")
        ).scalar_one()
    assert "(reporter_id, created_at DESC)" in index_definition


def test_database_itself_rejects_an_unknown_role_and_an_uppercase_email(clean_db):
    def attempt(**overrides):
        fields = {"email": "a@example.com", "full_name": "A", "hashed_password": "x", "role": "user"}
        fields.update(overrides)
        with SessionLocal() as session:
            session.add(User(**fields))
            with pytest.raises(IntegrityError) as exc_info:
                session.commit()
            session.rollback()
        return str(exc_info.value)

    assert "ck_users_role_valid" in attempt(role="superuser")
    assert "ck_users_email_lowercase" in attempt(email="Upper@Example.com")


def test_database_itself_rejects_invalid_case_rows(make_user):
    reporter = make_user(email="reporter@example.com")

    def attempt(**overrides):
        fields = {
            "reporter_id": reporter.id,
            "name": "Test Person",
            "age": 12,
            "gender": "female",
            "contact_phone": "+15550100199",
            "contact_email": None,
            "city": "Sampletown",
            "state": "Samplestate",
            "address": "1 Sample Street",
            "last_seen_location": "Sample Market",
            "last_seen_date": date(2026, 1, 15),
            "status": "pending",
        }
        fields.update(overrides)
        with SessionLocal() as session:
            session.add(MissingCase(**fields))
            with pytest.raises(IntegrityError) as exc_info:
                session.commit()
            session.rollback()
        return str(exc_info.value)

    assert "ck_missing_cases_status_valid" in attempt(status="archived")
    assert "ck_missing_cases_gender_valid" in attempt(gender="robot")
    assert "ck_missing_cases_age_range" in attempt(age=200)
    assert "ck_missing_cases_age_range" in attempt(age=-1)
    assert "ck_missing_cases_contact_present" in attempt(contact_phone=None, contact_email=None)
    assert "fk_missing_cases_reporter_id_users" in attempt(reporter_id=uuid.uuid4())


def test_a_user_who_has_cases_cannot_be_deleted(db, make_user, make_case):
    user = make_user(email="reporter@example.com")
    make_case(user)
    with pytest.raises(IntegrityError) as exc_info:
        db.execute(delete(User).where(User.id == user.id))
        db.commit()
    db.rollback()
    assert "fk_missing_cases_reporter_id_users" in str(exc_info.value)


def test_migration_0002_downgrades_and_upgrades_cleanly(alembic_cfg):
    try:
        command.downgrade(alembic_cfg, "0001")                # also undoes 0003, 0004, 0005 and 0006
        assert _current_revision() == "0001"
        tables = inspect(engine).get_table_names()
        assert "found_person_reports" not in tables
        assert "users" not in tables
        assert "missing_cases" not in tables
        assert "case_photos" not in tables
        assert "case_status_events" not in tables
        assert "detections" not in tables
        assert "audit_events" not in tables
    finally:
        command.upgrade(alembic_cfg, "head")                  # always leave the test database at head
    assert _current_revision() == HEAD
    tables = inspect(engine).get_table_names()
    assert "users" in tables
    assert "missing_cases" in tables
    assert "case_photos" in tables


def test_migration_0003_downgrades_and_upgrades_cleanly(alembic_cfg):
    try:
        command.downgrade(alembic_cfg, "0002")                # also undoes 0004, 0005 and 0006
        assert _current_revision() == "0002"
        tables = inspect(engine).get_table_names()
        assert "users" in tables
        assert "missing_cases" not in tables
        assert "case_photos" not in tables
    finally:
        command.upgrade(alembic_cfg, "head")
    assert _current_revision() == HEAD
    assert "missing_cases" in inspect(engine).get_table_names()


# ------------------------------------------------------------------ 0004: case_photos
def test_case_photos_table_has_the_expected_columns_constraints_and_indexes():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("case_photos")}
    assert set(columns) == {
        "id", "case_id", "storage_key", "content_type", "size_bytes", "width", "height", "sha256", "created_at",
        "embedding_status", "embedding_error",                                  # added by migration 0006
    }
    assert {name for name, column in columns.items() if column["nullable"]} == {"embedding_error"}
    assert columns["created_at"]["type"].timezone is True
    assert "file_name" not in columns and "filename" not in columns          # the original name is never stored

    assert inspector.get_pk_constraint("case_photos")["name"] == "pk_case_photos"
    assert {c["name"] for c in inspector.get_unique_constraints("case_photos")} == {
        "uq_case_photos_storage_key",
        "uq_case_photos_case_id_sha256",
    }
    assert {c["name"] for c in inspector.get_check_constraints("case_photos")} == {
        "ck_case_photos_content_type_valid",
        "ck_case_photos_size_positive",
        "ck_case_photos_width_positive",
        "ck_case_photos_height_positive",
        "ck_case_photos_sha256_format",
        "ck_case_photos_embedding_status_valid",                                # added by migration 0006
    }
    foreign_keys = inspector.get_foreign_keys("case_photos")
    assert len(foreign_keys) == 1
    assert foreign_keys[0]["name"] == "fk_case_photos_case_id_missing_cases"
    assert foreign_keys[0]["referred_table"] == "missing_cases"
    assert foreign_keys[0]["options"].get("ondelete") == "CASCADE"
    plain_indexes = [index for index in inspector.get_indexes("case_photos") if "duplicates_constraint" not in index]
    assert [index["name"] for index in plain_indexes] == ["ix_case_photos_case_id_created_at"]
    assert plain_indexes[0]["column_names"] == ["case_id", "created_at"]


def test_the_model_and_the_migration_use_the_same_names_for_case_photos():
    table = CasePhoto.__table__
    inspector = inspect(engine)
    database_names = (
        {inspector.get_pk_constraint("case_photos")["name"]}
        | {c["name"] for c in inspector.get_unique_constraints("case_photos")}
        | {c["name"] for c in inspector.get_check_constraints("case_photos")}
        | {fk["name"] for fk in inspector.get_foreign_keys("case_photos")}
    )
    assert {str(constraint.name) for constraint in table.constraints if constraint.name} == database_names
    plain_indexes = [index["name"] for index in inspector.get_indexes("case_photos") if "duplicates_constraint" not in index]
    assert {index.name for index in table.indexes} == set(plain_indexes)


def _add_photo(case_id, **overrides):
    fields = {
        "case_id": case_id,
        "storage_key": f"cases/{case_id}/{uuid.uuid4().hex}.jpg",
        "content_type": "image/jpeg",
        "size_bytes": 1234,
        "width": 200,
        "height": 150,
        "sha256": uuid.uuid4().hex + uuid.uuid4().hex,        # 64 lowercase hex characters
    }
    fields.update(overrides)
    return CasePhoto(**fields)


def test_database_itself_rejects_invalid_photo_rows(make_user, make_case):
    case = make_case(make_user(email="reporter@example.com"))

    def attempt(*, first_photo=None, **overrides):
        with SessionLocal() as session:
            if first_photo is not None:
                session.add(first_photo)
                session.commit()
            session.add(_add_photo(overrides.pop("case_id", case.id), **overrides))
            with pytest.raises(IntegrityError) as exc_info:
                session.commit()
            session.rollback()
        return str(exc_info.value)

    assert "ck_case_photos_sha256_format" in attempt(sha256="A" * 64)             # upper case
    assert "ck_case_photos_sha256_format" in attempt(sha256="a" * 63)             # too short
    assert "ck_case_photos_sha256_format" in attempt(sha256="a" * 63 + "g")      # not hex
    assert "ck_case_photos_size_positive" in attempt(size_bytes=0)
    assert "ck_case_photos_width_positive" in attempt(width=0)
    assert "ck_case_photos_height_positive" in attempt(height=-1)
    assert "ck_case_photos_content_type_valid" in attempt(content_type="image/png")
    assert "fk_case_photos_case_id_missing_cases" in attempt(case_id=uuid.uuid4())

    first = _add_photo(case.id, storage_key="cases/shared/key.jpg", sha256="b" * 64)
    assert "uq_case_photos_storage_key" in attempt(first_photo=first, storage_key="cases/shared/key.jpg")
    with SessionLocal() as session:
        session.execute(delete(CasePhoto))
        session.commit()
    first = _add_photo(case.id, sha256="c" * 64)
    assert "uq_case_photos_case_id_sha256" in attempt(first_photo=first, sha256="c" * 64)


def test_deleting_a_case_row_deletes_its_photo_rows(db, make_user, make_case):
    case = make_case(make_user(email="reporter@example.com"))
    other_case = make_case(make_user(email="other@example.com"))
    db.add_all([_add_photo(case.id), _add_photo(case.id), _add_photo(other_case.id)])
    db.commit()
    db.execute(delete(MissingCase).where(MissingCase.id == case.id))
    db.commit()
    remaining = db.scalars(sa.select(CasePhoto.case_id)).all()
    assert remaining == [other_case.id]


def test_migration_0004_downgrades_and_upgrades_cleanly(alembic_cfg):
    try:
        command.downgrade(alembic_cfg, "0003")
        assert _current_revision() == "0003"
        tables = inspect(engine).get_table_names()
        assert "case_photos" not in tables
        assert "missing_cases" in tables and "users" in tables      # earlier stages are untouched
    finally:
        command.upgrade(alembic_cfg, "head")
    assert _current_revision() == HEAD
    assert "case_photos" in inspect(engine).get_table_names()
