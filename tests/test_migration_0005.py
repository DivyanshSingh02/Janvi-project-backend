"""Migration 0005: the review columns on missing_cases and the case_status_events table."""
import uuid
from datetime import date

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.runtime.migration import MigrationContext
from sqlalchemy import delete, inspect, text
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal, engine
from app.models.case import MissingCase
from app.models.case_status_event import CaseStatusEvent
from app.models.user import User

HEAD = "0008"
CK_REJECTED_NEEDS_REASON = "ck_missing_cases_rejected_requires_reason"


def _current_revision() -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def _case_fields(reporter_id, **overrides):
    fields = {
        "reporter_id": reporter_id,
        "name": "Test Person",
        "age": 12,
        "gender": "female",
        "contact_phone": "+15550100199",
        "city": "Sampletown",
        "state": "Samplestate",
        "address": "1 Sample Street",
        "last_seen_location": "Sample Market",
        "last_seen_date": date(2026, 1, 15),
        "status": "pending",
    }
    fields.update(overrides)
    return fields


def _commit_error(*objects) -> str:
    """Add the objects in a fresh session, expect the commit to be refused, return the error text."""
    with SessionLocal() as session:
        session.add_all(objects)
        with pytest.raises(IntegrityError) as exc_info:
            session.commit()
        session.rollback()
    return str(exc_info.value)


# ------------------------------------------------------------------ structure
def test_missing_cases_has_the_three_review_columns_and_they_are_optional():
    columns = {column["name"]: column for column in inspect(engine).get_columns("missing_cases")}
    assert isinstance(columns["review_reason"]["type"], sa.Text)
    assert isinstance(columns["reviewed_by_id"]["type"], sa.Uuid)
    assert isinstance(columns["reviewed_at"]["type"], sa.DateTime) and columns["reviewed_at"]["type"].timezone is True
    for name in ("review_reason", "reviewed_by_id", "reviewed_at"):
        assert columns[name]["nullable"] is True


def test_the_rejected_check_and_the_reviewer_foreign_key_have_explicit_names():
    inspector = inspect(engine)
    checks = {check["name"]: check["sqltext"] for check in inspector.get_check_constraints("missing_cases")}
    assert "rejected" in checks[CK_REJECTED_NEEDS_REASON] and "review_reason" in checks[CK_REJECTED_NEEDS_REASON]
    foreign_keys = {fk["name"]: fk for fk in inspector.get_foreign_keys("missing_cases")}
    reviewer = foreign_keys["fk_missing_cases_reviewed_by_id_users"]
    assert reviewer["constrained_columns"] == ["reviewed_by_id"]
    assert reviewer["referred_table"] == "users"
    assert reviewer["options"].get("ondelete") == "RESTRICT"


def test_case_status_events_has_the_expected_columns_constraints_and_index():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("case_status_events")}
    assert set(columns) == {"id", "case_id", "from_status", "to_status", "changed_by_id", "reason", "created_at"}
    assert {name for name, column in columns.items() if column["nullable"]} == {"reason"}
    assert isinstance(columns["reason"]["type"], sa.Text)
    assert columns["created_at"]["type"].timezone is True

    assert inspector.get_pk_constraint("case_status_events")["name"] == "pk_case_status_events"
    assert {check["name"] for check in inspector.get_check_constraints("case_status_events")} == {
        "ck_case_status_events_from_status_valid",
        "ck_case_status_events_to_status_valid",
    }
    foreign_keys = {fk["name"]: fk for fk in inspector.get_foreign_keys("case_status_events")}
    assert set(foreign_keys) == {
        "fk_case_status_events_case_id_missing_cases",
        "fk_case_status_events_changed_by_id_users",
    }
    assert foreign_keys["fk_case_status_events_case_id_missing_cases"]["referred_table"] == "missing_cases"
    assert foreign_keys["fk_case_status_events_case_id_missing_cases"]["options"].get("ondelete") == "CASCADE"
    assert foreign_keys["fk_case_status_events_changed_by_id_users"]["referred_table"] == "users"
    assert foreign_keys["fk_case_status_events_changed_by_id_users"]["options"].get("ondelete") == "RESTRICT"
    indexes = inspector.get_indexes("case_status_events")
    assert [index["name"] for index in indexes] == ["ix_case_status_events_case_id_created_at"]
    assert indexes[0]["column_names"] == ["case_id", "created_at"]


@pytest.mark.parametrize("model", [MissingCase, CaseStatusEvent], ids=["missing_cases", "case_status_events"])
def test_the_models_and_the_migration_use_the_same_names(model):
    table = model.__table__
    inspector = inspect(engine)
    database_names = (
        {inspector.get_pk_constraint(table.name)["name"]}
        | {c["name"] for c in inspector.get_unique_constraints(table.name)}
        | {c["name"] for c in inspector.get_check_constraints(table.name)}
        | {fk["name"] for fk in inspector.get_foreign_keys(table.name)}
    )
    assert {str(constraint.name) for constraint in table.constraints if constraint.name} == database_names
    plain_indexes = {i["name"] for i in inspector.get_indexes(table.name) if "duplicates_constraint" not in i}
    assert {index.name for index in table.indexes} == plain_indexes


# ------------------------------------------------------------------ what the database itself enforces
def test_the_database_refuses_a_rejected_case_without_a_reason(make_user):
    reporter = make_user(email="reporter@example.com")
    assert CK_REJECTED_NEEDS_REASON in _commit_error(MissingCase(**_case_fields(reporter.id, status="rejected")))
    assert CK_REJECTED_NEEDS_REASON in _commit_error(
        MissingCase(**_case_fields(reporter.id, status="rejected", review_reason=None))
    )


def test_the_database_accepts_a_rejected_case_with_a_reason_and_other_statuses_without_one(make_user):
    reporter = make_user(email="reporter@example.com")
    with SessionLocal() as session:
        session.add(MissingCase(**_case_fields(reporter.id, status="rejected", review_reason="Fictional reason")))
        for status in ("pending", "approved", "found", "closed"):
            session.add(MissingCase(**_case_fields(reporter.id, status=status)))
        session.commit()                                          # no IntegrityError


def test_changing_an_existing_case_to_rejected_needs_a_reason_too(make_user, make_case):
    case = make_case(make_user(email="reporter@example.com"))
    with SessionLocal() as session:
        stored = session.get(MissingCase, case.id)
        stored.status = "rejected"
        with pytest.raises(IntegrityError) as exc_info:
            session.commit()
        session.rollback()
    assert CK_REJECTED_NEEDS_REASON in str(exc_info.value)


def test_the_database_refuses_unknown_statuses_and_unknown_references_in_the_history(make_user, make_case):
    admin = make_user(email="boss@example.com", role="admin")
    case = make_case(make_user(email="reporter@example.com"))

    def event(**overrides):
        fields = {"case_id": case.id, "from_status": "pending", "to_status": "approved", "changed_by_id": admin.id}
        fields.update(overrides)
        return CaseStatusEvent(**fields)

    assert "ck_case_status_events_from_status_valid" in _commit_error(event(from_status="archived"))
    assert "ck_case_status_events_to_status_valid" in _commit_error(event(to_status="archived"))
    assert "fk_case_status_events_case_id_missing_cases" in _commit_error(event(case_id=uuid.uuid4()))
    assert "fk_case_status_events_changed_by_id_users" in _commit_error(event(changed_by_id=uuid.uuid4()))


def test_a_user_who_changed_a_status_cannot_be_deleted(db, make_user, make_case):
    admin = make_user(email="boss@example.com", role="admin")
    case = make_case(make_user(email="reporter@example.com"))
    db.add(CaseStatusEvent(case_id=case.id, from_status="pending", to_status="approved", changed_by_id=admin.id))
    db.commit()
    with pytest.raises(IntegrityError) as exc_info:
        db.execute(delete(User).where(User.id == admin.id))
        db.commit()
    db.rollback()
    assert "fk_case_status_events_changed_by_id_users" in str(exc_info.value)


def test_a_user_who_reviewed_a_case_cannot_be_deleted(db, make_user, make_case):
    admin = make_user(email="boss@example.com", role="admin")
    case = make_case(make_user(email="reporter@example.com"))
    case.reviewed_by_id = admin.id
    db.commit()
    with pytest.raises(IntegrityError) as exc_info:
        db.execute(delete(User).where(User.id == admin.id))
        db.commit()
    db.rollback()
    assert "fk_missing_cases_reviewed_by_id_users" in str(exc_info.value)


def test_deleting_a_case_row_deletes_its_history(db, make_user, make_case):
    admin = make_user(email="boss@example.com", role="admin")
    reporter = make_user(email="reporter@example.com")
    case, other = make_case(reporter), make_case(reporter)
    db.add_all(
        [
            CaseStatusEvent(case_id=case.id, from_status="pending", to_status="approved", changed_by_id=admin.id),
            CaseStatusEvent(case_id=case.id, from_status="approved", to_status="closed", changed_by_id=admin.id),
            CaseStatusEvent(case_id=other.id, from_status="pending", to_status="approved", changed_by_id=admin.id),
        ]
    )
    db.commit()
    db.execute(delete(MissingCase).where(MissingCase.id == case.id))
    db.commit()
    assert db.scalars(sa.select(CaseStatusEvent.case_id)).all() == [other.id]


# ------------------------------------------------------------------ upgrade and downgrade
def test_migration_0005_downgrades_and_upgrades_cleanly(alembic_cfg):
    try:
        command.downgrade(alembic_cfg, "0004")
        assert _current_revision() == "0004"
        inspector = inspect(engine)
        assert "case_status_events" not in inspector.get_table_names()
        assert {"case_photos", "missing_cases", "users"} <= set(inspector.get_table_names())    # earlier stages untouched
        columns = {column["name"] for column in inspector.get_columns("missing_cases")}
        assert columns.isdisjoint({"review_reason", "reviewed_by_id", "reviewed_at"})
        assert CK_REJECTED_NEEDS_REASON not in {c["name"] for c in inspector.get_check_constraints("missing_cases")}
        assert "fk_missing_cases_reviewed_by_id_users" not in {
            fk["name"] for fk in inspector.get_foreign_keys("missing_cases")
        }
    finally:
        command.upgrade(alembic_cfg, "head")                       # always leave the test database at head
    assert _current_revision() == HEAD
    assert "case_status_events" in inspect(engine).get_table_names()


def test_upgrading_fixes_a_case_that_was_set_to_rejected_by_hand_before_0005(alembic_cfg, clean_db):
    user_id, case_id = uuid.uuid4(), uuid.uuid4()
    try:
        command.downgrade(alembic_cfg, "0004")
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO users (id, email, full_name, hashed_password) VALUES (:id, :email, 'Old Reporter', 'x')"),
                {"id": user_id, "email": "old.reporter@example.com"},
            )
            connection.execute(
                text(
                    "INSERT INTO missing_cases (id, reporter_id, name, age, gender, contact_phone, city, state, "
                    "address, last_seen_location, last_seen_date, status) VALUES (:id, :reporter, 'Old Case', 12, "
                    "'female', '+15550100199', 'Sampletown', 'Samplestate', '1 Sample Street', 'Sample Market', "
                    "'2026-01-15', 'rejected')"
                ),
                {"id": case_id, "reporter": user_id},
            )
    finally:
        command.upgrade(alembic_cfg, "head")                       # must not fail on the hand-made rejected row
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT status, review_reason FROM missing_cases WHERE id = :id"), {"id": case_id}
            ).one()
        assert row.status == "rejected"
        assert row.review_reason                                    # a neutral placeholder, so the CHECK holds
    finally:
        with engine.begin() as connection:
            connection.execute(text('TRUNCATE TABLE "case_status_events", "case_photos", "missing_cases", "users" CASCADE'))
