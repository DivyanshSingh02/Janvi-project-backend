import uuid
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import delete, inspect
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal, engine
from app.models.detection import Detection
from app.models.case import MissingCase
from app.models.user import User

HEAD = "0008"


def _revision():
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def test_migration_chain_ends_at_0008(alembic_cfg):
    script = ScriptDirectory.from_config(alembic_cfg)
    assert script.get_heads() == [HEAD]
    assert script.get_revision("0008").down_revision == "0007"
    assert script.get_revision("0007").down_revision == "0006"
    assert _revision() == HEAD


def test_detections_table_columns_and_types():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("detections")}
    assert set(columns) == {
        "id", "case_id", "detected_at", "source", "camera_id", "location_text", "latitude", "longitude",
        "confidence", "frame_reference", "review_status", "reviewed_by_id", "reviewed_at", "review_note",
        "recorded_by_id", "created_at",
    }
    assert columns["detected_at"]["type"].timezone is True
    assert columns["created_at"]["type"].timezone is True
    assert columns["reviewed_at"]["type"].timezone is True
    assert columns["camera_id"]["type"].length == 64
    assert columns["location_text"]["type"].length == 200
    assert columns["frame_reference"]["type"].length == 255
    assert "unreviewed" in str(columns["review_status"]["default"])


def test_detection_constraints_foreign_keys_and_indexes_are_named():
    inspector = inspect(engine)
    checks = {c["name"] for c in inspector.get_check_constraints("detections")}
    assert checks == {
        "ck_detections_source_valid", "ck_detections_confidence_range", "ck_detections_coordinates_pair",
        "ck_detections_latitude_range", "ck_detections_longitude_range", "ck_detections_location_present",
        "ck_detections_camera_source_requires_camera_id", "ck_detections_review_status_valid",
    }
    fks = {fk["name"]: fk for fk in inspector.get_foreign_keys("detections")}
    assert set(fks) == {
        "fk_detections_case_id_missing_cases", "fk_detections_reviewed_by_id_users",
        "fk_detections_recorded_by_id_users",
    }
    assert all(fk["options"].get("ondelete") == "RESTRICT" for fk in fks.values())
    indexes = {i["name"]: i for i in inspector.get_indexes("detections")}
    assert set(indexes) == {
        "ix_detections_case_id_detected_at", "ix_detections_detected_at", "ix_detections_review_status",
    }
    assert indexes["ix_detections_review_status"]["column_names"] == ["review_status"]
    with engine.connect() as connection:
        definitions = dict(connection.execute(sa.text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'detections'"
        )).all())
    assert "detected_at DESC" in definitions["ix_detections_case_id_detected_at"]
    assert "detected_at DESC" in definitions["ix_detections_detected_at"]


def test_model_and_database_constraint_names_match():
    table = Detection.__table__
    inspector = inspect(engine)
    db_names = (
        {inspector.get_pk_constraint("detections")["name"]}
        | {c["name"] for c in inspector.get_check_constraints("detections")}
        | {fk["name"] for fk in inspector.get_foreign_keys("detections")}
    )
    assert {str(c.name) for c in table.constraints if c.name} == db_names
    assert {i.name for i in table.indexes} == {i["name"] for i in inspector.get_indexes("detections")}


def test_database_enforces_detection_checks(make_user, make_case):
    admin = make_user(email="admin@example.com", role="admin")
    case = make_case(make_user(email="reporter@example.com"))
    def insert(**overrides):
        fields = {
            "case_id": case.id, "detected_at": datetime.now(timezone.utc), "source": "manual",
            "location_text": "Fictional Location", "confidence": 0.5, "recorded_by_id": admin.id,
        }
        fields.update(overrides)
        with SessionLocal() as session:
            session.add(Detection(**fields))
            with pytest.raises(IntegrityError) as error:
                session.commit()
            session.rollback()
        return str(error.value)
    assert "ck_detections_source_valid" in insert(source="other")
    assert "ck_detections_confidence_range" in insert(confidence=-0.1)
    assert "ck_detections_confidence_range" in insert(confidence=1.1)
    assert "ck_detections_confidence_range" in insert(confidence=float("nan"))
    assert "ck_detections_coordinates_pair" in insert(latitude=10.0)
    assert "ck_detections_coordinates_pair" in insert(longitude=10.0)
    assert "ck_detections_latitude_range" in insert(latitude=91.0, longitude=10.0)
    assert "ck_detections_longitude_range" in insert(latitude=10.0, longitude=181.0)
    assert "ck_detections_location_present" in insert(location_text=None)
    assert "ck_detections_camera_source_requires_camera_id" in insert(source="camera", camera_id=None)
    assert "ck_detections_review_status_valid" in insert(review_status="unknown")


def test_detection_foreign_keys_and_restrict_delete(make_user, make_case, db):
    reporter = make_user(email="reporter@example.com")
    admin = make_user(email="admin@example.com", role="admin")
    case = make_case(reporter)
    row = Detection(
        case_id=case.id, detected_at=datetime.now(timezone.utc), source="manual",
        location_text="Fictional Location", confidence=0.5, recorded_by_id=admin.id,
    )
    db.add(row)
    db.commit()
    with pytest.raises(IntegrityError) as recorded_error:
        db.execute(delete(User).where(User.id == admin.id))
        db.commit()
    db.rollback()
    assert "fk_detections_recorded_by_id_users" in str(recorded_error.value)
    with pytest.raises(IntegrityError) as case_error:
        db.execute(delete(MissingCase).where(MissingCase.id == case.id))
        db.commit()
    db.rollback()
    assert "fk_detections_case_id_missing_cases" in str(case_error.value)



def test_database_enforces_detection_foreign_keys(make_user, make_case):
    admin = make_user(email="admin@example.com", role="admin")
    case = make_case(make_user(email="reporter@example.com"))

    def attempt(**overrides):
        values = {
            "case_id": case.id,
            "detected_at": datetime.now(timezone.utc),
            "source": "manual",
            "location_text": "Fictional Location",
            "confidence": 0.5,
            "recorded_by_id": admin.id,
        }
        values.update(overrides)
        with SessionLocal() as session:
            session.add(Detection(**values))
            with pytest.raises(IntegrityError) as error:
                session.commit()
            session.rollback()
        return str(error.value)

    assert "fk_detections_case_id_missing_cases" in attempt(case_id=uuid.uuid4())
    assert "fk_detections_recorded_by_id_users" in attempt(recorded_by_id=uuid.uuid4())
    assert "fk_detections_reviewed_by_id_users" in attempt(
        reviewed_by_id=uuid.uuid4(), reviewed_at=datetime.now(timezone.utc)
    )

def test_migration_0007_downgrades_and_upgrades_cleanly(alembic_cfg):
    try:
        command.downgrade(alembic_cfg, "0006")
        assert _revision() == "0006"
        assert "detections" not in inspect(engine).get_table_names()
        assert "audit_events" not in inspect(engine).get_table_names()
        assert {"users", "missing_cases", "case_photos", "face_embeddings"} <= set(inspect(engine).get_table_names())
    finally:
        command.upgrade(alembic_cfg, "head")
    assert _revision() == HEAD
    assert "detections" in inspect(engine).get_table_names()
