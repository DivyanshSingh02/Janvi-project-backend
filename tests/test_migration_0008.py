import uuid
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal, engine
from app.models.audit_event import AuditEvent
from app.models.user import User

HEAD = "0008"


def _revision():
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def test_migration_0008_follows_0007_and_is_head(alembic_cfg):
    script = ScriptDirectory.from_config(alembic_cfg)
    assert script.get_heads() == [HEAD]
    assert script.get_revision("0008").down_revision == "0007"
    assert _revision() == HEAD


def test_audit_events_columns_types_indexes_and_foreign_keys():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("audit_events")}
    assert set(columns) == {"id", "actor_id", "action", "target_type", "target_id", "details", "created_at"}
    assert columns["details"]["type"].__class__.__name__.lower() == "jsonb"
    assert columns["created_at"]["type"].timezone is True
    assert columns["action"]["type"].length == 64
    assert columns["target_type"]["type"].length == 40
    assert inspector.get_pk_constraint("audit_events")["name"] == "pk_audit_events"
    fks = inspector.get_foreign_keys("audit_events")
    assert len(fks) == 1
    assert fks[0]["name"] == "fk_audit_events_actor_id_users"
    assert fks[0]["options"].get("ondelete") == "RESTRICT"
    indexes = {item["name"]: item for item in inspector.get_indexes("audit_events")}
    assert set(indexes) == {"ix_audit_events_created_at", "ix_audit_events_action", "ix_audit_events_actor_id"}
    with engine.connect() as connection:
        defs = dict(connection.execute(sa.text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'audit_events'"
        )).all())
    assert "created_at DESC" in defs["ix_audit_events_created_at"]
    assert {index.name for index in AuditEvent.__table__.indexes} == set(indexes)


def test_audit_target_id_has_no_foreign_key_and_event_survives_target_deletion(db, make_user, make_case):
    from app.services import audit
    admin = make_user(email="admin@example.test", role="admin")
    owner = make_user(email="owner@example.test")
    case = make_case(owner)
    audit.record_event(db, actor_id=admin.id, action="case_hard_deleted", target_type="case", target_id=case.id,
                       details={"cases_deleted": 1})
    db.commit()
    db.delete(case)
    db.commit()
    row = db.query(AuditEvent).filter_by(action="case_hard_deleted").one()
    assert row.target_id == case.id


def test_audit_actor_foreign_key_is_restrict(db, make_user):
    actor = make_user(email="admin@example.test", role="admin")
    db.add(AuditEvent(actor_id=actor.id, action="user_deactivated", target_type="user", target_id=uuid.uuid4()))
    db.commit()
    db.delete(actor)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_migration_0008_downgrades_and_upgrades_cleanly(alembic_cfg):
    try:
        command.downgrade(alembic_cfg, "0007")
        assert _revision() == "0007"
        assert "audit_events" not in inspect(engine).get_table_names()
    finally:
        command.upgrade(alembic_cfg, "head")
    assert _revision() == HEAD
    assert "audit_events" in inspect(engine).get_table_names()
