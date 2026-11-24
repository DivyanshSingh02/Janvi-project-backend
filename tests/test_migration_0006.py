"""Migration 0006: embedding status on case_photos, face_embeddings and face_search_logs."""
import uuid

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal, engine
from app.models.face_embedding import FaceEmbedding
from app.models.face_search_log import FaceSearchLog
from app.models.case import MissingCase
from app.models.photo import CasePhoto
from app.models.user import User

HEAD = "0008"


def _current_revision() -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def _commit_error(*objects) -> str:
    """Add the objects in a fresh session, expect the commit to be refused, return the error text."""
    with SessionLocal() as session:
        session.add_all(objects)
        with pytest.raises(IntegrityError) as exc_info:
            session.commit()
        session.rollback()
    return str(exc_info.value)


def _embedding(photo, **overrides) -> FaceEmbedding:
    fields = {
        "photo_id": photo.id,
        "case_id": photo.case_id,
        "model_name": "fake-face-model",
        "model_version": "1",
        "dimension": 4,
        "vector": b"\x00" * 16,
    }
    fields.update(overrides)
    return FaceEmbedding(**fields)


@pytest.fixture()
def photo(make_user, make_case, make_photos):
    owner = make_user(email="reporter@example.com")
    return make_photos(owner, make_case(owner), 1)[0]


# ------------------------------------------------------------------ the chain
def test_head_revision_is_0008_and_0006_follows_0005(alembic_cfg):
    script = ScriptDirectory.from_config(alembic_cfg)
    assert script.get_heads() == [HEAD]
    assert script.get_revision("0008").down_revision == "0007"
    assert script.get_revision("0007").down_revision == "0006"
    assert script.get_revision("0006").down_revision == "0005"
    assert _current_revision() == HEAD


# ------------------------------------------------------------------ structure
def test_case_photos_has_the_embedding_columns_and_a_named_check():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("case_photos")}
    assert isinstance(columns["embedding_status"]["type"], sa.String) and columns["embedding_status"]["type"].length == 20
    assert columns["embedding_status"]["nullable"] is False
    assert "pending" in str(columns["embedding_status"]["default"])
    assert isinstance(columns["embedding_error"]["type"], sa.String) and columns["embedding_error"]["type"].length == 50
    assert columns["embedding_error"]["nullable"] is True
    assert "ck_case_photos_embedding_status_valid" in {c["name"] for c in inspector.get_check_constraints("case_photos")}


def test_face_embeddings_has_the_expected_columns_constraints_and_index():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("face_embeddings")}
    assert set(columns) == {
        "id", "photo_id", "case_id", "model_name", "model_version", "dimension", "vector", "created_at",
    }
    assert all(column["nullable"] is False for column in columns.values())
    assert isinstance(columns["vector"]["type"], sa.LargeBinary)
    assert inspector.get_pk_constraint("face_embeddings")["name"] == "pk_face_embeddings"
    assert [c["name"] for c in inspector.get_unique_constraints("face_embeddings")] == [
        "uq_face_embeddings_photo_model_version"
    ]
    assert {c["name"] for c in inspector.get_check_constraints("face_embeddings")} == {
        "ck_face_embeddings_dimension_positive",
        "ck_face_embeddings_vector_length_matches_dimension",
    }
    foreign_keys = {fk["name"]: fk for fk in inspector.get_foreign_keys("face_embeddings")}
    assert set(foreign_keys) == {"fk_face_embeddings_photo_id_case_photos", "fk_face_embeddings_case_id_missing_cases"}
    assert foreign_keys["fk_face_embeddings_photo_id_case_photos"]["referred_table"] == "case_photos"
    assert foreign_keys["fk_face_embeddings_case_id_missing_cases"]["referred_table"] == "missing_cases"
    for foreign_key in foreign_keys.values():
        assert foreign_key["options"].get("ondelete") == "CASCADE"
    plain_indexes = [i for i in inspector.get_indexes("face_embeddings") if "duplicates_constraint" not in i]
    assert [(i["name"], i["column_names"]) for i in plain_indexes] == [("ix_face_embeddings_case_id", ["case_id"])]


def test_face_search_logs_has_the_expected_columns_and_foreign_keys():
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("face_search_logs")}
    assert set(columns) == {
        "id", "admin_id", "searched_at", "result_count", "top_case_id", "top_similarity", "model_name", "model_version",
    }
    assert {name for name, column in columns.items() if column["nullable"]} == {"top_case_id", "top_similarity"}
    assert inspector.get_pk_constraint("face_search_logs")["name"] == "pk_face_search_logs"
    assert {c["name"] for c in inspector.get_check_constraints("face_search_logs")} == {
        "ck_face_search_logs_result_count_non_negative"
    }
    foreign_keys = {fk["name"]: fk for fk in inspector.get_foreign_keys("face_search_logs")}
    assert set(foreign_keys) == {"fk_face_search_logs_admin_id_users", "fk_face_search_logs_top_case_id_missing_cases"}
    assert foreign_keys["fk_face_search_logs_admin_id_users"]["options"].get("ondelete") == "RESTRICT"
    assert foreign_keys["fk_face_search_logs_top_case_id_missing_cases"]["options"].get("ondelete") == "SET NULL"


@pytest.mark.parametrize("model", [CasePhoto, FaceEmbedding, FaceSearchLog], ids=["case_photos", "face_embeddings", "face_search_logs"])
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
def test_the_vector_length_must_be_dimension_times_four_bytes(photo):
    assert "ck_face_embeddings_vector_length_matches_dimension" in _commit_error(_embedding(photo, vector=b"\x00" * 15))
    assert "ck_face_embeddings_vector_length_matches_dimension" in _commit_error(_embedding(photo, vector=b"\x00" * 17))
    assert "ck_face_embeddings_vector_length_matches_dimension" in _commit_error(
        _embedding(photo, dimension=8, vector=b"\x00" * 16)
    )
    assert "ck_face_embeddings_dimension_positive" in _commit_error(_embedding(photo, dimension=0, vector=b""))
    with SessionLocal() as session:                                              # the right length is accepted
        session.add(_embedding(photo, dimension=8, vector=b"\x00" * 32))
        session.commit()


def test_one_embedding_per_photo_and_model_version(photo):
    with SessionLocal() as session:
        session.add(_embedding(photo))
        session.commit()
    assert "uq_face_embeddings_photo_model_version" in _commit_error(_embedding(photo))
    with SessionLocal() as session:                                              # another version (or name) may coexist
        session.add_all([_embedding(photo, model_version="2"), _embedding(photo, model_name="another-model")])
        session.commit()


def test_the_embedding_status_must_be_one_of_the_three_values(photo, db):
    assert (photo.embedding_status, photo.embedding_error) == ("pending", None)      # the default for a new photo
    for bad in ("bogus", "Done", ""):
        db.get(CasePhoto, photo.id).embedding_status = bad
        with pytest.raises(IntegrityError) as exc_info:
            db.commit()
        db.rollback()
        assert "ck_case_photos_embedding_status_valid" in str(exc_info.value), bad
    for good in ("done", "failed", "pending"):
        db.get(CasePhoto, photo.id).embedding_status = good
        db.commit()


def test_embeddings_follow_their_photo_and_their_case_when_those_rows_are_deleted(db, make_user, make_case, make_photos):
    owner = make_user(email="reporter@example.com")
    first_case, second_case = make_case(owner), make_case(owner)
    first_photo, second_photo = make_photos(owner, first_case, 1)[0], make_photos(owner, second_case, 1, first_seed=40)[0]
    db.add_all([_embedding(first_photo), _embedding(second_photo)])
    db.commit()

    db.execute(delete(CasePhoto).where(CasePhoto.id == first_photo.id))          # the photo row goes ...
    db.commit()
    assert db.scalars(select(FaceEmbedding.photo_id)).all() == [second_photo.id]  # ... and its embedding with it
    db.execute(delete(MissingCase).where(MissingCase.id == second_case.id))       # the whole case goes ...
    db.commit()
    assert db.scalars(select(FaceEmbedding.photo_id)).all() == []                 # ... and so does the rest


def test_the_search_log_keeps_its_admin_and_forgets_a_deleted_case(db, make_user, make_case):
    admin = make_user(email="boss@example.com", role="admin")
    case = make_case(make_user(email="reporter@example.com"))
    db.add(FaceSearchLog(admin_id=admin.id, result_count=1, top_case_id=case.id, top_similarity=0.5,
                         model_name="fake-face-model", model_version="1"))
    db.commit()

    with pytest.raises(IntegrityError) as exc_info:                                # an admin with a search history stays
        db.execute(delete(User).where(User.id == admin.id))
        db.commit()
    db.rollback()
    assert "fk_face_search_logs_admin_id_users" in str(exc_info.value)

    db.execute(delete(MissingCase).where(MissingCase.id == case.id))
    db.commit()
    row = db.scalars(select(FaceSearchLog)).one()
    assert row.top_case_id is None and row.result_count == 1                       # the row stays, the pointer is cleared


def test_a_search_log_cannot_have_a_negative_result_count(make_user):
    admin = make_user(email="boss@example.com", role="admin")
    error = _commit_error(
        FaceSearchLog(admin_id=admin.id, result_count=-1, model_name="fake-face-model", model_version="1")
    )
    assert "ck_face_search_logs_result_count_non_negative" in error


# ------------------------------------------------------------------ upgrade and downgrade
def test_migration_0006_downgrades_and_upgrades_cleanly(alembic_cfg):
    try:
        command.downgrade(alembic_cfg, "0005")
        assert _current_revision() == "0005"
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert not {"face_embeddings", "face_search_logs"} & tables
        assert {"case_photos", "missing_cases", "users", "case_status_events"} <= tables      # earlier stages untouched
        assert {c["name"] for c in inspector.get_columns("case_photos")}.isdisjoint({"embedding_status", "embedding_error"})
        assert "ck_case_photos_embedding_status_valid" not in {
            c["name"] for c in inspector.get_check_constraints("case_photos")
        }
    finally:
        command.upgrade(alembic_cfg, "head")                          # always leave the test database at head
    assert _current_revision() == HEAD
    assert {"face_embeddings", "face_search_logs"} <= set(inspect(engine).get_table_names())


def test_upgrading_marks_photos_that_already_exist_as_pending(alembic_cfg, clean_db):
    user_id, case_id, photo_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    try:
        command.downgrade(alembic_cfg, "0005")
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
                    "'2026-01-15', 'approved')"
                ),
                {"id": case_id, "reporter": user_id},
            )
            connection.execute(
                text(
                    "INSERT INTO case_photos (id, case_id, storage_key, content_type, size_bytes, width, height, sha256) "
                    "VALUES (:id, :case_id, 'cases/old/photo.jpg', 'image/jpeg', 100, 200, 200, :sha)"
                ),
                {"id": photo_id, "case_id": case_id, "sha": "a" * 64},
            )
    finally:
        command.upgrade(alembic_cfg, "head")
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT embedding_status, embedding_error FROM case_photos WHERE id = :id"), {"id": photo_id}
            ).one()
        assert (row.embedding_status, row.embedding_error) == ("pending", None)
    finally:
        with engine.begin() as connection:
            connection.execute(
                text('TRUNCATE TABLE "face_search_logs", "face_embeddings", "case_status_events", "case_photos", '
                     '"missing_cases", "users" CASCADE')
            )
