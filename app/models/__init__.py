"""Import every model module here so Alembic can see it on Base.metadata."""
from app.db.base import Base  # noqa: F401
from app.models.case import MissingCase  # noqa: F401
from app.models.case_status_event import CaseStatusEvent  # noqa: F401
from app.models.face_embedding import FaceEmbedding  # noqa: F401
from app.models.face_search_log import FaceSearchLog  # noqa: F401
from app.models.photo import CasePhoto  # noqa: F401
from app.models.user import User  # noqa: F401

from app.models.detection import Detection  # noqa: F401

from app.models.audit_event import AuditEvent  # noqa: F401
from app.models.found_person_report import FoundPersonReport  # noqa: F401
