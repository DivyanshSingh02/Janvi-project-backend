from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AuditEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    actor_id: UUID
    action: str
    target_type: str
    target_id: UUID | None
    details: dict
    created_at: datetime


class SearchLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    admin_id: UUID
    searched_at: datetime
    result_count: int
    top_case_id: UUID | None
    top_similarity: float | None
    model_name: str
    model_version: str
