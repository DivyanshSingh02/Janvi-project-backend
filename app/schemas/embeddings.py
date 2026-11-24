from uuid import UUID

from pydantic import BaseModel

from app.services.embeddings import EmbeddingSummary, EnrolmentResult


class ModelRead(BaseModel):
    name: str
    version: str


class EmbeddingSummaryRead(BaseModel):
    """How many of a case's photos have an embedding (generated), still wait for one (pending) or could not get
    one (failed). The vectors themselves are never part of any response."""

    generated: int
    pending: int
    failed: int

    @classmethod
    def from_summary(cls, summary: EmbeddingSummary) -> "EmbeddingSummaryRead":
        return cls(generated=summary.generated, pending=summary.pending, failed=summary.failed)


class PhotoEmbeddingRead(BaseModel):
    photo_id: UUID
    status: str                  # pending | done | failed
    error: str | None            # a short code such as no_face, multiple_faces, file_missing


class EnrolmentRead(EmbeddingSummaryRead):
    case_id: UUID
    model: ModelRead
    photos: list[PhotoEmbeddingRead]

    @classmethod
    def from_result(cls, result: EnrolmentResult) -> "EnrolmentRead":
        return cls(
            generated=result.summary.generated,
            pending=result.summary.pending,
            failed=result.summary.failed,
            case_id=result.case_id,
            model=ModelRead(name=result.model_name, version=result.model_version),
            photos=[PhotoEmbeddingRead(photo_id=p.photo_id, status=p.status, error=p.error) for p in result.photos],
        )
