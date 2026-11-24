from datetime import date
from uuid import UUID

from pydantic import BaseModel

from app.models.case import CaseStatus, Gender
from app.schemas.embeddings import ModelRead
from app.services.search import DISCLAIMER, SearchOutcome


class CandidateCase(BaseModel):
    """The few case fields a reviewer needs to follow up a lead. No contact details, no address."""

    id: UUID
    name: str
    age: int
    gender: Gender
    city: str
    state: str
    last_seen_date: date
    status: CaseStatus


class SearchResultRead(BaseModel):
    rank: int
    similarity: float              # the raw cosine similarity rounded to 4 decimals - NOT a probability or a confidence
    similarity_percent: float      # similarity * 100 for display; still NOT a probability or a confidence
    matched_photo_id: UUID
    case: CandidateCase


class SearchResponse(BaseModel):
    disclaimer: str
    model: ModelRead
    candidates_considered: int
    stale_embeddings_skipped: int
    results: list[SearchResultRead]

    @classmethod
    def from_outcome(cls, outcome: SearchOutcome) -> "SearchResponse":
        return cls(
            disclaimer=DISCLAIMER,
            model=ModelRead(name=outcome.model.name, version=outcome.model.version),
            candidates_considered=outcome.candidates_considered,
            stale_embeddings_skipped=outcome.stale_embeddings_skipped,
            results=[
                SearchResultRead(
                    rank=candidate.rank,
                    similarity=candidate.similarity,
                    similarity_percent=round(candidate.similarity * 100, 2),
                    matched_photo_id=candidate.matched_photo_id,
                    case=CandidateCase(
                        id=candidate.case.id,
                        name=candidate.case.name,
                        age=candidate.case.age,
                        gender=candidate.case.gender,
                        city=candidate.case.city,
                        state=candidate.case.state,
                        last_seen_date=candidate.case.last_seen_date,
                        status=candidate.case.status,
                    ),
                )
                for candidate in outcome.candidates
            ],
        )
