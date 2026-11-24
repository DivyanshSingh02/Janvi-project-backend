import logging
from uuid import UUID

from fastapi import APIRouter, Request

from app.api.deps import AdminUser, DbSession
from app.api.rate_limits import limit_admin
from app.api.routes.photos import StorageDep
from app.recognition import RecognitionDep
from app.schemas.embeddings import EnrolmentRead
from app.services import embeddings as embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.post("/cases/{case_id}/embeddings", response_model=EnrolmentRead)
def generate_embeddings(case_id: UUID, admin: AdminUser, db: DbSession, recognition: RecognitionDep,
                        storage: StorageDep, request: Request):
    """Admin only. Processes every photo of an APPROVED case that is pending, failed, or only has embeddings from
    another model version. A photo without exactly one face is marked failed while the others carry on. Safe to
    repeat: nothing is duplicated. With no recognition model it answers 503 and changes nothing."""
    limit_admin(request, admin.id, embeddings=True)
    result = embedding_service.enrol_case(db, storage, recognition, case_id)
    logger.info(
        "embeddings_generated",
        extra={
            "case_id": str(case_id),
            "admin_id": str(admin.id),
            "generated": result.summary.generated,
            "pending": result.summary.pending,
            "failed": result.summary.failed,
        },
    )
    return EnrolmentRead.from_result(result)
