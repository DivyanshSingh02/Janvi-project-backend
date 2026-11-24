"""Face search: one query image against the embeddings of APPROVED cases.

A result is a LEAD, never proof of identity: a human must review every candidate. The ranking is plain cosine
similarity between the query face and the stored faces made by the SAME model name and version. The API also exposes
that raw value multiplied by 100 for display; neither value is a probability or a confidence.
"""
import logging
import uuid
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.case import CaseStatus, MissingCase
from app.models.face_embedding import FaceEmbedding
from app.models.face_search_log import FaceSearchLog
from app.models.user import User
from app.recognition import ModelInfo, RecognitionService, extract_single_face, vectors
from app.services import images
from app.services import photos as photo_service

logger = logging.getLogger(__name__)

SIMILARITY_DECIMALS = 4
DISCLAIMER = "Candidates are not proof of identity. Every match must be reviewed by a human before any action is taken."


@dataclass(frozen=True)
class Candidate:
    rank: int
    similarity: float            # the raw cosine value, rounded to 4 decimals
    matched_photo_id: uuid.UUID  # the best-matching photo of that case
    case: MissingCase


@dataclass(frozen=True)
class SearchOutcome:
    model: ModelInfo
    candidates_considered: int       # stored embeddings that were compared (same model, approved cases)
    stale_embeddings_skipped: int    # embeddings of approved cases made by another model name/version
    candidates: list[Candidate]


def search_by_face(
    db: Session,
    admin: User,
    recognition: RecognitionService,
    stream: BinaryIO,
    *,
    limit: int,
    min_similarity: float | None = None,
) -> SearchOutcome:
    """The caller has already been authorised as an admin. Order of the work: validate and normalise the image
    (Stage 4 pipeline, same limits as photo uploads), then ask the recognition service. Raises the image errors,
    RecognitionUnavailableError, NoFaceDetectedError or MultipleFacesError; nothing is written in those cases."""
    # The upload limits are the same as for case photos; photo_service owns the settings -> limits mapping.
    processed = images.process_upload(stream, photo_service._limits())
    info = recognition.model_info()
    face = extract_single_face(recognition, processed.data)

    approved = MissingCase.status == CaseStatus.APPROVED.value
    rows = db.execute(
        select(FaceEmbedding.photo_id, FaceEmbedding.case_id, FaceEmbedding.dimension, FaceEmbedding.vector)
        .join(MissingCase, MissingCase.id == FaceEmbedding.case_id)
        .where(approved, FaceEmbedding.model_name == info.name, FaceEmbedding.model_version == info.version)
        .order_by(FaceEmbedding.created_at, FaceEmbedding.id)
    ).all()
    stale = db.scalar(
        select(func.count())
        .select_from(FaceEmbedding)
        .join(MissingCase, MissingCase.id == FaceEmbedding.case_id)
        .where(approved, or_(FaceEmbedding.model_name != info.name, FaceEmbedding.model_version != info.version))
    ) or 0

    usable, unpacked = [], []
    for row in rows:
        try:
            if row.dimension != info.dimension:
                raise vectors.InvalidVectorError("wrong size")
            unpacked.append(vectors.unpack(row.vector, row.dimension))
            usable.append(row)
        except vectors.InvalidVectorError:
            stale += 1                               # same model but unusable data: skipped, and counted as such
    best: dict[uuid.UUID, tuple[float, uuid.UUID]] = {}
    considered = 0
    if usable:
        similarities = vectors.cosine_similarities(np.stack(unpacked), face.embedding)
        for row, similarity in zip(usable, similarities):
            if np.isnan(similarity):
                continue
            considered += 1
            current = best.get(row.case_id)
            if current is None or similarity > current[0]:     # on an exact tie the older photo is kept
                best[row.case_id] = (float(similarity), row.photo_id)

    ranked = sorted(best.items(), key=lambda item: (-item[1][0], str(item[0])))        # best first; ties by case id
    chosen = []
    for case_id, (similarity, photo_id) in ranked:
        rounded = round(similarity, SIMILARITY_DECIMALS)
        if min_similarity is not None and rounded < min_similarity:
            continue
        chosen.append((case_id, rounded, photo_id))
        if len(chosen) == limit:
            break
    cases = {}
    if chosen:
        cases = {case.id: case for case in db.scalars(select(MissingCase).where(MissingCase.id.in_([c[0] for c in chosen])))}
    candidates = [
        Candidate(rank=rank, similarity=similarity, matched_photo_id=photo_id, case=cases[case_id])
        for rank, (case_id, similarity, photo_id) in enumerate(chosen, start=1)
    ]

    # One audit row per search: counts only. The query image and its embedding are never stored.
    top = candidates[0] if candidates else None
    db.add(
        FaceSearchLog(
            admin_id=admin.id,
            result_count=len(candidates),
            top_case_id=top.case.id if top else None,
            top_similarity=top.similarity if top else None,
            model_name=info.name,
            model_version=info.version,
        )
    )
    db.commit()
    return SearchOutcome(model=info, candidates_considered=considered, stale_embeddings_skipped=stale, candidates=candidates)
