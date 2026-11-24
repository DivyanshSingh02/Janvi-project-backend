"""Re-enrol the photos of ALL APPROVED cases with the currently configured recognition model.

Run by an administrator, on the server, from the project root, after switching the recognition backend or its weights:

    python -m scripts.reembed_cases

It goes through the existing embedding service (app/services/embeddings.py), so the rules are the same as for
POST /admin/cases/{id}/embeddings: exactly one face per photo, one transaction per case, photos that already have an
embedding from the current model name and version are skipped (so running it twice is harmless), and photos with
embeddings from another model version get new ones that REPLACE the old ones.

Output is counts only, never a name, an id or a file name:

    cases=3 cases_skipped=0 cases_failed=0 photos_done=7 photos_failed=1 photos_pending=0

photos_* describe the state of the processed cases' photos AFTER the run (done = has an embedding from the current
model; failed = no usable single face or file missing; pending = not processed). Exit code 0 when no case failed,
1 when the model is unavailable or at least one case could not be processed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from sqlalchemy import select

from app.models.case import CaseStatus, MissingCase
from app.recognition import RecognitionUnavailableError, get_recognition_service
from app.services import cases as case_service
from app.services import embeddings as embedding_service
from app.storage import get_storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReembedCounts:
    cases: int = 0              # approved cases found
    cases_skipped: int = 0      # changed (or vanished) while the script ran
    cases_failed: int = 0       # could not be processed (storage problem or unexpected error); nothing saved for them
    photos_done: int = 0
    photos_failed: int = 0
    photos_pending: int = 0

    def line(self) -> str:
        return (
            f"cases={self.cases} cases_skipped={self.cases_skipped} cases_failed={self.cases_failed} "
            f"photos_done={self.photos_done} photos_failed={self.photos_failed} photos_pending={self.photos_pending}"
        )


def reembed_all_approved(db, storage, recognition) -> ReembedCounts:
    """Raises RecognitionUnavailableError when the model cannot be used (the whole run stops: nothing would work).
    A problem with ONE case is counted and the run carries on with the next."""
    case_ids = list(
        db.scalars(
            select(MissingCase.id)
            .where(MissingCase.status == CaseStatus.APPROVED.value)
            .order_by(MissingCase.created_at, MissingCase.id)
        )
    )
    skipped = failed = done = photos_failed = pending = 0
    for case_id in case_ids:
        try:
            result = embedding_service.enrol_case(db, storage, recognition, case_id)
        except RecognitionUnavailableError:
            raise
        except (case_service.CaseNotFoundError, embedding_service.CaseNotApprovedError):
            skipped += 1
            continue
        except Exception as exc:
            failed += 1
            logger.warning("reembed_case_failed", extra={"case_id": str(case_id), "reason": type(exc).__name__})
            continue
        done += result.summary.generated
        photos_failed += result.summary.failed
        pending += result.summary.pending
    return ReembedCounts(len(case_ids), skipped, failed, done, photos_failed, pending)


def _session_context():
    from app.db.session import SessionLocal

    return SessionLocal()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reembed_cases",
        description="Re-enrol all approved cases with the configured recognition model. Prints counts only.",
    )
    parser.parse_args(argv)
    try:
        recognition = get_recognition_service()
        recognition.model_info()                      # fail early and clearly when there is no usable model
    except RecognitionUnavailableError:
        print("Error: the configured recognition model is unavailable. Check RECOGNITION_BACKEND, the weights "
              "and the installed packages (docs/recognition.md).")
        return 1
    try:
        with _session_context() as db:
            counts = reembed_all_approved(db, get_storage(), recognition)
    except RecognitionUnavailableError:
        print("Error: the recognition model became unavailable during the run. Counts so far were not printed; "
              "run the script again (finished cases are skipped).")
        return 1
    except Exception:
        print("Error: re-embedding could not be completed.")
        return 1
    print(counts.line())
    return 1 if counts.cases_failed else 0


if __name__ == "__main__":
    sys.exit(main())
