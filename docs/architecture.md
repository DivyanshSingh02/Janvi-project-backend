# Architecture

## Runtime layers

```text
HTTP client
  -> FastAPI middleware (request ID, request-size cap, safe access log, security headers)
  -> CORS / trusted-host middleware
  -> API routers and dependencies (authentication, current-user / admin authorization)
  -> service layer (business rules, transaction boundaries, audit events)
  -> SQLAlchemy 2.0 models and PostgreSQL
  -> local storage interface for image bytes
  -> RecognitionService interface (unconfigured by default; optional CPU FaceNet backend, docs/recognition.md)
```

## Main modules

| Area | Files | Responsibility |
| --- | --- | --- |
| Configuration/security | `app/core/config.py`, `app/core/security.py` | Validated environment settings, password hashing, signed access tokens |
| HTTP controls | `app/api/middleware.py`, `app/api/errors.py`, `app/api/rate_limits.py`, `app/core/rate_limit.py` | Request IDs, standard errors, security headers, body cap, rate limits |
| Authentication | `app/api/deps.py`, `app/api/routes/auth.py` | DB-backed user identity, active-account and admin checks |
| Cases/review | `app/services/cases.py`, `app/services/review.py` | Case ownership, locking, status transitions, hard deletion |
| Photos | `app/services/photos.py`, `app/services/images.py`, `app/storage/` | Validate and normalise image files; private storage abstraction |
| Recognition | `app/recognition/`, `app/services/embeddings.py`, `app/services/search.py` | Model contract, optional FaceNet backend, enrolment, similarity search |
| Detections | `app/services/detections.py`, `app/api/routes/admin_detections.py` | Admin-recorded observation history and review |
| Audit | `app/models/audit_event.py`, `app/services/audit.py`, `app/api/routes/admin_audit.py` | Append-only admin actions and paginated audit/search history |

## Database relationships

- `users` owns `missing_cases` through `reporter_id` (`RESTRICT`).
- `missing_cases` owns `case_photos`, `face_embeddings`, `case_status_events`, and `detections` (the latter rows are explicitly deleted in FK-safe order by the privacy-delete service).
- `face_search_logs.top_case_id` uses `ON DELETE SET NULL`, so the aggregate search record remains without pointing to a deleted case.
- `audit_events.actor_id` references `users.id` with `ON DELETE RESTRICT`.
- `audit_events.target_id` intentionally has **no foreign key**, allowing the audit record to survive deletion of its target.

## Transaction boundaries

- Case status and its Stage 5 `case_status_events` row are committed together. The Stage 8B audit marker is added in the same transaction; the status event remains the detailed history source, including the review reason.
- User role/activity changes and their audit events commit together. Active administrator rows are locked before checking the last-admin invariant.
- Hard deletion locks the case, removes detections, embeddings, status events, photos, and the case in one database transaction, and commits an audit event with counts. Only after commit does it remove image files. A file-delete failure cannot restore the committed database rows; its photo ID is returned and logged.
- Search log rows contain only administrator ID, counts, top candidate reference/similarity, and model identifiers. Query images and query embeddings are not stored.

## Migrations

Alembic is the only schema-management path. Migration IDs are sequential and linear: `0001` through `0008`. Importing `app.models` registers model metadata for Alembic autogeneration and inspection; it does not create tables at runtime.
