# Stage 8A — HTTP hardening

## Rate limiting deployment note

The sliding-window limiter is in-memory and shared only by requests handled by one Python process. It is deterministic under tests through its injectable clock. **Run a single application worker when relying on this limiter. Multiple Uvicorn/Gunicorn workers require a shared Redis-backed limiter** so limits cannot be bypassed by distributing requests between processes.

Limits and windows are configured with `LOGIN_IP_LIMIT`, `LOGIN_EMAIL_LIMIT`, `LOGIN_WINDOW_SECONDS`, `REGISTRATION_IP_LIMIT`, `REGISTRATION_WINDOW_SECONDS`, `FACE_SEARCH_ADMIN_LIMIT`, `EMBEDDING_ADMIN_LIMIT`, `ADMIN_RECOGNITION_WINDOW_SECONDS`, `PHOTO_UPLOAD_USER_LIMIT`, and `PHOTO_UPLOAD_WINDOW_SECONDS`.

## Request and origin controls

`MAX_REQUEST_BODY_BYTES` caps every request body, including chunked bodies. `CORS_ALLOWED_ORIGINS` is a JSON list; an empty list disables cross-origin access. `TRUSTED_HOSTS` is a JSON list of accepted Host values. Use explicit production origins and hosts; do not use wildcards in production CORS.

## Production startup

With `APP_ENV=production`, startup fails if either credential is the placeholder `CHANGE_ME`, if CORS contains `*`, or if `ENABLE_API_DOCS=true`. Set `ENABLE_API_DOCS=false` and supply real secrets before deployment.

# Stage 8B — administrative audit and privacy operations

Migration `0008` adds `audit_events`. Its actor foreign key is `ON DELETE RESTRICT`; the target UUID has no foreign key so the audit marker survives deletion of the target. Audit details are restricted in the service helper to non-negative counts, booleans, and short machine codes.

Administrator role changes and deactivations, case review decisions, status changes, and permanent case deletion write audit events in the same transaction as the corresponding database change. Stage 5 `case_status_events` remains the source of detailed transition history and review reasons; audit rows do not duplicate free-form reasons.

`DELETE /admin/cases/{case_id}?confirm=true` removes dependent detections, embeddings, status events, photos, and the case in one transaction. It commits before removing stored files. If a file deletion fails, the response includes its photo UUID and the server log records that UUID; operators must investigate the storage orphan. Search logs survive with `top_case_id = NULL` where the deleted case was previously the top result.

The admin-only `/admin/audit-events` endpoint is paginated and filterable by action, actor, and creation date. `/admin/search-logs` exposes the existing Stage 6 aggregate search records with pagination and administrator/date filters.

Developer quality commands:

```powershell
ruff check .
pytest --cov=app --cov-report=term-missing
```
