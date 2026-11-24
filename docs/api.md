# API reference

All errors use the Stage 8A envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "Case not found",
    "request_id": "..."
  }
}
```

Validation errors additionally include `fields`, each with `field` and a safe `message`. Rejected values are never echoed. `X-Request-ID` matches `error.request_id`. Common error statuses are `400`, `401`, `403`, `404`, `409`, `413`, `415`, `422`, `429`, `500`, and `503` as applicable. Authentication failures retain `WWW-Authenticate`; rate-limit errors retain `Retry-After`.

## Authentication and system

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `GET /health` | Public | `200` | `503` when database or storage is unavailable; recognition is reported as `ready` only when a model is loaded and usable, otherwise `unconfigured`, and that alone does not fail health |
| `POST /auth/register` | Public | `201` | `409` duplicate email; `422` invalid payload; `429` registration limit |
| `POST /auth/login` | Public | `200` | `401` bad credentials; `422` malformed form; `429` IP/email limit |
| `GET /auth/me` | Active user | `200` | `401` missing/invalid/expired token or deactivated account |

## Cases

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `POST /cases` | Active user | `201` | `401`, `422` |
| `GET /cases` | Active user (own cases) | `200` | `401`, `422` pagination/filter errors |
| `GET /cases/{case_id}` | Owner | `200` | `401`, `404` (also for another user's case) |
| `GET /cases/{case_id}/detections` | Owner | `200` | `401`, `404`; only confirmed detections are returned |

## Photos

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `POST /cases/{case_id}/photos` | Owner | `201` | `401`, `404`, `409`, `413`, `415`, `422`, `429`, `503` |
| `GET /cases/{case_id}/photos` | Owner | `200` | `401`, `404` |
| `GET /cases/{case_id}/photos/{photo_id}` | Owner | JPEG bytes `200` | `401`, `404`, `503` |
| `DELETE /cases/{case_id}/photos/{photo_id}` | Owner | `204` | `401`, `404`, `409` |

## Found-person reports

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `POST /found-person-reports` | Active user; multipart image field `file` | `201` with report ID and review status only | `401`, `413`, `415`, `422`, `429`, `503` |
| `GET /found-person-reports` | Active user (own submissions only) | `200` paginated status list | `401`, `422` |
| `GET /admin/found-person-reports` | Admin | `200` paginated review queue; defaults to `review_status=pending` | `401`, `403`, `422` |
| `GET /admin/found-person-reports/{report_id}/photo` | Admin | JPEG bytes `200` | `401`, `403`, `404`, `503` |
| `POST /admin/found-person-reports/{report_id}/search` | Admin | `200` ranked candidate leads | `401`, `403`, `404`, `413`, `415`, `422`, `429`, `503` |
| `PATCH /admin/found-person-reports/{report_id}/review` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `DELETE /admin/found-person-reports/{report_id}?confirm=true` | Admin | `200` with `photo_deleted` result | `401`, `403`, `404`, `422` |

The submitting user receives no match results or case details. Admins can search the submitted photo against approved cases, review the candidate leads, and use the submitter's account contact details to follow up. Review status can move from `pending` to `reviewed`, `contacted`, or `dismissed`, and from `reviewed` to `contacted` or `dismissed`. The app does not send notifications or messages; an admin must contact the submitter outside the app. Admins can permanently delete a report and its photo after review. If `photo_deleted` is false, investigate the storage cleanup failure.

## Administrator case review and privacy deletion

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `GET /admin/cases` | Admin | `200` | `401`, `403`, `422` |
| `GET /admin/cases/{case_id}` | Admin | `200` | `401`, `403`, `404` |
| `GET /admin/cases/{case_id}/photos/{photo_id}` | Admin | JPEG bytes `200` | `401`, `403`, `404`, `503` |
| `POST /admin/cases/{case_id}/approve` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `POST /admin/cases/{case_id}/reject` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `PATCH /admin/cases/{case_id}/status` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `DELETE /admin/cases/{case_id}?confirm=true` | Admin | `200` with deletion counts and failed photo IDs | `401`, `403`, `404`, `422` if `confirm=true` is absent/false |
| `POST /admin/cases/{case_id}/embeddings` | Admin | `200` | `401`, `403`, `404`, `409`, `429`, `503` |

The permanent-delete operation commits row deletion before attempting file deletion. `failed_photo_ids` must be checked by the operator; a non-empty list is not a successful complete file purge.

## Administrator user management, audit and search history

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `GET /admin/users` | Admin | `200` | `401`, `403`, `422` |
| `PATCH /admin/users/{user_id}` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |
| `GET /admin/audit-events` | Admin | `200` paginated | `401`, `403`, `422` |
| `GET /admin/search-logs` | Admin | `200` paginated | `401`, `403`, `422` |

`PATCH /admin/users/{user_id}` accepts `{ "role": "user" | "admin", "is_active": true | false }`, with either field optional. Unknown fields and an empty update are rejected. An administrator cannot change their own role or deactivate themselves; the last active admin cannot be demoted/deactivated.

`GET /admin/audit-events` supports `action`, `actor_id`, `created_from`, `created_to`, `limit` (1..100), and `offset` (0+). `GET /admin/search-logs` supports `admin_id`, `searched_from`, `searched_to`, `limit`, and `offset`. Date filters must include a timezone and the start cannot be later than the end.

## Face search and detections

| Method / path | Role | Success | Other relevant statuses |
| --- | --- | --- | --- |
| `POST /search` | Admin | `200` ranked candidate leads | `401`, `403`, `413`, `415`, `422`, `429`, `503` |
| `POST /admin/cases/{case_id}/detections` | Admin | `201` | `401`, `403`, `404`, `409`, `422` |
| `GET /admin/cases/{case_id}/detections` | Admin | `200` paginated | `401`, `403`, `404`, `422` |
| `GET /admin/detections` | Admin | `200` paginated | `401`, `403`, `422` |
| `GET /admin/detections/{detection_id}` | Admin | `200` | `401`, `403`, `404` |
| `PATCH /admin/detections/{detection_id}/review` | Admin | `200` | `401`, `403`, `404`, `409`, `422` |

Face search and embedding generation are rate-limited per admin; photo uploads per user. A detection is an observation/lead, not a confirmed identity.

Each search result includes `similarity` (raw cosine similarity, from -1 to 1, rounded to four decimal places) and
`similarity_percent` (that value multiplied by 100 and rounded to two decimal places, for display only). For example,
`0.4252` is displayed as `42.52%` similarity. This is not a probability or confidence that the candidate is the same
person; candidates require human review.

## Query defaults

Paginated routes default to `limit=20&offset=0`. Detection lists and audit/search history cap `limit` at 100. Face search defaults to `SEARCH_DEFAULT_LIMIT` and is capped by `SEARCH_MAX_LIMIT`.
