# Security and privacy

## Threat model

The service handles sensitive reports and face images. The principal risks are account takeover, privilege escalation, cross-user case access, unauthorised photo access, oversized/malformed uploads, abuse of login/recognition endpoints, host/origin spoofing, leakage through logs, and accidental retention after a privacy deletion request.

## Access controls

- Passwords are hashed; plaintext passwords are never stored.
- Access tokens identify a user, but the API reloads that user from PostgreSQL on each request. `is_active` and `role` are checked in the database, so deactivation immediately invalidates existing tokens.
- Public registration cannot assign an admin role.
- Admin routes require an active database-backed admin account.
- Case-owner routes scope queries to the authenticated reporter; unknown and other-owner cases are intentionally indistinguishable.
- An admin cannot change their own role/deactivate themselves. The last active admin cannot be removed from the active-admin set.

## HTTP protections

Stage 8A adds a consistent error envelope, request IDs, validation errors that omit rejected values, generic 500 responses, security headers, trusted-host checks, explicit CORS allow-lists, a global streamed body cap, and rate limits. The limiter is per process; multiple workers require Redis-backed shared state. Production startup refuses placeholder secrets, wildcard CORS, and enabled API docs.

## Privacy decisions

- Original client filenames are discarded. Stored image keys are server-generated and are not returned by the API.
- Images are normalised to JPEG with metadata removed before storage/recognition.
- Found-person report images and search results are restricted to admins; submitters can see only their own report status. Admins should delete reports when no longer needed.
- Face embeddings and query images are never returned or written to search logs.
- Recognition results are leads, not identity proof. Human review is mandatory.
- `audit_events.details` is restricted to non-negative counts, booleans, and short codes. Do not add names, emails, addresses, free-form notes, image data, embeddings, tokens, or credentials.
- The Stage 5 status-event table remains the detailed case-status history; Stage 8B adds only an audit action marker, avoiding duplication of the review reason.
- Hard deletion permanently removes the case, its photos, embeddings, detections, and status events. Search-log references are nulled by the existing FK. Audit records survive as actor/action/target UUID/count metadata; they do not retain case contents.

## What is logged

Application logs may contain request ID, HTTP method/path/status/duration, opaque database IDs, event codes, and aggregate counts. They must not contain names, email addresses, phone numbers, street/location details, review text, original filenames, storage keys/paths, photo bytes, embeddings, access tokens, or passwords. A hard-delete storage failure is logged with the photo UUID only.

## Retention

The application does not currently run automatic retention/purge jobs for audit events or search logs. Operators must define and document a lawful retention period, restrict database/log access, and implement a reviewed retention process before production. Do not describe indefinite retention as a legal requirement; it is only the current software behavior.

## Data requests and deletion

1. Verify the requester's authority and applicable legal basis outside the API.
2. An administrator may permanently delete a case using `DELETE /admin/cases/{case_id}?confirm=true`.
3. The response reports database row counts and any photo IDs whose stored files could not be deleted. A failed file deletion is not silently treated as success; investigate it with the storage operator.
4. The case and dependent database rows are not recoverable through the API after commit. Ensure backups and external storage replicas follow the organisation's separate retention/deletion policy.
5. Audit metadata remains for accountability, but the case target has no FK and the case contents are not copied into audit details.
6. Permanently delete a found-person report with `DELETE /admin/found-person-reports/{report_id}?confirm=true`. Check `photo_deleted`; a false value requires storage cleanup.

## Face recognition backend (optional FaceNet, Stage 6B)

- **Supply chain.** The weights file is downloaded once, explicitly, by an operator (`scripts/download_recognition_model.py`),
  never by the API. The script prints the size and SHA-256 of what it saved and hard-codes no checksum: record the value
  and compare it when the file is copied. The API loads the file with `weights_only=True`, so a tampered file cannot run
  code through pickle; a wrong or corrupt file is refused (`weights_invalid`) and recognition answers 503.
- **Install.** `facenet-pytorch==2.6.0` is installed with `--no-deps` because its metadata conflicts with the project's
  numpy and Pillow pins (docs/recognition.md). That bypasses pip's resolver for one package: keep all five optional packages pinned, install
  only from PyPI, and re-test on the target machine. `pip check` will keep reporting the stale bounds.
- **File permissions.** Keep `RECOGNITION_MODEL_DIR` readable by the API account only and writable by nobody while it runs;
  keep it, and any evaluation dataset, out of source control (`models/recognition/`, `evaluation-datasets/`, `*.pt` are git-ignored).
- **Licence and consent.** The weights derive from the VGGFace2 dataset: review its terms before any real deployment.
  Evaluate only on public data whose licence allows it or on photos with explicit informed consent; never on real case photos.
- **Resource abuse.** Inference is CPU-heavy. Only admins reach enrolment and search, both are rate limited, and inference is
  serialised per process; images are bounded by the upload limits and shrunk before detection. Each worker process holds its
  own models.
- **Logs and errors.** Recognition logs contain reason codes (`weights_missing`, `weights_invalid`, `packages_missing`,
  `inference` error type names) and ids/counts only. Paths, third-party exception text, image bytes and embeddings never
  reach logs or API responses (`tests/test_canary_logging.py`). The 503 body is generic.
- **Scripts print counts only.** `reembed_cases` and `evaluate_recognition` never print names, ids, file names or paths.
- **Results are leads, not proof.** A score is not a probability, thresholds are not validated until calibrated on your own
  data, accuracy can differ between groups of people, and a human must review every candidate (docs/recognition.md).

## Reporting a security issue

Do not include real people's records, images, passwords, or tokens in a public issue. Report suspected vulnerabilities privately to the system owner/security contact designated by the deploying organisation. Preserve only the minimum technical evidence needed to reproduce the issue, redact personal data, and rotate any exposed credentials immediately.

## Known limitations

- Recognition is unconfigured by default. The optional FaceNet CPU backend requires separately installed dependencies,
  locally downloaded weights and a licence review before real deployment.
- The rate limiter is process-local; it is not suitable for horizontally scaled/multi-worker deployments without Redis.
- Local file storage is a single-host backend, not a replicated object store.
- Hard deletion can commit while one or more file deletions fail; those photo IDs are reported, but an operator-controlled storage cleanup procedure is required.
- No automatic audit/search-log retention scheduler or legal-basis/consent workflow is implemented.
- Found-person report images are retained until an admin deletes the report; no automatic report-retention job is implemented.
- Face search is brute-force and is not calibrated as an identity probability. FaceNet results are leads only and
  require human review.
- Face-recognition performance may vary with age gaps, lighting, pose, masks, image quality and across demographic
  groups. Evaluate subgroup performance on representative, lawfully collected and consented data.
- Similarity thresholds are dataset- and use-case-dependent. Calibrate false-accept/false-reject trade-offs on the
  deploying organisation's own evaluation data; never use a score as proof of identity.
- FaceNet weights are downloaded explicitly by an operator and loaded from a local directory. The API does not
  download model files at runtime. Review the VGGFace2 dataset/model terms before any real deployment.
