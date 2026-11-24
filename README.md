# Missing Person API

A Python 3.11 / FastAPI backend for case registration, administrator review, photo storage, face-enrolment/search interfaces, and detection history. The recognition backend is deliberately unconfigured by default: no embeddings, matches, or identities are fabricated. An optional CPU FaceNet backend (local weights, explicit one-time download) is described in [docs/recognition.md](docs/recognition.md).

> **Privacy and safety:** face-search results are leads, not proof of identity. An administrator must review a result before taking action. Use only with a lawful basis, clear notice, strict access control, and an approved retention policy.

## Requirements

- Windows 10/11 or another supported OS
- Python **3.11** (use the 64-bit installer)
- PostgreSQL 14 or newer
- Git (optional)

## Windows PowerShell setup

```powershell
# From the project folder
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation for the current terminal, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Edit `.env`. Set `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, and `POSTGRES_DB` to your local PostgreSQL values. Generate a signing key:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Put the generated value in `SECRET_KEY`. Never commit `.env`. The provided `.env.example` is a template, not a deployable secret configuration.

## PostgreSQL

Create a database and a least-privilege application user using pgAdmin or `psql`. Example (run as a PostgreSQL administrator and replace the example password):

```sql
CREATE ROLE missing_person_app LOGIN PASSWORD 'replace-with-a-strong-password';
CREATE DATABASE missing_person OWNER missing_person_app;
```

For tests, create a separate database whose name ends in `_test`, for example `missing_person_test`. The test fixture refuses to run against a database without that suffix and uses a temporary storage directory.

## Migrations

All schema changes are Alembic migrations. Do **not** use `Base.metadata.create_all()`.

```powershell
alembic upgrade head
alembic current
alembic history
```

Stage 8B adds migration `0008` after `0007`; it creates `audit_events`. Existing migrations are not rewritten.

## Run the API

```powershell
uvicorn app.main:app --reload
```

- Health: `http://127.0.0.1:8000/health`
- API docs in development: `http://127.0.0.1:8000/docs`
- Register: `POST /auth/register`; login: `POST /auth/login`

Create the first administrator using the controlled script (not public registration):

```powershell
python -m scripts.create_admin --email admin@example.com --full-name "System Administrator"
```

Follow the prompts and protect the resulting credentials. Public registration always creates a normal `user`.

## Tests and coverage

Use a separate PostgreSQL test database and set `POSTGRES_DB=missing_person_test` in the environment before running tests. `tests/conftest.py` selects a throwaway storage directory and refuses a non-test database.

```powershell
pytest
pytest --cov=app --cov-report=term-missing
```

The migration tests use the real Alembic chain and exercise upgrade/downgrade. Never point tests at a development or production database.

## Lint

```powershell
ruff check .
```

The Ruff configuration is in `pyproject.toml`; the pinned developer dependency is in `requirements-dev.txt`.

## Deployment notes

- `APP_ENV=production` requires real `SECRET_KEY` and `POSTGRES_PASSWORD`, explicit CORS origins (no `*`), and `ENABLE_API_DOCS=false`.
- Set `TRUSTED_HOSTS` to the actual host names and `CORS_ALLOWED_ORIGINS` to the exact browser origins that need access.
- The Stage 8A rate limiter is in-memory and **per process**. Use one worker or replace it with a shared Redis limiter before running multiple workers.
- Configure TLS at the reverse proxy. HSTS is emitted only in production mode.
- See [architecture](docs/architecture.md), [API reference](docs/api.md), [security](docs/security.md), [recognition](docs/recognition.md), and [hardening](docs/hardening.md).
