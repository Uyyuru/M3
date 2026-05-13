"""
Service-layer contracts.

These are the seams Milestone 3 leans on from Milestones 1 & 2. Your existing
code probably already exposes equivalent functions — if names differ, only
this file needs to be re-pointed. The goal is a single import surface so the
workflow code doesn't reach into infra directly.

Replace the bodies with thin re-exports from your real modules.
"""

from __future__ import annotations

from typing import Any


# ===========================================================================
# config.py — Pydantic-settings instance with all env-driven knobs
# ===========================================================================

class Settings:  # placeholder — replace with your real BaseSettings subclass
    openai_model: str = "gpt-5.4"
    openai_timeout_s: int = 60
    clarification_cap: int = 3
    langgraph_sqlite_path: str = "./data/state/langgraph_checkpoints.sqlite"
    data_root: str = "./data"


settings = Settings()


# ===========================================================================
# vector_store.py — re-exports from your Milestone 1 ChromaDB layer
# ===========================================================================

def get_documents_collection() -> Any:
    """
    Return the ChromaDB `documents` collection.

    Required metadata fields per chunk (from Milestone 1):
        document_id, section_id, section_title, page, kind, project_id
    """
    raise NotImplementedError("Wire to your existing ChromaDB documents collection.")


def get_patterns_collection() -> Any:
    """Return the ChromaDB `patterns` collection (Milestone 2)."""
    raise NotImplementedError("Wire to your existing ChromaDB patterns collection.")


# ===========================================================================
# document_store.py — SQLite-backed document/section reads
# ===========================================================================

async def get_document_sections(
    *, project_id: str, document_id: str
) -> list[dict[str, Any]] | None:
    """
    Return the section outline for one document, or None if the document
    doesn't exist within the project.

    Each section: {section_id, section_title, page, kind}.
    The project_id check enforces 404-not-403 (no existence leaks).
    """
    raise NotImplementedError


# ===========================================================================
# auth.py — JWT validation + project-scoped authorization
# ===========================================================================

from dataclasses import dataclass


@dataclass
class AuthContext:
    user_id: str
    token_claims: dict[str, Any]


async def require_auth(*args: Any, **kwargs: Any) -> AuthContext:
    """FastAPI dependency that validates the Bearer JWT and returns the user."""
    raise NotImplementedError


async def authorize_project(auth: AuthContext, project_id: str) -> None:
    """
    Raise HTTPException(404) if the project doesn't exist OR doesn't belong
    to this user. 404 (not 403) by design — see Project-Centric Resource Model.
    """
    raise NotImplementedError


async def authorize_project_ws(
    websocket: Any, project_id: str, *, token_query: str | None
) -> AuthContext | None:
    """
    Validate the JWT on a WebSocket connection. Supports both subprotocol
    (`Sec-WebSocket-Protocol: bearer.<jwt>`) and `?token=` query string,
    per the spec.

    Returns AuthContext on success. On failure, closes the socket with code
    1008 (policy violation) and returns None.
    """
    raise NotImplementedError


# ===========================================================================
# file_store.py — local filesystem abstraction
# ===========================================================================

from pathlib import Path


class FileStore:
    """
    Project-scoped FileStore. Per the spec:
      'The FileStore API takes project_id as a required argument; there is no
       global file-store namespace.'
    """

    def __init__(self, *, project_id: str) -> None:
        self.project_id = project_id
        self.root = Path(settings.data_root) / "projects" / project_id

    async def write_text(self, relpath: str, content: str) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        # In production this would be aiofiles; sync write is fine for small artifacts.
        path.write_text(content, encoding="utf-8")
        return path

    async def read_text(self, relpath: str) -> str | None:
        path = self.root / relpath
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")


# ===========================================================================
# run_repository.py — runs table CRUD
# ===========================================================================

class RunRepository:
    """SQLAlchemy-async wrapper over the `runs` table."""

    async def create(
        self, *, run_id: str, project_id: str, user_id: str,
        workflow: str, document_ids: list[str],
    ) -> None:
        raise NotImplementedError

    async def get(self, *, run_id: str, project_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def mark_running(self, *, run_id: str, project_id: str, started_at: float) -> None:
        raise NotImplementedError

    async def mark_succeeded(
        self, *, run_id: str, finished_at: float,
        requirements_md_path: str | None, requirements_json_path: str | None,
    ) -> None:
        raise NotImplementedError

    async def mark_failed(self, *, run_id: str, finished_at: float, error: str) -> None:
        raise NotImplementedError

    async def mark_cancelled(self, *, run_id: str, finished_at: float) -> None:
        raise NotImplementedError


# ===========================================================================
# usage_recorder.py — token/cost tracking
# ===========================================================================

async def record_usage(
    *, run_id: str, project_id: str, node: str | None,
    model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int,
) -> None:
    """
    Persist a usage row + roll up to the run-level totals. Cost is computed
    here using a configurable pricing table (env-driven per the spec).
    """
    raise NotImplementedError
