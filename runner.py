"""
Runner for Workflow 1.

Owns:
  - The single-active-run-per-project lock (in-memory; spec acknowledges
    in-memory state is acceptable for the assignment scope).
  - The async loop that drives `graph.astream`, detects `__interrupt__`,
    hands the payload to the HITL broker, awaits a resume value, and
    feeds it back via `Command(resume=...)`.
  - Lifecycle bookkeeping in the `runs` table (status, started_at,
    finished_at, error).

This module is intentionally agnostic of FastAPI — it's pure async Python
that can be triggered from a BackgroundTasks call or from a worker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langgraph.types import Command

from app.core.hitl_broker import get_broker
from app.schemas.requirements import RequirementsState
from app.services.config import settings
from app.services.run_repository import RunRepository
from app.workflows.requirements.graph import get_graph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project locks — single active run per project
# ---------------------------------------------------------------------------

class ProjectLockManager:
    """
    Maps project_id -> (run_id of the holder). Used by the API layer to
    return 409 Conflict if a second run is started while one is active.
    """

    def __init__(self) -> None:
        self._locks: dict[str, str] = {}
        self._mu = asyncio.Lock()

    async def acquire(self, project_id: str, run_id: str) -> bool:
        async with self._mu:
            if project_id in self._locks:
                return False
            self._locks[project_id] = run_id
            return True

    async def release(self, project_id: str, run_id: str) -> None:
        async with self._mu:
            if self._locks.get(project_id) == run_id:
                self._locks.pop(project_id, None)

    def holder(self, project_id: str) -> str | None:
        return self._locks.get(project_id)


_lock_mgr = ProjectLockManager()


def get_lock_manager() -> ProjectLockManager:
    return _lock_mgr


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_requirements_workflow(
    *,
    project_id: str,
    run_id: str,
    user_id: str,
    document_ids: list[str],
) -> None:
    """
    Top-level entrypoint. Designed to be launched via `asyncio.create_task`
    (or BackgroundTasks). Never raises — terminal errors are written to the
    `runs` table and surfaced via SSE.
    """
    graph = get_graph()
    broker = get_broker()
    repo = RunRepository()

    config = {
        "configurable": {
            "thread_id": run_id,           # thread_id == run_id ties checkpoints to the run
            "project_id": project_id,
            "user_id": user_id,
        }
    }

    initial_state: RequirementsState = {
        "project_id": project_id,
        "run_id": run_id,
        "user_id": user_id,
        "document_ids": document_ids,
        "messages": [],
        "retrieved_chunks": [],
        "draft": None,
        "critique": None,
        "clarification_round": 0,
        "clarification_cap": settings.clarification_cap,
        "clarification_answers": {},
        "approval_feedback": None,
        "requirements_md_path": None,
        "requirements_json_path": None,
        "error": None,
    }

    await repo.mark_running(run_id=run_id, project_id=project_id, started_at=time.time())
    await broker.emit_event(run_id, {"event": "run_started", "run_id": run_id})

    # The first astream call uses the initial state.
    # Subsequent calls (after a resume) pass a Command, not state.
    next_input: Any = initial_state

    try:
        while True:
            interrupt_payload: dict[str, Any] | None = None

            async for chunk in graph.astream(
                next_input,
                config=config,
                stream_mode="updates",
                # version="v2",  # uncomment once your langgraph supports it as a kwarg
                subgraphs=True,
            ):
                # `chunk` shape with subgraphs=True is (ns, update_dict).
                # We don't need ns for routing here.
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    _ns, update = chunk
                else:
                    update = chunk

                if not isinstance(update, dict):
                    continue

                if "__interrupt__" in update:
                    interrupts = update["__interrupt__"]
                    # We never emit parallel interrupts in this graph, so [0] is safe.
                    interrupt_payload = interrupts[0].value
                    break

                # Forward each node update as a low-volume SSE event for the UI
                for node_name, node_update in update.items():
                    if node_name.startswith("__"):
                        continue
                    await broker.emit_event(run_id, {
                        "event": "node_update",
                        "node": node_name,
                        "keys": list((node_update or {}).keys()),
                    })

            if interrupt_payload is None:
                # astream exited cleanly — the run is done.
                break

            # Hand the interrupt to the broker and wait for the user.
            request_id = interrupt_payload.get("request_id")
            if not request_id:
                raise RuntimeError(
                    f"Interrupt payload missing request_id: {interrupt_payload!r}"
                )

            await broker.emit_event(run_id, {
                "event": "hitl_open",
                "type": interrupt_payload.get("type"),
                "request_id": request_id,
            })

            resume_value = await broker.open_interrupt(
                run_id=run_id,
                request_id=request_id,
                payload=interrupt_payload,
            )

            await broker.emit_event(run_id, {
                "event": "hitl_resolved",
                "request_id": request_id,
            })

            next_input = Command(resume=resume_value)

        # ------------- success path -------------
        final_state = await graph.aget_state(config)
        values = final_state.values
        await repo.mark_succeeded(
            run_id=run_id,
            finished_at=time.time(),
            requirements_md_path=values.get("requirements_md_path"),
            requirements_json_path=values.get("requirements_json_path"),
        )
        await broker.emit_event(run_id, {
            "event": "run_succeeded",
            "run_id": run_id,
            "requirements_md_url": (
                f"/projects/{project_id}/runs/{run_id}/artifacts/requirements.md"
            ),
            "requirements_json_url": (
                f"/projects/{project_id}/runs/{run_id}/artifacts/requirements.json"
            ),
        })

    except asyncio.CancelledError:
        log.warning("run %s cancelled", run_id)
        await repo.mark_cancelled(run_id=run_id, finished_at=time.time())
        await broker.emit_event(run_id, {"event": "run_cancelled", "run_id": run_id})
        raise

    except Exception as exc:
        log.exception("run %s failed", run_id)
        await repo.mark_failed(run_id=run_id, finished_at=time.time(), error=str(exc))
        await broker.emit_event(run_id, {
            "event": "run_failed",
            "run_id": run_id,
            "error": str(exc),
        })

    finally:
        await _lock_mgr.release(project_id, run_id)
