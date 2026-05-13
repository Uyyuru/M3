"""
FastAPI routes for Workflow 1 — Requirements Gathering.

All endpoints under /projects/{project_id}/... are project-scoped per the
Project-Centric Resource Model in the spec. Every handler:
  1. Resolves project_id from the path.
  2. Authorizes the JWT subject against the project.
  3. Returns 404 (never 403) on a project_id mismatch — no existence leaks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.hitl_broker import get_broker
from app.schemas.requirements import (
    ApprovalResponse,
    ClarificationResponse,
)
from app.services.auth import (
    AuthContext,
    authorize_project,
    authorize_project_ws,
    require_auth,
)
from app.services.run_repository import RunRepository
from app.workflows.requirements.runner import (
    get_lock_manager,
    run_requirements_workflow,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}", tags=["workflow:requirements"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TriggerRequirementsBody(BaseModel):
    document_ids: list[str] = Field(
        ..., min_length=1,
        description="The document_ids (within this project) to ground the agent on.",
    )


class TriggerRequirementsResponse(BaseModel):
    run_id: str
    status: str = "accepted"


class ClarificationRESTBody(BaseModel):
    request_id: str
    answers: list[dict[str, str]]  # [{id, answer}]


class RejectBody(BaseModel):
    feedback: str = Field(..., min_length=1, max_length=4000)


# ---------------------------------------------------------------------------
# 1. Trigger a run
# ---------------------------------------------------------------------------

@router.post(
    "/workflows/requirements",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TriggerRequirementsResponse,
)
async def trigger_requirements(
    body: TriggerRequirementsBody,
    background: BackgroundTasks,
    response: Response,
    project_id: str = Path(...),
    auth: AuthContext = Depends(require_auth),
) -> TriggerRequirementsResponse:
    """
    202 Accepted with Location header pointing at the run resource.
    409 Conflict if a run is already active for this project (idempotency).
    """
    await authorize_project(auth, project_id)

    run_id = f"run_{uuid.uuid4().hex[:16]}"
    lock_mgr = get_lock_manager()

    acquired = await lock_mgr.acquire(project_id, run_id)
    if not acquired:
        existing = lock_mgr.holder(project_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": "RUN_ALREADY_ACTIVE",
                    "message": "A workflow run is already active for this project.",
                    "details": {"active_run_id": existing},
                }
            },
        )

    # Create the run row before launching the task — so the SSE/status endpoints
    # immediately see something even if the task hasn't started yet.
    await RunRepository().create(
        run_id=run_id,
        project_id=project_id,
        user_id=auth.user_id,
        workflow="requirements",
        document_ids=body.document_ids,
    )

    background.add_task(
        run_requirements_workflow,
        project_id=project_id,
        run_id=run_id,
        user_id=auth.user_id,
        document_ids=body.document_ids,
    )

    response.headers["Location"] = f"/projects/{project_id}/runs/{run_id}"
    return TriggerRequirementsResponse(run_id=run_id)


# ---------------------------------------------------------------------------
# 2. Run status
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}")
async def get_run(
    project_id: str = Path(...),
    run_id: str = Path(...),
    auth: AuthContext = Depends(require_auth),
) -> dict[str, Any]:
    await authorize_project(auth, project_id)
    run = await RunRepository().get(run_id=run_id, project_id=project_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


# ---------------------------------------------------------------------------
# 3. SSE event stream
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/events")
async def stream_events(
    project_id: str = Path(...),
    run_id: str = Path(...),
    auth: AuthContext = Depends(require_auth),
) -> StreamingResponse:
    await authorize_project(auth, project_id)
    run = await RunRepository().get(run_id=run_id, project_id=project_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    broker = get_broker()
    queue = broker.get_event_queue(run_id)

    async def event_gen():
        # Send a hello so clients know the channel is live
        yield f"event: open\ndata: {json.dumps({'run_id': run_id})}\n\n"
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: {ev.get('event', 'message')}\ndata: {json.dumps(ev)}\n\n"
                    if ev.get("event") in {"run_succeeded", "run_failed", "run_cancelled"}:
                        break
                except asyncio.TimeoutError:
                    # heartbeat — keeps proxies from killing the connection
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# 4. WebSocket HITL channel — the primary path
# ---------------------------------------------------------------------------

@router.websocket("/runs/{run_id}/hitl")
async def hitl_websocket(
    websocket: WebSocket,
    project_id: str = Path(...),
    run_id: str = Path(...),
    token: str | None = Query(default=None),
) -> None:
    """
    WebSocket protocol (documented here in lieu of Swagger):

    Server → client:
      { "type": "clarification_request", "request_id": "...", "questions": [...] }
      { "type": "approval_request",      "request_id": "...", "artifact": {...} }
      { "type": "clarification_failed",  "request_id": "...", "reason": "...", "rounds_used": N }
      { "type": "error",                 "code": "...", "message": "..." }

    Client → server:
      { "type": "clarification_response", "request_id": "...", "answers": [{id, answer}, ...] }
      { "type": "approval_response",      "request_id": "...", "decision": "approve" }
      { "type": "approval_response",      "request_id": "...", "decision": "reject", "feedback": "..." }
    """
    # Auth via subprotocol (`bearer.<jwt>`) or ?token= query
    auth = await authorize_project_ws(websocket, project_id, token_query=token)
    if auth is None:
        return  # `authorize_project_ws` already closed the socket

    broker = get_broker()

    # On (re)connect: if there's a pending interrupt for this run, re-push it.
    pending = broker.get_pending(run_id)
    if pending is not None:
        try:
            await websocket.send_text(json.dumps(pending.payload))
        except Exception:
            log.exception("failed to re-push pending interrupt for run %s", run_id)
            return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error", "code": "BAD_JSON", "message": "Invalid JSON."
                }))
                continue

            mtype = msg.get("type")

            if mtype == "clarification_response":
                try:
                    parsed = ClarificationResponse.model_validate(msg)
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error", "code": "VALIDATION", "message": str(exc),
                    }))
                    continue
                ok = await broker.resolve(
                    run_id=run_id,
                    request_id=parsed.request_id,
                    value={"answers": [a.model_dump() for a in parsed.answers],
                           "request_id": parsed.request_id},
                )
                if not ok:
                    await websocket.send_text(json.dumps({
                        "type": "error", "code": "NO_PENDING",
                        "message": "No pending clarification matching this request_id.",
                    }))

            elif mtype == "approval_response":
                try:
                    parsed = ApprovalResponse.model_validate(msg)
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error", "code": "VALIDATION", "message": str(exc),
                    }))
                    continue
                ok = await broker.resolve(
                    run_id=run_id,
                    request_id=parsed.request_id,
                    value={"decision": parsed.decision, "feedback": parsed.feedback,
                           "request_id": parsed.request_id},
                )
                if not ok:
                    await websocket.send_text(json.dumps({
                        "type": "error", "code": "NO_PENDING",
                        "message": "No pending approval matching this request_id.",
                    }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error", "code": "UNKNOWN_TYPE",
                    "message": f"Unsupported message type: {mtype!r}",
                }))

    except WebSocketDisconnect:
        log.info("WS disconnected run=%s (run remains paused at checkpoint)", run_id)
        return


# ---------------------------------------------------------------------------
# 5. REST fallbacks — same state injection as the WS, for automation/replay
# ---------------------------------------------------------------------------

@router.post("/runs/{run_id}/clarifications", status_code=204)
async def post_clarifications(
    body: ClarificationRESTBody,
    project_id: str = Path(...),
    run_id: str = Path(...),
    auth: AuthContext = Depends(require_auth),
) -> Response:
    await authorize_project(auth, project_id)
    ok = await get_broker().resolve(
        run_id=run_id,
        request_id=body.request_id,
        value={"answers": body.answers, "request_id": body.request_id},
    )
    if not ok:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "NO_PENDING", "message": "No matching pending clarification."}},
        )
    return Response(status_code=204)


@router.post("/runs/{run_id}/approve", status_code=204)
async def post_approve(
    project_id: str = Path(...),
    run_id: str = Path(...),
    auth: AuthContext = Depends(require_auth),
) -> Response:
    await authorize_project(auth, project_id)
    # request_id is the currently-pending approval, if any.
    pending = get_broker().get_pending(run_id)
    if pending is None or pending.payload.get("type") != "approval_request":
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "NO_PENDING_APPROVAL"}},
        )
    await get_broker().resolve(
        run_id=run_id,
        request_id=pending.request_id,
        value={"decision": "approve", "feedback": None, "request_id": pending.request_id},
    )
    return Response(status_code=204)


@router.post("/runs/{run_id}/reject", status_code=204)
async def post_reject(
    body: RejectBody,
    project_id: str = Path(...),
    run_id: str = Path(...),
    auth: AuthContext = Depends(require_auth),
) -> Response:
    await authorize_project(auth, project_id)
    pending = get_broker().get_pending(run_id)
    if pending is None or pending.payload.get("type") != "approval_request":
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "NO_PENDING_APPROVAL"}},
        )
    await get_broker().resolve(
        run_id=run_id,
        request_id=pending.request_id,
        value={"decision": "reject", "feedback": body.feedback,
               "request_id": pending.request_id},
    )
    return Response(status_code=204)
