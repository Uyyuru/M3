"""
HITL broker — the bridge between LangGraph `interrupt()` and the per-run WebSocket.

Why this exists
---------------
`interrupt()` pauses the graph and returns control to the caller of `astream`.
The caller (the workflow runner) needs to:

  1. Surface the interrupt payload to the user over WebSocket as a typed message.
  2. Wait for the user's reply (also over WebSocket).
  3. Resume the graph with `Command(resume=<reply>)`.

But the WebSocket lifecycle is INDEPENDENT of the run lifecycle — the assignment
spec says explicitly: "If the WebSocket disconnects mid-interrupt, the run stays
paused at the checkpoint; on reconnect the server re-pushes the pending
clarification_request (idempotent by request_id)."

So we can't tie the wait directly to a single websocket. Instead:

  - The runner registers each interrupt with the broker: (run_id, request_id, payload).
  - The runner awaits an `asyncio.Future` from the broker.
  - The WebSocket handler, on receiving a `clarification_response` / `approval_response`,
    resolves the future via the broker.
  - On WS connect (or reconnect), the handler asks the broker for any *pending*
    interrupt on this run and re-pushes it.

This also gives us the REST fallback for free: `POST /clarifications` and
`POST /approve` just call `broker.resolve(...)` the same way the WS does.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class PendingInterrupt:
    """A single outstanding interrupt waiting on a user response."""
    run_id: str
    request_id: str
    payload: dict[str, Any]                # exact wire payload (clarification_request or approval_request)
    future: asyncio.Future[Any] = field(default_factory=asyncio.Future)


class HITLBroker:
    """
    Process-singleton (within one FastAPI worker). The spec acknowledges
    that in-memory state is acceptable for the assignment scope, with the
    trade-off documented (single-process only, state lost on restart).
    """

    def __init__(self) -> None:
        # run_id -> PendingInterrupt (only one open interrupt per run at a time)
        self._pending: dict[str, PendingInterrupt] = {}
        # run_id -> asyncio.Queue for SSE-style fan-out of non-blocking events
        # (the runner pushes progress events here, the SSE endpoint drains it)
        self._event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Interrupt lifecycle                                                #
    # ------------------------------------------------------------------ #

    async def open_interrupt(
        self,
        run_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> Any:
        """
        Called by the runner when the graph emits an interrupt.

        Registers the pending payload and BLOCKS until either:
          - the WS/REST handler calls `resolve(run_id, request_id, value)`, or
          - the run is cancelled (future is cancelled externally).

        Returns the resume value that should be fed into `Command(resume=...)`.
        """
        async with self._lock:
            if run_id in self._pending:
                # Idempotency: if there's already a pending interrupt for this run
                # AND the request_id matches, return the existing future (reconnect case).
                existing = self._pending[run_id]
                if existing.request_id == request_id:
                    log.info("HITL: reusing existing pending interrupt %s for run %s",
                             request_id, run_id)
                    return await existing.future
                # Different request_id but old one still open — that's a bug.
                log.warning(
                    "HITL: run %s opening interrupt %s while %s still pending; "
                    "cancelling old.", run_id, request_id, existing.request_id,
                )
                existing.future.cancel()

            pending = PendingInterrupt(run_id=run_id, request_id=request_id, payload=payload)
            self._pending[run_id] = pending

        log.info("HITL: opened interrupt run=%s request=%s type=%s",
                 run_id, request_id, payload.get("type"))

        try:
            return await pending.future
        finally:
            async with self._lock:
                # Only clear if this is still the same pending interrupt;
                # don't clobber a new one that opened during a race.
                if self._pending.get(run_id) is pending:
                    self._pending.pop(run_id, None)

    async def resolve(self, run_id: str, request_id: str, value: Any) -> bool:
        """
        Called by the WS handler (or REST fallback) when a user response arrives.
        Returns True if a pending interrupt was matched, False otherwise.
        """
        async with self._lock:
            pending = self._pending.get(run_id)
            if pending is None:
                log.warning("HITL: no pending interrupt for run %s", run_id)
                return False
            if pending.request_id != request_id:
                log.warning(
                    "HITL: request_id mismatch for run %s (pending=%s, got=%s)",
                    run_id, pending.request_id, request_id,
                )
                return False
            if pending.future.done():
                return False
            pending.future.set_result(value)
            return True

    async def cancel(self, run_id: str, reason: str = "cancelled") -> None:
        """Cancel any pending interrupt for the run (e.g. run aborted)."""
        async with self._lock:
            pending = self._pending.pop(run_id, None)
        if pending and not pending.future.done():
            pending.future.set_exception(asyncio.CancelledError(reason))

    def get_pending(self, run_id: str) -> PendingInterrupt | None:
        """
        Used by the WS handler on (re)connect to re-push the outstanding
        clarification_request / approval_request payload.
        """
        return self._pending.get(run_id)

    # ------------------------------------------------------------------ #
    # Non-blocking event fan-out (used by hooks for SSE)                 #
    # ------------------------------------------------------------------ #

    def get_event_queue(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        q = self._event_queues.get(run_id)
        if q is None:
            q = asyncio.Queue(maxsize=256)
            self._event_queues[run_id] = q
        return q

    def drop_event_queue(self, run_id: str) -> None:
        self._event_queues.pop(run_id, None)

    async def emit_event(self, run_id: str, event: dict[str, Any]) -> None:
        q = self._event_queues.get(run_id)
        if q is None:
            return  # no SSE listeners; drop on the floor (events are advisory)
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("HITL: event queue full for run %s, dropping", run_id)


# Module-level singleton. Acceptable per spec; production would inject via DI.
_broker: HITLBroker | None = None


def get_broker() -> HITLBroker:
    global _broker
    if _broker is None:
        _broker = HITLBroker()
    return _broker
