"""
StateGraph wiring for Workflow 1 — Requirements Gathering.

The graph is compiled with `AsyncSqliteSaver` so runs are durable across
restarts (until the SQLite db is wiped), and resumable via thread_id == run_id.

The compiled graph is created lazily on first import via `get_graph()` so the
checkpointer connection is owned by the FastAPI lifespan rather than module
import time.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from app.schemas.requirements import RequirementsState
from app.services.config import settings
from app.workflows.requirements.nodes import (
    approval_gate_node,
    approval_router,
    clarification_failed_node,
    clarification_gate_node,
    critic_node,
    critic_router,
    drafter_node,
    persist_node,
    refiner_node,
    retrieve_node,
)

log = logging.getLogger(__name__)


def build_graph_definition() -> StateGraph:
    """
    Pure graph topology — no checkpointer attached. Useful for tests and for
    rendering the graph PNG (per spec: 'rendered graph PNGs are stored under
    a configurable project root').
    """
    g: StateGraph = StateGraph(RequirementsState)

    # ---- nodes ----
    g.add_node("retrieve", retrieve_node)
    g.add_node("drafter", drafter_node)
    g.add_node("critic", critic_node)
    g.add_node("clarification_gate", clarification_gate_node)
    g.add_node("clarification_failed", clarification_failed_node)
    g.add_node("refiner", refiner_node)
    g.add_node("persist", persist_node)
    g.add_node("approval_gate", approval_gate_node)

    # ---- edges ----
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "drafter")
    g.add_edge("drafter", "critic")

    # critic ─► persist | clarification_gate | clarification_failed
    g.add_conditional_edges(
        "critic",
        critic_router,
        {
            "persist": "persist",
            "clarification_gate": "clarification_gate",
            "clarification_failed": "clarification_failed",
        },
    )

    g.add_edge("clarification_gate", "refiner")
    g.add_edge("refiner", "critic")           # Reflection loop closure
    g.add_edge("clarification_failed", END)

    g.add_edge("persist", "approval_gate")

    # approval_gate ─► refiner (reject) | END (approve)
    g.add_conditional_edges(
        "approval_gate",
        approval_router,
        {"refiner": "refiner", "__end__": END},
    )

    return g


# ---------------------------------------------------------------------------
# Compiled graph singleton (per FastAPI worker)
# ---------------------------------------------------------------------------

_compiled: Any = None
_exit_stack: AsyncExitStack | None = None


async def init_graph() -> None:
    """Call from FastAPI startup. Opens the SQLite checkpointer connection."""
    global _compiled, _exit_stack
    if _compiled is not None:
        return

    _exit_stack = AsyncExitStack()
    checkpointer = await _exit_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(settings.langgraph_sqlite_path)
    )
    g = build_graph_definition()
    _compiled = g.compile(checkpointer=checkpointer)
    log.info("Requirements graph compiled (checkpointer=%s)", settings.langgraph_sqlite_path)


async def shutdown_graph() -> None:
    """Call from FastAPI shutdown."""
    global _compiled, _exit_stack
    if _exit_stack is not None:
        await _exit_stack.aclose()
        _exit_stack = None
    _compiled = None


def get_graph() -> Any:
    if _compiled is None:
        raise RuntimeError(
            "Requirements graph not initialised. Call init_graph() in FastAPI lifespan."
        )
    return _compiled
