"""
Graph nodes for the Requirements Gathering workflow.

Topology (high level):

    START
      │
      ▼
    retrieve ──► drafter ──► critic ──┐
                                       │ is_complete?
                                       │
                   ┌───── no ──────────┘
                   ▼
              clarification_gate  (interrupt)
                   │
                   ▼
                refine ──► critic   (loop, cap = 3)
                                       │
                   ┌───── yes ─────────┘
                   ▼
                persist
                   │
                   ▼
            approval_gate  (interrupt_before)
                   │
            ┌──────┴──────┐
            ▼             ▼
         approve        reject ──► refine (with feedback)
            │
            ▼
           END
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt

from app.core.hitl_broker import get_broker
from app.core.hooks import TokenCostHandler, traced_node
from app.schemas.requirements import (
    ClarificationFailed,
    ClarificationRequest,
    CritiqueReport,
    RequirementsArtifact,
    RequirementsState,
)
from app.services.config import settings
from app.services.file_store import FileStore
from app.workflows.requirements.prompts import (
    CRITIC_SYSTEM,
    DRAFTER_SYSTEM,
    REFINER_SYSTEM,
)
from app.workflows.requirements.tools import (
    REQUIREMENTS_TOOLS,
    retrieve_document_context,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory — one place to configure GPT-5.4 / GPT-5.5 per spec
# ---------------------------------------------------------------------------

def _llm(node: str, run_id: str, project_id: str, *, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,           # e.g. "gpt-5.4"
        temperature=temperature,
        callbacks=[TokenCostHandler(run_id=run_id, project_id=project_id, node=node)],
        timeout=settings.openai_timeout_s,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# 1. Retrieve — seed the conversation with broad-stroke context
# ---------------------------------------------------------------------------

@traced_node("retrieve")
async def retrieve_node(state: RequirementsState) -> dict[str, Any]:
    """
    Pull a broad initial set of chunks across the user's documents so the
    Drafter has something to chew on without needing to call the tool first.

    The Drafter can (and should) still call `retrieve_document_context` for
    targeted lookups during drafting.
    """
    project_id = state["project_id"]
    document_ids = state.get("document_ids") or []

    # Three coarse seed queries cover the typical BRD/PRD/TRD axes
    seed_queries = [
        "business goals, problem statement, success metrics",
        "user personas, user journeys, functional requirements",
        "non-functional requirements, constraints, technical architecture",
    ]

    chunks: list[dict[str, Any]] = []
    for q in seed_queries:
        # Direct invocation — we bypass the LLM here, this is a deterministic seed
        result = retrieve_document_context.invoke({
            "query": q,
            "top_k": 5,
            "project_id": project_id,
            "document_ids": document_ids or None,
        })
        chunks.extend(result)

    # Dedupe by section_id to avoid feeding the same chunk three times
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in chunks:
        sid = c.get("section_id")
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(c)

    return {"retrieved_chunks": deduped}


# ---------------------------------------------------------------------------
# 2. Drafter — produces the first structured artifact
# ---------------------------------------------------------------------------

def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Compact rendering of retrieved chunks for prompt injection."""
    lines = []
    for c in chunks:
        header = (
            f"[section_id={c.get('section_id')} | "
            f"kind={c.get('kind')} | "
            f"title={c.get('section_title')}]"
        )
        lines.append(f"{header}\n{c.get('text', '')}")
    return "\n\n---\n\n".join(lines)


@traced_node("drafter")
async def drafter_node(state: RequirementsState) -> dict[str, Any]:
    project_id = state["project_id"]
    run_id = state["run_id"]

    llm = _llm("drafter", run_id, project_id, temperature=0.1)
    # Give the Drafter the retrieval tool AND structured output enforcement.
    structured = llm.bind_tools(REQUIREMENTS_TOOLS).with_structured_output(
        RequirementsArtifact, method="function_calling", include_raw=False,
    )

    context_block = _format_chunks_for_prompt(state.get("retrieved_chunks") or [])

    messages = [
        SystemMessage(content=DRAFTER_SYSTEM),
        HumanMessage(content=(
            f"project_id: {project_id}\nrun_id: {run_id}\n\n"
            f"Seed context from RAG (use the tool for deeper lookups):\n\n"
            f"{context_block}\n\n"
            f"Produce the initial RequirementsArtifact."
        )),
    ]

    artifact: RequirementsArtifact = await structured.ainvoke(messages)
    # Force identity fields — the LLM should not be inventing these
    artifact.project_id = project_id
    artifact.run_id = run_id

    return {
        "draft": artifact.model_dump(),
        "messages": [HumanMessage(content="[Drafter] initial draft complete")],
    }


# ---------------------------------------------------------------------------
# 3. Critic — Reflection / Self-Critique
# ---------------------------------------------------------------------------

@traced_node("critic")
async def critic_node(state: RequirementsState) -> dict[str, Any]:
    project_id = state["project_id"]
    run_id = state["run_id"]
    draft = state.get("draft") or {}
    context_block = _format_chunks_for_prompt(state.get("retrieved_chunks") or [])
    answered_so_far = state.get("clarification_answers") or {}

    llm = _llm("critic", run_id, project_id, temperature=0.0)
    structured = llm.with_structured_output(CritiqueReport, method="function_calling")

    answered_summary = (
        "\n".join(f"- {qid}: {ans}" for qid, ans in answered_so_far.items())
        or "(none yet)"
    )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=(
            f"SOURCE CHUNKS:\n{context_block}\n\n"
            f"CURRENT DRAFT:\n{json.dumps(draft, indent=2)}\n\n"
            f"PREVIOUSLY ANSWERED (do not re-ask):\n{answered_summary}\n\n"
            f"Produce the CritiqueReport."
        )),
    ]

    report: CritiqueReport = await structured.ainvoke(messages)
    return {"critique": report.model_dump()}


def critic_router(state: RequirementsState) -> str:
    """Conditional edge: where do we go after critique?"""
    critique = state.get("critique") or {}
    round_n = state.get("clarification_round", 0)
    cap = state.get("clarification_cap", 3)

    if critique.get("is_complete"):
        return "persist"

    # Has blockers — but have we exhausted our clarification budget?
    if round_n >= cap:
        return "clarification_failed"

    return "clarification_gate"


# ---------------------------------------------------------------------------
# 4. Clarification gate — HITL interrupt
# ---------------------------------------------------------------------------

@traced_node("clarification_gate")
async def clarification_gate_node(state: RequirementsState) -> dict[str, Any]:
    """
    Surfaces the Critic's gaps to the user via `interrupt()`. The runner
    bridges this to the WebSocket and waits for the typed
    `clarification_response`. When resumed, we merge the answers into
    accumulated state and bump the round counter.
    """
    critique = state.get("critique") or {}
    gaps = critique.get("gaps") or []

    # Only ask about blocker + major gaps. Minor goes straight to open_questions.
    askable = [g for g in gaps if g.get("severity") in ("blocker", "major")]
    if not askable:
        # Nothing to ask but critic said is_complete=False; shouldn't happen,
        # but if it does, treat as complete to avoid infinite loops.
        return {"clarification_round": state.get("clarification_round", 0) + 1}

    request_id = f"clr_{uuid.uuid4().hex[:12]}"
    payload = ClarificationRequest(
        request_id=request_id,
        questions=[
            {"id": g["id"], "question": g["question"], "context": g.get("context", "")}
            for g in askable
        ],
    ).model_dump()

    # `interrupt()` returns whatever the resume value is. The runner will
    # set resume = {"answers": {qid: answer_text, ...}, "request_id": request_id}
    response = interrupt(payload)

    if not isinstance(response, dict) or "answers" not in response:
        raise ValueError(
            f"Malformed clarification_response: expected dict with 'answers', got {response!r}"
        )

    new_answers = {a["id"]: a["answer"] for a in response["answers"]}
    merged = {**(state.get("clarification_answers") or {}), **new_answers}

    return {
        "clarification_answers": merged,
        "clarification_round": state.get("clarification_round", 0) + 1,
        "messages": [HumanMessage(content=(
            f"[User clarifications round {state.get('clarification_round', 0) + 1}]\n"
            + "\n".join(f"- {qid}: {a}" for qid, a in new_answers.items())
        ))],
    }


# ---------------------------------------------------------------------------
# 5. Refiner — integrates new answers (or rejection feedback) into the draft
# ---------------------------------------------------------------------------

@traced_node("refiner")
async def refiner_node(state: RequirementsState) -> dict[str, Any]:
    project_id = state["project_id"]
    run_id = state["run_id"]
    draft = state.get("draft") or {}
    answers = state.get("clarification_answers") or {}
    feedback = state.get("approval_feedback")

    llm = _llm("refiner", run_id, project_id, temperature=0.1)
    structured = llm.with_structured_output(RequirementsArtifact, method="function_calling")

    answers_block = "\n".join(f"- {qid}: {a}" for qid, a in answers.items()) or "(none)"
    feedback_block = f"REVIEWER FEEDBACK (must address):\n{feedback}\n\n" if feedback else ""

    messages = [
        SystemMessage(content=REFINER_SYSTEM),
        HumanMessage(content=(
            f"PREVIOUS DRAFT:\n{json.dumps(draft, indent=2)}\n\n"
            f"CLARIFICATION ANSWERS (cumulative):\n{answers_block}\n\n"
            f"{feedback_block}"
            f"Produce the updated RequirementsArtifact. Preserve stable IDs."
        )),
    ]

    artifact: RequirementsArtifact = await structured.ainvoke(messages)
    artifact.project_id = project_id
    artifact.run_id = run_id

    return {
        "draft": artifact.model_dump(),
        # Clear feedback once consumed; keep the answers (they're cumulative).
        "approval_feedback": None,
        "messages": [HumanMessage(content="[Refiner] draft updated")],
    }


# ---------------------------------------------------------------------------
# 6. Clarification-failed terminal node
# ---------------------------------------------------------------------------

@traced_node("clarification_failed")
async def clarification_failed_node(state: RequirementsState) -> dict[str, Any]:
    """Cap exceeded. Push a typed failure to the broker and end the run."""
    run_id = state["run_id"]
    rounds = state.get("clarification_round", 0)
    payload = ClarificationFailed(
        request_id=f"clf_{uuid.uuid4().hex[:12]}",
        reason=f"Clarification cap of {state.get('clarification_cap', 3)} exceeded.",
        rounds_used=rounds,
    ).model_dump()
    await get_broker().emit_event(run_id, {"event": "clarification_failed", **payload})
    return {"error": payload["reason"]}


# ---------------------------------------------------------------------------
# 7. Persist — write requirements.md + requirements.json to FileStore
# ---------------------------------------------------------------------------

def _render_markdown(artifact: RequirementsArtifact) -> str:
    md: list[str] = [f"# Requirements — {artifact.project_id} / run {artifact.run_id}", ""]

    md.append("## Goals")
    md.extend(f"- {g}" for g in artifact.goals)
    md.append("")

    if artifact.personas:
        md.append("## Personas")
        for p in artifact.personas:
            md.append(f"### {p.name} — {p.role}")
            md.extend(f"- {n}" for n in p.needs)
            md.append("")

    md.append("## Functional Requirements")
    for fr in artifact.functional_requirements:
        md.append(f"### {fr.id} — {fr.title} (`{fr.priority}`)")
        md.append(fr.description)
        if fr.acceptance_criteria:
            md.append("\n**Acceptance criteria:**")
            md.extend(f"- {a}" for a in fr.acceptance_criteria)
        if fr.source_section_ids:
            md.append(f"\n_Sources: {', '.join(fr.source_section_ids)}_")
        md.append("")

    md.append("## Non-Functional Requirements")
    for nfr in artifact.non_functional_requirements:
        md.append(f"### {nfr.id} — {nfr.category}")
        md.append(nfr.statement)
        if nfr.measurable_target:
            md.append(f"\n_Target: {nfr.measurable_target}_")
        md.append("")

    if artifact.constraints:
        md.append("## Constraints")
        md.extend(f"- {c}" for c in artifact.constraints)
        md.append("")

    if artifact.out_of_scope:
        md.append("## Out of Scope")
        md.extend(f"- {o}" for o in artifact.out_of_scope)
        md.append("")

    if artifact.assumptions:
        md.append("## Assumptions")
        md.extend(f"- {a}" for a in artifact.assumptions)
        md.append("")

    if artifact.open_questions:
        md.append("## Open Questions")
        for oq in artifact.open_questions:
            tag = " **(blocking)**" if oq.blocking else ""
            md.append(f"- `{oq.id}`{tag}: {oq.question}")
        md.append("")

    return "\n".join(md)


@traced_node("persist")
async def persist_node(state: RequirementsState) -> dict[str, Any]:
    project_id = state["project_id"]
    run_id = state["run_id"]
    artifact = RequirementsArtifact.model_validate(state["draft"])

    md = _render_markdown(artifact)
    json_blob = artifact.model_dump_json(indent=2)

    store = FileStore(project_id=project_id)
    md_path = await store.write_text(f"runs/{run_id}/requirements.md", md)
    json_path = await store.write_text(f"runs/{run_id}/requirements.json", json_blob)

    return {
        "requirements_md_path": str(md_path),
        "requirements_json_path": str(json_path),
    }


# ---------------------------------------------------------------------------
# 8. Approval gate — interrupt_before, so the user must explicitly approve
# ---------------------------------------------------------------------------

@traced_node("approval_gate")
async def approval_gate_node(state: RequirementsState) -> dict[str, Any]:
    """
    Pushes approval_request and waits. On approve, the run ends here.
    On reject, writes feedback into state and routes back to the refiner.
    """
    request_id = f"apr_{uuid.uuid4().hex[:12]}"
    payload = {
        "type": "approval_request",
        "request_id": request_id,
        "artifact": {
            # URLs are constructed by the API layer — we just expose the
            # logical paths; the API maps these to GET /projects/.../artifacts
            "requirements_md_url": (
                f"/projects/{state['project_id']}/runs/{state['run_id']}"
                f"/artifacts/requirements.md"
            ),
            "requirements_json_url": (
                f"/projects/{state['project_id']}/runs/{state['run_id']}"
                f"/artifacts/requirements.json"
            ),
        },
    }

    response = interrupt(payload)

    if not isinstance(response, dict) or "decision" not in response:
        raise ValueError(f"Malformed approval_response: {response!r}")

    if response["decision"] == "approve":
        return {"messages": [HumanMessage(content="[User] approved.")]}

    # Reject path — write feedback into state; the router sends us back to refiner.
    return {
        "approval_feedback": response.get("feedback") or "(no feedback provided)",
        "messages": [HumanMessage(content=(
            f"[User] rejected. Feedback: {response.get('feedback')}"
        ))],
    }


def approval_router(state: RequirementsState) -> str:
    if state.get("approval_feedback"):
        return "refiner"
    return "__end__"
