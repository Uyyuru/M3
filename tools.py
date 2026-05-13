"""
Retrieval tools for the Requirements Gathering workflow.

Two tools are exposed to the agent:

1. `retrieve_document_context` — semantic search over the ChromaDB `documents`
   collection, MANDATORILY filtered by `project_id`. The project_id is injected
   from graph state via `ToolRuntime` so the LLM cannot escape the project
   sandbox even if it tries (defense in depth — the spec calls this out
   explicitly under "Project-Centric Resource Model (Non-Negotiable)").

2. `list_document_sections` — structural lookup that returns the section
   outline for a given document. Useful when the Drafter wants to know "what
   sections exist in the BRD" before drilling into RAG.

Both tools are designed to be consumed by `create_agent` or by manual
`ToolNode` wiring inside the StateGraph.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.services.vector_store import get_documents_collection
from app.services.document_store import get_document_sections


# ---------------------------------------------------------------------------
# Input schemas (give the LLM rich, validated descriptions)
# ---------------------------------------------------------------------------

class RetrieveDocumentContextArgs(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural-language query describing the information you need. "
            "Be specific — 'authentication requirements' beats 'security'."
        ),
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Default 6, max 20.",
    )
    kind_filter: str | None = Field(
        default=None,
        description=(
            "Optional document-kind filter: one of 'BRD', 'PRD', 'TRD'. "
            "Use this when you specifically want business vs product vs technical context."
        ),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("retrieve_document_context", args_schema=RetrieveDocumentContextArgs)
def retrieve_document_context(
    query: str,
    top_k: int = 6,
    kind_filter: str | None = None,
    # `InjectedToolArg` keeps these out of the JSON schema shown to the LLM;
    # they're filled in by the graph at runtime.
    project_id: Annotated[str, InjectedToolArg] = "",
    document_ids: Annotated[list[str] | None, InjectedToolArg] = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant chunks from the project's uploaded BRD/PRD/TRD
    documents. Always filtered by project_id and (optionally) the specific
    document_ids the user triggered the run with.

    Returns a list of {section_id, section_title, document_id, kind, page, text, score}.
    """
    if not project_id:
        raise ValueError(
            "project_id is required but was not injected. "
            "This indicates a graph-wiring bug — tools must be bound with project_id."
        )

    collection = get_documents_collection()

    # Mandatory project filter; never trust the LLM with this.
    where_clauses: dict[str, Any] = {"project_id": project_id}
    if kind_filter:
        where_clauses["kind"] = kind_filter.upper()
    if document_ids:
        where_clauses["document_id"] = {"$in": document_ids}

    # Chroma's combined-filter syntax requires $and when there are multiple keys
    if len(where_clauses) > 1:
        where = {"$and": [{k: v} for k, v in where_clauses.items()]}
    else:
        where = where_clauses

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
    )

    # Chroma returns parallel arrays — flatten to a list of dicts
    out: list[dict[str, Any]] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, distances):
        out.append({
            "section_id": meta.get("section_id"),
            "section_title": meta.get("section_title"),
            "document_id": meta.get("document_id"),
            "kind": meta.get("kind"),
            "page": meta.get("page"),
            "text": text,
            # Chroma returns cosine *distance*; convert to similarity for the LLM
            "score": round(1.0 - float(dist), 4) if dist is not None else None,
        })
    return out


class ListSectionsArgs(BaseModel):
    document_id: str = Field(..., description="The document_id whose outline you want")


@tool("list_document_sections", args_schema=ListSectionsArgs)
def list_document_sections(
    document_id: str,
    project_id: Annotated[str, InjectedToolArg] = "",
) -> list[dict[str, Any]]:
    """
    Return the structured section outline for a single document
    (section_id, section_title, page, kind). Use this when you need to
    understand the structure of a document before doing targeted RAG.

    Raises 404-equivalent if the document does not belong to project_id.
    """
    if not project_id:
        raise ValueError("project_id must be injected; graph wiring bug.")

    sections = get_document_sections(project_id=project_id, document_id=document_id)
    if sections is None:
        raise ValueError(
            f"Document {document_id} not found within project {project_id}."
        )
    return sections


# Convenience: the canonical tool list the workflow binds to its agent / ToolNode
REQUIREMENTS_TOOLS = [retrieve_document_context, list_document_sections]
