"""
FastAPI lifespan integration for Workflow 1.

Wire these into your existing app/main.py:

    from contextlib import asynccontextmanager
    from fastapi import FastAPI

    from app.api.v1.workflow_requirements import router as workflow_requirements_router
    from app.workflows.requirements.lifespan import init_workflow_requirements, shutdown_workflow_requirements

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_workflow_requirements()
        try:
            yield
        finally:
            await shutdown_workflow_requirements()

    app = FastAPI(lifespan=lifespan)
    app.include_router(workflow_requirements_router)
"""

from __future__ import annotations

from app.workflows.requirements.graph import init_graph, shutdown_graph


async def init_workflow_requirements() -> None:
    await init_graph()


async def shutdown_workflow_requirements() -> None:
    await shutdown_graph()
