from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    search,
    experiments,
    github,
    jobs,
    ai_scientist,
    agent_lab,
    unified_workflow,
    research_tree,
    ai_generation,
    llm_monitor,
    repomaster,
    memory,
)


def create_app() -> FastAPI:
    app = FastAPI(title="uagent API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(search.router, prefix="/api")
    app.include_router(experiments.router, prefix="/api")
    app.include_router(github.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(ai_scientist.router, prefix="/api")
    app.include_router(agent_lab.router, prefix="/api")
    app.include_router(unified_workflow.router, prefix="/api")
    app.include_router(research_tree.router, prefix="/api")
    app.include_router(ai_generation.router, prefix="/api")
    app.include_router(llm_monitor.router, prefix="/api")
    app.include_router(repomaster.router, prefix="/api")
    app.include_router(memory.router)
    return app


app = create_app()
