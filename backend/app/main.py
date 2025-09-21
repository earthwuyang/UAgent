"""Main FastAPI application for UAgent system"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.llm_client import create_llm_client
from .core.cache import create_cache
from .core.smart_router import SmartRouter
from .core.research_engines import (
    DeepResearchEngine,
    CodeResearchEngine,
    ScientificResearchEngine
)
from .core.openhands import OpenHandsClient
from .core.session_manager import ResearchSessionManager
from .core.app_state import set_app_state, clear_app_state
from .connectors import ArxivClient, CrossrefClient, OpenAlexClient, PubMedClient
from .pipelines import ClaimVerifier, EvidenceRetriever, EvidenceSynthesizer
from .services import ArtifactStore, PlaywrightCaptureService, QwenVisionAnalyzer, ResearchGraphService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting UAgent system...")

    # Initialize LLM client
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_key:
        logger.warning("DASHSCOPE_API_KEY not found - some features may not work")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DASHSCOPE_API_KEY environment variable is required"
        )

    llm_client = create_llm_client("dashscope", api_key=dashscope_key)

    # Initialize cache
    cache = create_cache("memory")

    # Initialize OpenHands client
    workspace_dir = os.getenv("WORKSPACE_DIR", "/tmp/uagent_workspaces")
    openhands_client = OpenHandsClient(base_workspace_dir=workspace_dir)

    # Initialize session manager
    session_manager = ResearchSessionManager()

    # Initialize research engines
    deep_engine = DeepResearchEngine(llm_client)
    code_engine = CodeResearchEngine(llm_client, openhands_client=openhands_client)
    scientific_engine = ScientificResearchEngine(
        llm_client=llm_client,
        deep_research_engine=deep_engine,
        code_research_engine=code_engine,
        openhands_client=openhands_client,
        config={
            "max_iterations": int(os.getenv("MAX_ITERATIONS", "3")),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
        }
    )

    # Initialize scientific research helpers
    claim_verifier = ClaimVerifier(llm_client)
    synthesizer = EvidenceSynthesizer(llm_client)
    artifact_root = os.getenv("ARTIFACT_STORE_ROOT", "./artifacts")
    artifact_store = ArtifactStore(artifact_root)
    research_graph_path = os.getenv("RESEARCH_GRAPH_PATH", "./data/research_graph.db")
    research_graph = ResearchGraphService(research_graph_path)
    playwright_headless = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() != "false"
    browser_service = PlaywrightCaptureService(headless=playwright_headless)
    vision_analyzer = None
    try:
        vision_analyzer = QwenVisionAnalyzer(api_key=dashscope_key)
    except Exception as vision_exc:
        logger.warning("Qwen-VL analyzer unavailable: %s", vision_exc)
    connectors = {
        "arxiv": ArxivClient(),
        "openalex": OpenAlexClient(mailto=os.getenv("OPENALEX_MAILTO")),
        "pubmed": PubMedClient(api_key=os.getenv("PUBMED_API_KEY"), email=os.getenv("PUBMED_EMAIL")),
        "crossref": CrossrefClient(),
    }

    # Initialize smart router
    smart_router = SmartRouter(llm_client=llm_client, cache=cache)

    # Store in global state
    set_app_state({
        "llm_client": llm_client,
        "cache": cache,
        "engines": {
            "deep": deep_engine,
            "code": code_engine,
            "scientific": scientific_engine
        },
        "smart_router": smart_router,
        "session_manager": session_manager,
        "science_tools": {
            "connectors": connectors,
            "retriever_factory": EvidenceRetriever,
            "claim_verifier": claim_verifier,
            "synthesizer": synthesizer,
            "artifact_store": artifact_store,
            "research_graph": research_graph,
            "browser_service": browser_service,
            "vision_analyzer": vision_analyzer,
        },
    })

    logger.info("UAgent system started successfully")

    yield

    # Shutdown
    logger.info("Shutting down UAgent system...")
    if cache:
        await cache.clear()
    await session_manager.shutdown()
    clear_app_state()
    logger.info("UAgent system shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Universal Agent (UAgent) API",
    description="Intelligent research system with multi-engine orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173", "http://101.6.5.211:3000", "http://101.6.5.211:3001"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"}
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "engines": {
            "deep_research": "active",
            "code_research": "active",
            "scientific_research": "active"
        },
        "smart_router": "active"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Universal Agent (UAgent) API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "research": "/api/research/",
            "router": "/api/router/",
            "engines": "/api/engines/"
        }
    }


# Import and include routers after app creation to avoid circular imports
from .routers import research, science, smart_router as router_endpoints, websocket

app.include_router(research.router, prefix="/api/research", tags=["research"])
app.include_router(science.router, prefix="/api/science", tags=["science"])
app.include_router(router_endpoints.router, prefix="/api/router", tags=["smart_router"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
