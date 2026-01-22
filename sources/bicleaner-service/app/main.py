"""FastAPI application for bicleaner-service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import settings
from .routers.v1 import router as v1_router
from .schemas import HealthResponse
from .service import BicleanerService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model at startup, cleanup at shutdown."""
    logger.info("Starting bicleaner-service...")

    # Startup: Load model
    service = BicleanerService()
    try:
        service.load()
        app.state.service = service
        logger.info("Model loaded, service ready")
    except Exception as e:
        logger.exception("Failed to load model")
        app.state.service = None

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down bicleaner-service...")
    if app.state.service:
        app.state.service.shutdown()


app = FastAPI(
    title="Bicleaner Service",
    description="HTTP microservice for bicleaner-ai parallel corpus quality scoring",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns 503 if model is not loaded.
    """
    service = app.state.service

    if service is None or not service._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return HealthResponse(
        status="healthy",
        model=settings.model_type,
        model_path=settings.model_path,
    )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "bicleaner-service",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
