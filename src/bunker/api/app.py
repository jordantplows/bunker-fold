"""FastAPI application for biological model inference."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bunker.api.routes import embeddings, models, structures
from bunker.api.schemas import ErrorResponse, ErrorDetail


# Model cache - loaded models are kept in memory
model_cache: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup
    app.state.model_cache = model_cache
    app.state.start_time = time.time()
    yield
    # Shutdown
    model_cache.clear()


# Create FastAPI app
app = FastAPI(
    title="Bunker API",
    description=(
        "OpenAI-compatible API for biological foundation models. "
        "Supports protein embeddings, structure prediction, and sequence design."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=exc.detail,
                    type="invalid_request_error",
                    code=str(exc.status_code),
                )
            ).model_dump(),
        )

    # Unhandled exceptions
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc),
                type="internal_server_error",
                code="500",
            )
        ).model_dump(),
    )


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    start_time = getattr(app.state, "start_time", time.time())
    model_cache = getattr(app.state, "model_cache", {})
    return {
        "status": "healthy",
        "uptime": time.time() - start_time,
        "models_loaded": len(model_cache),
    }


# Include routers
app.include_router(models.router, prefix="/v1", tags=["Models"])
app.include_router(embeddings.router, prefix="/v1", tags=["Embeddings"])
app.include_router(structures.router, prefix="/v1", tags=["Structures"])


@app.get("/", tags=["System"])
async def root():
    """Root endpoint."""
    return {
        "message": "Bunker API - Biological Foundation Models",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
