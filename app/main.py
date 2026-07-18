"""FastAPI application entry point.

Exposes an app factory (:func:`create_app`) plus a module-level ``app`` instance
so it can be served with ``uvicorn app.main:app``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.ctgov.client import CTGovClient
from app.ctgov.introspection import load_introspection
from app.schemas.request import QueryRequest
from app.schemas.response import VisualizeResponse
from app.services.pipeline import run_visualization

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run one-time API introspection at startup.

    Failures inside introspection are handled internally (per-endpoint
    fallbacks), so a network hiccup at boot degrades gracefully rather than
    preventing the app from starting.
    """
    try:
        async with CTGovClient() as client:
            intro = await load_introspection(client)
        logger.info("Startup introspection complete: sources=%s", intro.sources)
    except Exception as exc:  # noqa: BLE001 — never block startup
        logger.warning("Startup introspection could not run (%s).", exc)
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    # Ensure application INFO logs (e.g. startup introspection) are emitted.
    # basicConfig is a no-op if the root logger is already configured.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = FastAPI(
        title="Cheiron",
        description=(
            "Turns natural-language questions about clinical trials into "
            "structured visualization specifications, backed by live "
            "ClinicalTrials.gov data."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Permissive CORS for local use (so the static frontend can call the API
    # directly, including from a file:// origin).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the static frontend at the root so http://host:port/ shows the UI."""
        return FileResponse(frontend)

    @app.post("/visualize", response_model=VisualizeResponse)
    async def visualize(request: QueryRequest) -> VisualizeResponse:
        """Turn a natural-language query into a visualization specification.

        Runs the six-stage pipeline: interpret → validate → compile →
        fetch+aggregate → viz shaping → assemble.
        """
        try:
            return await run_visualization(request)
        except ValueError as exc:
            # Enum/shape validation failures are client errors.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
