"""frevia-ai — service AI nội bộ (FastAPI)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import errors, llm as llm_mod, routes
from .config import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(level=settings.log_level)

    llm_client = llm_mod.OllamaClient(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        llm_client.close()

    app = FastAPI(title="frevia-ai", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.llm = llm_client

    app.include_router(routes.router)

    @app.exception_handler(errors.ApiError)
    async def api_error_handler(_request: Request, exc: errors.ApiError):
        return JSONResponse(
            status_code=exc.status,
            content={"detail": exc.detail, "message": exc.message},
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        ok = request.app.state.llm.ping()
        return {"status": "ok" if ok else "unavailable"}

    return app


app = create_app()