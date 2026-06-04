"""
FastAPI application factory.

Structure is intentionally kept thin here — business logic lives in the
individual routers and the persistence module.

Inference engine hook
---------------------
The placeholder below (``InferenceEngine``) shows exactly where the
TensorFlow Transformer model will be injected once it is ready.  Import the
model loader, attach it to ``app.state.model`` inside ``lifespan``, then
consume ``request.app.state.model`` from any endpoint that needs inference.
"""

from __future__ import annotations
from app.inference import load_inference_assets
import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import rest_landmarks, ws_landmarks
from app.persistence import STORAGE_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown logic
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Runs once on server startup and once on shutdown.

    Startup
    -------
    • Guarantees the .npy storage directory exists before any request arrives.
    • **[INFERENCE HOOK ACTIVATE]** Loads the trained Keras model and the
      dynamically generated label map into ``app.state`` so any endpoint can
      call ``request.app.state.model`` / ``request.app.state.label_map``.

    Shutdown
    --------
    • Gracefully release any model resources.
    """
    # --- Storage directory ---
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Storage directory ready: %s", STORAGE_DIR)

    # --- [INFERENCE HOOK ACTIVATE] ------------------------------------------
    try:
        model, label_map = load_inference_assets()
        app.state.model = model
        app.state.label_map = label_map
        logger.info(
            "Inference assets loaded. Classes (%d): %s",
            len(label_map),
            ", ".join(label_map),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Could not load inference assets. {exc}")
        app.state.model = None
        app.state.label_map = []
        logger.warning(
            "Running without inference model. Train a model and restart. (%s)", exc
        )
    # -------------------------------------------------------------------------

    logger.info("Application startup complete.")

    yield  # server is now running

    # Shutdown
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gesture Recognition API",
        description=(
            "Real-time landmark streaming (WebSocket) and batch persistence (REST) "
            "layer for the sign-language gesture recognition system."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    # Tighten ``allow_origins`` to the exact frontend origin in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(ws_landmarks.router)
    app.include_router(rest_landmarks.router)

    # ── Health check ──────────────────────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Quick liveness probe — returns 200 OK when the server is up."""
        return {"status": "ok"}

    return app


app = create_app()
