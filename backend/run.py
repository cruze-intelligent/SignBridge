"""
Entry-point for running the server directly:
    python run.py

For production (Render/Railway/Fly.io) the PORT env var is respected
automatically so the platform's health checks connect on the correct port.
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    is_dev = os.environ.get("ENV", "development") == "development"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=is_dev,
        reload_dirs=["app", "ml"] if is_dev else [],
        reload_excludes=[".venv", ".venv/*", "models", "storage"],
        log_level="info",
        # Uncomment for multi-process production (disable reload first):
        # workers=4,
    )
