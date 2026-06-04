"""
Entry-point for running the server directly:
    python run.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # hot-reload on code changes during development
        log_level="info",
        # workers=4           # uncomment for multi-process production mode
        #                     # (disable reload first)
    )
