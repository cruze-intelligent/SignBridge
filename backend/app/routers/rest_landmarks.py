"""
REST router — /api/landmarks/batch

Accepts pre-recorded gesture sequences for intermittent / bulk submissions
and persists them as .npy files.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.persistence import save_sequence
from app.schemas import BatchLandmarkRequest, BatchLandmarkResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/landmarks", tags=["landmarks"])


@router.post(
    "/batch",
    response_model=BatchLandmarkResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a batch of pre-recorded landmark sequences",
    description=(
        "Accepts one or more landmark frames for a named gesture class and "
        "persists them as a serialised NumPy array for model retraining."
    ),
)
async def batch_landmarks(body: BatchLandmarkRequest) -> BatchLandmarkResponse:
    """
    Persist a batch of landmark frames to the `.npy` store.

    - **gesture_label**: class name used for directory and filename.
    - **sequences**: list of frames, each being 63 floats
      (21 hand landmarks × x, y, z).
    """
    try:
        saved_path = await save_sequence(
            label=body.gesture_label,
            sequence=body.sequences,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except OSError as exc:
        logger.exception("Disk I/O error while saving batch sequence.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist sequence: {exc}",
        ) from exc

    return BatchLandmarkResponse(
        status="saved",
        saved_file=str(saved_path),
        frames_saved=len(body.sequences),
    )
