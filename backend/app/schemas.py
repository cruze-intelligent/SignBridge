"""
Pydantic schemas for request / response validation.
"""

from __future__ import annotations

from typing import Annotated, List

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Each landmark frame: a flat array of 225 floats
#   Right Hand : 21 landmarks × (x, y, z) =  63 floats
#   Left  Hand : 21 landmarks × (x, y, z) =  63 floats
#   Pose       : 33 landmarks × (x, y, z) =  99 floats
#   Total                                  = 225 floats
# ---------------------------------------------------------------------------
LandmarkFrame = Annotated[
    List[float],
    Field(
        min_length=225,
        max_length=225,
        description=(
            "Flat list of 225 floats: "
            "Right Hand (63) + Left Hand (63) + Pose (99)."
        ),
    ),
]


class BatchLandmarkRequest(BaseModel):
    """Payload for the REST batch endpoint."""

    gesture_label: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable label for the gesture class being submitted.",
    )
    sequences: List[LandmarkFrame] = Field(
        ...,
        min_length=1,
        description="One or more landmark frames that form a gesture sequence.",
    )


class BatchLandmarkResponse(BaseModel):
    """Acknowledgement returned after a successful batch save."""

    status: str
    saved_file: str
    frames_saved: int
