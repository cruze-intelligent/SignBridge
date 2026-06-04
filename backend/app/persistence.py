"""
Data-persistence utility.

Saves landmark sequences as serialised NumPy (.npy) arrays so they can be
used for future Transformer model retraining without any further conversion.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage root – resolved relative to this file so it works from any cwd.
# ---------------------------------------------------------------------------
STORAGE_DIR: Path = (
    Path(__file__).resolve().parent.parent / "storage" / "sequences"
)


def _ensure_storage_dir(label: str) -> Path:
    """
    Return a per-label sub-directory, creating it if necessary.

    Keeps sequences grouped by gesture class which makes dataset management
    straightforward when building training pipelines later.
    """
    target: Path = STORAGE_DIR / label
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_filename(label: str) -> str:
    """
    Deterministic, collision-resistant filename:
    ``<label>_<ISO-timestamp>_<uuid4-short>.npy``
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:8]
    # Sanitise label so it is safe to use in a file name.
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return f"{safe_label}_{ts}_{uid}.npy"


def save_sequence_sync(label: str, sequence: list[list[float]]) -> Path:
    """
    Serialise *sequence* to a ``.npy`` file and return the saved path.

    Parameters
    ----------
    label:
        Gesture class label used both for directory organisation and as part
        of the filename.
    sequence:
        List of landmark frames. Each frame is a flat list of 63 floats
        (21 hand keypoints × x, y, z).

    Returns
    -------
    Path
        Absolute path of the newly written file.

    Raises
    ------
    ValueError
        If any frame does not contain exactly 63 coordinates.
    """
    for idx, frame in enumerate(sequence):
        if len(frame) != 225:
            raise ValueError(
                f"Frame {idx} has {len(frame)} values; expected 225 "
                "(Right Hand 63 + Left Hand 63 + Pose 99)."
            )

    array = np.array(sequence, dtype=np.float32)  # shape: (N, 63)

    target_dir = _ensure_storage_dir(label)
    filename = build_filename(label)
    file_path = target_dir / filename

    np.save(file_path, array)
    logger.info(
        "Saved sequence | label=%s | frames=%d | file=%s",
        label,
        len(sequence),
        file_path,
    )
    return file_path


async def save_sequence(label: str, sequence: list[list[float]]) -> Path:
    """
    Async wrapper around :func:`save_sequence_sync`.

    Offloads the blocking NumPy I/O to a thread-pool executor so the event
    loop is never stalled during high-concurrency WebSocket sessions.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, save_sequence_sync, label, sequence
    )
