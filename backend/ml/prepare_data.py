"""
prepare_data.py
===============
Data-preparation pipeline for the Acholi / English Sign-Language Recogniser.

Pipeline
--------
1. Walk ``SEQUENCES_DIR`` – each sub-directory name is a gesture label.
2. Load every ``.npy`` file inside each label directory.
3. Validate each array has shape ``(30, 225)`` – reject malformed files with
   a warning rather than crashing the whole run.
4. Map string labels → integer indices → one-hot encoded vectors.
5. Split into stratified train / test sets via scikit-learn.
6. Persist the four arrays (X_train, X_test, y_train, y_test) plus the label
   map to ``DATASET_DIR`` so ``train.py`` can consume them immediately.

Usage
-----
    python backend/ml/prepare_data.py [--test-size 0.2] [--random-state 42]

Outputs
-------
    backend/ml/dataset/X_train.npy
    backend/ml/dataset/X_test.npy
    backend/ml/dataset/y_train.npy
    backend/ml/dataset/y_test.npy
    backend/ml/dataset/label_map.npy   # 1-D array of string labels ordered by index
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
_HERE: Path = Path(__file__).resolve().parent
SEQUENCES_DIR: Path = _HERE.parent / "storage" / "sequences"
DATASET_DIR: Path = _HERE / "dataset"

# ---------------------------------------------------------------------------
# Constants matching the frontend MediaPipe extraction contract
# ---------------------------------------------------------------------------
EXPECTED_FRAMES: int = 30
EXPECTED_FEATURES: int = 225  # Right-Hand 63 + Left-Hand 63 + Pose 99

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sequences(sequences_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """
    Recursively scan *sequences_dir* for ``.npy`` files.

    Each immediate sub-directory of *sequences_dir* is treated as a gesture
    label.  Files are validated against ``(EXPECTED_FRAMES, EXPECTED_FEATURES)``
    and invalid files are skipped with a warning.

    Returns
    -------
    sequences : list[np.ndarray]
        Valid sequence arrays, each shaped ``(30, 225)``.
    labels : list[str]
        Corresponding string label for every entry in *sequences*.
    """
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    skipped: int = 0

    if not sequences_dir.exists():
        raise FileNotFoundError(
            f"Sequences directory not found: {sequences_dir}\n"
            "Ensure the FastAPI backend has been used to record at least one gesture."
        )

    label_dirs = sorted(
        [d for d in sequences_dir.iterdir() if d.is_dir()]
    )

    if not label_dirs:
        raise RuntimeError(
            f"No label sub-directories found inside: {sequences_dir}\n"
            "Record gesture sequences via the frontend before running this script."
        )

    logger.info("Found %d label class(es) in %s", len(label_dirs), sequences_dir)

    for label_dir in label_dirs:
        label: str = label_dir.name
        npy_files = sorted(label_dir.glob("*.npy"))

        if not npy_files:
            logger.warning("Label '%s' has no .npy files – skipping.", label)
            continue

        class_loaded = 0
        for npy_path in npy_files:
            try:
                arr: np.ndarray = np.load(npy_path, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load '%s': %s – skipping.", npy_path, exc)
                skipped += 1
                continue

            # ----------------------------------------------------------------
            # Shape validation: must be exactly (30, 225)
            # ----------------------------------------------------------------
            if arr.shape != (EXPECTED_FRAMES, EXPECTED_FEATURES):
                logger.warning(
                    "Rejected '%s': shape %s ≠ expected (%d, %d).",
                    npy_path.name,
                    arr.shape,
                    EXPECTED_FRAMES,
                    EXPECTED_FEATURES,
                )
                skipped += 1
                continue

            sequences.append(arr.astype(np.float32))
            labels.append(label)
            class_loaded += 1

        logger.info(
            "  %-30s | loaded: %3d  |  skipped: %3d",
            label,
            class_loaded,
            len(npy_files) - class_loaded,
        )

    logger.info(
        "Total sequences loaded: %d  |  total skipped: %d",
        len(sequences),
        skipped,
    )

    if not sequences:
        raise RuntimeError(
            "No valid sequences were loaded.  "
            "Ensure files have shape (%d, %d)." % (EXPECTED_FRAMES, EXPECTED_FEATURES)
        )

    return sequences, labels


def encode_labels(labels: list[str]) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode string labels to one-hot vectors.

    Returns
    -------
    y_onehot : np.ndarray   shape ``(N, num_classes)``
    y_int    : np.ndarray   shape ``(N,)`` – integer class indices
    encoder  : LabelEncoder – fitted encoder (use ``encoder.classes_`` for the map)
    """
    encoder = LabelEncoder()
    y_int: np.ndarray = encoder.fit_transform(labels)  # shape (N,)
    num_classes: int = len(encoder.classes_)

    y_onehot = np.eye(num_classes, dtype=np.float32)[y_int]  # shape (N, C)
    return y_onehot, y_int, encoder


def save_dataset(
    dataset_dir: Path,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_classes: np.ndarray,
) -> None:
    """Persist the finalised training splits to *dataset_dir*."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.save(dataset_dir / "X_train.npy", X_train)
    np.save(dataset_dir / "X_test.npy", X_test)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "y_test.npy", y_test)
    # Save label map as a plain NumPy array of strings for easy loading
    np.save(dataset_dir / "label_map.npy", label_classes)

    # Also write a human-readable JSON mapping for quick inspection
    label_map_json = {int(i): str(cls) for i, cls in enumerate(label_classes)}
    with open(dataset_dir / "label_map.json", "w", encoding="utf-8") as fh:
        json.dump(label_map_json, fh, indent=2, ensure_ascii=False)

    logger.info("Dataset saved to: %s", dataset_dir)
    logger.info(
        "  X_train: %s  |  X_test: %s", X_train.shape, X_test.shape
    )
    logger.info(
        "  y_train: %s  |  y_test: %s", y_train.shape, y_test.shape
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare .npy gesture sequences for model training."
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=SEQUENCES_DIR,
        help=f"Root directory of labelled .npy sequences. Default: {SEQUENCES_DIR}",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Output directory for train/test splits. Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data reserved for the test set. Default: 0.20",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits. Default: 42",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Sign-Language Recogniser — Data Preparation")
    logger.info("=" * 60)
    logger.info("Sequences dir : %s", args.sequences_dir)
    logger.info("Dataset dir   : %s", args.dataset_dir)
    logger.info("Test size     : %.0f%%", args.test_size * 100)
    logger.info("Random state  : %d", args.random_state)
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # 1. Load sequences
    # ------------------------------------------------------------------
    raw_sequences, raw_labels = load_sequences(args.sequences_dir)

    # ------------------------------------------------------------------
    # 2. Stack into a single NumPy tensor: (N, 30, 225)
    # ------------------------------------------------------------------
    X: np.ndarray = np.stack(raw_sequences, axis=0)  # (N, 30, 225)
    logger.info("Combined feature tensor X shape: %s", X.shape)

    # ------------------------------------------------------------------
    # 3. Encode labels → one-hot
    # ------------------------------------------------------------------
    y_onehot, y_int, encoder = encode_labels(raw_labels)
    num_classes: int = len(encoder.classes_)
    logger.info(
        "Classes (%d): %s",
        num_classes,
        ", ".join(str(c) for c in encoder.classes_),
    )

    # ------------------------------------------------------------------
    # 4. Stratified train / test split
    #    Falls back to non-stratified if any class has < 2 samples.
    # ------------------------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_onehot,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_int,
        )
        logger.info("Stratified split applied.")
    except ValueError as exc:
        logger.warning(
            "Stratified split failed (%s). "
            "Falling back to non-stratified split – collect more samples per class.",
            exc,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_onehot,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    # ------------------------------------------------------------------
    # 5. Save finalised splits
    # ------------------------------------------------------------------
    save_dataset(
        dataset_dir=args.dataset_dir,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_classes=encoder.classes_,
    )

    logger.info("-" * 60)
    logger.info("Data preparation complete.  Run train.py to start training.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
