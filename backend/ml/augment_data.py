"""
augment_data.py
===============
Synthetically multiplies existing gesture sequences using landmark-aware
augmentation transforms.  Reads every real ``.npy`` sequence in
``storage/sequences/<label>/``, applies one or more transforms, and saves
the augmented copies alongside the originals so that
``prepare_data.py`` / ``train.py`` can consume them immediately.

225-float frame layout (must match frontend + collect_data.py):
    [0  : 63 ] Right Hand  — 21 landmarks × (x-nose, y-nose, z)  nose-relative
    [63 : 126] Left  Hand  — 21 landmarks × (x-nose, y-nose, z)  nose-relative
    [126: 225] Pose        — 33 landmarks × (x, y, z)             raw normalised

Available augmentation strategies
----------------------------------
1. gaussian_noise     — adds per-frame positional jitter (σ ≈ 0.010)
2. temporal_warp      — resamples frames to simulate signing faster / slower
3. spatial_scale      — scales hand coords by ±20 % (hand distance variation)
4. mirror_flip        — negates x-coords and swaps RH ↔ LH blocks
5. temporal_shift     — shifts the gesture 2–5 frames earlier / later with edge padding
6. rotation_jitter    — small 2D rotation (±12°) around wrist to simulate head/hand tilt
7. landmark_dropout   — randomly zeroes 1–4 landmarks per frame (tracking-loss resilience)
8. coord_jitter       — independent per-landmark noise (σ varies by landmark type)
9. speed_perturbation — non-linear temporal warping (variable speed within the sequence)

Usage
-----
    python backend/ml/augment_data.py [--target 100] [--seed 42]

    --target  : desired number of sequences per class (default: 100)
    --seed    : random seed for reproducibility (default: 42)
    --dry-run : print what would be saved without writing anything
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d  # scipy ships with standard ML envs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE: Path = Path(__file__).resolve().parent
SEQUENCES_DIR: Path = _HERE.parent / "storage" / "sequences"

# ---------------------------------------------------------------------------
# Frame-layout constants  (must stay in sync with collect_data.py / app.js)
# ---------------------------------------------------------------------------
NUM_FRAMES:    int = 30
NUM_FEATURES:  int = 225

RH_START:  int = 0     # right-hand block start index
RH_END:    int = 63    # right-hand block end index   (exclusive)
LH_START:  int = 63    # left-hand block start index
LH_END:    int = 126   # left-hand block end index    (exclusive)
POSE_START: int = 126  # pose block start index
POSE_END:   int = 225  # pose block end index          (exclusive)

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


# ===========================================================================
#  Augmentation transforms
#  Each function accepts a (30, 225) float32 array and returns a new one.
# ===========================================================================

def aug_gaussian_noise(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Add per-frame Gaussian noise to all coordinates.

    Simulates natural hand tremor and mediapipe tracking micro-jitter.
    Noise std is moderate (0.010) to force the model to learn gesture
    *shape* rather than memorising exact coordinate values.
    """
    noise = rng.normal(loc=0.0, scale=0.010, size=seq.shape).astype(np.float32)
    # Zero-padded frames (all zeros) should stay zero — don't noise them
    zero_mask = (seq == 0.0)
    result = seq + noise
    result[zero_mask] = 0.0
    return result


def aug_temporal_warp(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Resample the 30-frame sequence at a randomly altered playback rate.

    Simulates signing slightly faster (rate > 1) or slower (rate < 1).
    The output is always exactly 30 frames via linear interpolation.

    Rate range: [0.75 ... 1.30]  (25 % slower to 30 % faster)
    """
    rate = rng.uniform(0.75, 1.30)

    # Original frame indices
    original_indices = np.linspace(0, NUM_FRAMES - 1, NUM_FRAMES)
    # New sampling positions in the original timeline
    num_source = max(2, int(round(NUM_FRAMES * rate)))
    source_indices = np.linspace(0, NUM_FRAMES - 1, num_source)

    # Interpolate feature-by-feature using the source positions
    interpolator = interp1d(
        original_indices, seq, axis=0, kind="linear", fill_value="extrapolate"
    )
    stretched = interpolator(source_indices)  # shape: (num_source, 225)

    # Re-sample back to exactly 30 frames
    resampler = interp1d(
        np.linspace(0, 1, num_source),
        stretched,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    result = resampler(np.linspace(0, 1, NUM_FRAMES)).astype(np.float32)
    return result


def aug_spatial_scale(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Scale hand landmark coordinates by a random factor in [0.80, 1.25].

    Simulates the signer's hand being closer to or farther from the camera,
    changing the apparent hand size relative to the body.

    Only the hand blocks (RH + LH) are scaled — pose stays untouched so the
    body anchor reference remains consistent.
    """
    scale = rng.uniform(0.80, 1.25)
    result = seq.copy()
    # Scale both hand blocks together so relative RH/LH proportions hold
    result[:, RH_START:LH_END] *= scale  # covers [0:126] = RH + LH
    return result


def aug_mirror_flip(seq: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    """
    Produce a left-right mirrored version of the sequence.

    In the normalised coordinate space:
      - Negate the x-component of every landmark (mirror across the centre)
      - Swap the Right-Hand and Left-Hand blocks (a mirrored right hand
        becomes a left hand and vice-versa)
      - Negate the x-component of pose landmarks

    This effectively doubles your dataset by adding the signer's mirror image,
    which is useful if the training data was recorded predominantly with one
    dominant hand.
    """
    result = seq.copy()

    # Helper: negate the x-coordinate of every (x, y, z) triplet in a slice
    def negate_x(arr: np.ndarray, start: int, end: int) -> None:
        # x is at indices 0, 3, 6, … within the slice → every 3rd starting at 0
        for i in range(start, end, 3):
            arr[:, i] *= -1.0

    # 1. Negate x for both hand blocks BEFORE swapping
    negate_x(result, RH_START, RH_END)
    negate_x(result, LH_START, LH_END)

    # 2. Swap RH ↔ LH blocks
    rh_copy = result[:, RH_START:RH_END].copy()
    result[:, RH_START:RH_END] = result[:, LH_START:LH_END]
    result[:, LH_START:LH_END] = rh_copy

    # 3. Negate x for pose landmarks
    negate_x(result, POSE_START, POSE_END)

    return result


def aug_temporal_shift(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Shift the gesture a few frames earlier or later, padding at the edges.

    A positive shift moves the gesture later (pads start with the first frame).
    A negative shift moves the gesture earlier (pads end with the last frame).

    Shift range: [-5 … +5] frames.
    """
    shift = int(rng.integers(-5, 6))  # inclusive on both ends
    if shift == 0:
        return seq.copy()

    result = np.empty_like(seq)
    if shift > 0:
        # Gesture starts later — repeat first frame as silent padding
        result[:shift]  = seq[0]
        result[shift:]  = seq[:NUM_FRAMES - shift]
    else:
        # Gesture starts earlier — repeat last frame as padding at the end
        abs_shift = abs(shift)
        result[:NUM_FRAMES - abs_shift] = seq[abs_shift:]
        result[NUM_FRAMES - abs_shift:] = seq[-1]

    return result.astype(np.float32)


# ===========================================================================
#  NEW augmentation transforms (v2 — low-data regime boosters)
# ===========================================================================

def aug_rotation_jitter(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a small 2D rotation (±12°) to hand landmarks around the wrist.

    Simulates natural wrist rotation and slight camera/head tilt.  Only the
    (x, y) components of each hand landmark are rotated; z is untouched.
    Pose landmarks are left unchanged so the body anchor stays stable.
    """
    angle_deg = rng.uniform(-12.0, 12.0)
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    result = seq.copy()

    for block_start, block_end in [(RH_START, RH_END), (LH_START, LH_END)]:
        # Compute the centroid of the hand block to rotate around it
        # (effectively rotating around the wrist/palm centre)
        x_indices = list(range(block_start, block_end, 3))      # x at 0, 3, 6, …
        y_indices = list(range(block_start + 1, block_end, 3))  # y at 1, 4, 7, …

        for t in range(NUM_FRAMES):
            # Skip zero-padded frames (hand not detected)
            if np.all(result[t, block_start:block_end] == 0.0):
                continue

            xs = result[t, x_indices]
            ys = result[t, y_indices]
            cx, cy = xs.mean(), ys.mean()

            # Rotate around centroid
            dx, dy = xs - cx, ys - cy
            result[t, x_indices] = cx + dx * cos_a - dy * sin_a
            result[t, y_indices] = cy + dx * sin_a + dy * cos_a

    return result


def aug_landmark_dropout(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly zero-out 1–4 landmarks per frame to simulate MediaPipe tracking loss.

    In production, MediaPipe frequently loses individual finger-tip landmarks
    for 1-2 frames — especially in low-light or fast-motion conditions.  By
    training on sequences with sporadic landmark dropout, the model learns to
    be robust to these partial-visibility frames rather than misclassifying.

    Only hand landmarks are dropped (not pose) because pose is rarely lost
    while hands are visible.
    """
    result = seq.copy()
    n_drop = int(rng.integers(1, 5))  # 1–4 landmarks to drop per frame

    # Total hand landmarks: 21 (RH) + 21 (LH) = 42, each with 3 coords
    hand_landmark_count = (RH_END - RH_START + LH_END - LH_START) // 3  # = 42

    for t in range(NUM_FRAMES):
        # Pick which landmarks to drop (indices 0–41 across both hands)
        drop_indices = rng.choice(hand_landmark_count, size=n_drop, replace=False)

        for lm_idx in drop_indices:
            if lm_idx < 21:
                # Right hand landmark
                start = RH_START + lm_idx * 3
            else:
                # Left hand landmark
                start = LH_START + (lm_idx - 21) * 3
            result[t, start:start + 3] = 0.0

    return result


def aug_coord_jitter(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Per-landmark noise with varying sigma by landmark type.

    Finger tips (landmarks 4, 8, 12, 16, 20 in each hand) get 2× the noise
    of palm/wrist landmarks because MediaPipe tracks them less reliably.
    Pose landmarks get very low noise (σ = 0.003) to preserve body anchor.
    """
    result = seq.copy()

    # Finger-tip landmark indices within each 21-landmark hand block
    finger_tip_offsets = [4, 8, 12, 16, 20]

    for block_start, block_end in [(RH_START, RH_END), (LH_START, LH_END)]:
        for lm in range(21):
            col_start = block_start + lm * 3
            col_end = col_start + 3
            sigma = 0.016 if lm in finger_tip_offsets else 0.008
            noise = rng.normal(0.0, sigma, size=(NUM_FRAMES, 3)).astype(np.float32)
            # Don't noise zero-padded landmarks
            mask = np.all(result[:, col_start:col_end] == 0.0, axis=1)
            noise[mask] = 0.0
            result[:, col_start:col_end] += noise

    # Light noise on pose
    pose_noise = rng.normal(0.0, 0.003, size=(NUM_FRAMES, POSE_END - POSE_START)).astype(np.float32)
    pose_mask = np.all(result[:, POSE_START:POSE_END] == 0.0, axis=1)
    pose_noise[pose_mask] = 0.0
    result[:, POSE_START:POSE_END] += pose_noise

    return result


def aug_speed_perturbation(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Non-linear temporal warping — variable speed within the same sequence.

    Instead of a uniform rate change (like temporal_warp), this splits the
    30-frame sequence into 3 segments and applies a different speed factor
    to each.  The result simulates realistic signing patterns: slow start,
    fast middle, slow end (or vice versa).
    """
    n_segments = 3
    seg_len = NUM_FRAMES // n_segments  # 10 frames per segment

    # Generate per-segment speed factors
    speeds = rng.uniform(0.65, 1.45, size=n_segments)

    # Build a non-linear time mapping
    original_times = np.linspace(0, 1, NUM_FRAMES)
    new_times = []
    for seg_idx in range(n_segments):
        start_frame = seg_idx * seg_len
        end_frame = start_frame + seg_len if seg_idx < n_segments - 1 else NUM_FRAMES
        n = end_frame - start_frame
        # Map this segment's frames through a speed factor
        segment_duration = n / speeds[seg_idx]
        new_times.extend(np.linspace(0, 1, max(2, int(round(segment_duration)))))

    # Normalise new_times to [0, 1] and resample back to exactly 30 frames
    new_times = np.array(new_times)
    new_times = (new_times - new_times[0]) / (new_times[-1] - new_times[0] + 1e-8)

    # Resample to exactly NUM_FRAMES positions
    target_times = np.linspace(0, 1, NUM_FRAMES)

    from scipy.interpolate import interp1d as _interp1d
    # Build interpolator from new_times → feature values
    # First map new_times back to original frame positions
    frame_positions = np.linspace(0, NUM_FRAMES - 1, len(new_times))
    orig_interpolator = _interp1d(
        np.linspace(0, 1, NUM_FRAMES), seq, axis=0,
        kind="linear", fill_value="extrapolate",
    )

    # Sample at the warped positions
    warped_positions = np.interp(target_times, new_times, frame_positions / (NUM_FRAMES - 1))
    warped_positions = warped_positions * (NUM_FRAMES - 1)

    result_interpolator = _interp1d(
        np.arange(NUM_FRAMES), seq, axis=0,
        kind="linear", fill_value="extrapolate",
    )
    result = result_interpolator(np.clip(warped_positions, 0, NUM_FRAMES - 1))

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry: maps transform name → function
# All transforms have the same signature: (seq, rng) → augmented_seq
# ---------------------------------------------------------------------------
TRANSFORMS: dict = {
    "noise":    aug_gaussian_noise,
    "warp":     aug_temporal_warp,
    "scale":    aug_spatial_scale,
    "flip":     aug_mirror_flip,
    "shift":    aug_temporal_shift,
    "rotate":   aug_rotation_jitter,
    "drop":     aug_landmark_dropout,
    "jitter":   aug_coord_jitter,
    "speed":    aug_speed_perturbation,
}


# ===========================================================================
#  Core augmentation pipeline
# ===========================================================================

def load_class_sequences(label_dir: Path) -> list[np.ndarray]:
    """Load and validate all real .npy sequences in a label directory."""
    seqs: list[np.ndarray] = []
    for npy_path in sorted(label_dir.glob("*.npy")):
        # Skip files we already generated (they are prefixed with "AUG_")
        if npy_path.stem.startswith("AUG_"):
            continue
        try:
            arr = np.load(npy_path, allow_pickle=False)
        except Exception as exc:
            logger.warning("  Skipping '%s': %s", npy_path.name, exc)
            continue

        if arr.shape != (NUM_FRAMES, NUM_FEATURES):
            logger.warning(
                "  Skipping '%s': shape %s ≠ (%d, %d)",
                npy_path.name, arr.shape, NUM_FRAMES, NUM_FEATURES,
            )
            continue

        seqs.append(arr.astype(np.float32))
    return seqs


def augment_class(
    label_dir: Path,
    target_count: int,
    rng: np.random.Generator,
    dry_run: bool = False,
) -> int:
    """
    Bring the sequence count for one class up to *target_count*.

    Strategy
    --------
    1. Count existing real sequences.
    2. If already at or above target, skip.
    3. Otherwise repeatedly sample a random real sequence, apply a randomly
       chosen transform (or two, chained), and save the result.

    Returns
    -------
    Number of new sequences written (0 on dry-run).
    """
    real_seqs = load_class_sequences(label_dir)
    n_real = len(real_seqs)

    # Count pre-existing augmented files so we don't overshoot on re-runs
    n_existing_aug = len(list(label_dir.glob("AUG_*.npy")))
    n_total = n_real + n_existing_aug

    if n_total >= target_count:
        logger.info(
            "  %-15s : %d real + %d aug = %d  (>= target %d) - skipping",
            label_dir.name, n_real, n_existing_aug, n_total, target_count,
        )
        return 0

    if n_real == 0:
        logger.warning("  %-15s : no real sequences found — skipping", label_dir.name)
        return 0

    needed = target_count - n_total
    transform_names = list(TRANSFORMS.keys())
    written = 0

    logger.info(
        "  %-15s : %d real, %d existing aug -> need %d more",
        label_dir.name, n_real, n_existing_aug, needed,
    )

    for _ in range(needed):
        # 1. Pick a random source sequence
        source = real_seqs[rng.integers(0, n_real)]

        # 2. Pick 1–3 transforms to chain (chaining adds combinatorial diversity)
        #    With 9 transforms, chaining 2 or 3 is safe and dramatically
        #    increases the effective augmentation space.
        n_transforms = rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15])
        chosen = rng.choice(transform_names, size=n_transforms, replace=False)

        augmented = source.copy()
        tag_parts = []
        for name in chosen:
            augmented = TRANSFORMS[name](augmented, rng)
            tag_parts.append(name[:3])  # short tag for the filename

        tag = "_".join(tag_parts)
        uid = uuid.uuid4().hex[:8]
        out_path = label_dir / f"AUG_{tag}_{uid}.npy"

        if not dry_run:
            np.save(out_path, augmented)
        written += 1

    return written


# ===========================================================================
#  Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment gesture sequences to reach a target count per class.\n"
            "Reads from storage/sequences/ and writes AUG_*.npy files there."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=int,
        default=200,
        help="Target number of sequences per class (real + augmented). Default: 200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=SEQUENCES_DIR,
        help=f"Root directory of labelled sequences. Default: {SEQUENCES_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be saved without writing any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    logger.info("=" * 60)
    logger.info("SignBridge — Gesture Data Augmentation")
    logger.info("=" * 60)
    logger.info("Sequences dir : %s", args.sequences_dir)
    logger.info("Target count  : %d per class", args.target)
    logger.info("Random seed   : %d", args.seed)
    logger.info("Dry run       : %s", args.dry_run)
    logger.info("Transforms    : %s", ", ".join(TRANSFORMS.keys()))
    logger.info("-" * 60)

    if not args.sequences_dir.exists():
        logger.error("Sequences directory not found: %s", args.sequences_dir)
        sys.exit(1)

    label_dirs = sorted([d for d in args.sequences_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        logger.error("No label sub-directories found in %s", args.sequences_dir)
        sys.exit(1)

    total_written = 0
    for label_dir in label_dirs:
        written = augment_class(
            label_dir=label_dir,
            target_count=args.target,
            rng=rng,
            dry_run=args.dry_run,
        )
        total_written += written

    logger.info("-" * 60)
    if args.dry_run:
        logger.info("DRY RUN complete. %d files would be written.", total_written)
    else:
        logger.info("Augmentation complete. %d new sequences written.", total_written)
    logger.info(
        "Next steps:\n"
        "  1. python backend/ml/prepare_data.py\n"
        "  2. python backend/ml/train.py"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
