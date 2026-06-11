"""
train.py
========
Model-training script for the Acholi / English Sign-Language Recogniser.

Architecture
------------
A stacked Bidirectional-LSTM network inspired by the temporal-sequence pattern
used in sign-language recognition projects.
The model is updated to accept our 225-feature holistic landmark schema:

    Input  : (batch, 30, 225)
              30 frames × (Right-Hand 63 + Left-Hand 63 + Pose 99)
    Layer 1: Bidirectional LSTM 128 units, return_sequences=True
    Layer 2: Bidirectional LSTM  64 units, return_sequences=True
    Layer 3: LSTM  64 units, return_sequences=False
    Dense 1: 128 units, ReLU + BatchNorm + Dropout(0.4)
    Dense 2: 64  units, ReLU + BatchNorm + Dropout(0.3)
    Output : Dense(num_classes, softmax)

The number of output units is determined dynamically from the loaded dataset.

Usage
-----
    python backend/ml/train.py [--epochs 150] [--batch-size 32] [--lr 0.001]

Outputs
-------
    backend/models/sign_language_model_v2.h5   (best checkpoint, H5)
    backend/models/training_history.json        (epoch-by-epoch metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = _HERE / "dataset"
MODELS_DIR: Path = _HERE.parent / "models"
MODEL_SAVE_PATH: Path = MODELS_DIR / "sign_language_model_v2.h5"
HISTORY_SAVE_PATH: Path = MODELS_DIR / "training_history.json"

# ---------------------------------------------------------------------------
# Input shape contract (must match prepare_data.py and frontend extraction)
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH: int = 30    # number of frames per gesture window
NUM_FEATURES: int = 225      # floats per frame: RH-63 + LH-63 + Pose-99

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
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the training split produced by ``prepare_data.py``.

    Returns
    -------
    X_train    : np.ndarray  shape ``(N, 30, 225)``
    y_train    : np.ndarray  shape ``(N, num_classes)``  one-hot
    label_map  : list[str]   ordered class names
    """
    required = ["X_train.npy", "y_train.npy", "label_map.npy"]
    for fname in required:
        fpath = dataset_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {fpath}\n"
                "Run prepare_data.py first to generate the dataset."
            )

    X_train: np.ndarray = np.load(dataset_dir / "X_train.npy", allow_pickle=False)
    y_train: np.ndarray = np.load(dataset_dir / "y_train.npy", allow_pickle=False)
    label_map_arr: np.ndarray = np.load(
        dataset_dir / "label_map.npy", allow_pickle=True
    )
    label_map: list[str] = [str(cls) for cls in label_map_arr]

    logger.info("X_train shape : %s  dtype: %s", X_train.shape, X_train.dtype)
    logger.info("y_train shape : %s  dtype: %s", y_train.shape, y_train.dtype)
    logger.info("Classes (%d)  : %s", len(label_map), ", ".join(label_map))

    # Sanity checks
    assert X_train.ndim == 3, f"Expected 3-D X_train, got {X_train.ndim}-D"
    assert X_train.shape[1] == SEQUENCE_LENGTH, (
        f"Sequence length mismatch: expected {SEQUENCE_LENGTH}, "
        f"got {X_train.shape[1]}"
    )
    assert X_train.shape[2] == NUM_FEATURES, (
        f"Feature count mismatch: expected {NUM_FEATURES}, "
        f"got {X_train.shape[2]}"
    )
    assert y_train.shape[1] == len(label_map), (
        "One-hot width ≠ number of labels"
    )

    return X_train, y_train, label_map


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> "Model":  # noqa: F821
    """
    Construct and return the compiled Bidirectional-LSTM model.
    """
    import tensorflow as tf
    import tensorflow.compat.v2
    
    # Bypass tf.keras bug by importing directly from keras
    from keras import layers
    from keras.models import Model
    from keras import optimizers
    from keras import metrics

    logger.info(
        "TensorFlow version: %s  |  GPU devices: %s",
        tf.__version__,
        tf.config.list_physical_devices("GPU"),
    )

    # ------------------------------------------------------------------
    # Enable GPU memory growth to avoid OOM errors on shared machines
    # ------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logger.warning("Could not set GPU memory growth: %s", exc)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    inputs = layers.Input(
        shape=(SEQUENCE_LENGTH, NUM_FEATURES),
        name="landmark_sequence",
    )

    # ------------------------------------------------------------------
    # Recurrent layers
    # ------------------------------------------------------------------
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1",
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2",
    )(x)

    x = layers.LSTM(64, return_sequences=False, name="lstm_3")(x)

    # ------------------------------------------------------------------
    # Dense classification head
    # ------------------------------------------------------------------
    x = layers.Dense(128, name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.Dropout(0.40, name="dropout_1")(x)

    x = layers.Dense(64, name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)

    # ------------------------------------------------------------------
    # Output layer: dynamically sized to num_classes
    # ------------------------------------------------------------------
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="gesture_output",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="SignLanguageLSTM_v2")

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: "Model",  # noqa: F821
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float,
) -> "callbacks.History":  # noqa: F821
    """
    Fit the model with EarlyStopping and ModelCheckpoint callbacks.
    """
    import tensorflow as tf
    import tensorflow.compat.v2
    
    # Bypass tf.keras bug by importing callbacks directly
    from keras import callbacks

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        # Stop training when val_loss has not improved for 15 epochs.
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save the best model checkpoint to disk.
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Reduce LR on plateau for smoother convergence.
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard logs.
        callbacks.TensorBoard(
            log_dir=str(MODELS_DIR / "logs"),
            histogram_freq=1,
        ),
    ]

    logger.info(
        "Starting training | epochs=%d  batch=%d  val_split=%.0f%%",
        epochs,
        batch_size,
        validation_split * 100,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks_list,
        shuffle=True,
        verbose=2,  # One line per epoch – clean for CI/CD logs
    )
    return history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_history(history: "callbacks.History") -> None:  # noqa: F821
    """Persist epoch-by-epoch metrics as JSON for later analysis."""
    serialisable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    HISTORY_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Training history saved to: %s", HISTORY_SAVE_PATH)


def evaluate_on_test(
    model: "Model",  # noqa: F821
    dataset_dir: Path,
) -> None:
    """Load the test split and evaluate the best-saved model."""
    x_test_path = dataset_dir / "X_test.npy"
    y_test_path = dataset_dir / "y_test.npy"

    if not x_test_path.exists() or not y_test_path.exists():
        logger.warning("Test split not found – skipping evaluation.")
        return

    X_test: np.ndarray = np.load(x_test_path, allow_pickle=False)
    y_test: np.ndarray = np.load(y_test_path, allow_pickle=False)

    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names
    for name, value in zip(metric_names, results):
        logger.info("Test %-25s : %.4f", name, value)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Sign-Language LSTM model on prepared .npy datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Directory containing X_train.npy / y_train.npy. Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum number of training epochs. Default: 150",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size. Default: 32",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial Adam learning rate. Default: 0.001",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Fraction of training data used for validation. Default: 0.20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Sign-Language Recogniser — Model Training")
    logger.info("=" * 60)
    logger.info("Dataset dir   : %s", args.dataset_dir)
    logger.info("Model output  : %s", MODEL_SAVE_PATH)
    logger.info("Epochs        : %d", args.epochs)
    logger.info("Batch size    : %d", args.batch_size)
    logger.info("Learning rate : %g", args.lr)
    logger.info("Val split     : %.0f%%", args.val_split * 100)
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # 1. Load prepared dataset
    # ------------------------------------------------------------------
    X_train, y_train, label_map = load_dataset(args.dataset_dir)
    num_classes: int = len(label_map)

    # ------------------------------------------------------------------
    # 2. Build model (input_shape updated to 225 features per frame)
    # ------------------------------------------------------------------
    model = build_model(num_classes=num_classes)

    # ------------------------------------------------------------------
    # Override compiled LR if a custom value was supplied via --lr
    # ------------------------------------------------------------------
    if args.lr != 1e-3:
        # We can still safely import tf here to access the backend assignment
        import tensorflow as tf  # noqa: PLC0415

        model.optimizer.learning_rate.assign(args.lr)
        logger.info("Learning rate overridden to: %g", args.lr)

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    history = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
    )

    # ------------------------------------------------------------------
    # 4. Persist history
    # ------------------------------------------------------------------
    save_history(history)

    # ------------------------------------------------------------------
    # 5. Evaluate on held-out test split
    # ------------------------------------------------------------------
    evaluate_on_test(model, args.dataset_dir)

    logger.info("-" * 60)
    logger.info("Best model saved to : %s", MODEL_SAVE_PATH)
    logger.info("Training complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()