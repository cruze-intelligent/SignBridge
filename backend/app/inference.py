import os
import json
import shutil
import tempfile
import numpy as np
import tensorflow as tf

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_language_model_v3.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "ml", "dataset", "label_map.npy")


def _strip_time_major(obj):
    """
    Recursively walk a JSON-deserialized model config and remove every
    occurrence of `time_major` from LSTM / BiLSTM config dicts.

    TF < 2.16 serialised this kwarg explicitly; TF 2.16+ rejects it.
    Removing it from the stored config lets `load_model` succeed without
    retraining or re-saving the weights.
    """
    if isinstance(obj, dict):
        obj.pop("time_major", None)
        for v in obj.values():
            _strip_time_major(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_time_major(item)
    return obj


def _load_model_compat(model_path: str):
    """
    Load a Keras .h5 model, transparently stripping any `time_major` kwarg
    that TF 2.16+ no longer accepts in LSTM / BiLSTM constructors.

    Strategy
    --------
    1.  Open the .h5 file and extract the JSON model config stored in its
        root attributes.
    2.  Strip `time_major` recursively from the config dict.
    3.  Write the patched config back into a *temporary copy* of the .h5 file
        (the original weights are untouched).
    4.  Load from the temp copy with ``compile=False`` so we bypass the
        optimiser config, which may also reference removed arguments.
    5.  Delete the temp file and return the loaded model.
    """
    import h5py  # bundled with tensorflow; always available

    # ── 1. Read & patch the model config ──────────────────────────────────
    with h5py.File(model_path, "r") as f:
        raw_cfg = f.attrs.get("model_config", None)

    if raw_cfg is None:
        # No stored config — just try loading directly (may still work)
        return tf.keras.models.load_model(model_path, compile=False)

    # h5py may return bytes or a numpy string scalar
    if isinstance(raw_cfg, (bytes, bytearray)):
        raw_cfg = raw_cfg.decode("utf-8")
    elif hasattr(raw_cfg, "item"):
        raw_cfg = raw_cfg.item()

    cfg = json.loads(raw_cfg)
    _strip_time_major(cfg)
    patched_cfg = json.dumps(cfg)

    # ── 2. Write to a temp file (copy weights, patch config) ──────────────
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    try:
        shutil.copy2(model_path, tmp_path)
        with h5py.File(tmp_path, "r+") as f:
            # h5py stores attrs as numpy strings in some versions
            f.attrs["model_config"] = patched_cfg

        # ── 3. Load from the patched copy ─────────────────────────────────
        model = tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        os.unlink(tmp_path)

    return model


def load_inference_assets():
    """Loads the trained model and the dynamically generated label map."""
    print(f"Loading model from {MODEL_PATH}...")

    model = _load_model_compat(MODEL_PATH)

    # Recompile with a minimal setup so model() calls work
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Warm-up: one dummy forward pass to JIT-compile the compute graph
    dummy_input = np.zeros((1, 30, 225), dtype=np.float32)
    model.predict(dummy_input, verbose=0)

    print(f"Loading label map from {LABEL_MAP_PATH}...")
    labels = np.load(LABEL_MAP_PATH, allow_pickle=True)
    label_map = [str(lbl) for lbl in labels]

    print(f"Inference assets loaded! Classes: {label_map}")
    return model, label_map


def predict_sequence(model, sequence_array, label_map):
    """
    Takes the (30, 225) float array, runs inference, and returns the string label.
    """
    input_data = np.expand_dims(sequence_array, axis=0)
    
    # Run the model
    predictions = model(input_data, training=False).numpy()[0]
    
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    
    # ---------------------------------------------------------
    # Diagnostics: print the top-3 predictions to the terminal
    # ---------------------------------------------------------
    top3_idx = np.argsort(predictions)[::-1][:3]
    top3_str  = "  |  ".join(
        f"{label_map[i]} {predictions[i]*100:.1f}%" for i in top3_idx
    )
    print(f"🧠 Top-3: [ {top3_str} ]")

    # --- Confidence threshold: must be above 70% ---
    # The model scores 97.5% on test data, so low confidence = genuine ambiguity.
    CONFIDENCE_THRESHOLD = 0.70

    # --- Margin guard: top-2 gap must be > 20% ---
    # Prevents near-tie false positives (e.g. 'you' vs 'thank_you' at 42% each).
    second_idx = top3_idx[1]
    margin = confidence - predictions[second_idx]
    MARGIN_THRESHOLD = 0.20

    if confidence > CONFIDENCE_THRESHOLD and margin > MARGIN_THRESHOLD:
        return label_map[predicted_idx]

    return "..."