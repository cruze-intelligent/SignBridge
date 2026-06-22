import os
import json
import shutil
import tempfile
import numpy as np
import tensorflow as tf

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_language_model_v4_static.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "ml", "dataset", "label_map.npy")





def load_inference_assets():
    """Loads the trained model and the dynamically generated label map."""
    print(f"Loading model from {MODEL_PATH}...")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Recompile with a minimal setup so model() calls work
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Warm-up: one dummy forward pass to JIT-compile the compute graph
    dummy_input = np.zeros((1, 225), dtype=np.float32)
    model.predict(dummy_input, verbose=0)

    print(f"Loading label map from {LABEL_MAP_PATH}...")
    labels = np.load(LABEL_MAP_PATH, allow_pickle=True)
    label_map = [str(lbl) for lbl in labels]

    print(f"Inference assets loaded! Classes: {label_map}")
    return model, label_map


def predict_frame(model, frame_array, label_map):
    """
    Takes a (225,) float array, runs inference, 
    and returns the string label (Static Recognition Mode).
    """
    input_data = np.expand_dims(frame_array, axis=0)
    
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
    print(f"[Top-3] [ {top3_str} ]")

    # --- Confidence threshold: increased for dense static recognition ---
    CONFIDENCE_THRESHOLD = 0.70

    if confidence > CONFIDENCE_THRESHOLD:
        return label_map[predicted_idx]

    return "..."