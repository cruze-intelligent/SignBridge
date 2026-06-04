import os
import numpy as np
import tensorflow as tf

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_language_model_v2.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "ml", "dataset", "label_map.npy")

def load_inference_assets():
    """Loads the trained model and the dynamically generated label map."""
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Perform a dummy prediction to 'warm up' the model in memory
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
    # THE FIX: Print the model's exact thoughts to the terminal
    # ---------------------------------------------------------
    print(f"🧠 Model sees: '{label_map[predicted_idx]}' at {confidence * 100:.1f}% confidence")
    
    # Lowered the threshold from 0.70 to 0.40 to allow for real-world variance
    if confidence > 0.40:
        return label_map[predicted_idx]
        
    return "..."