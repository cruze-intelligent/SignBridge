import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
from datetime import datetime

# --- Configuration ---
NUM_FRAMES = 30
EXPECTED_FEATURES = 225
STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'storage', 'sequences')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_225_floats(results):
    """Matches the exact extraction logic of the Javascript frontend."""
    # 1. Resolve Nose Anchor
    nose_x, nose_y = 0.0, 0.0
    pose_detected = results.pose_landmarks is not None
    if pose_detected:
        nose = results.pose_landmarks.landmark[0] # NOSE_IDX
        nose_x, nose_y = nose.x, nose.y

    # 2. Right Hand (63 floats)
    rh_coords = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh_coords.extend([lm.x - nose_x, lm.y - nose_y, lm.z])
    else:
        rh_coords = [0.0] * 63

    # 3. Left Hand (63 floats)
    lh_coords = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh_coords.extend([lm.x - nose_x, lm.y - nose_y, lm.z])
    else:
        lh_coords = [0.0] * 63

    # 4. Pose (99 floats) - Raw normalized coords
    pose_coords = []
    if pose_detected:
        for lm in results.pose_landmarks.landmark:
            pose_coords.extend([lm.x, lm.y, lm.z])
    else:
        pose_coords = [0.0] * 99

    # 5. Assemble
    frame_data = rh_coords + lh_coords + pose_coords
    return np.array(frame_data, dtype=np.float32)


def main():
    print("="*50)
    print("SignBridge - Standalone Data Collector")
    print("="*50)
    label = input("Enter the gesture label (e.g., 'hello', 'apwoyo'): ").strip().replace(' ', '_')
    
    label_dir = os.path.join(STORAGE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"\nSaving to: {label_dir}")
    print("Press 'R' to start recording a 30-frame sequence.")
    print("Press 'Q' to quit.\n")

    cap = cv2.VideoCapture(0)
    
    # Lower resolution for better framerate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sequence_buffer = []
    recording = False
    sequences_collected = len(os.listdir(label_dir))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip horizontally for a selfie-view
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw Landmarks (Debugging)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Handle Recording Logic
            if recording:
                frame_data = extract_225_floats(results)
                sequence_buffer.append(frame_data)
                
                cv2.putText(image, f"RECORDING: {len(sequence_buffer)}/30", (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                if len(sequence_buffer) == NUM_FRAMES:
                    # Save the sequence
                    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                    uid = uuid.uuid4().hex[:6]
                    filename = f"{label}_{timestamp}_{uid}.npy"
                    filepath = os.path.join(label_dir, filename)
                    
                    np.save(filepath, np.array(sequence_buffer))
                    sequences_collected += 1
                    print(f"Saved: {filename} (Total for '{label}': {sequences_collected})")
                    
                    sequence_buffer = []
                    recording = False
            else:
                cv2.putText(image, f"Label: {label} | Collected: {sequences_collected}", (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Press 'R' to Record | 'Q' to Quit", (15, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('SignBridge Data Collector', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('r') and not recording:
                recording = True
                sequence_buffer = []
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()