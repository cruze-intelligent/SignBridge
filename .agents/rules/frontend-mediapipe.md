---
trigger: always_on
---

# Client Node (Frontend) Rules

- **Data Acquisition:** Implement webcam video stream capture via Vanilla JavaScript.
- **Pre-processing (Holistic):** Integrate MediaPipe Holistic (or Hands + Pose simultaneously) directly within the browser environment to minimize network latency.
- **Landmark Extraction (225 Floats):** The agent MUST configure MediaPipe to extract exactly 225 floats per frame, structured strictly as:
  - Right Hand: 21 landmarks (x, y, z) = 63 floats.
  - Left Hand: 21 landmarks (x, y, z) = 63 floats.
  - Pose/Face: 33 landmarks (x, y, z) = 99 floats.
- **Missing Data & Zero-Padding:** If MediaPipe loses tracking of a hand (returns null), the system MUST zero-pad that specific 63-float block to ensure the total array length remains exactly 225 floats before transmission.
- **Spatial Normalization:** Implement logic to calculate hand landmark coordinates *relative* to a central anchor point on the Pose (e.g., the nose). Do not send raw screen pixel coordinates.
- **Sequence Buffer Logic:** Implement a client-side sequence buffer that groups the 225-float arrays into a continuous 30-frame "window" before transmission.
- **Accessibility Standards:** The UI MUST prioritize accessibility. Use large, high-contrast text and simple buttons with intuitive icons. 
- **User Onboarding:** Include a step-by-step beginner tutorial and floating help icons to assist users with camera positioning and navigation.
- **Bilingual Output:** Ensure the UI components support dual-mode text translation for both English and Acholi.