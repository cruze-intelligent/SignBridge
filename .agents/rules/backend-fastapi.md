---
trigger: always_on
---

# Server Node (Backend) Rules

- [cite_start]**Framework:** Use FastAPI to build the API and communication layer[cite: 99]. [cite_start]Ensure it is configured to handle high-volume concurrent asynchronous requests[cite: 100].
- [cite_start]**Real-Time Communication:** Implement WebSocket endpoints as the primary method for continuous, low-latency streaming of landmark sequences from the client[cite: 102].
- [cite_start]**Batch Processing:** Include secondary RESTful endpoints for intermittent or batch submissions[cite: 103].
- [cite_start]**Data Persistence:** Implement storage logic to save processed gesture sequences as serialized arrays in the `.npy` (NumPy) file format[cite: 112, 113]. [cite_start]This is non-negotiable, as these are required for future Transformer model retraining[cite: 114].
- [cite_start]**Inference Engine Placeholder:** Structure the application to allow the TensorFlow Transformer model to be easily injected later for sequence inference[cite: 104, 105]. [cite_start]Ensure it is optimized to leverage hardware acceleration (GPU/high-core CPU)[cite: 107, 108].