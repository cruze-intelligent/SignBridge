<div align="center">

# 🤟 Sign Language Recogniser

**A real-time, decoupled Acholi / English Sign Language recognition pipeline**  
*MediaPipe · WebSockets · FastAPI · Bidirectional LSTM · TensorFlow*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-0097A7?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-blueviolet)](LICENSE)

</div>

---

## 📖 Overview

The Sign Language Recogniser is a production-grade, low-latency gesture recognition system designed to bridge communication gaps for Acholi and English speakers. The pipeline runs entirely in the browser for pre-processing (zero server-side camera access required) and streams compact, normalised landmark arrays to a high-throughput Python backend for real-time inference.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CLIENT  (Browser)                         │
│                                                              │
│  Webcam  ──►  MediaPipe Holistic  ──►  225-float array/frame │
│                                                              │
│  • Right Hand : 21 landmarks × (x,y,z) = 63 floats          │
│  • Left Hand  : 21 landmarks × (x,y,z) = 63 floats          │
│  • Pose       : 33 landmarks × (x,y,z) = 99 floats          │
│                                                              │
│  Zero-pad missing hands • Nose-anchor normalisation          │
│  30-frame sliding window buffer                              │
│                                                              │
│         ▼  WebSocket (binary JSON)  ▼                        │
└──────────────────────────────────────────────────────────────┘
                         │
                    ws://localhost:8000
                         │
┌──────────────────────────────────────────────────────────────┐
│                    SERVER  (Python)                          │
│                                                              │
│  FastAPI  ──►  WebSocket router  ──►  Inference Engine       │
│                    │                      │                   │
│            REST batch endpoint      Bidirectional LSTM        │
│                    │                 (30 × 225 → softmax)     │
│                    ▼                                          │
│           .npy sequence store  (backend/storage/sequences/)  │
└──────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Concern | Decision | Rationale |
|---|---|---|
| Landmark extraction | Browser-side MediaPipe Holistic | Eliminates raw video transmission; only 225 floats/frame cross the wire |
| Transport | WebSocket (persistent) | Sub-20 ms round-trip vs HTTP polling |
| Normalisation | Coordinates relative to nose anchor | Invariant to camera distance and subject position |
| Missing tracking | Zero-pad 63-float block | Maintains strict tensor shape `(30, 225)` at all times |
| Persistence | NumPy `.npy` files | Optimal for downstream Transformer model retraining |
| ML framework | TensorFlow / Keras | Native GPU acceleration; straightforward LSTM API |

---

## 🧠 ML Model

The inference model is a stacked **Bidirectional LSTM** network (`SignLanguageLSTM_v2`) trained on gesture sequences captured through the data-collection pipeline.

```
Input  : (batch, 30, 225)
         └─ 30 frames × (RH-63 + LH-63 + Pose-99)

BiLSTM-1 : 128 units, return_sequences=True
BiLSTM-2 :  64 units, return_sequences=True
LSTM-3   :  64 units, return_sequences=False

Dense-1  : 128  → BatchNorm → ReLU → Dropout(0.4)
Dense-2  :  64  → BatchNorm → ReLU → Dropout(0.3)
Output   :  N   → Softmax   (N = dynamic class count)
```

Training callbacks: `EarlyStopping (patience=15)`, `ModelCheckpoint (best val_accuracy)`, `ReduceLROnPlateau`.

---

## 📁 Project Structure

```
Final Year Project/
├── frontend/
│   ├── index.html          # Vanilla HTML — bilingual UI (English / Acholi)
│   ├── style.css           # Mobile-first, high-contrast accessible design
│   └── app.js              # MediaPipe Holistic integration, WS client, buffer logic
│
└── backend/
    ├── run.py              # Uvicorn entry-point
    ├── requirements.txt
    ├── app/
    │   ├── main.py         # FastAPI factory, CORS, lifespan hooks
    │   ├── inference.py    # Model loader — inject Keras model here
    │   ├── persistence.py  # .npy sequence serialisation & storage
    │   ├── schemas.py      # Pydantic request/response models
    │   └── routers/
    │       ├── ws_landmarks.py    # WebSocket endpoint (real-time streaming)
    │       └── rest_landmarks.py  # REST endpoint (batch submission)
    ├── ml/
    │   ├── collect_data.py  # Step 1 — capture gesture sequences
    │   ├── prepare_data.py  # Step 2 — build train/test splits
    │   └── train.py         # Step 3 — train Bidirectional LSTM
    ├── models/
    │   └── .gitkeep         # Weights committed separately / ignored by git
    └── storage/
        └── sequences/       # Runtime .npy files — git-ignored
```

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10
- A modern browser with WebRTC support (Chrome / Edge recommended)
- *(Optional)* CUDA-capable GPU for faster training

### 1 — Backend setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/sign-language-recogniser.git
cd sign-language-recogniser

# Create and activate virtual environment
python -m venv backend/.venv
# Windows
backend\.venv\Scripts\activate
# macOS / Linux
source backend/.venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2 — Start the server

```bash
python backend/run.py
# FastAPI + Uvicorn starts on http://localhost:8000
# Interactive API docs: http://localhost:8000/docs
```

### 3 — Open the frontend

Simply open `frontend/index.html` in your browser.  
No build step required — pure Vanilla JS.

---

## 🔬 Training Your Own Model

```bash
# Step 1 — Collect gesture sequences (launches the capture UI)
python backend/ml/collect_data.py

# Step 2 — Prepare train / test splits
python backend/ml/prepare_data.py

# Step 3 — Train the Bidirectional LSTM
python backend/ml/train.py --epochs 150 --batch-size 32 --lr 0.001

# Outputs:
#   backend/models/sign_language_model_v2.h5
#   backend/models/training_history.json
```

Restart the server after training — the model is hot-loaded from `backend/models/` on startup via the `[INFERENCE HOOK ACTIVATE]` block in `app/main.py`.

---

## 🌐 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Interactive Swagger UI |
| `WS` | `/ws/landmarks` | Real-time 30-frame window stream |
| `POST` | `/landmarks/batch` | Single-shot batch sequence submission |

WebSocket message format (client → server):

```json
{
  "sequence": [[f1, f2, ..., f225], ...],   // 30 × 225 floats
  "label": "hello"                           // optional — for data collection
}
```

---

## ♿ Accessibility

- High-contrast UI with WCAG-AA compliant colour ratios
- Large tap targets (≥ 44 × 44 px) for mobile use
- Step-by-step onboarding tutorial with floating help icons
- Bilingual output: **English** and **Acholi** gesture labels

---

## 🗺️ Roadmap

- [ ] Transformer encoder replacing the LSTM backbone
- [ ] On-device TensorFlow Lite inference (offline mode)
- [ ] Expanded Acholi gesture vocabulary (community-sourced dataset)
- [ ] Progressive Web App (PWA) packaging
- [ ] Continuous retraining pipeline via GitHub Actions

---

## 🤝 Contributing

Pull requests are welcome. Please ensure all Python follows **PEP 8** and JavaScript is formatted with **Prettier** before opening a PR.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
