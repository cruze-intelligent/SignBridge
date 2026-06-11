<div align="center">

# 🤟 SignBridge

**Real-time Uganda Sign Language recognition — English & Acholi translation**  
*MediaPipe · WebSockets · FastAPI · Bidirectional LSTM · TensorFlow*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-0097A7?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-blueviolet)](LICENSE)

</div>

---

## 📖 Overview

SignBridge bridges communication gaps for Acholi and English speakers by translating Uganda Sign Language (USL) gestures into bilingual text in real time.  All landmark extraction happens in the browser — only 225 floats per frame cross the wire to a lightweight Python inference backend.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CLIENT  (Browser / GitHub Pages)          │
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
│         ▼  WebSocket (wss://)  ▼                             │
└──────────────────────────────────────────────────────────────┘
                         │
               wss://<your-backend>
                         │
┌──────────────────────────────────────────────────────────────┐
│                    SERVER  (Python — Render / Railway)       │
│                                                              │
│  FastAPI  ──►  WebSocket router  ──►  Inference Engine       │
│                    │                      │                   │
│            REST batch endpoint      Bidirectional LSTM        │
│                    │                 (30 × 225 → softmax)     │
│                    ▼                                          │
│           .npy sequence store  (backend/storage/sequences/)  │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SignBridge/
├── frontend/
│   ├── index.html          # Bilingual UI (English / Acholi) — deployable to GitHub Pages
│   ├── style.css           # Mobile-first dark theme, WCAG-AA compliant
│   └── app.js              # MediaPipe pipeline, WS client, settings panel
│
└── backend/
    ├── run.py              # Uvicorn entry-point (reads PORT env var)
    ├── Procfile            # Render / Railway deployment
    ├── .env.example        # Environment variable reference
    ├── requirements.txt
    ├── app/
    │   ├── main.py         # FastAPI factory, CORS (ALLOWED_ORIGINS env var), lifespan
    │   ├── inference.py    # Model loader + predict_sequence()
    │   ├── persistence.py  # .npy sequence serialisation
    │   ├── schemas.py      # Pydantic request/response models
    │   └── routers/
    │       ├── ws_landmarks.py    # WebSocket endpoint — real-time inference
    │       └── rest_landmarks.py  # REST endpoint — batch submission
    └── ml/
        ├── usl_vocabulary.py  # ★ Canonical USL sign list + collection guide
        ├── collect_data.py    # Step 1 — capture gesture sequences via webcam
        ├── prepare_data.py    # Step 2 — build train/test splits
        └── train.py           # Step 3 — train Bidirectional LSTM
```

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10, ≤ 3.12 (TensorFlow not yet compatible with 3.13+)
- A modern browser with WebRTC (Chrome / Edge recommended)
- *(Optional)* CUDA GPU for faster model training

### 1 — Backend setup

```bash
git clone https://github.com/<your-org>/SignBridge.git
cd SignBridge

python -m venv backend/.venv
source backend/.venv/bin/activate   # Windows: backend\.venv\Scripts\activate

pip install -r backend/requirements.txt
```

### 2 — Start the backend

```bash
python backend/run.py
# API available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 3 — Open the frontend

Open `frontend/index.html` directly in your browser.  
No build step required — pure Vanilla JS.

---

## 🌐 Deployment (GitHub Pages + Render)

The frontend is a static site — it deploys to GitHub Pages with zero configuration.  
The backend runs as a free web service on Render (or Railway / Fly.io).

### Frontend → GitHub Pages

1. Push the `frontend/` directory to a GitHub repository.
2. Go to **Settings → Pages → Source: Deploy from branch** and select `main` / `frontend`.
3. Your app will be live at `https://<username>.github.io/<repo>/`.

### Backend → Render (free tier)

1. Create a new **Web Service** on [render.com](https://render.com) pointing at this repo.
2. Set **Root Directory** to `backend`.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
5. Add these **Environment Variables**:

   | Key               | Value                                          |
   |---|---|
   | `ENV`             | `production`                                   |
   | `ALLOWED_ORIGINS` | `https://<username>.github.io`                 |

6. Copy the `https://your-service.onrender.com` URL.

### Connect the frontend to the live backend

Option A — **URL query param** (no rebuild needed):
```
https://<username>.github.io/<repo>/?backend=wss://your-service.onrender.com/ws/landmarks
```

Option B — **In-app Settings panel**:  
Tap the ⚙️ **Settings** icon in the bottom nav, paste the `wss://` URL, and tap **Save & Reconnect**.

> **Note:** Render's free tier spins down after 15 minutes of inactivity.  
> The first connection after spin-down may take ~30 seconds while the model loads.  
> A [paid plan](https://render.com/pricing) or [Railway](https://railway.app) eliminates cold starts.

---

## 🧠 ML Model

Stacked **Bidirectional LSTM** (`SignLanguageLSTM_v2`):

```
Input     : (batch, 30, 225)    — 30 frames × (RH-63 + LH-63 + Pose-99)
BiLSTM-1  : 128 units, return_sequences=True
BiLSTM-2  :  64 units, return_sequences=True
LSTM-3    :  64 units, return_sequences=False
Dense-1   : 128  → BatchNorm → ReLU → Dropout(0.4)
Dense-2   :  64  → BatchNorm → ReLU → Dropout(0.3)
Output    :  N   → Softmax   (N = number of trained classes)
```

---

## 🎯 Uganda Sign Language Vocabulary

The current training target covers **42 core USL gestures** organised by priority.  
Run the vocabulary guide at any time:

```bash
python backend/ml/usl_vocabulary.py
# Add --alphabet to include the A–Z finger-spelling set
```

| Priority | Category              | Signs |
|---|---|---|
| 1 | **Greetings**         | hello, goodbye, good_morning, good_night, how_are_you, i_am_fine, thank_you, please, sorry, welcome, congratulations, nice_to_meet_you |
| 2 | **Basic Communication** | yes, no, help, stop, go, come, wait, understand, repeat, my_name_is |
| 3 | **Feelings**          | good, bad, happy, love, hungry, tired |
| 4 | **People**            | mother, father, brother, sister, friend, family, child |
| 5 | **Numbers**           | one – ten |
| 6 | **Everyday Nouns**    | water, food, home, school, hospital, road |
| 7 | **Alphabet** *(optional)* | A – Z finger-spelling |

### Data collection guidelines

- Collect **≥ 30 sequences per sign** (60–100 recommended for production).
- Use **multiple signers** to improve model generalisation.
- Record in varied lighting and backgrounds.
- Each sequence is 30 frames (~1 second at 30 fps).

---

## 🔬 Training Your Own Model

```bash
# Step 1 — View the USL vocabulary checklist
python backend/ml/usl_vocabulary.py

# Step 2 — Collect gesture sequences (launches a webcam window)
python backend/ml/collect_data.py

# Step 3 — Prepare train / test splits
python backend/ml/prepare_data.py

# Step 4 — Train the model
python backend/ml/train.py --epochs 150 --batch-size 32 --lr 0.001

# Outputs:
#   backend/models/sign_language_model_v2.h5
#   backend/models/training_history.json
```

Restart the backend after training — the model hot-loads on startup.

---

## 🌐 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status":"ok"}` |
| `GET` | `/docs` | Interactive Swagger UI |
| `WS` | `/ws/landmarks` | Real-time 30-frame landmark stream |
| `POST` | `/landmarks/batch` | Single-shot batch submission |

WebSocket message (client → server):
```json
{ "frame": [f0, f1, ..., f224] }
```

WebSocket response (server → client):
```json
{ "status": "translated", "text": "hello" }
```

---

## 🎮 Games Integration

The bottom navigation includes a **Games** button that opens [SignMaster](https://cruze-intelligent.github.io/SignMaster/) — an interactive USL practice game.  
SignMaster also serves as the reference dataset for USL gesture labels used during training.

---

## ♿ Accessibility

- High-contrast dark UI with WCAG-AA colour ratios
- Large tap targets (≥ 44 × 44 px) for mobile use
- Step-by-step onboarding wizard + floating help button
- Bilingual output: **English** and **Acholi (Luo)**
- Web Speech API TTS reads each translation aloud

---

## 🗺️ Roadmap

- [ ] TensorFlow Lite export for fully offline inference
- [ ] Progressive Web App (PWA) packaging
- [ ] Expanded USL vocabulary via community-sourced recording sessions
- [ ] Continuous retraining pipeline via GitHub Actions
- [ ] Transformer encoder replacing the LSTM backbone

---

## 🤝 Contributing

PRs are welcome. Python should follow PEP 8; JavaScript should be formatted with Prettier.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
