# 🛡️ Vi-SAFE: Real-Time AI Violence Detection System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Metal](https://img.shields.io/badge/Accelerate-Apple%20Silicon-gold.svg)](https://developer.apple.com/metal/)

A high-performance, real-time violence detection system designed for university campus security. This system leverages **YOLOv8** for spatial human detection and a custom **CNN-LSTM (MobileNetV2)** architecture for temporal action classification, optimized for Apple Silicon (M1/M2/M3) using the MPS backend.

---

## 🚀 Key Features

*   **⚡ Multi-Camera Dashboard:** Simulate a control room with side-by-side feeds and an integrated alert history panel.
*   **🧠 High Accuracy:** Trained on real-world datasets achieving **91.7% validation accuracy**.
*   **🎯 Trained Weights:** Automatically loads `violence_classifier.pt` on startup — no random weights.
*   **🌊 Optical Flow Suppression:** Intelligent motion filtering that reduces false positives when the scene is static.
*   **🍎 Apple Silicon Native:** Built with `torch.device("mps")` for GPU-accelerated inference on Mac.
*   **📝 Structured Alerting:** Auto-logs alerts with timestamps, confidence scores, and duration into `alerts.jsonl`.
*   **🏗️ Training Pipeline:** Includes a complete script to download datasets and train the LSTM from scratch.

---

## 🛠️ Architecture

The system operates in a dual-stage pipeline:

1.  **Spatial Stage (YOLOv8):** Detects humans in the frame and crops the Region of Interest (ROI).
2.  **Temporal Stage (MobileNetV2 + LSTM):** Takes a 16-frame sequence of the ROI to classify the behavior as "Violent" or "Normal".

```
Camera Feed → YOLOv8 (Person Detection) → ROI Crop
                                              ↓
                               16-Frame Buffer (112×112)
                                              ↓
                          MobileNetV2 (Feature Extraction)
                                              ↓
                              2-Layer LSTM (Temporal Reasoning)
                                              ↓
                         Optical Flow Suppression (Motion Gate)
                                              ↓
                          Violence Score → Alert if score > 0.55
```

---

## ⚙️ Detection Parameters

These parameters control detection sensitivity and can be tuned in `main.py` and `multicam.py`:

| Parameter | Value | Description |
|---|---|---|
| `VIOLENCE_THRESHOLD` | `0.55` | Score above which an alert is triggered. Lowered from 0.75 to catch self-hitting and mild violence. |
| `MOTION_THRESHOLD` | `0.35` | Optical flow magnitude below which the scene is considered "still". Lowered so moderate motion (self-hitting) isn't suppressed. |
| `MOTION_SUPPRESS` | `0.85` | Score multiplier when motion is below threshold. Eased from 0.70 to reduce over-suppression. |
| `YOLO_CONFIDENCE` | `0.40` | Minimum YOLO confidence to detect a person. Lowered to catch partially visible persons. |
| `FRAME_BUFFER_SIZE` | `16` | Number of frames fed into the LSTM for each inference pass (~0.5s at 30fps). |
| `ALERT_COOLDOWN` | `10s` | Minimum seconds between repeated alerts for the same camera. |

> **Tuning tip:** If you're getting too many false positives, raise `VIOLENCE_THRESHOLD` toward `0.65`. If violence is being missed, lower it toward `0.45`.

---

## 📥 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JapjeetSingh17/Vi-SAFE-Inspired--Real-Time-AI-Violence-Detection-System..git
    cd Vi-SAFE-Inspired--Real-Time-AI-Violence-Detection-System.
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

### Control Room Dashboard
Launch the multi-camera simulation:
```bash
python multicam.py
```

### Standard Single Feed
Launch the primary detection system (with live score debug output in terminal):
```bash
python main.py
```

### Training
To retrain the model on the latest datasets:
```bash
python train.py --epochs 75
```
The best checkpoint is saved as `violence_classifier_trained.pt` and copied to `violence_classifier.pt` automatically.

---

## 📊 Performance & Alerts

Alerts are triggered when the violence score exceeds **0.55** (after optional motion suppression). Every alert records:

*   `timestamp`: When the event occurred.
*   `location`: Camera identifier.
*   `confidence`: Model probability score (0.0 – 1.0).
*   `duration_seconds`: How long the violence has been detected.

Sample `alerts.jsonl`:
```json
{"timestamp": "2026-04-20 16:30:00", "location": "Library - Floor 2", "confidence": 0.612, "duration_seconds": 3.2}
```

Alerts are also written as plain text to `alerts.log` for backward compatibility.

---

## 🌐 Deploying to Vercel & Cloud

### 1. Deploy Frontend on Vercel
1. Import this repository into **[Vercel](https://vercel.com/)**.
2. Vercel will automatically detect the Next.js app via `vercel.json` (or set **Root Directory** to `frontend`).
3. In your Vercel Project Settings under **Environment Variables**, add:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: Your backend server URL (e.g., `https://your-api.onrender.com` or `http://YOUR_SERVER_IP:8000`).
4. Click **Deploy**.

### 2. Deploy AI FastAPI Backend
Because heavy PyTorch, OpenCV, and streaming WebSockets require long-lived execution, host `api_server.py` using Docker / Cloud Host (e.g. Render, Railway, Hugging Face Spaces, or AWS):
```bash
# Run backend with Docker
docker build -t vi-safe-backend .
docker run -p 8000:8000 vi-safe-backend
```

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Acknowledgements

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
*   [Hugging Face Datasets](https://huggingface.co/datasets)
*   [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
