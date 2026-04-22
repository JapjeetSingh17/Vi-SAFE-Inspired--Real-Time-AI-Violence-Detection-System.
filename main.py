import cv2
import torch
import numpy as np
import json
from collections import deque
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
from datetime import datetime
import time
import os

# ── Config ──────────────────────────────────────────
CAMERA_ID          = 0
CAMERA_LOCATION    = "Library - Floor 2"
FRAME_BUFFER_SIZE  = 16
VIOLENCE_THRESHOLD = 0.25     # lowered to 25%: trigger more easily on live feed
ALERT_COOLDOWN     = 10       # seconds
YOLO_CONFIDENCE    = 0.4      # slightly lower: detect person even when partially visible
MOTION_THRESHOLD   = 0.20     # lowered: consider smaller interactions as motion
MOTION_SUPPRESS    = 0.95     # eased: almost no penalty for low motion
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Output files ────────────────────────────────────
ALERTS_LOG   = "alerts.log"
ALERTS_JSONL = "alerts.jsonl"

# ── Model Definition ─────────────────────────────────
class QuickViolenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1280, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.pool(self.features(x)).squeeze(-1).squeeze(-1)
        x = x.view(B, T, -1)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1]))

# ── Load Models ──────────────────────────────────────
print("Loading YOLOv8...")
yolo = YOLO('yolov8n.pt')

print("Loading Violence Classifier...")
classifier = QuickViolenceNet().to(DEVICE)

CLASSIFIER_WEIGHTS = "violence_classifier.pt"
if os.path.exists(CLASSIFIER_WEIGHTS):
    classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
    print(f"✅ Loaded trained weights from '{CLASSIFIER_WEIGHTS}'")
else:
    print(f"⚠️  WARNING: '{CLASSIFIER_WEIGHTS}' not found — running with random/untrained weights!")
    print("   Run 'python train.py' first to generate trained weights.")

classifier.eval()
print("Models ready.")

# ── Preprocessing ────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Alert ────────────────────────────────────────────
last_alert_time   = 0
violence_start_t  = None   # when score first crossed threshold

def trigger_alert(confidence, location, duration_seconds):
    """Fire an alert: prints, logs to .log and .jsonl, plays sound."""
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return
    last_alert_time = now

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (f"[ALERT] {timestamp} | Location: {location} | "
           f"Confidence: {confidence:.1%} | Duration: {duration_seconds:.1f}s")
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60 + "\n")

    # Plain-text log (backward compatible)
    with open(ALERTS_LOG, "a") as f:
        f.write(msg + "\n")

    # Structured JSON Lines log (Task 4)
    record = {
        "timestamp":        timestamp,
        "location":         location,
        "confidence":       round(confidence, 4),
        "duration_seconds": round(duration_seconds, 1),
    }
    with open(ALERTS_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")

    os.system("afplay /System/Library/Sounds/Sosumi.aiff &")

# ── Main Loop ────────────────────────────────────────
print(f"\nSystem running | Camera: {CAMERA_LOCATION}")
print(f"Alerts: {ALERTS_LOG}  |  {ALERTS_JSONL}")
print("Press Q to quit\n")

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open camera. Check camera permissions in System Settings.")
    exit()

frame_buffer  = deque(maxlen=FRAME_BUFFER_SIZE)
violence_score = 0.0
frame_count    = 0
prev_gray      = None    # for optical flow
motion_mag     = 1.0     # default: assume motion present

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Lost camera feed.")
        break

    frame_count += 1

    # ── 1. Optical flow — compute motion magnitude ──────────
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        flow       = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _     = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mag = float(np.mean(mag))
    prev_gray = curr_gray.copy()

    # ── 2. Detect people with YOLOv8 ──────────────────────
    results = yolo(frame, classes=[0], conf=YOLO_CONFIDENCE, verbose=False)
    boxes   = results[0].boxes

    # ── 3. Crop largest person ROI ─────────────────────────
    roi = frame
    if boxes is not None and len(boxes) > 0:
        xyxy  = boxes.xyxy.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
        best  = xyxy[np.argmax(areas)].astype(int)
        x1, y1, x2, y2 = best
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]

    # ── 4. Add frame to buffer ─────────────────────────────
    try:
        rgb    = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb)
        frame_buffer.append(tensor)
    except Exception as e:
        print(f"Frame processing error: {e}")
        continue

    # ── 5. Run classifier when buffer is full ──────────────
    if len(frame_buffer) == FRAME_BUFFER_SIZE:
        with torch.no_grad():
            clip   = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)
            logits = classifier(clip)
            probs  = torch.softmax(logits, dim=1)
            raw_score = probs[0][1].item()

        # Optical flow suppression
        if motion_mag < MOTION_THRESHOLD:
            raw_score *= MOTION_SUPPRESS
        violence_score = raw_score

        # Debug: always print score so you can see what the model sees
        suppressed_marker = " [suppressed]" if motion_mag < MOTION_THRESHOLD else ""
        print(f"  Score: {violence_score:.3f}{suppressed_marker}  motion={motion_mag:.2f}  "
              f"{'⚠️  ALERT' if violence_score > VIOLENCE_THRESHOLD else ''}")

        # Task 4 — Track how long score has been above threshold
        if violence_score > VIOLENCE_THRESHOLD:
            if violence_start_t is None:
                violence_start_t = time.time()
            duration = round(time.time() - violence_start_t, 1)
            trigger_alert(violence_score, CAMERA_LOCATION, duration)
        else:
            violence_start_t = None

    # ── 6. Draw UI overlay ─────────────────────────────────
    annotated    = results[0].plot()
    is_violent   = violence_score > VIOLENCE_THRESHOLD
    status_color = (0, 60, 220) if is_violent else (20, 180, 60)
    status_text  = "!! VIOLENCE DETECTED !!" if is_violent else "NORMAL"
    motion_label = f"Motion: {'HIGH' if motion_mag >= MOTION_THRESHOLD else 'LOW '} ({motion_mag:.2f})"
    suppressed   = motion_mag < MOTION_THRESHOLD

    # Header bar
    h_w = annotated.shape[1]
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (h_w, 75), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    cv2.putText(annotated, f"Status: {status_text}",
                (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2)
    cv2.putText(annotated, f"Score: {violence_score:.2f}{'  [suppressed]' if suppressed else ''}  |  {CAMERA_LOCATION}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(annotated, motion_label,
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (100, 200, 100) if motion_mag >= MOTION_THRESHOLD else (100, 100, 200), 1)

    # Flashing border when violent
    if is_violent and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(annotated, (2, 2), (h_w - 2, annotated.shape[0] - 2), (0, 0, 255), 3)

    cv2.imshow("Vi-SAFE Violence Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nSystem stopped.")
print(f"Alerts (plain text) : {ALERTS_LOG}")
print(f"Alerts (structured) : {ALERTS_JSONL}")