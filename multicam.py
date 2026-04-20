"""
Vi-SAFE Multi-Camera Control Room Dashboard
============================================
Simulates a university campus security control room with two independent
camera feeds displayed side-by-side in one window.

Both cameras use cv2.VideoCapture(0) (same physical camera) with different
location labels, running independent violence detection pipelines that share
one LSTM model instance.

Features:
  - Independent 16-frame buffers + violence scores per feed
  - Per-feed optical flow motion suppression (reduces false positives)
  - Shared alerts.log + alerts.jsonl (both cameras write with their label)
  - Rich HUD: status banner, score bar, motion level, location badge
  - Press Q to quit

Usage:
    python multicam.py
"""

import cv2
import torch
import numpy as np
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from ultralytics import YOLO
import torch.nn as nn

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEVICE             = "mps" if torch.backends.mps.is_available() else "cpu"
FRAME_BUFFER_SIZE  = 16
VIOLENCE_THRESHOLD = 0.55     # lowered: catch self-hitting & mild violence
ALERT_COOLDOWN     = 10       # seconds between alerts per camera
YOLO_CONFIDENCE    = 0.4      # slightly lower: detect person even when partially visible
MOTION_THRESHOLD   = 0.35     # lowered: self-hitting is moderate motion, not high
MOTION_SUPPRESS    = 0.85     # eased: don't penalise moderate motion so aggressively
DISPLAY_WIDTH      = 640      # per-feed display width (total = 2×)
DISPLAY_HEIGHT     = 480

CAMERAS = [
    {"id": 0, "location": "Campus Main Gate"},
    {"id": 0, "location": "Library - Floor 2"},
]

LOG_FILE   = Path("alerts.log")
JSONL_FILE = Path("alerts.jsonl")

print(f"[Config] Device : {DEVICE}")
print(f"[Config] Cameras: {len(CAMERAS)}")


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class QuickViolenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.lstm     = nn.LSTM(1280, 128, num_layers=2, batch_first=True)
        self.fc       = nn.Linear(128, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x   = x.view(B * T, C, H, W)
        x   = self.pool(self.features(x)).squeeze(-1).squeeze(-1)
        x   = x.view(B, T, -1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
# ALERT SYSTEM
# ─────────────────────────────────────────────────────────────
class AlertManager:
    """Per-camera alert manager with cooldown and JSON logging."""

    def __init__(self, location):
        self.location         = location
        self.last_alert_time  = 0
        self.violence_start   = None  # when score first crossed threshold
        self.alert_history    = []    # list of recent alerts for HUD display

    def update(self, score):
        """Call each frame with current score. Returns True if alert fired."""
        now = time.time()
        if score > VIOLENCE_THRESHOLD:
            if self.violence_start is None:
                self.violence_start = now
        else:
            self.violence_start = None

        if score > VIOLENCE_THRESHOLD and (now - self.last_alert_time) >= ALERT_COOLDOWN:
            duration = round(now - (self.violence_start or now), 1)
            self._fire(score, duration)
            self.last_alert_time = now
            return True
        return False

    def _fire(self, confidence, duration_seconds):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (f"[ALERT] {ts} | Location: {self.location} | "
               f"Confidence: {confidence:.1%} | Duration: {duration_seconds}s")

        print("\n" + "═" * 65)
        print(msg)
        print("═" * 65 + "\n")

        # alerts.log (plain text)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")

        # alerts.jsonl (structured)
        record = {
            "timestamp":        ts,
            "location":         self.location,
            "confidence":       round(confidence, 4),
            "duration_seconds": duration_seconds,
        }
        with open(JSONL_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Sound alert
        os.system("afplay /System/Library/Sounds/Sosumi.aiff &")

        self.alert_history.insert(0, f"{ts[-8:]} {self.location[:12]} {confidence:.0%}")
        self.alert_history = self.alert_history[:3]   # keep last 3

    def get_recent_alerts(self):
        return self.alert_history


# ─────────────────────────────────────────────────────────────
# OPTICAL FLOW HELPER
# ─────────────────────────────────────────────────────────────
def compute_motion(prev_gray, curr_gray):
    """Returns mean optical flow magnitude (0 = no motion)."""
    if prev_gray is None:
        return 1.0   # default: assume motion present until we have two frames
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


# ─────────────────────────────────────────────────────────────
# HUD RENDERING
# ─────────────────────────────────────────────────────────────
def draw_hud(frame, location, score, motion_mag, is_violent, alert_mgr):
    h, w = frame.shape[:2]

    # ── Top banner ──
    banner_h = 72
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    status_col  = (0, 60, 220)  if is_violent else (20, 180, 60)
    status_text = "!! VIOLENCE !!" if is_violent else "NORMAL"

    cv2.putText(frame, status_text,
                (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.85, status_col, 2)
    cv2.putText(frame, location,
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # ── Violence score bar (right side of banner) ──
    bar_x, bar_y, bar_w, bar_h = w - 160, 12, 140, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    fill = int(bar_w * score)
    bar_col = (0, int(255 * (1 - score)), int(255 * score))   # green→red
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      bar_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (100, 100, 100), 1)
    cv2.putText(frame, f"Score {score:.2f}",
                (bar_x, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # ── Motion indicator ──
    motion_label = "Motion: "
    motion_col   = (100, 200, 100) if motion_mag >= MOTION_THRESHOLD else (100, 100, 200)
    motion_val   = f"{'HIGH' if motion_mag >= MOTION_THRESHOLD else ' LOW'} ({motion_mag:.2f})"
    cv2.putText(frame, motion_label + motion_val,
                (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, motion_col, 1)

    # ── Recent alerts panel (bottom) ──
    recents = alert_mgr.get_recent_alerts()
    for i, txt in enumerate(recents):
        cv2.putText(frame, f"ALT: {txt}",
                    (10, h - 16 - i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 255), 1)

    # ── Flashing red border when violent ──
    if is_violent and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 3)

    return frame


# ─────────────────────────────────────────────────────────────
# CAMERA STATE
# ─────────────────────────────────────────────────────────────
class CameraState:
    def __init__(self, cam_id, location):
        self.cap           = cv2.VideoCapture(cam_id)
        self.location      = location
        self.frame_buffer  = deque(maxlen=FRAME_BUFFER_SIZE)
        self.violence_score= 0.0
        self.prev_gray     = None
        self.motion_mag    = 1.0
        self.alert_mgr     = AlertManager(location)
        self.last_frame    = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

    def is_open(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main():
    print("\nLoading models...")
    yolo       = YOLO("yolov8n.pt")
    classifier = QuickViolenceNet().to(DEVICE)

    CLASSIFIER_WEIGHTS = "violence_classifier.pt"
    if os.path.exists(CLASSIFIER_WEIGHTS):
        classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
        print(f"✅ Loaded trained weights from '{CLASSIFIER_WEIGHTS}'")
    else:
        print(f"⚠️  WARNING: '{CLASSIFIER_WEIGHTS}' not found — running with random/untrained weights!")
        print("   Run 'python train.py' first to generate trained weights.")

    classifier.eval()
    print("Models ready.\n")

    # Open cameras
    cameras = []
    for cfg in CAMERAS:
        cam = CameraState(cfg["id"], cfg["location"])
        if not cam.is_open():
            print(f"[WARN] Could not open camera {cfg['id']} for {cfg['location']}")
        else:
            print(f"[OK] Camera opened: {cfg['location']}")
        cameras.append(cam)

    if not any(c.is_open() for c in cameras):
        print("[ERROR] No cameras available.")
        return

    frame_count = 0
    WINDOW_NAME = "Vi-SAFE | Campus Control Room"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print(f"\nControl Room LIVE | {len(cameras)} cameras")
    print("Press Q to quit\n")

    while True:
        feed_frames = []

        for cam in cameras:
            if not cam.is_open():
                feed_frames.append(cam.last_frame)
                continue

            ret, frame = cam.cap.read()
            if not ret:
                feed_frames.append(cam.last_frame)
                continue

            display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # ── Optical flow ──
            curr_gray    = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            cam.motion_mag = compute_motion(cam.prev_gray, curr_gray)
            cam.prev_gray  = curr_gray.copy()

            # ── YOLO person detection ──
            results = yolo(display, classes=[0], conf=YOLO_CONFIDENCE, verbose=False)
            boxes   = results[0].boxes
            roi     = display
            if boxes is not None and len(boxes) > 0:
                xyxy  = boxes.xyxy.cpu().numpy()
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in xyxy]
                best  = xyxy[np.argmax(areas)].astype(int)
                x1, y1, x2, y2 = best
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(display.shape[1], x2), min(display.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    roi = display[y1:y2, x1:x2]

            # ── Frame buffer ──
            try:
                rgb    = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                tensor = TRANSFORM(rgb)
                cam.frame_buffer.append(tensor)
            except Exception:
                pass

            # ── Violence inference ──
            if len(cam.frame_buffer) == FRAME_BUFFER_SIZE:
                with torch.no_grad():
                    clip   = torch.stack(list(cam.frame_buffer)).unsqueeze(0).to(DEVICE)
                    logits = classifier(clip)
                    probs  = torch.softmax(logits, dim=1)
                    raw_score = probs[0][1].item()

                # Optical flow suppression
                if cam.motion_mag < MOTION_THRESHOLD:
                    raw_score *= MOTION_SUPPRESS

                cam.violence_score = raw_score
                cam.alert_mgr.update(cam.violence_score)

            # ── HUD overlay on YOLO-annotated frame ──
            annotated    = results[0].plot()
            annotated    = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            is_violent   = cam.violence_score > VIOLENCE_THRESHOLD
            annotated    = draw_hud(annotated, cam.location, cam.violence_score,
                                    cam.motion_mag, is_violent, cam.alert_mgr)
            cam.last_frame = annotated
            feed_frames.append(annotated)

        # ── Composite: side by side ──
        if len(feed_frames) == 2:
            # Divider line
            div = np.zeros((DISPLAY_HEIGHT, 4, 3), dtype=np.uint8)
            div[:] = (80, 80, 80)
            composite = np.hstack([feed_frames[0], div, feed_frames[1]])
        elif feed_frames:
            composite = feed_frames[0]
        else:
            composite = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        # ── Header bar ──
        header_h = 32
        header   = np.zeros((header_h, composite.shape[1], 3), dtype=np.uint8)
        header[:] = (18, 18, 30)
        ts_str   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(header, f"Vi-SAFE Control Room Dashboard       {ts_str}",
                    (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 180, 230), 1)
        composite = np.vstack([header, composite])

        cv2.imshow(WINDOW_NAME, composite)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()
    print("\nControl Room stopped.")
    print(f"Alerts saved to: {LOG_FILE} and {JSONL_FILE}")


if __name__ == "__main__":
    main()
