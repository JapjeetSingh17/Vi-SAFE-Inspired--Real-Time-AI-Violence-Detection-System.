import cv2
import torch
import numpy as np
import json
import time
import os
import threading
from collections import deque
from datetime import datetime
from typing import Set, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
import uvicorn

# ── Config & Globals ─────────────────────────────────
app = FastAPI(title="Vi-SAFE Real-Time AI Violence Detection Server")

# Allow CORS for Next.js deployed on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Server starting on device: {DEVICE}")

# Output file paths
ALERTS_LOG = "alerts.log"
ALERTS_JSONL = "alerts.jsonl"

# Active WebSockets clients
active_websockets: Set[WebSocket] = set()

# Global settings (dynamically editable)
global_settings = {
    "violence_threshold": 0.35,  # Alert threshold
    "motion_threshold": 0.25,     # Motion gate threshold
    "motion_suppress": 0.90,      # Score multiplier for low motion
    "yolo_confidence": 0.40,      # Minimum person detection conf
    "alert_cooldown": 10.0,       # Cooldown in seconds between alerts
    "camera_location": "Library - Floor 2",
    "camera_id": 0,
    "is_running": True
}

# Real-time state
current_state = {
    "fps": 0.0,
    "violence_score": 0.0,
    "motion_mag": 0.0,
    "is_violent": False,
    "camera_status": "Starting...",
    "using_mock": True
}

latest_frame_bytes = None
frame_lock = threading.Lock()

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

# Load models
print("Loading YOLOv8...")
yolo_model = YOLO('yolov8n.pt')

print("Loading LSTM Classifier...")
classifier = QuickViolenceNet().to(DEVICE)
CLASSIFIER_WEIGHTS = "violence_classifier.pt"

if os.path.exists(CLASSIFIER_WEIGHTS):
    classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
    print(f"✅ Loaded weights from '{CLASSIFIER_WEIGHTS}'")
else:
    print(f"⚠️  WARNING: '{CLASSIFIER_WEIGHTS}' not found — running with untrained weights!")

classifier.eval()
print("Models ready.")

# Preprocessing transforms
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Alert Dispatcher ─────────────────────────────────
last_alert_time = 0.0

def dispatch_alert(confidence: float, location: str, duration_seconds: float):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < global_settings["alert_cooldown"]:
        return
    last_alert_time = now

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[ALERT] {timestamp} | Location: {location} | Confidence: {confidence:.1%} | Duration: {duration_seconds:.1f}s"
    print(f"\n🚨 {msg}\n")

    # Write to local logs
    with open(ALERTS_LOG, "a") as f:
        f.write(msg + "\n")

    alert_record = {
        "timestamp": timestamp,
        "location": location,
        "confidence": round(confidence, 4),
        "duration_seconds": round(duration_seconds, 1),
    }

    with open(ALERTS_JSONL, "a") as f:
        f.write(json.dumps(alert_record) + "\n")

    # Broadcast to WebSocket clients
    broadcast_message({
        "type": "alert",
        "data": alert_record
    })

def broadcast_message(message: Dict[str, Any]):
    disconnected = set()
    message_str = json.dumps(message)
    for ws in active_websockets:
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(ws.send_text(message_str), main_event_loop)
        except Exception:
            disconnected.add(ws)
    
    for ws in disconnected:
        active_websockets.discard(ws)

# ── Simulation & Mock Feed Generator ──────────────────
def generate_synthetic_frame(width=640, height=480, frame_idx=0):
    """Generates a beautiful futuristic security grid frame with telemetry overlays."""
    # Create dark futuristic background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw background Grid lines
    grid_size = 40
    for y in range(0, height, grid_size):
        cv2.line(frame, (0, y), (width, y), (15, 20, 25), 1)
    for x in range(0, width, grid_size):
        cv2.line(frame, (x, 0), (x, height), (15, 20, 25), 1)

    # Simulated motion coordinates (bouncing around)
    t = frame_idx * 0.05
    cx1 = int(width / 2 + np.cos(t) * 150)
    cy1 = int(height / 2 + np.sin(t * 1.5) * 100)
    w1, h1 = 120, 240
    
    cx2 = int(width / 2 + np.sin(t * 0.8) * 200)
    cy2 = int(height / 2 + np.cos(t * 1.2) * 80)
    w2, h2 = 100, 200

    # Draw bounding boxes (simulating YOLOv8 output)
    cv2.rectangle(frame, (cx1 - w1//2, cy1 - h1//2), (cx1 + w1//2, cy1 + h1//2), (0, 180, 0), 2)
    cv2.putText(frame, "Person 1: 0.94", (cx1 - w1//2, cy1 - h1//2 - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

    cv2.rectangle(frame, (cx2 - w2//2, cy2 - h2//2), (cx2 + w2//2, cy2 + h2//2), (0, 180, 0), 2)
    cv2.putText(frame, "Person 2: 0.89", (cx2 - w2//2, cy2 - h2//2 - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

    # Simulating a mock conflict event every 350 frames (~15 seconds at 20fps)
    event_cycle = frame_idx % 350
    is_fighting = 100 < event_cycle < 220
    
    # Draw red highlight if fighting
    if is_fighting:
        # Draw dynamic action boxes overlapping
        cv2.rectangle(frame, (cx1 - w1//2 - 10, cy1 - h1//2), (cx2 + w2//2 + 10, max(cy1+h1//2, cy2+h2//2)), (0, 0, 220), 2)
        cv2.putText(frame, "SCUFFLE DETECTED", (cx1 - w1//2 - 10, cy1 - h1//2 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
        
        # Bouncing conflict indicators
        for _ in range(3):
            rx = np.random.randint(min(cx1, cx2) - 20, max(cx1, cx2) + 20)
            ry = np.random.randint(min(cy1, cy2) - 50, max(cy1, cy2) + 50)
            cv2.circle(frame, (rx, ry), np.random.randint(5, 15), (0, 0, 240), -1)

    # Dynamic scanning bar
    scan_y = int((frame_idx * 4) % height)
    cv2.line(frame, (0, scan_y), (width, scan_y), (100, 100, 100), 1)
    
    # HUD details
    cv2.putText(frame, "SIMULATED SECURITY FEED - STANDBY MODE", (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 140, 160), 1)
    
    return frame, is_fighting

# ── Core Video Processing Thread ──────────────────────
def video_capture_loop():
    global latest_frame_bytes, current_state
    
    # Initialize variables
    frame_buffer = deque(maxlen=16)
    violence_score = 0.0
    violence_start_t = None
    prev_gray = None
    motion_mag = 0.0
    
    last_fps_calc = time.time()
    frame_count = 0
    fps = 0.0

    current_camera_id = None
    cap = None

    while global_settings["is_running"]:
        target_camera_id = global_settings["camera_id"]
        
        # Switch camera if ID changed
        if target_camera_id != current_camera_id:
            if cap is not None:
                cap.release()
                cap = None
            
            # Check if camera ID is numeric or a video file path
            try:
                cam_source = int(target_camera_id)
            except ValueError:
                cam_source = target_camera_id  # file path
                
            print(f"[Core] Connecting to camera source: {cam_source}")
            cap = cv2.VideoCapture(cam_source)
            current_camera_id = target_camera_id
            
            if cap.isOpened():
                print("✅ [Core] Camera opened successfully!")
                current_state["using_mock"] = False
                current_state["camera_status"] = "Active"
            else:
                print("⚠️ [Core] Camera access failed. Running simulation fallback.")
                current_state["using_mock"] = True
                current_state["camera_status"] = "Simulated Feed"

        start_time = time.time()
        is_mock = current_state["using_mock"]

        if is_mock or cap is None or not cap.isOpened():
            # Generate mock frame
            frame, is_fighting_mock = generate_synthetic_frame(frame_idx=frame_count)
            frame_count += 1
            
            # Add synthetic motion
            motion_mag = 0.85 if is_fighting_mock else 0.12
            
            # Compute simulated violence score
            if is_fighting_mock:
                target_score = 0.68 + np.sin(frame_count * 0.1) * 0.15
            else:
                target_score = 0.05 + np.sin(frame_count * 0.02) * 0.04
            
            # Smooth violence score
            violence_score = violence_score * 0.8 + target_score * 0.2
            
            # Simulate processing delay to match FPS (e.g. 20 FPS)
            time.sleep(max(0.01, 0.05 - (time.time() - start_time)))
        else:
            # Read real webcam frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Lost camera feed, retrying...")
                current_state["camera_status"] = "Feed Interrupted"
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # 1. Optical Flow (Motion Check)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mag = float(np.mean(mag))
            else:
                motion_mag = 1.0
            prev_gray = curr_gray.copy()

            # 2. Run Person Detection
            results = yolo_model(frame, classes=[0], conf=global_settings["yolo_confidence"], verbose=False)
            boxes = results[0].boxes
            
            # 3. Crop largest person
            roi = frame
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
                best = xyxy[np.argmax(areas)].astype(int)
                x1, y1, x2, y2 = best
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]

            # 4. Add to Buffer and Classify
            try:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                tensor_roi = transform_pipeline(rgb_roi)
                frame_buffer.append(tensor_roi)
            except Exception as e:
                print(f"[Core] Preprocessing error: {e}")
                continue

            if len(frame_buffer) == 16:
                with torch.no_grad():
                    clip = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)
                    logits = classifier(clip)
                    probs = torch.softmax(logits, dim=1)
                    raw_score = probs[0][1].item()

                # Optical flow suppression
                if motion_mag < global_settings["motion_threshold"]:
                    raw_score *= global_settings["motion_suppress"]
                
                violence_score = raw_score
            else:
                violence_score = 0.0

            # Render YOLO standard bounding boxes on real frame
            frame = results[0].plot()

        # ── Handle Alerts ────────────────────────────────
        is_violent = violence_score > global_settings["violence_threshold"]
        if is_violent:
            if violence_start_t is None:
                violence_start_t = time.time()
            duration = time.time() - violence_start_t
            dispatch_alert(violence_score, global_settings["camera_location"], duration)
        else:
            violence_start_t = None

        # ── FPS Calculation ──────────────────────────────
        now = time.time()
        if now - last_fps_calc >= 1.0:
            fps = frame_count / (now - last_fps_calc)
            frame_count = 0
            last_fps_calc = now

        # ── Overlay HUD ──────────────────────────────────
        annotated_frame = frame.copy()
        h_w = annotated_frame.shape[1]
        
        # Header box overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (h_w, 75), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.80, annotated_frame, 0.20, 0, annotated_frame)

        status_color = (0, 60, 220) if is_violent else (20, 180, 60)
        status_text = "!! DANGER: VIOLENCE !!" if is_violent else "NORMAL"
        
        cv2.putText(annotated_frame, f"System HUD: {status_text}",
                    (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, status_color, 2)
        
        suppressed_label = " [Motion Suppression Active]" if motion_mag < global_settings["motion_threshold"] else ""
        cv2.putText(annotated_frame, f"Score: {violence_score:.1%} {suppressed_label}  |  Loc: {global_settings['camera_location']}",
                    (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 240, 255), 1)
        
        cv2.putText(annotated_frame, f"Motion: {motion_mag:.2f} (Gate: {global_settings['motion_threshold']:.2f})  |  FPS: {fps:.1f}",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 180), 1)

        # Flashing red border on danger
        if is_violent and int(time.time() * 3) % 2 == 0:
            cv2.rectangle(annotated_frame, (3, 3), (h_w - 3, annotated_frame.shape[0] - 3), (0, 0, 255), 4)

        # ── Encode Frame as JPEG ─────────────────────────
        ret_enc, jpeg_bytes = cv2.imencode(".jpg", annotated_frame)
        if ret_enc:
            with frame_lock:
                latest_frame_bytes = jpeg_bytes.tobytes()

        # Update global state for endpoints
        current_state.update({
            "fps": round(fps, 1),
            "violence_score": round(violence_score, 4),
            "motion_mag": round(motion_mag, 2),
            "is_violent": is_violent,
        })

    # Cleanup when exiting loop
    if cap is not None:
        cap.release()
    print("[Core] Video processing thread ended.")

# ── API Routes ───────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Returns general server status, loaded settings, and current metrics."""
    return {
        "status": "online",
        "device": DEVICE,
        "settings": global_settings,
        "metrics": current_state
    }

@app.post("/api/settings")
def update_settings(new_settings: Dict[str, Any]):
    """Allows client to dynamically update detection parameters."""
    for key, val in new_settings.items():
        if key in global_settings:
            # Type casting to match config values
            if isinstance(global_settings[key], float):
                global_settings[key] = float(val)
            elif isinstance(global_settings[key], int):
                global_settings[key] = int(val)
            else:
                global_settings[key] = val
    
    # Broadcast configuration changes to websocket clients
    broadcast_message({
        "type": "settings",
        "data": global_settings
    })
    
    print(f"[Server] Settings updated: {global_settings}")
    return {"status": "success", "settings": global_settings}

@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    """Reads historical alert records from alerts.jsonl and returns them in JSON format."""
    alerts = []
    if os.path.exists(ALERTS_JSONL):
        try:
            with open(ALERTS_JSONL, "r") as f:
                lines = f.readlines()
                # Get the last 'limit' records
                for line in lines[-limit:]:
                    if line.strip():
                        alerts.append(json.loads(line.strip()))
        except Exception as e:
            return {"status": "error", "message": f"Failed to read alerts: {str(e)}"}
    
    # Return reversed order to show most recent first
    return {"status": "success", "alerts": list(reversed(alerts))}

# MJPEG Stream generator
def mjpeg_frame_generator():
    global latest_frame_bytes
    while global_settings["is_running"]:
        if latest_frame_bytes is not None:
            with frame_lock:
                frame_data = latest_frame_bytes
            
            # Construct MJPEG boundary chunk
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        # Limit frame streaming rate (~25fps max) to prevent network congestion
        time.sleep(0.04)

@app.get("/api/stream")
def get_stream():
    """Returns the live annotated MJPEG stream of the active camera feed."""
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── WebSockets Alert Hub ──────────────────────────────
@app.websocket("/api/alerts/ws")
async def websocket_alerts_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    print(f"🔌 WebSocket client connected! Total clients: {len(active_websockets)}")
    
    # Send current settings and status immediately on connection
    await websocket.send_text(json.dumps({
        "type": "welcome",
        "data": {
            "settings": global_settings,
            "metrics": current_state
        }
    }))
    
    try:
        while True:
            # Wait for any incoming message (keep-alive, commands, etc.)
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            # Simple ping-pong
            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif data.get("type") == "update_settings":
                # Handle settings change directly through WebSocket if wanted
                update_settings(data.get("data", {}))
                
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
        print("🔌 WebSocket client disconnected.")
    except Exception as e:
        active_websockets.discard(websocket)
        print(f"⚠️ WebSocket error: {e}")

# ── Lifetime Management & Main Runner ──────────────────
@app.on_event("startup")
def startup_event():
    global main_event_loop
    # Keep reference to the main thread's asyncio loop for WebSocket sending
    import asyncio
    main_event_loop = asyncio.get_event_loop()
    
    # Start the core background processing loop
    thread = threading.Thread(target=video_capture_loop, daemon=True)
    thread.start()
    print("🚀 Video capture thread started in background.")

@app.on_event("shutdown")
def shutdown_event():
    global_settings["is_running"] = False
    print("🛑 Shutting down server...")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
