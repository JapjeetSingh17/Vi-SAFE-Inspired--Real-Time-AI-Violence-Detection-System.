# System Architecture & Inference Optimization

This document outlines the system architecture for deploying the Vi-SAFE based violence detection model on live CCTV feeds, fulfilling the requirements for real-time edge processing.

## 1. Real-Time Pipeline Design

The system processes an RTSP stream (CCTV), performs temporal batching, and infers violence events synchronously to minimize latency. 

**System Flow:**
`Camera (RTSP) → Frame Extractor (OpenCV) → Preprocessing (Resize/Norm) → Vi-SAFE Model (TensorRT/TorchScript) → Prediction (FastAPI) → Alert Backend`

### Edge vs. Cloud Architecture
- **Edge Node (Near Cameras):** Handles frame extraction, frame skipping, model inference, and local thresholding. Using an edge GPU (e.g., NVIDIA Jetson Orin Nano / RTX 4060) prevents sending heavy 1080p raw video feeds over the network, achieving the `< 2 seconds detection delay` target.
- **Cloud/Core Server:** Acts as a centralized backend. It receives the lightweight JSON payload (Detection Events) and only the 5-10 second video snippet around the violence event for dashboarding and alerting.

## 2. Optimizing Inference (Target < 2s Latency)

### Frame Skipping Strategy
Analyzing every frame at 30 FPS is computationally wasteful. The model requires 16 frames.
- **Action:** Sample at 8 FPS (Skip 3-4 frames). This captures 2 seconds of temporal context (16 frames) perfectly tailored for Vi-SAFE's temporal receptive window while drastically lowering I/O latency.

### Batch vs Single Frame Inference
- Use a **Sliding Window Batch Queue**: Use a moving `deque` of 16 frames. 
- Avoid single-frame processing since Vi-SAFE relies on 3D temporal convolutions. Pass the entire 16-frame tensor in one mini-batch. 

### GPU vs CPU Trade-offs
- **CPU:** High latency (500ms+ per inference block). Suitable only for 1-2 cameras at 2-3 fps.
- **GPU:** Low latency (10-30ms per inference block). Essential for scaling to multiple CCTV streams concurrently. Use `FP16` precision (Half-Tensor) for up to 2x speedup on NVIDIA GPUs.

---

## 3. Deployment Code Snippets (FastAPI + OpenCV)

This snippet demonstrates capturing RTSP streams, managing the temporal queue, and serving via FastAPI. 
*(A fully functional script `serve.py` has also been added to the repository).*

```python
import cv2
import torch
import collections
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()
model = load_visafe_model().half().cuda() # Load optimized FP16 model
model.eval()

# Temporal queue for 16 frames
frame_buffer = collections.deque(maxlen=16)

def infer_stream(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        # Frame skipping - process 8 FPS (skip every 3 frames out of 30)
        if frame_count % 4 != 0:
            continue
            
        processed_frame = preprocess(frame)
        frame_buffer.append(processed_frame)
        
        # When buffer is full, infer
        if len(frame_buffer) == 16:
            input_tensor = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).half().cuda()
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0][1].item() # Violence class prob
                
            if prob > 0.85: # High confidence threshold
                trigger_alert(prob, rtsp_url)

@app.post("/start_stream")
def start_detection(rtsp_url: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(infer_stream, rtsp_url)
    return {"status": "started", "camera": rtsp_url}
```

### Docker Deployment
Containerize the edge inferencer for scale using NVIDIA runtime:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt fastapi uvicorn opencv-python-headless
COPY . .
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```
Run with: `docker run --gpus all -p 8000:8000 my-video-inference`
