import cv2
import torch
import collections
from fastapi import FastAPI, BackgroundTasks
import uvicorn

# We reuse the model definition from your existing codebase
from model import ViolenceDetectionModel
from torchvision import transforms

app = FastAPI(title="Vi-SAFE Real-Time Violence Detection API")

# Load model globally so it's ready for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViolenceDetectionModel(num_classes=2, pretrained=False)

try:
    model.load_state_dict(torch.load("violence_detection_model.pth", map_location=device))
except FileNotFoundError:
    print("Warning: violence_detection_model.pth not found. Using randomly initialized weights for demonstration.")

model.to(device)
model.eval()

# Temporal queue for Vi-SAFE (requires sliding window of 16 frames)
# Since we might have multiple cameras, we keep a dict of deques
active_streams = {}

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

def notify_backend(camera_val, confidence):
    """
    Mock function to represent triggering an alert to the Node.js backend.
    In production, this would be an HTTP POST to the backend API.
    """
    print(f"🚨 [ALERT] Violence Detected on {camera_val}! Confidence: {confidence:.2f}")
    # requests.post("http://backend-server/api/v1/detect", json={...})

def process_stream(rtsp_url: str):
    """
    Background task to continuously read RTSP stream, skip frames, and infer.
    """
    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0
    active_streams[rtsp_url] = collections.deque(maxlen=16)
    
    # Simple temporal debouncing (only alert if multiple frames are bad)
    violence_streak = 0
    
    print(f"Started Monitoring Stream: {rtsp_url}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print(f"Stream {rtsp_url} ended or failed.")
            break
            
        frame_count += 1
        
        # Frame Skipping Strategy to handle 30fps -> 8fps (target < 2s duration)
        if frame_count % 4 != 0:
            continue
            
        # Preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = preprocess(rgb_frame)
        active_streams[rtsp_url].append(tensor_frame)
        
        # Once buffer is full (2 seconds of temporal context), infer
        if len(active_streams[rtsp_url]) == 16:
            # Shape into (Batch, Channels, Temporal Depth, H, W)
            input_tensor = torch.stack(list(active_streams[rtsp_url]), dim=1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # Class 1 = Violence
                violence_prob = probabilities[0][1].item()
                
                # Confidence Threshold check (Avoid False Positives)
                if violence_prob > 0.85:
                    violence_streak += 1
                else:
                    violence_streak = 0
                    
                # Temporal Debouncing (Must be confident for 2 consecutive batches)
                if violence_streak >= 2:
                    notify_backend(rtsp_url, violence_prob)
                    violence_streak = 0 # reset to avoid spamming
                    # We might also dump the buffer to MP4 here and upload it

    cap.release()
    if rtsp_url in active_streams:
        del active_streams[rtsp_url]

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Vi-SAFE Inference API. The service is running.",
        "endpoints": {
            "POST /start_stream?rtsp_url=<url>": "Start inference on a video stream"
        }
    }

@app.post("/start_stream")
def start_detection(rtsp_url: str, background_tasks: BackgroundTasks):
    """
    API endpoint to trigger the processing of a new CCTV stream on Edge Node.
    Example: curl -X POST "http://localhost:8000/start_stream?rtsp_url=0" 
    """
    if rtsp_url in active_streams:
        return {"status": "error", "message": "Stream is already being processed."}
        
    background_tasks.add_task(process_stream, rtsp_url)
    return {"status": "started", "camera": rtsp_url, "message": "Inference running in background."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
