import cv2
import torch
import collections
from torchvision import transforms
from model import ViolenceDetectionModel

def start_live_demo():
    print("Initializing model...")
    # 1. Load the Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ViolenceDetectionModel(num_classes=2, pretrained=False)

    try:
        model.load_state_dict(torch.load("violence_detection_model.pth", map_location=device))
        print("Model weights loaded.")
    except FileNotFoundError:
        print("\n=======================================================")
        print("⚠️ WARNING: violence_detection_model.pth not found!")
        print("Using randomly initialized weights.")
        print("The detections will be RANDOM until you execute train.py.")
        print("=======================================================\n")

    model.to(device)
    model.eval()

    # 2. Setup OpenCV VideoCapture for Webcam (0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 3. Setup preprocessing and buffers
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

    frame_buffer = collections.deque(maxlen=16)
    frame_count = 0
    current_status = "NORMAL"
    status_color = (0, 255, 0) # Green
    
    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # We keep the display frame at full resolution
        display_frame = frame.copy()

        # Frame Skipping for Inference (Run Model on ~8 FPS)
        if frame_count % 4 == 0:
            # Preprocess the frame and add to buffer
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = preprocess(rgb_frame)
            frame_buffer.append(tensor_frame)

            # Once we have 16 frames (2 seconds of temporal context), predict!
            if len(frame_buffer) == 16:
                input_tensor = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    violence_prob = probabilities[0][1].item()

                if violence_prob > 0.85:
                    current_status = f"VIOLENCE DETECTED! ({violence_prob*100:.1f}%)"
                    status_color = (0, 0, 255) # Red
                    print(f"🚨 ALERT: {current_status}") 
                else:
                    current_status = "NORMAL"
                    status_color = (0, 255, 0) # Green

        # Overlay text on the video feed
        cv2.putText(display_frame, current_status, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('Vi-SAFE Real-Time Demo', display_frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_demo()
