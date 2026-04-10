import cv2
import numpy as np
import os

def create_dummy_video(filename, num_frames=16, height=112, width=112, color=(0, 0, 255)):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(filename, fourcc, 5.0, (width, height))

    for _ in range(num_frames):
        # Create a frame with random noise added to a base color
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        out.write(frame)

    out.release()

if __name__ == "__main__":
    os.makedirs("dataset/violence", exist_ok=True)
    os.makedirs("dataset/non_violence", exist_ok=True)

    print("Generating dummy violence videos...")
    for i in range(5):
        create_dummy_video(f"dataset/violence/dummy_{i}.mp4", color=(0, 0, 255)) # Red-ish

    print("Generating dummy non-violence videos...")
    for i in range(5):
        create_dummy_video(f"dataset/non_violence/dummy_{i}.mp4", color=(0, 255, 0)) # Green-ish
        
    print("Done generating 10 dummy videos.")
