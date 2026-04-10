# GPU Training Guide: Real Life Violence Dataset

This document provides the exact automated pipeline for teammates with capable GPU hardware to download the dataset and train the Vi-SAFE/ResNet3D model locally.

## Why We Need a GPU Node
1. **Training Takes Time:** Processing thousands of videos through a 3D Temporal Convolutional Network (Vi-SAFE/ResNet3D) on standard CPUs (like a MacBook Air) takes over 5-10 hours just for 5 epochs. A dedicated GPU shrinks this drastically.
2. **Authenticated Data:** World-class datasets like the *Real Life Violence Dataset* (the "Hello World" of violence detection) are hosted on platforms completely protected by authenticated APIs (like Kaggle) to prevent bots from stealing data. It requires an authenticated user command.

Follow the steps below to securely pull the data and immediately commence the heavy training cycle.

---

## The Automated Training Pipeline

### Step 1: Install the Kaggle Downloader
Run this in your terminal to get the official edge-downloader toolkit:
```bash
pip install kaggle
```

### Step 2: Configure Your Kaggle API Key
1. Create a free account at [Kaggle.com](https://www.kaggle.com/)
2. Navigate to your **Profile -> Settings -> Create New API Token**. This will download a file called `kaggle.json`.
3. Move that file into your system's hidden `.kaggle` folder and restrict permissions:
```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download & Train!
When you are ready to let your GPU matrix crunch, execute the following block of commands. 

This script natively grabs the massive 2GB dataset (containing 1,000+ CCTV fight/normal videos), structures it directly into our pipeline folders, and starts the full PyTorch training cycle.

```bash
cd "AI-Driven-Real-time-violence-detection-model"

# 1. Download the Real Life Violence Dataset securely
kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset

# 2. Extract it directly into our dataset format structure
unzip real-life-violence-situations-dataset.zip
mv "Real Life Violence Dataset/Violence" "dataset/violence"
mv "Real Life Violence Dataset/NonViolence" "dataset/non_violence"

# 3. Start the heavy training cycle on the GPU
python3 train.py
```

Once `train.py` completes, it will export the finalized **`violence_detection_model.pth`** weights file to the root directory.

You can then thoroughly test the model accuracy in real-time by launching the interactive webcam UI:
```bash
python3 live_demo.py
```
