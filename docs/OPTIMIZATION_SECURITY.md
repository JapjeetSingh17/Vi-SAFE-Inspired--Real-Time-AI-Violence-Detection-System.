# AI Optimization & Campus Security Policy

This document provides strategies for minimizing false positives, optimizing edge models, and ensuring data privacy for the university's Vi-SAFE surveillance deployment.

## 1. Decreasing False Positives
Violence detection in open spaces like hostels can easily be confused by students playing sports, dancing, or hugging.
- **Ensemble Methods:** Instead of relying entirely on 3D convolutions (Vi-SAFE), run a lightweight Object Detection pose-estimator (like YOLO or MoveNet) in parallel. If Vi-SAFE detects violence `prob=0.9` but the Object Detector sees 0 humans or peaceful poses, suppress the alert.
- **Temporal/Optical Flow Smoothing:** Check the velocity of objects across frames. Abnormal accelerations often correlate with physical altercations.

## 2. Model Optimization for Edge Hardware
Deploying PyTorch models directly on edge devices (Jetson, Raspberry Pi) is inefficient.
1. **Quantization:**
   - Convert FP32 (32-bit Float) models to **INT8** or **FP16** using TensorRT.
   - *Result:* 2x to 4x inference speedup with negligible (< 1%) drop in accuracy.
2. **Model Pruning:**
   - Eliminate redundant weights in the ResNet3D backbone that have values near zero. 

## 3. Continuous Learning Pipeline
A static model decays over time. The system should learn from university-specific behavior.
- **False Alert Loop:** When dashboard operators mark an alert as a "False Positive," the backend should securely save that 16-frame video clip into a `hard_negatives` AWS S3 bucket.
- **Periodic Retraining:** Every month, fine-tune the Vi-SAFE model on this locally collected data to improve contextual robustness.

## 4. Security & Privacy Architecture
CCTV in university campuses presents significant ethical and privacy concerns.

### Technical Security
- **API Security:** Edge nodes authenticate with the Cloud backend using rotating JWTs or mutual TLS (mTLS). 
- **Encryption:** RTSP Streams and WebRTC connections must be wrapped in SRTP (Secure Real-Time Protocol), preventing local network sniffing.

### Privacy Safeguards
- **No Long-term PII Storage:** Raw video streams should only exist in the volatile RAM of the edge device for the 2-second sliding window. Once analyzed, it is immediately deleted *unless* an alert is triggered.
- **Face Masking (Anonymization):** If policy requires, integrate a lightweight face-blurring filter (e.g. Haar Cascades + Gaussian Blur) at the edge *before* saving the alert clip to the cloud.

## Deployment Checklist
- [ ] TensorRT INT8 Model Export generated?
- [ ] Edge devices configured to use static IPs without public ingress?
- [ ] Temporal debouncing (4-second rule) implemented?
- [ ] End-to-end load tests confirming latency < 2 seconds?
- [ ] GDPR/Campus Privacy Policy approval obtained?
