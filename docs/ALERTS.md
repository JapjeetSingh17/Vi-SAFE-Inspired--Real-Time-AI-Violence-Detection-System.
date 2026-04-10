# Alert System & Backend Infrastructure

This document outlines the backend architecture required to notify campus authorities when the Vi-SAFE edge nodes detect violence. 

## 1. Alert System Design

**Tech Stack Recommendations:**
- **Backend:** Node.js (Express) or Django (Python). Node.js is excellent for real-time I/O (WebSockets).
- **Database:** PostgreSQL (for relational camera/location management and user data) + Redis (for real-time event debouncing and WebSocket pub/sub). Firebase is a good alternative if you want to offload push notifications completely.
- **Push Notifications:** Firebase Cloud Messaging (FCM) for PWA/Mobile app pushes.
- **SMS/Email:** Twilio (SMS) and SendGrid/Amazon SES (Email).

## 2. Smart Logic: Reducing False Alarms
A model will inevitably jitter. An alert should not be sent for a single 0.5s anomalous prediction.

### Thresholds & Temporal Debouncing
1. **Confidence Threshold:** Only log predictions where `confidence > 0.85`.
2. **Temporal Smoothing (The "X Seconds" Rule):** 
   Maintain an event state in Redis. Provide an alert ONLY if violence is detected sequentially for `> 2 seconds`.
   - *Logic:* Since 1 inference block = 2 seconds, wait for 2 consecutive inference blocks (total 4 seconds of context) to return `violence=True` before triggering the API push.

## 3. Backend APIs

Here is the specification for the primary API endpoints:

### 1. `POST /api/v1/detect`
*Consumed by the Edge Node (FastAPI/Inference server).*
**Request Body:**
```json
{
  "camera_id": "cam_hostel_04",
  "location": "North Wing Hallway",
  "timestamp": "2024-05-15T14:32:00Z",
  "confidence_score": 0.92,
  "clip_url": "https://s3.cloud/events/video_clip_xyz.mp4"
}
```
**Backend Action:** Saves to DB, checks temporal requirements, and if triggered, broadcasts to connected WebSockets and Firebase.

### 2. `GET /api/v1/alerts`
*Consumed by the Frontend Dashboard.*
**Query Params:** `?status=active&limit=50&sort=desc`
**Response:**
```json
[
  {
    "alert_id": "993a4b",
    "camera_id": "cam_hostel_04",
    "status": "UNRESOLVED",
    "timestamp": "2024-05-15T14:32:00Z",
    "clip_url": "..."
  }
]
```

## 4. Dashboard Structure (WebSockets)

A responsive security dashboard should consist of:
1. **Live Camera Grid (WebRTC/RTSP Over HLS):** Clean grid viewing campus zones.
2. **Alert Timeline (WebSocket Connected):** A dynamic sidebar. When `POST /detect` is triggered, the Node.js backend emits a `violent_event` over Socket.io to the frontend. The timeline updates instantly with a flashing red indicator and the 5-second video clip loops.
3. **Campus Map Tagging:** An interactive map with green dots mapping to camera nodes. A Dot turns red when an event occurs, allowing authorities to visually pinpoint the location.
