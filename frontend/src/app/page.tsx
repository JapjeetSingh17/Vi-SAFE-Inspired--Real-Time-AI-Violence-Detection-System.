"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Shield, 
  Activity, 
  AlertTriangle, 
  Settings, 
  ListRestart, 
  Bell, 
  BellOff, 
  Database,
  Radio, 
  Info,
  Calendar,
  Clock,
  Download,
  Trash2,
  Sliders,
  Tv,
  CheckCircle2,
  AlertCircle
} from "lucide-react";

// Mock locations for extra cameras
const EXTRA_CAMERAS = [
  { id: "cam-2", name: "Campus Main Gate", status: "Active", bgHue: 200, location: "Main Gate" },
  { id: "cam-3", name: "Student Cafeteria", status: "Active", bgHue: 280, location: "Cafeteria" },
  { id: "cam-4", name: "Science Lab Hallway", status: "Standby", bgHue: 340, location: "Science Hall" }
];

interface AlertRecord {
  timestamp: string;
  location: string;
  confidence: number;
  duration_seconds: number;
}

export default function SecurityControlRoom() {
  // Settings State
  const [backendUrl, setBackendUrl] = useState(process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000");
  const [wsConnected, setWsConnected] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [activeTab, setActiveTab] = useState("dashboard"); // dashboard, alerts, analytics, settings
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Model parameters state
  const [violenceThreshold, setViolenceThreshold] = useState(0.35);
  const [motionThreshold, setMotionThreshold] = useState(0.25);
  const [yoloConfidence, setYoloConfidence] = useState(0.40);
  const [alertCooldown, setAlertCooldown] = useState(10);
  const [cameraLocation, setCameraLocation] = useState("Library - Floor 2");
  const [cameraId, setCameraId] = useState("0");

  // Telemetry metrics
  const [metrics, setMetrics] = useState({
    fps: 0.0,
    violenceScore: 0.0,
    motionMag: 0.0,
    isViolent: false,
    cameraStatus: "Disconnected",
    usingMock: true
  });

  // Alerts log
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [alertFilter, setAlertFilter] = useState("all");
  const [minConfFilter, setMinConfFilter] = useState(0.0);
  
  // Flash effect on new threat
  const [recentAlertFlash, setRecentAlertFlash] = useState(false);

  // References
  const wsRef = useRef<WebSocket | null>(null);
  const canvasRefs = useRef<{ [key: string]: HTMLCanvasElement | null }>({});
  const animationFrameId = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Init Audio Context for native synthesizer beeps
  const playAlertSound = (frequency = 650, type: OscillatorType = "sine", duration = 0.3) => {
    if (!soundEnabled) return;
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      const ctx = audioContextRef.current;
      if (ctx.state === "suspended") {
        ctx.resume();
      }
      
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      
      osc.type = type;
      osc.frequency.setValueAtTime(frequency, ctx.currentTime);
      
      // Sweet envelope
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
      
      osc.connect(gain);
      gain.connect(ctx.destination);
      
      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch (e) {
      console.warn("Audio failed to play", e);
    }
  };

  // Double beep for critical alert
  const triggerDoubleBeep = () => {
    playAlertSound(780, "triangle", 0.15);
    setTimeout(() => {
      playAlertSound(700, "triangle", 0.25);
    }, 180);
  };

  // WebSocket Connection Lifecycle
  useEffect(() => {
    if (demoMode) {
      if (wsRef.current) {
        wsRef.current.close();
      }
      setWsConnected(false);
      return;
    }

    let wsHost = backendUrl.replace("http://", "").replace("https://", "");
    const wsProtocol = backendUrl.startsWith("https") ? "wss://" : "ws://";
    const wsUrl = `${wsProtocol}${wsHost}/api/alerts/ws`;
    
    console.log(`🔌 Attempting WebSocket connection to: ${wsUrl}`);
    
    const connectWs = () => {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log("✅ WebSocket connected successfully");
          setWsConnected(true);
          // Fetch historical alerts once backend is online
          fetchHistoricalAlerts();
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "welcome") {
              const s = data.data.settings;
              const m = data.data.metrics;
              setViolenceThreshold(s.violence_threshold);
              setMotionThreshold(s.motion_threshold);
              setYoloConfidence(s.yolo_confidence);
              setAlertCooldown(s.alert_cooldown);
              setCameraLocation(s.camera_location);
              setCameraId(s.camera_id.toString());
              
              setMetrics({
                fps: m.fps,
                violenceScore: m.violence_score,
                motionMag: m.motion_mag,
                isViolent: m.is_violent,
                cameraStatus: m.camera_status,
                usingMock: m.using_mock
              });
            } else if (data.type === "settings") {
              const s = data.data;
              setViolenceThreshold(s.violence_threshold);
              setMotionThreshold(s.motion_threshold);
              setYoloConfidence(s.yolo_confidence);
              setAlertCooldown(s.alert_cooldown);
              setCameraLocation(s.camera_location);
              setCameraId(s.camera_id.toString());
            } else if (data.type === "alert") {
              const newAlert: AlertRecord = data.data;
              setAlerts((prev) => [newAlert, ...prev]);
              setRecentAlertFlash(true);
              triggerDoubleBeep();
              setTimeout(() => setRecentAlertFlash(false), 2000);
            }
          } catch (e) {
            console.error("Failed to parse WS message", e);
          }
        };

        ws.onclose = () => {
          console.log("🔌 WebSocket closed, retrying in 3s...");
          setWsConnected(false);
          setTimeout(() => {
            if (!demoMode && wsRef.current === ws) {
              connectWs();
            }
          }, 3000);
        };

        ws.onerror = (err) => {
          console.error("WS error: ", err);
          ws.close();
        };

      } catch (e) {
        console.error("WebSocket setup failed", e);
      }
    };

    connectWs();

    // Poll current state periodically
    const statusInterval = setInterval(() => {
      if (!demoMode) {
        fetch(`${backendUrl}/api/status`)
          .then((res) => res.json())
          .then((data) => {
            if (data.status === "online") {
              const m = data.metrics;
              setMetrics({
                fps: m.fps,
                violenceScore: m.violence_score,
                motionMag: m.motion_mag,
                isViolent: m.is_violent,
                cameraStatus: m.camera_status,
                usingMock: m.using_mock
              });
            }
          })
          .catch(() => {
            // Silently swallow fetch errors when offline
          });
      }
    }, 1000);

    return () => {
      clearInterval(statusInterval);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [backendUrl, demoMode]);

  // Fetch historical alerts from backend
  const fetchHistoricalAlerts = () => {
    if (demoMode) return;
    fetch(`${backendUrl}/api/alerts?limit=50`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "success") {
          setAlerts(data.alerts);
        }
      })
      .catch((err) => console.error("Failed to fetch alerts", err));
  };

  // Sync settings with API backend
  const saveSettingsToBackend = (updates: any) => {
    if (demoMode) return;
    
    // Map parameter names to backend snake_case keys
    const backendPayload: any = {};
    if ("violenceThreshold" in updates) backendPayload.violence_threshold = parseFloat(updates.violenceThreshold);
    if ("motionThreshold" in updates) backendPayload.motion_threshold = parseFloat(updates.motionThreshold);
    if ("yoloConfidence" in updates) backendPayload.yolo_confidence = parseFloat(updates.yoloConfidence);
    if ("alertCooldown" in updates) backendPayload.alert_cooldown = parseFloat(updates.alertCooldown);
    if ("cameraLocation" in updates) backendPayload.camera_location = updates.cameraLocation;
    if ("cameraId" in updates) {
      // Send as number if convertible, otherwise string
      const parsed = parseInt(updates.cameraId);
      backendPayload.camera_id = isNaN(parsed) ? updates.cameraId : parsed;
    }

    fetch(`${backendUrl}/api/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(backendPayload)
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "success") {
          console.log("Settings successfully synced to backend");
        }
      })
      .catch((err) => console.error("Failed to update settings", err));
  };

  // Handle local parameter modification
  const handleSettingChange = (key: string, val: any) => {
    switch (key) {
      case "violenceThreshold":
        setViolenceThreshold(val);
        saveSettingsToBackend({ violenceThreshold: val });
        break;
      case "motionThreshold":
        setMotionThreshold(val);
        saveSettingsToBackend({ motionThreshold: val });
        break;
      case "yoloConfidence":
        setYoloConfidence(val);
        saveSettingsToBackend({ yoloConfidence: val });
        break;
      case "alertCooldown":
        setAlertCooldown(val);
        saveSettingsToBackend({ alertCooldown: val });
        break;
      case "cameraLocation":
        setCameraLocation(val);
        saveSettingsToBackend({ cameraLocation: val });
        break;
      case "cameraId":
        setCameraId(val);
        saveSettingsToBackend({ cameraId: val });
        break;
    }
  };

  // Mock data cycle simulation for pure client-side Demo Mode
  useEffect(() => {
    if (!demoMode) return;

    // Reset alert list to custom demo set initially
    setAlerts([
      { timestamp: new Date(Date.now() - 3600000 * 2).toISOString().replace("T", " ").slice(0, 19), location: "Campus Main Gate", confidence: 0.742, duration_seconds: 5.1 },
      { timestamp: new Date(Date.now() - 3600000 * 4).toISOString().replace("T", " ").slice(0, 19), location: "Library - Floor 2", confidence: 0.612, duration_seconds: 2.5 }
    ]);

    let frameCount = 0;
    const interval = setInterval(() => {
      frameCount++;
      
      // Bouncing values for score and motion
      const sinVal = Math.sin(frameCount * 0.1);
      const isEventActive = (frameCount % 180) > 100 && (frameCount % 180) < 160;

      let score = 0.05 + (sinVal + 1) * 0.04;
      let motion = 0.06 + (sinVal * sinVal) * 0.1;
      
      if (isEventActive) {
        score = 0.62 + Math.cos(frameCount * 0.2) * 0.12;
        motion = 0.72 + Math.sin(frameCount * 0.3) * 0.15;
      }

      setMetrics({
        fps: 20.0 + Math.random() * 0.5,
        violenceScore: score,
        motionMag: motion,
        isViolent: score > violenceThreshold,
        cameraStatus: "Demo Mode Active",
        usingMock: true
      });

      // Trigger alerts in Demo Mode
      if (score > violenceThreshold && frameCount % 30 === 0) {
        const demoAlert: AlertRecord = {
          timestamp: new Date().toISOString().replace("T", " ").slice(0, 19),
          location: "Library - Floor 2",
          confidence: score,
          duration_seconds: 1.2 + (frameCount % 5) * 1.5
        };
        setAlerts((prev) => [demoAlert, ...prev]);
        setRecentAlertFlash(true);
        triggerDoubleBeep();
        setTimeout(() => setRecentAlertFlash(false), 2000);
      }

    }, 150);

    return () => clearInterval(interval);
  }, [demoMode, violenceThreshold]);

  // Security Feeds Bounding Box & HUD Drawings
  useEffect(() => {
    let frameIdx = 0;

    const renderCanvasFeeds = () => {
      frameIdx++;
      
      // Render simulated cameras 2, 3, 4
      EXTRA_CAMERAS.forEach((cam) => {
        const canvas = canvasRefs.current[cam.id];
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;

        // Clear with dark blue hue
        ctx.fillStyle = `hsl(${cam.bgHue}, 30%, 6%)`;
        ctx.fillRect(0, 0, w, h);

        // Security Grid lines
        ctx.strokeStyle = `hsla(${cam.bgHue}, 20%, 30%, 0.1)`;
        ctx.lineWidth = 1;
        const grid = 30;
        for (let x = 0; x < w; x += grid) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        for (let y = 0; y < h; y += grid) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        // Bouncing mock humans
        const t = frameIdx * 0.015 + (cam.bgHue * 0.01);
        const cx = w/2 + Math.sin(t * 0.7) * (w/4);
        const cy = h/2 + Math.cos(t * 1.1) * (h/6);
        const pw = 45;
        const ph = 90;

        // Bounding boxes
        ctx.strokeStyle = "rgba(0, 230, 118, 0.65)";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(cx - pw/2, cy - ph/2, pw, ph);

        // Box label
        ctx.fillStyle = "rgba(0, 230, 118, 0.85)";
        ctx.font = "9px monospace";
        ctx.fillText("Person: 0.88", cx - pw/2, cy - ph/2 - 4);

        // Draw camera details HUD
        ctx.fillStyle = "rgba(226, 232, 240, 0.4)";
        ctx.font = "9px sans-serif";
        ctx.fillText(`CAM // ${cam.name.toUpperCase()}`, 10, 20);
        ctx.fillText(`FPS: 24.0  |  STATUS: ${cam.status}`, 10, h - 10);
        
        // REC circle
        ctx.fillStyle = frameIdx % 30 < 15 ? "#ff3838" : "transparent";
        ctx.beginPath();
        ctx.arc(w - 20, 17, 4, 0, 2 * Math.PI);
        ctx.fill();
      });

      // Render Camera 1 (Main Feed) canvas simulation ONLY when demoMode is active
      if (demoMode) {
        const canvas = canvasRefs.current["cam-1"];
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            const w = canvas.width;
            const h = canvas.height;

            // Clear
            ctx.fillStyle = "#090d16";
            ctx.fillRect(0, 0, w, h);

            // Grid lines
            ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
            ctx.lineWidth = 1;
            for (let x = 0; x < w; x += 40) {
              ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            }
            for (let y = 0; y < h; y += 40) {
              ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            }

            // Bounding boxes moving closer together
            const cycle = frameIdx % 180;
            const isFighting = cycle > 100 && cycle < 160;
            
            const px1 = w/2 - 60 + (isFighting ? Math.sin(frameIdx * 0.1)*5 : Math.sin(frameIdx * 0.04)*25);
            const py1 = h/2 + (isFighting ? Math.cos(frameIdx * 0.1)*3 : 0);

            const px2 = w/2 + 60 - (isFighting ? Math.sin(frameIdx * 0.12)*8 : Math.cos(frameIdx * 0.04)*30);
            const py2 = h/2 + (isFighting ? Math.sin(frameIdx * 0.1)*4 : 0);

            // Bounding box colors
            const strokeColor = isFighting ? "rgba(255, 56, 56, 0.85)" : "rgba(0, 230, 118, 0.7)";
            const labelColor = isFighting ? "rgba(255, 56, 56, 1)" : "rgba(0, 230, 118, 0.9)";
            
            // Draw person 1
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(px1 - 25, py1 - 60, 50, 120);
            ctx.fillStyle = labelColor;
            ctx.font = "10px monospace";
            ctx.fillText(isFighting ? "Person: VIOLENT" : "Person 1: 0.92", px1 - 25, py1 - 66);

            // Draw person 2
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(px2 - 25, py2 - 60, 50, 120);
            ctx.fillStyle = labelColor;
            ctx.fillText(isFighting ? "Person: VIOLENT" : "Person 2: 0.89", px2 - 25, py2 - 66);

            // Red highlights if scuffling
            if (isFighting) {
              ctx.strokeStyle = "rgba(255, 56, 56, 0.4)";
              ctx.lineWidth = 4;
              ctx.strokeRect(w/2 - 100, h/2 - 80, 200, 160);
              
              ctx.fillStyle = "rgba(255, 56, 56, 0.1)";
              ctx.fillRect(w/2 - 100, h/2 - 80, 200, 160);

              ctx.fillStyle = "#ff3838";
              ctx.font = "bold 12px sans-serif";
              ctx.fillText("CRITICAL alert: Violence Detected", w/2 - 85, h/2 - 92);
              
              // Draw some chaotic particle sparks
              for (let i = 0; i < 4; i++) {
                ctx.fillStyle = "rgba(255, 179, 0, 0.8)";
                const rx = w/2 + (Math.random() - 0.5) * 80;
                const ry = h/2 + (Math.random() - 0.5) * 100;
                ctx.beginPath();
                ctx.arc(rx, ry, Math.random() * 4 + 2, 0, 2 * Math.PI);
                ctx.fill();
              }
            }

            // HUD HUD HUD
            ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
            ctx.font = "10px sans-serif";
            ctx.fillText("CAM // 01 (MAIN) // " + cameraLocation.toUpperCase(), 15, 22);
            ctx.fillText(`FPS: ${metrics.fps.toFixed(1)}  |  Motion: ${metrics.motionMag.toFixed(2)}`, 15, h - 15);
            
            // Flashing RED Border if alert is triggered
            if (isFighting && Math.floor(frameIdx / 15) % 2 === 0) {
              ctx.strokeStyle = "#ff3838";
              ctx.lineWidth = 4;
              ctx.strokeRect(0, 0, w, h);
            }
          }
        }
      }

      animationFrameId.current = requestAnimationFrame(renderCanvasFeeds);
    };

    renderCanvasFeeds();

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [demoMode, cameraLocation, metrics.fps, metrics.motionMag]);

  // Helper functions
  const clearAlertsLog = () => {
    setAlerts([]);
    playAlertSound(300, "sine", 0.2);
  };

  const downloadAlertsJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(alerts, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `visafe_alerts_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const formatConfidence = (conf: number) => `${(conf * 100).toFixed(1)}%`;

  // Get active alerts based on filter
  const filteredAlerts = alerts.filter((alert) => {
    if (minConfFilter > 0 && alert.confidence < minConfFilter) return false;
    if (alertFilter === "all") return true;
    return alert.location.toLowerCase().includes(alertFilter.toLowerCase());
  });

  // Calculate stats for Analytics charts
  const totalAlertsCount = alerts.length;
  const averageConfidence = alerts.reduce((acc, curr) => acc + curr.confidence, 0) / (alerts.length || 1);
  
  // Group alerts by location
  const locationStats = alerts.reduce((acc: { [key: string]: number }, curr) => {
    acc[curr.location] = (acc[curr.location] || 0) + 1;
    return acc;
  }, {});

  // Group alerts by hour of day (simple simulated timeline based on alert logs)
  const hourlyStats = alerts.reduce((acc: { [key: number]: number }, curr) => {
    // Extract hour from format "2026-04-20 16:30:00"
    const parts = curr.timestamp.split(" ");
    if (parts.length > 1) {
      const hr = parseInt(parts[1].split(":")[0]);
      if (!isNaN(hr)) {
        acc[hr] = (acc[hr] || 0) + 1;
      }
    } else {
      // Fallback
      acc[12] = (acc[12] || 0) + 1;
    }
    return acc;
  }, {});

  // Generate hourly keys for chart (24 hour range)
  const hoursRange = Array.from({ length: 12 }, (_, i) => (9 + i) % 24); // 9:00 AM to 9:00 PM range representation

  return (
    <div className="flex h-screen overflow-hidden">
      
      {/* Alert Banner / Pulse overlay on critical threats */}
      {recentAlertFlash && (
        <div className="absolute inset-0 border-[6px] border-neon-red/70 pointer-events-none z-50 animate-pulse" />
      )}

      {/* Sidebar Navigation */}
      <div 
        className={`${
          sidebarOpen ? "w-64" : "w-16"
        } bg-[#0b0f19] border-r border-panel-border flex flex-col justify-between transition-all duration-300 z-30`}
      >
        <div>
          {/* Header Branding */}
          <div className="p-4 border-b border-panel-border flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-neon-red/10 flex items-center justify-center border border-neon-red/30">
              <Shield className="w-5 h-5 text-neon-red text-glow-red" />
            </div>
            {sidebarOpen && (
              <div>
                <h1 className="font-bold text-sm leading-none tracking-wide text-white">Vi-SAFE</h1>
                <span className="text-[10px] text-gray-500 font-mono">SECURE CCTV CORE v2.0</span>
              </div>
            )}
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1">
            <button
              onClick={() => setActiveTab("dashboard")}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                activeTab === "dashboard"
                  ? "bg-slate-800/60 text-white font-medium border-l-2 border-neon-blue"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/20"
              }`}
            >
              <Tv className="w-4 h-4" />
              {sidebarOpen && <span>Control Room</span>}
            </button>

            <button
              onClick={() => setActiveTab("alerts")}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                activeTab === "alerts"
                  ? "bg-slate-800/60 text-white font-medium border-l-2 border-neon-red"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/20"
              }`}
            >
              <AlertTriangle className="w-4 h-4" />
              {sidebarOpen && (
                <span className="flex items-center justify-between w-full">
                  <span>Alert Logs</span>
                  {alerts.length > 0 && (
                    <span className="bg-neon-red/20 text-neon-red border border-neon-red/30 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold">
                      {alerts.length}
                    </span>
                  )}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab("analytics")}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                activeTab === "analytics"
                  ? "bg-slate-800/60 text-white font-medium border-l-2 border-neon-green"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/20"
              }`}
            >
              <Activity className="w-4 h-4" />
              {sidebarOpen && <span>Analytics</span>}
            </button>

            <button
              onClick={() => setActiveTab("settings")}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                activeTab === "settings"
                  ? "bg-slate-800/60 text-white font-medium border-l-2 border-neon-amber"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/20"
              }`}
            >
              <Settings className="w-4 h-4" />
              {sidebarOpen && <span>Settings Hub</span>}
            </button>
          </nav>
        </div>

        {/* Sidebar Footer */}
        <div className="p-3 border-t border-panel-border space-y-2">
          {/* Sound toggle */}
          <button 
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="w-full flex items-center gap-3 px-3 py-2 hover:bg-slate-800/30 rounded-lg text-slate-400 hover:text-white text-xs transition-all"
          >
            {soundEnabled ? (
              <>
                <Bell className="w-4 h-4 text-neon-green" />
                {sidebarOpen && <span>Alarm Active</span>}
              </>
            ) : (
              <>
                <BellOff className="w-4 h-4 text-slate-500" />
                {sidebarOpen && <span className="text-slate-500">Alarm Muted</span>}
              </>
            )}
          </button>

          {/* Sidebar collapse button */}
          <button 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="w-full text-center py-1.5 hover:bg-slate-800/30 rounded text-[10px] tracking-widest text-slate-600 font-mono uppercase"
          >
            {sidebarOpen ? "« COLLAPSE" : "»"}
          </button>
        </div>
      </div>

      {/* Main Container Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#06080c] relative">
        
        {/* Top Status Header */}
        <header className="h-16 border-b border-panel-border bg-[#090d16]/40 flex items-center justify-between px-6 z-20">
          <div className="flex items-center gap-4">
            <h2 className="text-md font-semibold tracking-wide text-white uppercase">
              {activeTab === "dashboard" && "Control Room Dashboard"}
              {activeTab === "alerts" && "Real-Time System Log"}
              {activeTab === "analytics" && "Analytical Threat Intelligence"}
              {activeTab === "settings" && "Advanced Configuration"}
            </h2>
            
            {/* Mode badge */}
            <div className="flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-full ${demoMode ? "bg-neon-amber" : "bg-neon-green led-active"} led-active`} />
              <span className="text-xs font-mono font-bold tracking-wider uppercase">
                {demoMode ? (
                  <span className="text-neon-amber">Demo Standby</span>
                ) : (
                  <span className="text-neon-green">Live Active</span>
                )}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Backend connectivity indicator */}
            {!demoMode && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-panel-border">
                <Database className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-xs font-mono font-medium">
                  {wsConnected ? (
                    <span className="text-neon-green">API Connection Online</span>
                  ) : (
                    <span className="text-neon-red animate-pulse">API Offline / Reconnecting</span>
                  )}
                </span>
              </div>
            )}

            {/* Quick Demo toggle */}
            <button
              onClick={() => {
                setDemoMode(!demoMode);
                playAlertSound(440, "sine", 0.15);
              }}
              className={`px-3 py-1.5 rounded text-xs font-mono font-bold border transition-all ${
                demoMode 
                  ? "bg-neon-amber/10 border-neon-amber/30 text-neon-amber hover:bg-neon-amber/20" 
                  : "bg-slate-900 border-panel-border text-slate-400 hover:text-white hover:border-slate-700"
              }`}
            >
              {demoMode ? "SWITCH LIVE" : "SWITCH DEMO"}
            </button>
          </div>
        </header>

        {/* Panel Views Content */}
        <main className="flex-1 overflow-y-auto p-6 relative">
          
          {/* BACKGROUND FUTURE MATRIX GRID */}
          <div className="absolute inset-0 grid-overlay pointer-events-none z-0" />

          {/* TAB 1: DASHBOARD CONTROL ROOM */}
          {activeTab === "dashboard" && (
            <div className="space-y-6 relative z-10">
              
              {/* Telemetry Metrics Alert Ticker */}
              <div className={`p-4 rounded-xl border flex items-center justify-between ${
                metrics.isViolent 
                  ? "bg-neon-red/10 border-neon-red/40 alert-danger-active" 
                  : "bg-[#0b0f19]/60 border-panel-border"
              } transition-all`}>
                <div className="flex items-center gap-4">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    metrics.isViolent ? "bg-neon-red/20 text-neon-red" : "bg-neon-blue/15 text-neon-blue"
                  }`}>
                    {metrics.isViolent ? (
                      <AlertTriangle className="w-5 h-5 text-glow-red" />
                    ) : (
                      <Activity className="w-5 h-5" />
                    )}
                  </div>
                  <div>
                    <h3 className={`text-sm font-semibold uppercase ${metrics.isViolent ? "text-neon-red" : "text-white"}`}>
                      {metrics.isViolent ? "🚨 Threat Detected: Violence Action" : "System Status Normal"}
                    </h3>
                    <p className="text-xs text-slate-400 font-mono">
                      Feed: {cameraLocation} | Status: {metrics.cameraStatus} | Target Score: {(metrics.violenceScore * 100).toFixed(0)}% (Threshold: {(violenceThreshold * 100).toFixed(0)}%)
                    </p>
                  </div>
                </div>

                <div className="flex gap-6 items-center">
                  <div className="text-right">
                    <span className="text-[10px] block text-slate-500 font-mono uppercase">Violence Prob</span>
                    <span className={`text-lg font-bold font-mono ${metrics.isViolent ? "text-neon-red text-glow-red" : "text-neon-blue"}`}>
                      {(metrics.violenceScore * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="text-right border-l border-panel-border pl-6">
                    <span className="text-[10px] block text-slate-500 font-mono uppercase">Motion Mag</span>
                    <span className={`text-lg font-bold font-mono ${metrics.motionMag > motionThreshold ? "text-neon-amber" : "text-slate-400"}`}>
                      {metrics.motionMag.toFixed(2)}
                    </span>
                  </div>

                  <div className="text-right border-l border-panel-border pl-6">
                    <span className="text-[10px] block text-slate-500 font-mono uppercase">Active FPS</span>
                    <span className="text-lg font-bold font-mono text-neon-green">
                      {metrics.fps.toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Cameras Video Stream Grid */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                
                {/* Camera 1: Active Main Channel (FastAPI Streaming or Demo canvas simulation) */}
                <div className={`glass-panel overflow-hidden relative group border ${
                  metrics.isViolent ? "border-neon-red" : "border-panel-border"
                }`}>
                  <div className="p-3 border-b border-panel-border flex items-center justify-between bg-slate-950/40">
                    <div className="flex items-center gap-2">
                      <Radio className={`w-3.5 h-3.5 ${metrics.isViolent ? "text-neon-red" : "text-neon-blue animate-pulse"}`} />
                      <span className="text-xs font-semibold font-mono tracking-wider text-slate-300">
                        CAM 01 // {cameraLocation.toUpperCase()}
                      </span>
                    </div>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase border ${
                      demoMode 
                        ? "bg-neon-amber/10 border-neon-amber/20 text-neon-amber" 
                        : "bg-neon-green/10 border-neon-green/20 text-neon-green"
                    }`}>
                      {demoMode ? "DEMO FEED" : "LIVE CAPTURE"}
                    </span>
                  </div>

                  <div className="aspect-video bg-black relative flex items-center justify-center overflow-hidden">
                    {demoMode ? (
                      /* Demo Simulation Canvas */
                      <canvas 
                        ref={(el) => { canvasRefs.current["cam-1"] = el; }} 
                        width={640} 
                        height={480} 
                        className="w-full h-full object-cover" 
                      />
                    ) : (
                      /* Live Backend Stream */
                      <img 
                        src={`${backendUrl}/api/stream?t=${Date.now()}`} 
                        alt="Security camera live feed" 
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          // Fallback display if server stream fails to load
                          const parent = e.currentTarget.parentElement;
                          if (parent) {
                            e.currentTarget.style.display = 'none';
                            const el = document.getElementById("stream-error");
                            if (el) el.style.display = 'flex';
                          }
                        }}
                      />
                    )}
                    
                    {/* Error placeholder for live stream */}
                    <div 
                      id="stream-error" 
                      className="absolute inset-0 flex-col items-center justify-center bg-[#090d16]/95 text-center gap-2 z-10 hidden"
                    >
                      <AlertTriangle className="w-8 h-8 text-neon-red" />
                      <span className="text-sm font-semibold text-white">LIVE STREAM OFFLINE</span>
                      <span className="text-xs text-slate-500 font-mono px-6">
                        Could not resolve stream route: `{backendUrl}/api/stream`<br />
                        Start the python backend (`python api_server.py`) or switch to Demo Mode.
                      </span>
                    </div>

                    <div className="absolute top-3 left-3 bg-black/60 px-2 py-1 rounded text-[9px] font-mono tracking-widest text-white uppercase">
                      SYS DEV: {metrics.usingMock ? "EMULATOR" : "MPS ACCEL"}
                    </div>
                  </div>

                  {/* Progress Telemetry bars overlay */}
                  <div className="p-4 grid grid-cols-2 gap-4 border-t border-panel-border bg-slate-950/20">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-[10px] font-mono text-slate-500 uppercase">Violence Index</span>
                        <span className={`text-xs font-mono font-bold ${metrics.isViolent ? "text-neon-red" : "text-neon-blue"}`}>
                          {(metrics.violenceScore * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-150 ${metrics.isViolent ? "bg-neon-red" : "bg-neon-blue"}`}
                          style={{ width: `${Math.min(100, metrics.violenceScore * 100)}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-[10px] font-mono text-slate-500 uppercase">Motion Level</span>
                        <span className={`text-xs font-mono font-bold ${metrics.motionMag > motionThreshold ? "text-neon-amber" : "text-slate-400"}`}>
                          {metrics.motionMag.toFixed(2)}
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-neon-amber transition-all duration-150"
                          style={{ width: `${Math.min(100, metrics.motionMag * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Extra Camera channels */}
                {EXTRA_CAMERAS.map((cam) => (
                  <div key={cam.id} className="glass-panel overflow-hidden relative border border-panel-border">
                    <div className="p-3 border-b border-panel-border flex items-center justify-between bg-slate-950/40">
                      <span className="text-xs font-semibold font-mono tracking-wider text-slate-400">
                        CAM // {cam.name.toUpperCase()}
                      </span>
                      <span className="bg-slate-900 border border-panel-border px-1.5 py-0.5 rounded text-[9px] font-mono text-slate-500 uppercase">
                        {cam.status}
                      </span>
                    </div>

                    <div className="aspect-video relative overflow-hidden">
                      <canvas 
                        ref={(el) => { canvasRefs.current[cam.id] = el; }} 
                        width={640} 
                        height={480} 
                        className="w-full h-full object-cover filter brightness-[0.7] contrast-[1.1]" 
                      />
                      <div className="scan-line" />
                    </div>

                    <div className="p-3.5 bg-slate-950/20 text-xs font-mono text-slate-500 flex justify-between">
                      <span>LOCATION: {cam.location.toUpperCase()}</span>
                      <span className="text-neon-green">ACTIVE FEED STABLE</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Rolling Alerts Grid and quick config */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* Rolling Alerts Logs Ticker */}
                <div className="glass-panel lg:col-span-2 p-5 border border-panel-border flex flex-col h-80">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-sm font-semibold tracking-wide text-white uppercase flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-neon-red text-glow-red" />
                      Live Alert Log
                    </h4>
                    <span className="text-[10px] text-slate-500 font-mono uppercase">
                      Showing last 4 entries
                    </span>
                  </div>

                  <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                    {alerts.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 gap-1.5">
                        <CheckCircle2 className="w-7 h-7 text-neon-green" />
                        <span className="text-xs font-semibold">Security Matrix Intact</span>
                        <span className="text-[10px] font-mono">No alerts logged in the current session.</span>
                      </div>
                    ) : (
                      alerts.slice(0, 4).map((alert, idx) => (
                        <div 
                          key={idx}
                          className="p-3.5 rounded-lg bg-neon-red/5 border border-neon-red/15 hover:border-neon-red/35 transition-all flex items-center justify-between"
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-7 h-7 rounded bg-neon-red/10 flex items-center justify-center">
                              <AlertCircle className="w-4 h-4 text-neon-red" />
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-bold text-white">{alert.location}</span>
                                <span className="text-[9px] bg-neon-red/20 text-neon-red px-1 rounded font-mono font-bold uppercase">
                                  CRITICAL
                                </span>
                              </div>
                              <span className="text-[10px] text-slate-500 font-mono block mt-0.5">{alert.timestamp}</span>
                            </div>
                          </div>

                          <div className="flex gap-4 items-center">
                            <div className="text-right">
                              <span className="text-[9px] block text-slate-500 font-mono">CONFIDENCE</span>
                              <span className="text-xs font-bold font-mono text-neon-red">
                                {formatConfidence(alert.confidence)}
                              </span>
                            </div>
                            <div className="text-right border-l border-panel-border pl-4">
                              <span className="text-[9px] block text-slate-500 font-mono">DURATION</span>
                              <span className="text-xs font-bold font-mono text-slate-300">
                                {alert.duration_seconds.toFixed(1)}s
                              </span>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Quick thresholds tuning card */}
                <div className="glass-panel p-5 border border-panel-border flex flex-col justify-between">
                  <div>
                    <h4 className="text-sm font-semibold tracking-wide text-white uppercase flex items-center gap-2 mb-4">
                      <Sliders className="w-4 h-4 text-neon-amber" />
                      Quick Tuning HUD
                    </h4>
                    
                    <div className="space-y-4">
                      {/* Threshold Slider */}
                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1">
                          <span className="text-slate-400">Violence Trigger:</span>
                          <span className="text-neon-amber font-bold">{(violenceThreshold * 100).toFixed(0)}%</span>
                        </div>
                        <input 
                          type="range" 
                          min="0.10" 
                          max="0.85" 
                          step="0.05"
                          value={violenceThreshold} 
                          onChange={(e) => handleSettingChange("violenceThreshold", parseFloat(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neon-amber"
                        />
                      </div>

                      {/* Motion Gate Slider */}
                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1">
                          <span className="text-slate-400">Motion Suppression Gate:</span>
                          <span className="text-neon-amber font-bold">{motionThreshold.toFixed(2)}</span>
                        </div>
                        <input 
                          type="range" 
                          min="0.05" 
                          max="0.60" 
                          step="0.05"
                          value={motionThreshold} 
                          onChange={(e) => handleSettingChange("motionThreshold", parseFloat(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neon-amber"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 pt-4 border-t border-panel-border text-[10px] font-mono text-slate-500">
                    <span className="block">Changing parameters updates the active local inference engine in real time.</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: ALERTS LOG VIEWER */}
          {activeTab === "alerts" && (
            <div className="glass-panel p-6 space-y-6 relative z-10 border border-panel-border h-full flex flex-col">
              
              {/* Filter HUD Toolbar */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-panel-border">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-xs font-semibold text-slate-400 font-mono">FILTER BY LOCATION:</span>
                  <select 
                    value={alertFilter}
                    onChange={(e) => setAlertFilter(e.target.value)}
                    className="bg-[#090d16] border border-panel-border rounded px-3 py-1.5 text-xs text-slate-300 outline-none focus:border-neon-blue"
                  >
                    <option value="all">All Locations</option>
                    <option value="library">Library - Floor 2</option>
                    <option value="gate">Campus Main Gate</option>
                    <option value="cafeteria">Student Cafeteria</option>
                  </select>

                  <span className="text-xs font-semibold text-slate-400 font-mono ml-2">MIN CONFIDENCE:</span>
                  <select 
                    value={minConfFilter}
                    onChange={(e) => setMinConfFilter(parseFloat(e.target.value))}
                    className="bg-[#090d16] border border-panel-border rounded px-3 py-1.5 text-xs text-slate-300 outline-none focus:border-neon-blue"
                  >
                    <option value="0.0">No Limit</option>
                    <option value="0.40">Conf &gt; 40%</option>
                    <option value="0.60">Conf &gt; 60%</option>
                    <option value="0.80">Conf &gt; 80%</option>
                  </select>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    onClick={downloadAlertsJson}
                    className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-mono font-medium px-3.5 py-1.5 rounded transition-all border border-slate-700"
                  >
                    <Download className="w-3.5 h-3.5" />
                    EXPORT JSON
                  </button>

                  <button
                    onClick={clearAlertsLog}
                    className="flex items-center gap-2 bg-neon-red/10 hover:bg-neon-red/20 text-neon-red border border-neon-red/30 text-xs font-mono font-medium px-3.5 py-1.5 rounded transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    CLEAR LOGS
                  </button>
                </div>
              </div>

              {/* Scrollable Alert List */}
              <div className="flex-1 overflow-y-auto space-y-3 pr-2 min-h-0">
                {filteredAlerts.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 py-16 gap-3">
                    <Info className="w-8 h-8 text-slate-500" />
                    <span className="text-sm font-semibold">No Matching Records Found</span>
                    <span className="text-xs font-mono">Adjust your filters or generate new events.</span>
                  </div>
                ) : (
                  filteredAlerts.map((alert, idx) => {
                    const isSevere = alert.confidence > 0.65;
                    return (
                      <div 
                        key={idx}
                        className={`p-4 rounded-xl border transition-all flex items-center justify-between ${
                          isSevere 
                            ? "bg-neon-red/10 border-neon-red/30 hover:border-neon-red/50" 
                            : "bg-slate-900/40 border-panel-border hover:border-slate-700"
                        }`}
                      >
                        <div className="flex items-center gap-4">
                          <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                            isSevere ? "bg-neon-red/20 text-neon-red" : "bg-neon-amber/20 text-neon-amber"
                          }`}>
                            <AlertCircle className="w-5 h-5 animate-pulse" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2.5">
                              <span className="font-bold text-sm text-white">{alert.location}</span>
                              <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold uppercase ${
                                isSevere ? "bg-neon-red/20 text-neon-red" : "bg-neon-amber/20 text-neon-amber"
                              }`}>
                                {isSevere ? "SEVERE THREAT" : "MEDIUM THREAT"}
                              </span>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-slate-500 font-mono mt-1">
                              <span className="flex items-center gap-1">
                                <Clock className="w-3.5 h-3.5" />
                                {alert.timestamp}
                              </span>
                              <span>|</span>
                              <span>INDEX SHAPE: MobileNetV2-LSTM</span>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-6">
                          <div className="text-right">
                            <span className="text-[9px] block text-slate-500 font-mono">VIOLENCE LEVEL</span>
                            <span className={`text-md font-bold font-mono ${isSevere ? "text-neon-red" : "text-neon-amber"}`}>
                              {formatConfidence(alert.confidence)}
                            </span>
                          </div>

                          <div className="text-right border-l border-panel-border pl-6">
                            <span className="text-[9px] block text-slate-500 font-mono">DURATION MEASURED</span>
                            <span className="text-md font-bold font-mono text-slate-300">
                              {alert.duration_seconds.toFixed(1)}s
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Statistics Overlay bottom banner */}
              <div className="p-3 bg-slate-950/40 border border-panel-border rounded-lg text-xs font-mono text-slate-500 flex justify-between items-center">
                <span>TOTAL LOGGED RECORD COUNTS: {totalAlertsCount}</span>
                <span>FILTER MATCHES: {filteredAlerts.length}</span>
              </div>
            </div>
          )}

          {/* TAB 3: ANALYTICS DASHBOARD */}
          {activeTab === "analytics" && (
            <div className="space-y-6 relative z-10">
              
              {/* Telemetry Overview Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <div className="glass-panel p-5 border border-panel-border">
                  <span className="text-[10px] text-slate-500 font-mono uppercase block">Average Threat Severity</span>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold font-mono text-neon-amber">
                      {isNaN(averageConfidence) ? "0.0%" : formatConfidence(averageConfidence)}
                    </span>
                    <span className="text-xs text-slate-500 font-mono">probability</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono block mt-1">Based on global active threat indices.</span>
                </div>

                <div className="glass-panel p-5 border border-panel-border">
                  <span className="text-[10px] text-slate-500 font-mono uppercase block">Total System Alerts</span>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold font-mono text-neon-red text-glow-red">
                      {totalAlertsCount}
                    </span>
                    <span className="text-xs text-slate-500 font-mono">occurrences</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono block mt-1">Total count since server startup.</span>
                </div>

                <div className="glass-panel p-5 border border-panel-border">
                  <span className="text-[10px] text-slate-500 font-mono uppercase block">Active Camera Channels</span>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold font-mono text-neon-green">
                      4
                    </span>
                    <span className="text-xs text-slate-500 font-mono">channels online</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono block mt-1">1 live camera + 3 synthetic arrays.</span>
                </div>

              </div>

              {/* Dynamic SVG Charts Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* SVG Area Chart: Hourly Alerts Frequency */}
                <div className="glass-panel p-5 border border-panel-border flex flex-col h-80">
                  <h4 className="text-xs font-bold font-mono tracking-wider text-slate-300 uppercase mb-6 flex items-center gap-2">
                    <Clock className="w-4 h-4 text-neon-blue" />
                    Hourly Alert Distribution Timeline
                  </h4>

                  <div className="flex-1 w-full flex items-end justify-between px-4 pb-4">
                    {/* Native responsive SVG area graph */}
                    <div className="w-full h-full relative">
                      {alerts.length === 0 ? (
                        <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-xs font-mono">
                          NO ANALYTICAL RECORDS AVAILABLE
                        </div>
                      ) : (
                        <svg className="w-full h-full" viewBox="0 0 500 200" preserveAspectRatio="none">
                          <defs>
                            <linearGradient id="area-grad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#00b0ff" stopOpacity="0.4" />
                              <stop offset="100%" stopColor="#00b0ff" stopOpacity="0.0" />
                            </linearGradient>
                          </defs>
                          
                          {/* Map hourly bars into a smooth path */}
                          {(() => {
                            const dataPoints = hoursRange.map((hr) => hourlyStats[hr] || 0);
                            const maxVal = Math.max(...dataPoints, 1);
                            
                            // Build SVG polyline points
                            const stepX = 500 / (hoursRange.length - 1);
                            const points = hoursRange.map((hr, idx) => {
                              const val = hourlyStats[hr] || 0;
                              const x = idx * stepX;
                              const y = 180 - (val / (maxVal * 1.2)) * 150; // padding top
                              return { x, y };
                            });

                            const linePathStr = points.map(p => `${p.x},${p.y}`).join(" ");
                            const areaPathStr = `0,180 ${linePathStr} 500,180`;

                            return (
                              <>
                                {/* Area */}
                                <polygon points={areaPathStr} fill="url(#area-grad)" />
                                {/* Line */}
                                <polyline points={linePathStr} fill="none" stroke="#00b0ff" strokeWidth="2.5" />
                                {/* Dots */}
                                {points.map((p, idx) => (
                                  <circle key={idx} cx={p.x} cy={p.y} r="3.5" fill="#06080c" stroke="#00b0ff" strokeWidth="2" />
                                ))}
                              </>
                            );
                          })()}
                        </svg>
                      )}
                    </div>
                  </div>

                  {/* Chart timeline labels */}
                  <div className="flex justify-between px-4 text-[9px] font-mono text-slate-500 border-t border-panel-border/30 pt-2">
                    {hoursRange.map((hr, idx) => (
                      <span key={idx}>{hr}:00</span>
                    ))}
                  </div>
                </div>

                {/* SVG Bar Chart: Alerts Count per Location */}
                <div className="glass-panel p-5 border border-panel-border flex flex-col h-80">
                  <h4 className="text-xs font-bold font-mono tracking-wider text-slate-300 uppercase mb-6 flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-neon-green" />
                    Threat Hotspots per Location
                  </h4>

                  <div className="flex-1 overflow-y-auto space-y-4 flex flex-col justify-center pr-2">
                    {Object.keys(locationStats).length === 0 ? (
                      <div className="h-full flex items-center justify-center text-slate-600 text-xs font-mono">
                        NO LOCATION DATA CAPTURED
                      </div>
                    ) : (
                      Object.entries(locationStats).map(([loc, count], idx) => {
                        const maxCount = Math.max(...Object.values(locationStats), 1);
                        const pct = (count / maxCount) * 100;
                        return (
                          <div key={idx} className="space-y-1">
                            <div className="flex justify-between text-xs font-mono">
                              <span className="text-slate-300 font-bold">{loc}</span>
                              <span className="text-neon-green">{count} alerts</span>
                            </div>
                            <div className="w-full h-2.5 bg-slate-900 border border-panel-border rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-neon-green glow-green transition-all duration-500"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* TAB 4: ADVANCED SETTINGS PANEL */}
          {activeTab === "settings" && (
            <div className="glass-panel p-6 space-y-6 relative z-10 border border-panel-border">
              
              <div className="flex items-center gap-3 border-b border-panel-border pb-4">
                <Sliders className="w-5 h-5 text-neon-amber" />
                <div>
                  <h3 className="text-sm font-semibold uppercase text-white">System Variable Hub</h3>
                  <p className="text-xs text-slate-500 font-mono">Configure the AI models and the endpoint networking parameters.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Section A: Networking */}
                <div className="space-y-4">
                  <h4 className="text-xs font-bold font-mono uppercase text-neon-blue tracking-widest border-b border-panel-border/30 pb-2">
                    A. Network Endpoints
                  </h4>

                  {/* Backend Server URL */}
                  <div>
                    <label className="block text-xs font-mono text-slate-400 mb-1.5 uppercase">
                      FastAPI Base API URL:
                    </label>
                    <div className="flex gap-2">
                      <input 
                        type="text" 
                        value={backendUrl}
                        onChange={(e) => setBackendUrl(e.target.value)}
                        disabled={demoMode}
                        className="bg-[#090d16] border border-panel-border rounded px-3 py-2 text-xs text-slate-200 flex-1 outline-none focus:border-neon-blue disabled:opacity-40 disabled:cursor-not-allowed"
                      />
                      <button 
                        onClick={() => {
                          setDemoMode(false);
                          playAlertSound(440, "sine", 0.1);
                        }}
                        disabled={!demoMode}
                        className="bg-slate-800 hover:bg-slate-700 text-white text-xs font-mono px-3 py-2 rounded transition-all border border-slate-700 disabled:opacity-40"
                      >
                        CONNECT
                      </button>
                    </div>
                    <span className="text-[10px] text-slate-500 font-mono block mt-1">
                      Specify the URL where the local python server is running. Disabled in Demo Mode.
                    </span>
                  </div>

                  {/* Camera ID */}
                  <div>
                    <label className="block text-xs font-mono text-slate-400 mb-1.5 uppercase">
                      Hardware Camera ID or Video Path:
                    </label>
                    <input 
                      type="text" 
                      value={cameraId}
                      onChange={(e) => handleSettingChange("cameraId", e.target.value)}
                      className="bg-[#090d16] border border-panel-border rounded px-3 py-2 text-xs text-slate-200 w-full outline-none focus:border-neon-blue"
                    />
                    <span className="text-[10px] text-slate-500 font-mono block mt-1">
                      Use `0` for your Mac webcam. You can also specify an absolute video file path (e.g. `/Users/.../fight.mp4`) to feed a recorded clip to the model.
                    </span>
                  </div>

                  {/* Camera location label */}
                  <div>
                    <label className="block text-xs font-mono text-slate-400 mb-1.5 uppercase">
                      CCTV Location Tag:
                    </label>
                    <input 
                      type="text" 
                      value={cameraLocation}
                      onChange={(e) => handleSettingChange("cameraLocation", e.target.value)}
                      className="bg-[#090d16] border border-panel-border rounded px-3 py-2 text-xs text-slate-200 w-full outline-none focus:border-neon-blue"
                    />
                    <span className="text-[10px] text-slate-500 font-mono block mt-1">
                      The label used in alert logs and overlays to map where the camera is located.
                    </span>
                  </div>
                </div>

                {/* Section B: Model Parameters */}
                <div className="space-y-4">
                  <h4 className="text-xs font-bold font-mono uppercase text-neon-blue tracking-widest border-b border-panel-border/30 pb-2">
                    B. Machine Learning Controls
                  </h4>

                  {/* Threshold */}
                  <div>
                    <div className="flex justify-between items-center mb-1.5">
                      <label className="block text-xs font-mono text-slate-400 uppercase">
                        Violence Alert Threshold:
                      </label>
                      <span className="text-xs font-mono font-bold text-neon-amber">{(violenceThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.10" 
                      max="0.85" 
                      step="0.05"
                      value={violenceThreshold} 
                      onChange={(e) => handleSettingChange("violenceThreshold", parseFloat(e.target.value))}
                      className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neon-amber"
                    />
                    <span className="text-[10px] text-slate-500 font-mono block mt-1">
                      Lower values trigger alerts easily. Raise this toward 0.65+ to suppress false alarms.
                    </span>
                  </div>

                  {/* Motion Gate */}
                  <div>
                    <div className="flex justify-between items-center mb-1.5">
                      <label className="block text-xs font-mono text-slate-400 uppercase">
                        Optical Flow Motion Gate:
                      </label>
                      <span className="text-xs font-mono font-bold text-neon-amber">{motionThreshold.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.05" 
                      max="0.60" 
                      step="0.05"
                      value={motionThreshold} 
                      onChange={(e) => handleSettingChange("motionThreshold", parseFloat(e.target.value))}
                      className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neon-amber"
                    />
                    <span className="text-[10px] text-slate-500 font-mono block mt-1">
                      Optical flow magnitude boundary. If scene motion is below this, the violence score is suppressed to block static camera noise.
                    </span>
                  </div>

                  {/* Alert Cooldown */}
                  <div>
                    <div className="flex justify-between items-center mb-1.5">
                      <label className="block text-xs font-mono text-slate-400 uppercase">
                        Alert Dispatch Cooldown (seconds):
                      </label>
                      <span className="text-xs font-mono font-bold text-neon-amber">{alertCooldown}s</span>
                    </div>
                    <input 
                      type="range" 
                      min="5" 
                      max="60" 
                      step="5"
                      value={alertCooldown} 
                      onChange={(e) => handleSettingChange("alertCooldown", parseInt(e.target.value))}
                      className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neon-amber"
                    />
                    <span className="text-[10px] text-slate-500 font-mono block mt-1">
                      Minimum seconds that must elapse before another alert is fired for this location (prevents alert logs flooding).
                    </span>
                  </div>

                  {/* YOLO Confidence */}
                  <div>
                    <div className="flex justify-between items-center mb-1.5">
                      <label className="block text-xs font-mono text-slate-400 uppercase">
                        YOLOv8 Person Conf:
                      </label>
                      <span className="text-xs font-mono font-bold text-neon-amber">{(yoloConfidence * 100).toFixed(0)}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.20" 
                      max="0.80" 
                      step="0.05"
                      value={yoloConfidence} 
                      onChange={(e) => handleSettingChange("yoloConfidence", parseFloat(e.target.value))}
                      className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neon-amber"
                    />
                    <span className="text-[10px] text-slate-500 font-mono block mt-1">
                      Minimum spatial confidence required by YOLOv8 model to crop a person's bounding box region.
                    </span>
                  </div>
                </div>

              </div>

              {/* Configuration verification banner */}
              <div className="mt-8 p-4 rounded bg-[#0b1220]/70 border border-panel-border flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Settings changes are applied instantly and propagate to all active dashboard frames.</span>
                <span className="text-neon-green flex items-center gap-1.5 font-bold uppercase">
                  <CheckCircle2 className="w-3.5 h-3.5" /> SYSTEM SYNC COMPLETED
                </span>
              </div>

            </div>
          )}

        </main>

        {/* Global Footer telemetry bar */}
        <footer className="h-10 border-t border-panel-border bg-[#090d16]/30 flex items-center justify-between px-6 z-20 text-[10px] font-mono text-slate-500">
          <span>BUILT FOR APPLE SILICON MPS ACCELERATION // ARX-V250913210</span>
          <span>© 2026 VI-SAFE CORE AI SYSTEMS. ALL RIGHTS RESERVED.</span>
        </footer>

      </div>
    </div>
  );
}
