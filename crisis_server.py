"""SecureEye Crisis Command Server — Gate 1 AI pre-screen + Gate 2 human confirmation."""
from __future__ import annotations

import base64
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore[assignment]

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

if torch is not None and torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

try:
    import crisis_gate2 as _gate2_mod
    _GATE2_OK = True
except ImportError:
    _GATE2_OK = False


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos"
MODEL_DIR = ROOT / "models"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_FIRE_MODEL = MODEL_DIR / "best.pt"
DEFAULT_FALL_MODEL = MODEL_DIR / "yolov8n.pt"
STATUS_URL = "/status"


def _env_path(name: str, fallback: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else fallback


def _to_b64(frame) -> Optional[str]:
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _severity_from_conf(conf: float) -> str:
    if conf >= 0.82:
        return "CRITICAL"
    if conf >= 0.65:
        return "HIGH"
    if conf >= 0.45:
        return "MEDIUM"
    return "LOW"


def _profile_for_video(path: Path) -> str:
    name = path.name.lower()
    if "fire" in name:
        return "fire"
    if "fall" in name:
        return "fall"
    return "mixed"


@dataclass
class DetectionEvent:
    kind: str
    confidence: float
    label: str
    bbox: tuple[int, int, int, int]


class CrisisEngine:
    def __init__(self) -> None:
        self.fire_model_path = _env_path("CRISIS_FIRE_MODEL_PATH", DEFAULT_FIRE_MODEL)
        self.fall_model_path = _env_path("CRISIS_FALL_MODEL_PATH", DEFAULT_FALL_MODEL)
        self.video_paths = self._discover_videos()
        self.cuda_enabled = bool(torch is not None and torch.cuda.is_available())
        self.device = 0 if self.cuda_enabled else "cpu"
        self.state_lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.state = self._initial_state()
        self.logs: deque[dict] = deque(maxlen=60)
        self.alerts: deque[dict] = deque(maxlen=1)
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.active_video_index = 0
        self.frame_index = 0
        self.last_status_signature: Optional[tuple[str, str]] = None
        self.last_fps_ts = time.perf_counter()
        self.fps_ema = 0.0
        self.fire_model = self._load_model(self.fire_model_path)
        self.fall_model = self._load_model(self.fall_model_path)
        self.missing_fire_warned = False
        self.missing_fall_warned = False
        self.no_video_warned = False
        self._video_ended = False

        # Fall streak detector (triggers Gate 1 fall monitoring)
        self.fall_detection_streak = 0
        self.fall_conf_history: deque[float] = deque(maxlen=8)
        self.fall_alert_emitted = False
        self.fall_confirm_frames = 3
        self.fall_confirm_min_frame = 8
        self.fall_confirm_confidence = 0.18

        self.active_incident_decision: Optional[str] = None
        self.active_incident_reason: Optional[str] = None

        # Live JPEG frame served via /frame
        self.latest_jpeg: Optional[bytes] = None
        self.jpeg_lock = threading.Lock()
        self._last_events: list[DetectionEvent] = []

        # ── Fire Gate 1: temporal region-growth + flicker analysis ──────────────
        self.fire_gate1_streak = 0          # consecutive frames with fire conf >= 0.25
        self.fire_gate1_areas: deque[float] = deque(maxlen=60)
        self.fire_gate1_brightness: deque[float] = deque(maxlen=30)
        self.fire_gate1_passed = False       # True once Gate 1 fires (single-shot)
        self.fire_gate1_prelim_alerted = False  # True once preliminary alert is sent

        # ── Fall Gate 1: 9-second recovery monitoring window ────────────────────
        self.fall_gate1_monitoring = False
        self.fall_gate1_start_ts: float = 0.0
        self.fall_gate1_window: float = 3.0  # seconds to watch for recovery
        self.fall_gate1_suppressed = False   # True if person got back up (false positive)
        self.fall_gate1_cx: float = 320.0   # horizontal center of fall bbox
        self.fall_gate1_pre_jpegs: list[bytes] = []
        self.fall_gate1_peak_conf: float = 0.0

        # ── Gate 2: rolling pre-event buffer + post-event clip + email ───────────
        self.frame_buffer: deque[bytes] = deque(maxlen=90)   # ~3 s at 30 fps of JPEG frames
        self.gate2_pending = False
        self.gate2_event_id: Optional[str] = None
        self.gate2_event_type: Optional[str] = None          # "fire" or "fall"
        self.gate2_event_conf: float = 0.0
        self.gate2_pre_jpegs: list[bytes] = []
        self.gate2_post_collecting = False
        self.gate2_post_start_ts: float = 0.0
        self.gate2_post_jpegs: list[bytes] = []
        self.gate2_clip_duration: float = 5.0                # seconds of post-event to capture
        self.gate2_response: Optional[str] = None            # set by /gate2/response endpoint

    # ── asset helpers ─────────────────────────────────────────────────────────

    def _discover_videos(self) -> list[Path]:
        if not VIDEO_DIR.exists():
            return []
        files = [p for p in VIDEO_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        return sorted(files, key=lambda p: p.name.lower())

    def refresh_assets(self) -> None:
        self.video_paths = self._discover_videos()
        if self.fire_model is None and self.fire_model_path.exists():
            self.fire_model = self._load_model(self.fire_model_path)
        if self.fall_model is None and self.fall_model_path.exists():
            self.fall_model = self._load_model(self.fall_model_path)

    def _load_model(self, path: Path):
        if YOLO is None or not path.exists():
            return None
        try:
            model = YOLO(str(path))
            if self.cuda_enabled:
                try:
                    model.to("cuda")
                except Exception:
                    pass
            return model
        except Exception as exc:
            self._append_log("WARN", f"Model load failed for {path.name}: {exc}")
            return None

    # ── state helpers ─────────────────────────────────────────────────────────

    def _initial_state(self) -> dict[str, Any]:
        return {
            "frame": None,
            "status": "SAFE",
            "alerts": [],
            "videos": [],
            "activeVideo": None,
            "logs": [],
            "metrics": {
                "confidence": 0.0,
                "fps": 0.0,
                "latencyMs": 0.0,
                "modelsActive": 0,
                "uptime": "00d 00:00:00",
            },
            "location": "Unknown",
            "currentVideo": "None",
            "lifecycleState": "MONITORING",
            "confidenceExplanation": [],
            "decisionReason": None,
            "signals": [],
            "llmSummary": None,
            "llmLink": "https://github.com/Mohiee661/ai-crisis-response",
            "llmConfirmation": None,
            "systemHealth": {
                "model_status": "MISSING",
                "camera_status": "OFFLINE",
                "api_status": "ONLINE",
                "latency": "0ms",
            },
            "decision": None,
            "incidentLocked": False,
        }

    def _append_log(self, level: str, message: str) -> None:
        entry = {
            "id": f"log-{int(time.time() * 1000)}-{len(self.logs)}",
            "timestamp": _now_iso(),
            "level": level,
            "message": message,
        }
        self.logs.appendleft(entry)

    def _append_alert(self, kind: str, severity: str, camera_id: str, message: str) -> None:
        entry = {
            "id": f"alert-{int(time.time() * 1000)}-{len(self.alerts)}",
            "type": kind,
            "severity": severity,
            "cameraId": camera_id,
            "timestamp": _now_iso(),
            "message": message,
        }
        self.alerts.clear()
        self.alerts.appendleft(entry)

    def _uptime(self) -> str:
        elapsed = int(time.time() - START_TS)
        days, rem = divmod(elapsed, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{days:02d}d {hours:02d}:{minutes:02d}:{seconds:02d}"

    # ── reset helpers ─────────────────────────────────────────────────────────

    def _reset_fall_tracker(self) -> None:
        self.fall_detection_streak = 0
        self.fall_conf_history.clear()
        self.fall_alert_emitted = False

    def _reset_incident_state(self) -> None:
        self.active_incident_decision = None
        self.active_incident_reason = None

    def _reset_fire_gate1(self) -> None:
        self.fire_gate1_streak = 0
        self.fire_gate1_areas.clear()
        self.fire_gate1_brightness.clear()
        self.fire_gate1_passed = False
        self.fire_gate1_prelim_alerted = False

    def _reset_fall_gate1(self) -> None:
        self.fall_gate1_monitoring = False
        self.fall_gate1_start_ts = 0.0
        self.fall_gate1_suppressed = False
        self.fall_gate1_pre_jpegs = []
        self.fall_gate1_peak_conf = 0.0

    def _reset_gate2(self) -> None:
        self.gate2_pending = False
        self.gate2_event_id = None
        self.gate2_event_type = None
        self.gate2_event_conf = 0.0
        self.gate2_pre_jpegs = []
        self.gate2_post_collecting = False
        self.gate2_post_jpegs = []
        self.gate2_response = None

    # ── Gate 1: fire temporal analysis ───────────────────────────────────────

    def _gate1_fire_update(self, events: list[DetectionEvent], frame) -> bool:
        """Track fire detection history. Returns True exactly once when Gate 1 passes."""
        if self.fire_gate1_passed:
            return False

        fire_events = [e for e in events if e.kind == "fire" and e.confidence >= 0.25]
        if not fire_events:
            self.fire_gate1_streak = max(0, self.fire_gate1_streak - 2)
            return False

        best = max(fire_events, key=lambda e: e.confidence)
        x1, y1, x2, y2 = best.bbox
        area = float(max(1, (x2 - x1) * (y2 - y1)))
        self.fire_gate1_areas.append(area)
        self.fire_gate1_streak += 1

        # Brightness inside fire bbox — real fire flickers (high temporal variance)
        if _NUMPY and frame is not None:
            roi = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
            if roi.size > 0:
                self.fire_gate1_brightness.append(float(roi.mean()))

        # Always pass after 8 consecutive fire frames (~1 s) — sustained = real
        if self.fire_gate1_streak >= 8:
            self.fire_gate1_passed = True
            return True

        # Pass earlier when area is stable/growing AND brightness is flickering
        if self.fire_gate1_streak >= 5:
            areas = list(self.fire_gate1_areas)
            if len(areas) >= 10:
                early_avg = sum(areas[:5]) / 5
                late_avg  = sum(areas[-5:]) / 5
                growing = late_avg >= early_avg * 0.75
            else:
                growing = True

            if _NUMPY and len(self.fire_gate1_brightness) >= 10:
                bvals = np.array(list(self.fire_gate1_brightness)[-10:])
                flickering = float(np.var(bvals)) > 25.0
            else:
                flickering = True

            if growing and flickering:
                self.fire_gate1_passed = True
                return True

        return False

    # ── Gate 1: fall recovery check ───────────────────────────────────────────

    def _gate1_fall_check_recovery(self, events: list[DetectionEvent]) -> bool:
        """During the fall monitoring window, check if the person stood back up.
        Only 'person' events qualify — a lying bbox (kind='fall') is NOT recovery."""
        for event in events:
            if event.kind != "person":
                continue
            x1, y1, x2, y2 = event.bbox
            width      = max(1, x2 - x1)
            height     = max(1, y2 - y1)
            aspect_h_w = height / width   # > 1.2 = standing upright
            cx = (x1 + x2) / 2.0
            if abs(cx - self.fall_gate1_cx) < 250 and aspect_h_w >= 1.2:
                return True
        return False

    # ── Gate 2: clip build + email ────────────────────────────────────────────

    def _build_clip(self, pre_jpegs: list[bytes], post_jpegs: list[bytes], fps: float = 15.0) -> Optional[bytes]:
        import tempfile
        all_jpegs = pre_jpegs + post_jpegs
        if not all_jpegs or not _NUMPY:
            return None

        w, h = 640, 480
        for j in all_jpegs[:5]:
            arr   = np.frombuffer(j, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                h, w = frame.shape[:2]
                break

        tmp_path = Path(tempfile.mktemp(suffix=".mp4"))
        try:
            out = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for j in all_jpegs:
                arr   = np.frombuffer(j, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                    out.write(frame)
            out.release()
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception as exc:
            self._append_log("ERROR", f"Clip build failed: {exc}")
            return None
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _gate2_send_bg(self, pre_jpegs: list[bytes], post_jpegs: list[bytes],
                       event_id: str, event_type: str, confidence: float) -> None:
        """Build 7-second clip and send Gate 2 email (runs in background thread)."""
        try:
            clip = self._build_clip(pre_jpegs, post_jpegs)
            sev  = _severity_from_conf(confidence)
            if not _GATE2_OK:
                self._append_log("ERROR", "Gate 2: crisis_gate2 module not importable")
                return
            if event_type == "fire":
                ok, detail = _gate2_mod.send_fire_review(event_id, "CAM-01", confidence, sev, clip)
            else:
                ok, detail = _gate2_mod.send_fall_review(event_id, "CAM-01", confidence, clip)

            if ok:
                self._append_log("INFO", f"Gate 2 email SENT [{event_id}] — awaiting authority response")
            else:
                self._append_log("ERROR", f"Gate 2 email FAILED [{event_id}]: {detail}")
        except Exception as exc:
            self._append_log("ERROR", f"Gate 2 send crashed: {exc}")

    def _trigger_gate2(self, event_type: str, confidence: float, pre_jpegs: list[bytes]) -> str:
        """Activate Gate 2 pending state and begin post-event frame collection."""
        event_id = f"g2-{int(time.time()*1000) & 0xFFFF:04x}"
        self.gate2_event_id       = event_id
        self.gate2_event_type     = event_type
        self.gate2_event_conf     = confidence
        self.gate2_pre_jpegs      = list(pre_jpegs)
        self.gate2_post_jpegs     = []
        self.gate2_post_collecting = True
        self.gate2_post_start_ts  = time.time()
        self.gate2_pending        = True
        return event_id

    # ── YOLO inference ────────────────────────────────────────────────────────

    def _model_detection(self, model, frame, kind: str) -> list[DetectionEvent]:
        if model is None:
            return []
        try:
            result = model.predict(frame, conf=0.15, verbose=False,
                                   device=self.device, half=self.cuda_enabled)[0]
        except Exception as exc:
            self._append_log("ERROR", f"{kind} inference failed: {exc}")
            return []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = result.names if hasattr(result, "names") else {}
        kind_keywords = {
            "fire": ("fire", "flame", "smoke", "burn"),
            "fall": ("fall", "fallen", "down", "collapse", "lying"),
        }[kind]

        events: list[DetectionEvent] = []
        for box in boxes:
            cls_index  = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            label      = str(names.get(cls_index, kind)).lower()
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            if any(kw in label for kw in kind_keywords) or confidence >= 0.25:
                xyxy = box.xyxy[0].tolist()
                bbox = tuple(int(v) for v in xyxy)
                events.append(DetectionEvent(kind=kind, confidence=confidence, label=label, bbox=bbox))

        return events

    def _fall_person_events(self, model, frame) -> list[DetectionEvent]:
        if model is None:
            return []
        try:
            result = model.predict(frame, conf=0.18, verbose=False,
                                   device=self.device, half=self.cuda_enabled)[0]
        except Exception as exc:
            self._append_log("ERROR", f"fall inference failed: {exc}")
            return []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = result.names if hasattr(result, "names") else {}
        frame_height, frame_width = frame.shape[:2]
        events: list[DetectionEvent] = []

        for box in boxes:
            cls_index  = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            label      = str(names.get(cls_index, "person")).lower()
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            xyxy = box.xyxy[0].tolist()
            bbox = tuple(int(v) for v in xyxy)

            if any(kw in label for kw in ("fall", "fallen", "lying", "collapse")) and confidence >= 0.18:
                events.append(DetectionEvent(kind="fall", confidence=confidence, label=label, bbox=bbox))
                continue

            if label == "person" and confidence >= 0.18:
                x1, y1, x2, y2 = bbox
                width      = max(1, x2 - x1)
                height     = max(1, y2 - y1)
                area_ratio = (width * height) / max(1, frame_width * frame_height)
                aspect_h_w = height / width  # < 0.85 → lying down; > 1.2 → standing

                if area_ratio >= 0.008 and aspect_h_w < 0.85:
                    # Horizontal bbox = person lying down → fall candidate
                    events.append(DetectionEvent(kind="fall", confidence=confidence, label="fall_candidate", bbox=bbox))
                else:
                    events.append(DetectionEvent(kind="person", confidence=confidence, label="person", bbox=bbox))

        return events

    # ── annotation ────────────────────────────────────────────────────────────

    def _annotate(self, frame, events: list[DetectionEvent]):
        annotated = frame.copy()
        for event in events:
            x1, y1, x2, y2 = event.bbox
            if event.kind == "fire":
                color = (0, 0, 255)
            elif event.kind == "fall":
                color = (255, 180, 0)
            else:
                color = (90, 170, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_name = "PERSON" if event.kind == "person" else event.kind.upper()
            cv2.putText(annotated, f"{label_name} {event.confidence:.2f}",
                        (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return annotated

    # ── fall streak tracker ───────────────────────────────────────────────────

    def _update_fall_tracker(self, fall_conf: float, video: Path) -> bool:
        is_fall_video = _profile_for_video(video) == "fall"
        if is_fall_video and fall_conf >= self.fall_confirm_confidence:
            self.fall_detection_streak += 1
        else:
            self.fall_detection_streak = max(0, self.fall_detection_streak - 1)

        self.fall_conf_history.append(fall_conf if is_fall_video else 0.0)

        ready = (
            not self.fall_alert_emitted
            and is_fall_video
            and self.frame_index >= self.fall_confirm_min_frame
            and self.fall_detection_streak >= self.fall_confirm_frames
            and fall_conf >= self.fall_confirm_confidence
        )
        if ready:
            self.fall_alert_emitted = True
        return ready

    # ── video helpers ─────────────────────────────────────────────────────────

    def _current_location(self, video: Path) -> str:
        name = video.name.lower()
        if "fire1" in name:
            return "Lobby"
        if "fire2" in name:
            return "Warehouse"
        if "fall" in name:
            return "Stairwell"
        return "Unknown"

    def _open_current_video(self) -> bool:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if not self.video_paths:
            return False
        video = self.video_paths[self.active_video_index % len(self.video_paths)]
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            self._append_log("ERROR", f"Failed to open video: {video.name}")
            return False
        self.capture = cap
        self._video_ended = False
        self._append_log("INFO", f"Loaded video: {video.name}")
        return True


    # ── main state-update ─────────────────────────────────────────────────────

    def _update_state(self, frame, events: list[DetectionEvent], video: Path, process_ms: float) -> None:
        fire_conf = max((e.confidence for e in events if e.kind == "fire"), default=0.0)
        fall_conf = max((e.confidence for e in events if e.kind == "fall"), default=0.0)
        top_conf  = max(fire_conf, fall_conf)
        profile   = _profile_for_video(video)

        decision     = "IGNORE"
        reason       = "No high-confidence crisis signal detected"
        status       = "SAFE"
        lifecycle    = "MONITORING"
        alert_locked = False

        # ── Handle incoming Gate 2 response from authority ──────────────────
        resp = self.gate2_response
        if resp is not None:
            self.gate2_response = None
            self.gate2_pending  = False
            if resp == "dispatch_ambulance":
                decision = "ALERT_AMBULANCE"
                reason   = "Gate 2 confirmed: ambulance dispatched by authority"
                self.active_incident_decision = decision
                self.active_incident_reason   = reason
                self._append_alert("FALL", _severity_from_conf(self.gate2_event_conf), "CAM-01",
                    f"Authority confirmed fall — ambulance dispatched")
                self._append_log("INFO", "Gate 2 CONFIRMED → DISPATCH AMBULANCE")
            elif resp == "dispatch_fire":
                decision = "ALERT_FIRE_ENGINE"
                reason   = "Gate 2 confirmed: fire engine dispatched by authority"
                self.active_incident_decision = decision
                self.active_incident_reason   = reason
                self._append_alert("FIRE", _severity_from_conf(self.gate2_event_conf), "CAM-01",
                    f"Authority confirmed fire — fire engine dispatched")
                self._append_log("INFO", "Gate 2 CONFIRMED → DISPATCH FIRE ENGINE")
            elif resp == "onsite_team":
                decision = "ONSITE_TEAM"
                reason   = "Gate 2: on-site team dispatched by authority"
                self.active_incident_decision = decision
                self.active_incident_reason   = reason
                self._append_log("INFO", "Gate 2 → ON-SITE TEAM DISPATCHED")
            elif resp == "false_alarm":
                decision = "IGNORE"
                reason   = "Gate 2: authority confirmed false alarm"
                self._reset_fire_gate1()
                self._reset_fall_gate1()
                self._reset_fall_tracker()
                self.active_incident_decision = None
                self.active_incident_reason   = None
                self._append_log("INFO", "Gate 2 → FALSE ALARM — alert cleared")

        # ── Active incident locked by Gate 2 response or operator override ───
        if self.active_incident_decision is not None and resp is None:
            decision     = self.active_incident_decision
            reason       = self.active_incident_reason or reason
            alert_locked = True
            status       = "ALERT"

        # ── Gate 2 email sent, waiting for authority response ────────────────
        elif self.gate2_pending:
            decision     = "GATE2_PENDING"
            lifecycle    = "GATE2_PENDING"
            reason       = f"Awaiting authority response [{self.gate2_event_id}]"
            status       = "ALERT"
            alert_locked = True

        # ── Normal detection + Gate 1 analysis ──────────────────────────────
        else:
            # --- Fire Gate 1 ---
            if profile in ("fire", "mixed"):
                fire_gate1_just_passed = self._gate1_fire_update(events, frame)
                if fire_gate1_just_passed:
                    eid = self._trigger_gate2("fire", fire_conf, list(self.frame_buffer))
                    self._append_alert("FIRE", _severity_from_conf(fire_conf), "CAM-01",
                        f"Gate 1 passed — fire in {video.name} at {fire_conf:.2f}")
                    self._append_log("INFO", f"Fire Gate 1 passed → Gate 2 triggered [{eid}]")
                    decision  = "GATE2_PENDING"
                    lifecycle = "GATE2_PENDING"
                    reason    = f"Fire Gate 1 passed — email sent to authority [{eid}]"
                    status    = "ALERT"
                    alert_locked = True
                elif self.fire_gate1_streak >= 5:
                    status  = "MONITOR"
                    reason  = f"Fire Gate 1 in progress (streak={self.fire_gate1_streak}/8)"
                    # Show a preliminary alert card the first time we hit the threshold
                    if not self.fire_gate1_prelim_alerted:
                        self.fire_gate1_prelim_alerted = True
                        self._append_alert("FIRE", "LOW", "CAM-01",
                            f"Fire detected in {video.name} — Gate 1 AI analysis in progress")
            else:
                self.fire_gate1_streak = max(0, self.fire_gate1_streak - 1)

            # --- Fall Gate 1 monitoring window ---
            if self.fall_gate1_monitoring:
                elapsed   = time.time() - self.fall_gate1_start_ts
                recovered = self._gate1_fall_check_recovery(events)

                if recovered:
                    self._append_log("INFO",
                        "Gate 1 (fall): person recovered — false positive suppressed")
                    self.fall_gate1_monitoring = False
                    self.fall_gate1_suppressed = True
                    self._reset_fall_tracker()
                    decision  = "IGNORE"
                    reason    = "Gate 1 (fall): recovery detected, alert suppressed"
                    status    = "MONITOR" if status == "SAFE" else status
                    lifecycle = "MONITORING"

                elif elapsed >= self.fall_gate1_window:
                    # 9-second window expired, no recovery → Gate 2
                    self.fall_gate1_monitoring = False
                    if not self.gate2_pending:
                        eid = self._trigger_gate2("fall", self.fall_gate1_peak_conf,
                                                  self.fall_gate1_pre_jpegs)
                        self._append_alert("FALL", _severity_from_conf(self.fall_gate1_peak_conf),
                            "CAM-01", f"Gate 1 passed — fall in {video.name}, person did not recover")
                        self._append_log("INFO",
                            f"Fall Gate 1 passed (9s, no recovery) → Gate 2 triggered [{eid}]")
                    decision     = "GATE2_PENDING"
                    lifecycle    = "GATE2_PENDING"
                    reason       = f"Fall Gate 1 passed — email sent to authority [{self.gate2_event_id}]"
                    status       = "ALERT"
                    alert_locked = True

                else:
                    remaining = self.fall_gate1_window - elapsed
                    decision  = "GATE1_MONITORING"
                    lifecycle = "GATE1_MONITORING"
                    reason    = f"Gate 1 (fall): watching for recovery ({remaining:.0f}s remaining)"
                    status    = "MONITOR"

            elif not self.fall_gate1_suppressed:
                fall_ready = self._update_fall_tracker(fall_conf, video)
                if fall_ready:
                    # Enter Gate 1 fall monitoring window
                    best_fall = max(
                        (e for e in events if e.kind == "fall"),
                        key=lambda e: e.confidence, default=None
                    )
                    self.fall_gate1_cx = (
                        (best_fall.bbox[0] + best_fall.bbox[2]) / 2.0
                        if best_fall else 320.0
                    )
                    self.fall_gate1_monitoring  = True
                    self.fall_gate1_start_ts    = time.time()
                    self.fall_gate1_pre_jpegs   = list(self.frame_buffer)
                    self.fall_gate1_peak_conf   = fall_conf
                    self._append_log("INFO",
                        f"Fall Gate 1 monitoring started ({fall_conf:.2f} conf, 3s window)")
                    self._append_alert("FALL", "MEDIUM", "CAM-01",
                        f"Fall detected in {video.name} — monitoring for recovery (3s window)")
                    decision  = "GATE1_MONITORING"
                    lifecycle = "GATE1_MONITORING"
                    reason    = "Gate 1 (fall): monitoring for recovery (3s window)"
                    status    = "MONITOR"
                elif fall_conf >= 0.2 or fire_conf >= 0.2:
                    status = "MONITOR" if status == "SAFE" else status

        # ── Annotate frame and publish JPEG ──────────────────────────────────
        annotated = self._annotate(frame, events)
        ok2, buf  = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok2:
            jpeg = buf.tobytes()
            with self.jpeg_lock:
                self.latest_jpeg = jpeg
            self.frame_buffer.append(jpeg)

        # ── FPS EMA ──────────────────────────────────────────────────────────
        now   = time.perf_counter()
        delta = now - self.last_fps_ts
        fps_now = (1.0 / delta) if delta > 0 else 0.0
        self.fps_ema = fps_now if self.fps_ema == 0.0 else (self.fps_ema * 0.8 + fps_now * 0.2)
        self.last_fps_ts = now

        # ── Build state dict and publish ─────────────────────────────────────
        system_health = {
            "model_status":  "ACTIVE" if self.fire_model or self.fall_model else "MISSING",
            "camera_status": "CONNECTED" if self.capture and self.capture.isOpened() else "OFFLINE",
            "api_status":    "ONLINE",
            "latency":       f"{process_ms:.0f}ms",
        }
        confidence_explanation = [
            "Video sample: " + video.name,
            "Model profile: " + profile,
            "High-confidence detections: " + (
                ", ".join(f"{e.kind}:{e.confidence:.2f}" for e in events) if events else "none"
            ),
        ]
        if self.fall_gate1_monitoring:
            elapsed = time.time() - self.fall_gate1_start_ts
            confidence_explanation.append(
                f"Gate 1 fall window: {elapsed:.1f}s / {self.fall_gate1_window:.0f}s elapsed"
            )
        if self.gate2_pending:
            confidence_explanation.append(f"Gate 2 pending — event: {self.gate2_event_id}")

        signals     = [f"{e.kind.upper()} @{e.confidence:.2f}" for e in events]
        llm_summary = (
            f"Gate 2 email sent for {self.gate2_event_type} event. Awaiting authority response."
            if self.gate2_pending else
            f"Fire Gate 1 in progress — streak {self.fire_gate1_streak} frames."
            if self.fire_gate1_streak >= 5 and profile in ("fire", "mixed") else
            f"Gate 1 fall monitoring — person has {(self.fall_gate1_window - (time.time() - self.fall_gate1_start_ts)):.0f}s to recover."
            if self.fall_gate1_monitoring else
            f"Crisis confirmed — {self.active_incident_decision}."
            if self.active_incident_decision else
            "No active threat confirmed by the reasoning layer."
        )

        final_decision = decision
        if self.active_incident_decision is not None and resp is None:
            final_decision = self.active_incident_decision
            alert_locked   = True
            status         = "ALERT"

        with self.state_lock:
            self.state.update({
                "status":       "ALERT" if alert_locked else status,
                "alerts":       list(self.alerts),
                "videos":       [p.name for p in self.video_paths],
                "activeVideo":  video.name,
                "logs":         list(self.logs),
                "metrics": {
                    "confidence": round(top_conf * 100, 2),
                    "fps":        round(self.fps_ema if self.fps_ema > 0 else fps_now, 2),
                    "latencyMs":  round(process_ms, 2),
                    "modelsActive": int(self.fire_model is not None) + int(self.fall_model is not None),
                    "uptime":     self._uptime(),
                },
                "location":             self._current_location(video),
                "currentVideo":         video.name,
                "lifecycleState":       lifecycle,
                "confidenceExplanation": confidence_explanation,
                "decisionReason":       reason,
                "signals":              signals,
                "llmSummary":           llm_summary,
                "llmConfirmation":      "REVIEW REQUIRED" if alert_locked else "NO CORRELATION",
                "systemHealth":         system_health,
                "decision":             final_decision,
                "incidentLocked":       alert_locked,
            })

        sig = (video.name, final_decision or "IGNORE")
        if sig != self.last_status_signature:
            self._append_log("AGENT", f"{final_decision or 'IGNORE'} on {video.name} ({top_conf:.2f})")
            self.last_status_signature = sig

    # ── main processing loop ──────────────────────────────────────────────────

    def run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.refresh_assets()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _loop(self) -> None:
        if not self.video_paths:
            self._append_log("WARN", "No videos found in videos/")
            while not self.stop_event.is_set():
                with self.state_lock:
                    self.state["logs"] = list(self.logs)
                    self.state["systemHealth"]["camera_status"] = "MISSING"
                    self.state["status"] = "SAFE"
                    self.state["activeVideo"] = None
                    self.state["currentVideo"] = "None"
                time.sleep(1.0)
            return

        if not self._open_current_video():
            while not self.stop_event.is_set():
                time.sleep(1.0)
            return

        self._reset_fall_tracker()
        self._reset_incident_state()

        while not self.stop_event.is_set():
            with self.capture_lock:
                if self.capture is None:
                    if self._video_ended:
                        # Video finished — hold last frame, wait for operator to select a new one
                        time.sleep(0.5)
                        continue
                    if not self._open_current_video():
                        time.sleep(1.0)
                        continue
                cap = self.capture
                if cap is None:
                    time.sleep(0.05)
                    continue

                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    # End of video — stop here, hold last frame
                    self._append_log("INFO", f"Video ended: {video.name}")
                    self.capture.release()
                    self.capture = None
                    self._video_ended = True
                    continue

                self.frame_index += 1
                video   = self.video_paths[self.active_video_index % len(self.video_paths)]
                profile = _profile_for_video(video)

                if self.frame_index % 4 == 0:
                    # Full YOLO inference frame
                    events: list[DetectionEvent] = []
                    if profile in ("fire", "mixed") and self.fire_model is not None:
                        events.extend(self._model_detection(self.fire_model, frame, "fire"))
                    if profile in ("fall", "mixed") and self.fall_model is not None:
                        events.extend(self._fall_person_events(self.fall_model, frame))

                    if not events and profile == "fire" and self.fire_model is None and not self.missing_fire_warned:
                        self._append_log("WARN", "Fire model missing — cannot detect fire")
                        self.missing_fire_warned = True
                    if not events and profile == "fall" and self.fall_model is None and not self.missing_fall_warned:
                        self._append_log("WARN", "Fall model missing — cannot detect falls")
                        self.missing_fall_warned = True

                    self._last_events = events
                    process_ms = (time.perf_counter() - start) * 1000.0
                    self._update_state(frame, events, video, process_ms)

                else:
                    # Non-inference frame: annotate with last known events, push JPEG
                    annotated = self._annotate(frame, self._last_events)
                    ok2, buf  = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok2:
                        jpeg = buf.tobytes()
                        with self.jpeg_lock:
                            self.latest_jpeg = jpeg
                        self.frame_buffer.append(jpeg)

                # Collect post-event frames for Gate 2 clip
                if self.gate2_post_collecting:
                    with self.jpeg_lock:
                        if self.latest_jpeg:
                            self.gate2_post_jpegs.append(self.latest_jpeg)
                    if time.time() - self.gate2_post_start_ts >= self.gate2_clip_duration:
                        self.gate2_post_collecting = False
                        # Send email in background so we don't stall the loop
                        threading.Thread(
                            target=self._gate2_send_bg,
                            args=(
                                list(self.gate2_pre_jpegs),
                                list(self.gate2_post_jpegs),
                                self.gate2_event_id or "unknown",
                                self.gate2_event_type or "unknown",
                                self.gate2_event_conf,
                            ),
                            daemon=True,
                        ).start()

            fps   = self.capture.get(cv2.CAP_PROP_FPS) if self.capture is not None else 0
            delay = 1.0 / fps if fps and fps > 0 else 0.033
            time.sleep(max(0.01, min(delay, 0.05)))

    # ── public API ────────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        with self.state_lock:
            result = dict(self.state)
            result["videos"] = [p.name for p in self.video_paths]
            return result

    def select_video(self, name: str) -> dict[str, Any]:
        with self.capture_lock:
            self.refresh_assets()
            match_index = next(
                (idx for idx, p in enumerate(self.video_paths) if p.name == name), None
            )
            if match_index is None:
                raise ValueError(f"Unknown video: {name}")

            self.active_video_index = match_index
            self.frame_index = 0
            self.last_status_signature = None
            if not self._open_current_video():
                raise RuntimeError(f"Could not open video: {name}")

            self._reset_fall_tracker()
            self._reset_incident_state()
            self._reset_fire_gate1()
            self._reset_fall_gate1()
            self._reset_gate2()
            self._last_events = []

            with self.state_lock:
                self.alerts.clear()
                self.state["activeVideo"]    = name
                self.state["currentVideo"]   = name
                self.state["decisionReason"] = f"Operator selected video: {name}"
                self.state["lifecycleState"] = "MONITORING"
                self.state["incidentLocked"] = False
                self.state["alerts"]         = []
                self.state["logs"]           = list(self.logs)

            self._append_log("INFO", f"Selected video: {name}")
            return self.get_state()

    def override(self, action: str) -> dict[str, Any]:
        with self.state_lock:
            if action == "IGNORE":
                self._reset_fire_gate1()
                self._reset_fall_gate1()
                self._reset_fall_tracker()
                self._reset_gate2()
                self.active_incident_decision = None
                self.active_incident_reason   = None
                self.state["decision"]        = "IGNORE"
                self.state["lifecycleState"]  = "MONITORING"
                self.state["incidentLocked"]  = False
                self.state["decisionReason"]  = "Manual override by operator: IGNORE"
                self._append_log("INFO", "Operator cleared all alerts")

            elif action in ("ALERT_AMBULANCE", "ALERT_FIRE_ENGINE"):
                # Manual Gate 2 trigger — send review email to authority now
                event_type = "fall" if action == "ALERT_AMBULANCE" else "fire"
                raw_conf   = self.state.get("metrics", {}).get("confidence", 50)
                conf       = float(raw_conf) / 100.0

                # Reuse existing Gate 2 event if already pending, otherwise open a new one
                if not self.gate2_pending:
                    eid = self._trigger_gate2(event_type, conf, list(self.frame_buffer))
                    self.gate2_post_collecting = False  # send immediately, no post-event wait
                    pre  = list(self.gate2_pre_jpegs)
                    threading.Thread(
                        target=self._gate2_send_bg,
                        args=(pre, [], eid, event_type, conf),
                        daemon=True,
                    ).start()
                    alert_kind = "FALL" if event_type == "fall" else "FIRE"
                    self._append_alert(alert_kind, _severity_from_conf(conf), "CAM-01",
                        f"Manual Gate 2 review triggered by operator — {alert_kind} event")
                    self._append_log("INFO", f"Manual Gate 2 email sent [{eid}] — {event_type}")
                else:
                    self._append_log("INFO", f"Gate 2 already pending [{self.gate2_event_id}] — email already sent")

                self.state["decision"]        = "GATE2_PENDING"
                self.state["lifecycleState"]  = "GATE2_PENDING"
                self.state["incidentLocked"]  = True
                self.state["decisionReason"]  = f"Manual review triggered by operator — {action}"

            else:
                self.active_incident_decision = action
                self.active_incident_reason   = f"Manual override by operator: {action}"
                self.state["decision"]        = action
                self.state["lifecycleState"]  = "DISPATCHED"
                self.state["incidentLocked"]  = True
                self.state["decisionReason"]  = f"Manual override by operator: {action}"
                self._append_log("INFO", f"Operator override: {action}")
            self.state["logs"] = list(self.logs)
            return dict(self.state)

    def reset(self) -> dict[str, Any]:
        with self.state_lock:
            self.state = self._initial_state()
            self.alerts.clear()
            self.logs.clear()
            self.last_status_signature = None
            self._reset_fall_tracker()
            self._reset_incident_state()
            self._reset_fire_gate1()
            self._reset_fall_gate1()
            self._reset_gate2()
            self.missing_fire_warned = False
            self.missing_fall_warned = False
            self.no_video_warned     = False
            self._append_log("INFO", "Crisis state reset")
            self.state["logs"] = list(self.logs)
            return dict(self.state)

    def apply_gate2_response(self, event_id: str, action: str) -> bool:
        """Called by the /gate2/response endpoint. Returns True if event matched."""
        if self.gate2_event_id == event_id and self.gate2_pending:
            self.gate2_response = action
            return True
        return False


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Crisis Command Center")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TS = time.time()
engine   = CrisisEngine()


@app.on_event("startup")
def on_startup() -> None:
    VIDEO_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    engine.run()


@app.on_event("shutdown")
def on_shutdown() -> None:
    engine.stop()


@app.get("/frame")
def get_frame():
    with engine.jpeg_lock:
        data = engine.latest_jpeg
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get(STATUS_URL)
def get_status() -> dict[str, Any]:
    return engine.get_state()


@app.get("/videos")
def list_videos() -> dict[str, Any]:
    return {
        "videos": [p.name for p in engine.video_paths],
        "active": (
            engine.video_paths[engine.active_video_index % len(engine.video_paths)].name
            if engine.video_paths else None
        ),
    }


@app.post("/videos/select")
def select_video(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from fastapi import HTTPException
    name = str(payload.get("video", ""))
    try:
        return engine.select_video(name)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/override")
def override(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action = str(payload.get("action", "IGNORE"))
    return engine.override(action)


@app.post("/reset")
def reset() -> dict[str, Any]:
    return engine.reset()


@app.get("/gate2/response")
def gate2_response(id: str, action: str):
    """Gate 2 action-button endpoint — called when authority clicks a button in the email."""
    matched = engine.apply_gate2_response(id, action)
    action_label = {
        "dispatch_ambulance": "Dispatch Ambulance",
        "dispatch_fire":      "Dispatch Fire Engine",
        "onsite_team":        "On-site Team Dispatched",
        "false_alarm":        "Marked as False Alarm",
    }.get(action, action.replace("_", " ").title())

    if matched:
        body = f"""
<html><body style="font-family:sans-serif;background:#0f1724;color:#e8eef9;text-align:center;padding:80px 24px;max-width:480px;margin:0 auto">
  <div style="font-size:48px;margin-bottom:16px">✓</div>
  <h2 style="color:#3BF0A4;margin:0 0 8px">Response Recorded</h2>
  <p style="color:#94a3b8;margin:0 0 4px">Action: <strong style="color:#e8eef9">{action_label}</strong></p>
  <p style="color:#94a3b8;margin:0 0 24px">Event: <code style="color:#5AA9FF">{id}</code></p>
  <p style="color:#64748b;font-size:13px">The command center has been updated. You can close this tab.</p>
</body></html>"""
        return HTMLResponse(content=body)

    body = f"""
<html><body style="font-family:sans-serif;background:#0f1724;color:#e8eef9;text-align:center;padding:80px 24px;max-width:480px;margin:0 auto">
  <div style="font-size:48px;margin-bottom:16px">⚠</div>
  <h2 style="color:#FFB547;margin:0 0 8px">Event Not Found</h2>
  <p style="color:#94a3b8">Event <code style="color:#5AA9FF">{id}</code> may have already been resolved or expired.</p>
</body></html>"""
    return HTMLResponse(content=body, status_code=404)


@app.get("/gate2/test")
def gate2_test():
    """Open http://localhost:8000/gate2/test in a browser to diagnose email config."""
    if not _GATE2_OK:
        return HTMLResponse(content="<h2>crisis_gate2.py not importable</h2>", status_code=500)

    ok, detail = _gate2_mod.test_email()

    colour = "#3BF0A4" if ok else "#FF5E5B"
    icon   = "✓" if ok else "✗"
    rows   = "".join(
        f"<tr><td style='padding:6px 14px;color:#64748b;white-space:nowrap'>{k}</td>"
        f"<td style='padding:6px 14px;font-family:monospace'>{v}</td></tr>"
        for part in detail.split(" | ")
        for k, _, v in [part.partition("=")]
        if _
    )
    body = f"""
<html><body style="font-family:sans-serif;background:#0f1724;color:#e8eef9;padding:40px;max-width:640px;margin:0 auto">
  <h2 style="color:{colour}">{icon} Gate 2 Email Diagnostic</h2>
  <table style="width:100%;border-collapse:collapse;background:#131d2e;border-radius:8px;overflow:hidden;margin-top:16px">
    {rows}
  </table>
  <p style="color:#94a3b8;margin-top:20px;font-size:13px">
    {"Test emails sent. Check inboxes." if ok else "Email failed — see detail above. Fix .env and restart the server."}
  </p>
  <p style="color:#334155;font-size:11px">Raw detail: {detail}</p>
</body></html>"""
    engine._append_log("INFO" if ok else "ERROR", f"Gate 2 test: {detail}")
    return HTMLResponse(content=body, status_code=200 if ok else 500)


def main() -> None:
    import uvicorn
    uvicorn.run("crisis_server:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
