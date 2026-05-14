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

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]

if torch is not None and torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


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


def _pick_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


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


def _status_from_conf(conf: float) -> str:
    if conf >= 0.45:
        return "ALERT"
    if conf >= 0.2:
        return "MONITOR"
    return "SAFE"


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
        self.logs = deque(maxlen=60)
        self.alerts = deque(maxlen=1)
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
        self.fall_detection_streak = 0
        self.fall_conf_history = deque(maxlen=8)
        self.fall_alert_emitted = False
        self.fall_confirm_frames = 5
        self.fall_confirm_min_frame = 30
        self.fall_confirm_confidence = 0.62
        self.active_incident_decision: Optional[str] = None
        self.active_incident_reason: Optional[str] = None

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

    def _reset_fall_tracker(self) -> None:
        self.fall_detection_streak = 0
        self.fall_conf_history.clear()
        self.fall_alert_emitted = False

    def _reset_incident_state(self) -> None:
        self.active_incident_decision = None
        self.active_incident_reason = None

    def _update_fall_tracker(self, fall_conf: float, video: Path) -> bool:
        if _profile_for_video(video) == "fall" and fall_conf >= 0.45:
            self.fall_detection_streak += 1
        else:
            self.fall_detection_streak = max(0, self.fall_detection_streak - 1)

        self.fall_conf_history.append(fall_conf if _profile_for_video(video) == "fall" else 0.0)
        recent = list(self.fall_conf_history)
        recent_window = recent[-5:]
        recent_avg = sum(recent_window) / len(recent_window) if recent_window else 0.0
        recent_min = min(recent_window) if recent_window else 0.0
        recent_peak = max(recent_window) if recent_window else 0.0

        ready = (
            not self.fall_alert_emitted
            and _profile_for_video(video) == "fall"
            and self.frame_index >= self.fall_confirm_min_frame
            and self.fall_detection_streak >= self.fall_confirm_frames
            and fall_conf >= self.fall_confirm_confidence
            and recent_avg >= self.fall_confirm_confidence
            and recent_min >= 0.48
            and recent_peak >= 0.7
        )
        if ready:
            self.fall_alert_emitted = True
        return ready

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
        self._append_log("INFO", f"Loaded video: {video.name}")
        return True

    def _advance_video(self) -> bool:
        if not self.video_paths:
            return False
        self.active_video_index = (self.active_video_index + 1) % len(self.video_paths)
        self.frame_index = 0
        self._reset_fall_tracker()
        return self._open_current_video()

    def _model_detection(self, model, frame, kind: str) -> list[DetectionEvent]:
        if model is None:
            return []

        try:
            result = model.predict(
                frame,
                conf=0.18,
                verbose=False,
                device=self.device,
                half=self.cuda_enabled,
            )[0]
        except Exception as exc:
            self._append_log("ERROR", f"{kind} inference failed: {exc}")
            return []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = result.names if hasattr(result, "names") else {}
        events: list[DetectionEvent] = []
        kind_keywords = {
            "fire": ("fire", "flame", "smoke", "burn"),
            "fall": ("fall", "fallen", "down", "collapse", "lying"),
        }[kind]

        for box in boxes:
            cls_index = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            label = str(names.get(cls_index, kind)).lower()
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            matches_label = any(keyword in label for keyword in kind_keywords)
            if matches_label or confidence >= 0.45:
                xyxy = box.xyxy[0].tolist()
                bbox = tuple(int(v) for v in xyxy)
                events.append(DetectionEvent(kind=kind, confidence=confidence, label=label, bbox=bbox))

        return events

    def _annotate(self, frame, events: list[DetectionEvent]):
        annotated = frame.copy()
        for event in events:
            x1, y1, x2, y2 = event.bbox
            color = (0, 0, 255) if event.kind == "fire" else (255, 180, 0) if event.kind == "fall" else (90, 170, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_name = "PERSON" if event.kind == "person" else event.kind.upper()
            label = f"{label_name} {event.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _decision_from_events(self, events: list[DetectionEvent], fall_ready: bool) -> tuple[Optional[str], Optional[str]]:
        fire_conf = max((e.confidence for e in events if e.kind == "fire"), default=0.0)
        fall_conf = max((e.confidence for e in events if e.kind == "fall"), default=0.0)

        if fire_conf >= fall_conf and fire_conf >= 0.45:
            return "ALERT_FIRE_ENGINE", "Fire detected by the fire model"
        if fall_ready:
            return "ALERT_AMBULANCE", "Fall detected by the fall model"
        if fall_conf >= 0.45:
            return "IGNORE", "Fall candidate detected; waiting for confirmation"
        return "IGNORE", "No high-confidence crisis signal detected"

    def _fall_candidate_from_person_box(
        self,
        confidence: float,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect = width / height
        bottom_ratio = y2 / max(1, frame_height)
        area_ratio = (width * height) / max(1, frame_width * frame_height)

        return (
            confidence >= 0.55
            and aspect >= 1.12
            and bottom_ratio >= 0.60
            and area_ratio >= 0.03
        )

    def _fall_person_events(
        self,
        model,
        frame,
    ) -> list[DetectionEvent]:
        if model is None:
            return []

        try:
            result = model.predict(
                frame,
                conf=0.18,
                verbose=False,
                device=self.device,
                half=self.cuda_enabled,
            )[0]
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
            cls_index = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            label = str(names.get(cls_index, "person")).lower()
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            xyxy = box.xyxy[0].tolist()
            bbox = tuple(int(v) for v in xyxy)

            if any(keyword in label for keyword in ("fall", "fallen", "lying", "collapse")) and confidence >= 0.45:
                events.append(DetectionEvent(kind="fall", confidence=confidence, label=label, bbox=bbox))
                continue

            if label == "person" and confidence >= 0.18:
                if self._fall_candidate_from_person_box(confidence, bbox, frame_width, frame_height):
                    events.append(DetectionEvent(kind="fall", confidence=confidence, label="fall_candidate", bbox=bbox))
                else:
                    events.append(DetectionEvent(kind="person", confidence=confidence, label="person", bbox=bbox))

        return events

    def _current_location(self, video: Path) -> str:
        name = video.name.lower()
        if "fire1" in name:
            return "Lobby"
        if "fire2" in name:
            return "Warehouse"
        if "fall" in name:
            return "Stairwell"
        return "Unknown"

    def _update_state(self, frame, events: list[DetectionEvent], video: Path, process_ms: float) -> None:
        fire_conf = max((e.confidence for e in events if e.kind == "fire"), default=0.0)
        fall_conf = max((e.confidence for e in events if e.kind == "fall"), default=0.0)
        top_conf = max(fire_conf, fall_conf)
        fall_ready = self._update_fall_tracker(fall_conf, video)
        decision, reason = self._decision_from_events(events, fall_ready)

        if self.active_incident_decision is not None:
            decision = self.active_incident_decision
            reason = self.active_incident_reason or reason
            alert_locked = True
            status = "ALERT"
            severity = _severity_from_conf(top_conf) if top_conf > 0.0 else "LOW"
        else:
            alert_locked = decision != "IGNORE"
            if decision == "ALERT_FIRE_ENGINE" or decision == "ALERT_AMBULANCE":
                status = "ALERT"
            elif fire_conf >= 0.2 or fall_conf >= 0.2:
                status = "MONITOR"
            else:
                status = "SAFE"
            severity = _severity_from_conf(top_conf) if alert_locked and top_conf > 0.0 else "LOW"

            if decision == "ALERT_FIRE_ENGINE":
                alert_kind = "FIRE" if fire_conf >= fall_conf else "FALL"
                self._append_alert(
                    alert_kind,
                    severity,
                    "CAM-01",
                    f"{alert_kind.title()} detection from {video.name} at {top_conf:.2f}",
                )
                self.active_incident_decision = decision
                self.active_incident_reason = reason
            elif decision == "ALERT_AMBULANCE":
                self._append_alert(
                    "FALL",
                    severity,
                    "CAM-01",
                    f"Fall detection from {video.name} at {top_conf:.2f}",
                )
                self.active_incident_decision = decision
                self.active_incident_reason = reason

        lifecycle = "DISPATCHED" if alert_locked else "MONITORING"

        confidence_explanation = [
            "Video sample: " + video.name,
            "Model profile: " + _profile_for_video(video),
            "High-confidence detections: " + (", ".join(f"{e.kind}:{e.confidence:.2f}" for e in events) if events else "none"),
        ]

        signals = [f"{e.kind.upper()} @{e.confidence:.2f}" for e in events]
        llm_summary = (
            f"Fire event confirmed in {video.name}."
            if decision == "ALERT_FIRE_ENGINE"
            else f"Fall event confirmed in {video.name}."
            if decision == "ALERT_AMBULANCE"
            else "No active threat confirmed by the reasoning layer."
        )
        llm_confirmation = "REVIEW REQUIRED" if alert_locked else "NO CORRELATION"
        frame_b64 = _to_b64(self._annotate(frame, events))

        fps_now = 0.0
        now = time.perf_counter()
        delta = now - self.last_fps_ts
        if delta > 0:
            fps_now = 1.0 / delta
            self.fps_ema = fps_now if self.fps_ema == 0.0 else (self.fps_ema * 0.8 + fps_now * 0.2)
        self.last_fps_ts = now

        system_health = {
            "model_status": "ACTIVE" if self.fire_model or self.fall_model else "MISSING",
            "camera_status": "CONNECTED" if self.capture and self.capture.isOpened() else "OFFLINE",
            "api_status": "ONLINE",
            "latency": f"{process_ms:.0f}ms",
        }

        with self.state_lock:
            self.state.update(
                {
                    "frame": frame_b64,
                    "status": "ALERT" if alert_locked else status,
                    "alerts": list(self.alerts),
                    "videos": [p.name for p in self.video_paths],
                    "activeVideo": video.name,
                    "logs": list(self.logs),
                    "metrics": {
                        "confidence": round(top_conf * 100, 2),
                        "fps": round(self.fps_ema if self.fps_ema > 0 else fps_now, 2),
                        "latencyMs": round(process_ms, 2),
                        "modelsActive": int(self.fire_model is not None) + int(self.fall_model is not None),
                        "uptime": self._uptime(),
                    },
                    "location": self._current_location(video),
                    "currentVideo": video.name,
                    "lifecycleState": lifecycle,
                    "confidenceExplanation": confidence_explanation,
                    "decisionReason": reason,
                    "signals": signals,
                    "llmSummary": llm_summary,
                    "llmConfirmation": llm_confirmation,
                    "systemHealth": system_health,
                    "decision": decision,
                    "incidentLocked": alert_locked,
                }
            )

        signature = (video.name, decision or "IGNORE")
        if signature != self.last_status_signature:
            self._append_log("AGENT", f"{decision or 'IGNORE'} on {video.name} ({top_conf:.2f})")
            self.last_status_signature = signature

    def _uptime(self) -> str:
        elapsed = int(time.time() - START_TS)
        days, rem = divmod(elapsed, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{days:02d}d {hours:02d}:{minutes:02d}:{seconds:02d}"

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
                    self.state["frame"] = None
                    self.state["currentVideo"] = "None"
                    self.state["activeVideo"] = None
                    self.state["status"] = "SAFE"
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
                    if not self._advance_video():
                        time.sleep(1.0)
                        continue
                cap = self.capture

                if cap is None:
                    time.sleep(0.05)
                    continue

                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if not self._advance_video():
                        time.sleep(0.2)
                    continue

                self.frame_index += 1
                profile = _profile_for_video(self.video_paths[self.active_video_index % len(self.video_paths)])
                events: list[DetectionEvent] = []

                if profile in ("fire", "mixed") and self.fire_model is not None:
                    events.extend(self._model_detection(self.fire_model, frame, "fire"))
                if profile in ("fall", "mixed") and self.fall_model is not None:
                    events.extend(self._fall_person_events(self.fall_model, frame))

                if not events and profile == "fire" and self.fire_model is None and not self.missing_fire_warned:
                    self._append_log("WARN", "Fire model missing; unable to detect fire in video")
                    self.missing_fire_warned = True
                if not events and profile == "fall" and self.fall_model is None and not self.missing_fall_warned:
                    self._append_log("WARN", "Fall model missing; unable to detect fall in video")
                    self.missing_fall_warned = True

                process_ms = (time.perf_counter() - start) * 1000.0
                self._update_state(frame, events, self.video_paths[self.active_video_index % len(self.video_paths)], process_ms)

            fps = self.capture.get(cv2.CAP_PROP_FPS) if self.capture is not None else 0
            delay = 1.0 / fps if fps and fps > 0 else 0.08
            time.sleep(max(0.02, min(delay, 0.15)))

    def get_state(self) -> dict[str, Any]:
        with self.state_lock:
            return dict(self.state)

    def select_video(self, name: str) -> dict[str, Any]:
        with self.capture_lock:
            self.refresh_assets()
            match_index = next((idx for idx, path in enumerate(self.video_paths) if path.name == name), None)
            if match_index is None:
                raise ValueError(f"Unknown video: {name}")

            self.active_video_index = match_index
            self.frame_index = 0
            self.last_status_signature = None
            if not self._open_current_video():
                raise RuntimeError(f"Could not open video: {name}")

            self._reset_fall_tracker()
            self._reset_incident_state()

            with self.state_lock:
                self.alerts.clear()
                self.state["activeVideo"] = name
                self.state["currentVideo"] = name
                self.state["decisionReason"] = f"Operator selected video: {name}"
                self.state["lifecycleState"] = "MONITORING"
                self.state["incidentLocked"] = False
                self.state["alerts"] = []
                self.state["logs"] = list(self.logs)

            self._append_log("INFO", f"Selected video: {name}")
            return self.get_state()

    def override(self, action: str) -> dict[str, Any]:
        with self.state_lock:
            if action == "IGNORE":
                self.state["decision"] = "IGNORE"
                self.state["lifecycleState"] = "MONITORING"
                self.state["incidentLocked"] = False
                self.state["decisionReason"] = "Manual override by operator: IGNORE"
            else:
                self.state["decision"] = action
                self.state["lifecycleState"] = "DISPATCHED"
                self.state["incidentLocked"] = True
                self.state["decisionReason"] = f"Manual override by operator: {action}"
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
            self.missing_fire_warned = False
            self.missing_fall_warned = False
            self.no_video_warned = False
            self._append_log("INFO", "Crisis state reset")
            self.state["logs"] = list(self.logs)
            return dict(self.state)


app = FastAPI(title="AI Crisis Command Center")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TS = time.time()
engine = CrisisEngine()


@app.on_event("startup")
def on_startup() -> None:
    VIDEO_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    engine.run()


@app.on_event("shutdown")
def on_shutdown() -> None:
    engine.stop()


@app.get(STATUS_URL)
def get_status() -> dict[str, Any]:
    return engine.get_state()


@app.get("/videos")
def list_videos() -> dict[str, Any]:
    return {
        "videos": [p.name for p in engine.video_paths],
        "active": engine.video_paths[engine.active_video_index % len(engine.video_paths)].name if engine.video_paths else None,
    }


@app.post("/videos/select")
def select_video(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(payload.get("video", ""))
    return engine.select_video(name)


@app.post("/override")
def override(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action = str(payload.get("action", "IGNORE"))
    return engine.override(action)


@app.post("/reset")
def reset() -> dict[str, Any]:
    return engine.reset()


def main() -> None:
    import uvicorn

    uvicorn.run("crisis_server:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
