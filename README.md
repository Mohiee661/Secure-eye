# Aegis – Tamper-Resistant Surveillance System

AegisAI is a professional-grade surveillance platform for tamper-resistant surveillance and incident logging. The crisis command center is kept as a separate entry point, not merged into the tamper dashboard.

---

## Dashboard Overview

The real-time dashboard provides visualization of camera feeds, tamper metrics, and incident logs with toggleable defense modules.

| Primary Dashboard View | Video Integrity & Analytics |
|:---:|:---:|
| ![Main Dashboard](assets/frontend/dashboard_main.png) | ![Video Analysis](assets/frontend/dashboard_video.png) |

---

## Key Capabilities

- **Tamper Detection**: Real-time identification of blur, shake, and repositioning.
- **Glare Rescue**: Detail recovery from overexposed frames using CLAHE.
- **Cryptographic Security**: HMAC-SHA256 frame watermarking to prevent replay attacks.
- **Forensic Logging**: SQL-backed incident tracking and liveness verification.

### Separate Crisis Entry

The crisis-response dashboard is available as a separate frontend entry at `crisis.html`.

Drop your crisis sample videos into the `videos/` folder, and place your `.pt` model files in `models/` or whichever path you configure in the crisis backend.

Expected model mapping:

- `models/best.pt` for fire detection
- `models/yolov8n.pt` for fall detection

You can override those defaults with `CRISIS_FIRE_MODEL_PATH` and `CRISIS_FALL_MODEL_PATH`.

The crisis backend uses CUDA automatically when `torch.cuda.is_available()` is true.

The crisis backend cycles through files in `videos/` and uses the video name to choose the detector profile:

- names containing `fire` use the fire model
- names containing `fall` use the fall model
- everything else falls back to combined mode when both models are present

The crisis dashboard also includes direct buttons to select any uploaded video and play it immediately.

### Glare Rescue Performance
| Glare Recovery 1 | Glare Recovery 2 | Glare Recovery 3 |
|:---:|:---:|:---:|
| ![Glare Detection](https://github.com/user-attachments/assets/9d827772-dec1-4091-b6e7-204c0b49b8b5) | ![Recovery Process](https://github.com/user-attachments/assets/ad7a8f4b-ff9f-4455-8346-6fa55e4e8595) | ![Restored Output](https://github.com/user-attachments/assets/3284e225-613f-4894-b88c-80338ae5b1f1) |
---

## Technical Architecture

- **Backend**: Python/Flask-SocketIO handling computer vision (OpenCV) and cryptography.
- **Frontend**: HTML5/JS/CSS real-time monitoring dashboard.
- **Security**: HMAC-SHA256 token embedding for frame-level integrity.
- **Database**: SQLite for auditable incident logs and metadata.

---

## Quick Start

### Installation
```powershell
# Clone and setup
git clone https://github.com/ZeroDeaths7/AegisAI-tamper-resistent-surveillance-system.git
cd AegisAI-tamper-resistent-surveillance-system
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Execution
```powershell
python app.py
```
Access the dashboard at `http://localhost:5000`.

### Crisis Backend

```powershell
python crisis_server.py
```
This serves the crisis API on `http://localhost:8000/status`.

### Frontend Entries

```powershell
npm run dev
```
Open the tamper dashboard through the default app entry.

```powershell
http://localhost:4173/crisis.html
```
Open the crisis-response dashboard as a separate page.

---

## Testing
Run the automated test suite to verify detection and integrity modules:
```bash
pytest
```

---

## Authors & License
- **Authors**: Prateek, Mevin, Rajeev, Abhiram
- **License**: MIT License
