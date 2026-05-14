# AV26-001

AV26-001 is a split security-and-response demo with two independent dashboards that run from the same frontend launcher:

- `Tamper Resistance`
- `Crisis Response`

The project is designed to demonstrate two separate workflows without merging them into one mixed interface.

## What the project demonstrates

### 1. Tamper Resistance

The tamper side focuses on integrity monitoring and visual verification. It is intended to show how a system can watch for suspicious activity, surface security-related signals, and present them in a minimal operator dashboard.

Core ideas demonstrated here:

- tamper-focused UI and backend flow
- live monitoring style layout
- separate runtime from the crisis module
- camera/video-based evidence handling

### 2. Crisis Response

The crisis side is a separate emergency-response dashboard for fire and fall scenarios.

Core ideas demonstrated here:

- separate fire detection workflow
- separate fall detection workflow
- single alert card design for clean operator attention
- clear response status display
- action status such as ambulance or fire engine deployment

The crisis dashboard uses your local uploads:

- `videos/` for sample input clips
- `models/` for `.pt` weights

### 3. Shared launcher

When you open the root frontend, you see a brief project overview and two launch options:

- `Tamper Resistance`
- `Crisis Response`

This keeps the demos separate while still giving you one entry point.

## Project structure

The project is organized so the two demos stay isolated.

### Frontend entry points

- `src/pages/Home.tsx` - launcher page
- `src/pages/TamperSurveillance.tsx` - tamper dashboard
- `src/pages/CrisisResponse.tsx` - crisis dashboard
- `src/tamper-main.tsx` - tamper entry bundle
- `src/crisis-main.tsx` - crisis entry bundle

### Frontend routes and pages

- `index.html` - launcher route
- `tamper.html` - tamper demo route
- `crisis.html` - crisis demo route
- `vite.config.ts` - multi-page Vite config

### Crisis backend

- `crisis_server.py` - crisis detection server
- `src/hooks/use-crisis-status.ts` - crisis polling hook
- `src/lib/crisis-types.ts` - shared crisis data types
- `src/lib/crisis-mock.ts` - fallback/mock data helpers

### Tamper backend and support code

- `app.py` - main backend entry
- `backend/database.py` - persistence layer
- `backend/tamper_detector.py` - tamper detection logic
- `backend/watermark_*.py` - watermark embed/extract/validate flow
- `backend/glare_rescue.py` - glare-handling support
- `backend/pocketsphinx_recognizer.py` - speech recognizer support

### Utility scripts

- `scripts/dynamic_watermarker.py`
- `scripts/liveness.py`
- `scripts/low_light.py`
- `scripts/sensor_test.py`

## How it works

### Dashboard launch flow

Open the root frontend and choose one of the demos:

1. `Tamper Resistance`
2. `Crisis Response`

Each demo opens as a separate dashboard so the experiences remain isolated.

### Tamper runtime

The tamper side is intended for monitoring and verification tasks. It uses its own backend and frontend bundle and is not merged with the crisis pipeline.

### Crisis runtime

The crisis side processes uploaded videos and model files separately from tamper.

Model convention:

- `models/best.pt` for fire detection
- `models/yolov8n.pt` for fall detection

Video convention:

- add your sample clips to `videos/`
- select the clip from the crisis dashboard

The crisis dashboard is intentionally minimal:

- one active alert card
- one main video panel
- one response state area
- one control area for choosing clips

## Local setup

### Frontend

Install dependencies and run the app:

```bash
npm install
npm run dev
```

If the project is configured with the Vite dev server used in this repo, open:

- `http://localhost:4173/`

### Crisis backend

Run the crisis server separately if you want live crisis inference:

```bash
python crisis_server.py
```

### Tamper backend

Run the main backend separately for the tamper workflow:

```bash
python app.py
```

## Folder conventions

Use these folders for local assets:

- `videos/` - sample crisis clips
- `models/` - `.pt` model weights

The repository ignores runtime and cache files such as build output, Python caches, logs, and local database artifacts.

## Notes

- The two demos are intentionally separate.
- The launcher is the main entry point for both demos.
- The crisis response view is for fire/fall monitoring and alert presentation.
- The tamper view is for integrity and watermark-related monitoring.

## Repository goal

This repository is meant to demonstrate a practical multi-dashboard security project with:

- a clear launcher
- separate tamper and crisis experiences
- local model/video asset handling
- a minimal operator-focused UI

