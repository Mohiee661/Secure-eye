"""
AEGIS: Active Defense System - Main Flask Server
Streams live video feed with tamper detection and real-time alerts via Socket.IO
"""

from flask import Flask, render_template, Response, send_from_directory, send_file, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from backend import tamper_detector
from backend.tamper_detector import fix_blur_unsharp_mask
import os
import threading
import time
import speech_recognition
from werkzeug.utils import secure_filename

# Import backend modules
from backend.database import aegis_db, save_glare_image, get_incident_description
from backend.watermark_validator import validate_video
from backend.watermark_embedder import get_watermark_embedder
from backend.pocketsphinx_recognizer import get_pocketsphinx_recognizer, is_pocketsphinx_available

# Try to import glare rescue functions, but make them optional
try:
    # <--- FIX 1: Only import functions that exist in your new file ---
    from backend.glare_rescue import (
        apply_unsharp_mask,
        get_image_viability_stats,
        apply_msr,
    )
    GLARE_RESCUE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Glare rescue functions not available: {e}")
    GLARE_RESCUE_AVAILABLE = False
    # Dummy functions
    def apply_unsharp_mask(frame, amount=1.0):
        return frame
    def get_image_viability_stats(frame, **kwargs): return False, 0, 0, 0, None, None


# ============================================================================
# INITIALIZATION
# ============================================================================

app = Flask(__name__, 
            template_folder='./frontend',
            static_folder='./frontend',
            static_url_path='')

socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
BLUR_THRESHOLD = 25.0
SHAKE_THRESHOLD = 5.0
REPOSITION_THRESHOLD = 5.0  # Threshold for directional shift magnitude
CAMERA_INDEX = 0
BLUR_FIX_ENABLED = True
BLUR_FIX_STRENGTH = 5
GLARE_RESCUE_ENABLED = True  # This is controlled by sensor_config
GLARE_RESCUE_MODE = 'MSR'  # This is controlled by sensor_config

# <--- FIX 2: Use your tuned CLAHE parameters ---
if GLARE_RESCUE_AVAILABLE:
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
else:
    clahe = None

# Sensor enable/disable configuration - all enabled by default
sensor_config = {
    'blur': True,           # Blur detection
    'shake': True,          # Shake detection
    'glare': True,          # Glare detection
    'liveness': True,       # Liveness detection
    'reposition': True,     # Reposition detection
    'blur_fix': False,      # Blur correction (off by default — apply only when needed)
    'glare_rescue': True,   # Glare rescue
    'audio_alerts': True,    # Audio alerts/logging
    'glare_rescue_mode': 'CLAHE' # <--- NEW: Add glare_rescue_mode here
}
sensor_config_lock = threading.Lock()  # Thread-safe config updates

# Liveness detection configuration
LIVENESS_THRESHOLD = 3.0        # LOW threshold: detects freezing/static feed
BLACKOUT_BRIGHTNESS_THRESHOLD = 25.0  # Mean pixel intensity threshold for blackout (0-255 range)
LIVENESS_CHECK_INTERVAL = 3.0   # Time (in seconds) between capturing a new reference frame
LIVENESS_ACTIVATION_TIME = 10.0 # Time (s) after startup before "FROZEN FEED ALERT" becomes active

# Global variables
cap = None
prev_gray = None
current_frame = None  # Raw frame without text
processed_frame = None  # Frame with detection text
frame_lock = threading.Lock()
reposition_alert_active = False
reposition_alert_shown = False  # Track if we've already shown the alert for this event
reposition_alert_frames = 0

# Liveness detection globals
liveness_reference_frame = None
liveness_reference_time = None
liveness_startup_time = None
liveness_is_frozen = False
liveness_status_text = "INITIALIZING"

# Initialize frames with blank black images (640x480) so they display while loading
current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Audio logging globals
LOG_AUDIO_SUBTITLES = False
audio_logging_lock = threading.Lock()
detection_data_cache = None

# Incident tracking globals
current_incident_id = None
last_detection_timestamp = None
detection_tracking_lock = threading.Lock()

# Storage configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'storage', 'liveness_videos')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def initialize_camera():
    """Initialize the camera capture object."""
    global cap
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera initialized successfully.")
    return True

def record_detection(detection_type, timestamp):
    """
    Record a detection incident to the database.
    Groups same-type detections within 5 seconds as one incident.
    """
    global current_incident_id
    
    with detection_tracking_lock:
        description = get_incident_description(detection_type)
        incident_id = aegis_db.record_detection(detection_type, timestamp, description)
        current_incident_id = incident_id
    
    return incident_id

def audio_logging_thread():
    """
    Audio logging thread that continuously listens for speech when triggered.
    Uses PocketSphinx for fast, uncensored, offline speech recognition.
    Falls back to Google Speech Recognition if PocketSphinx unavailable.
    Runs in a separate daemon thread and sends recognized speech to the frontend.
    """
    global LOG_AUDIO_SUBTITLES
    
    # Try to use PocketSphinx first (fast, uncensored, offline)
    ps_recognizer = get_pocketsphinx_recognizer()
    use_pocketsphinx = is_pocketsphinx_available()
    
    if use_pocketsphinx:
        print("[AUDIO] [OK] Using PocketSphinx for speech recognition (fast, uncensored)")
    else:
        print("[AUDIO] [WARN] PocketSphinx not available, falling back to Google API")
        r = speech_recognition.Recognizer()
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.6
        try:
            with speech_recognition.Microphone() as _src:
                print("[AUDIO] Calibrating for ambient noise (1 second)...")
                r.adjust_for_ambient_noise(_src, duration=1)
                print("[AUDIO] Calibration complete.")
        except Exception as _e:
            print(f"[AUDIO] Calibration skipped: {_e}")
    
    print("[AUDIO] Audio logger thread started - waiting for tamper detection...")
    
    while True:
        # Thread-safe check of audio logging flag
        with audio_logging_lock:
            should_log = LOG_AUDIO_SUBTITLES
        
        if should_log:
            try:
                if use_pocketsphinx:
                    # Use PocketSphinx (fast, uncensored, offline)
                    text = ps_recognizer.listen_and_recognize(timeout_seconds=1.5)
                else:
                    # Fallback to Google Speech Recognition
                    try:
                        with speech_recognition.Microphone() as source:
                            print("[AUDIO] Listening for speech...")
                            audio = r.listen(source, timeout=None, phrase_time_limit=8)
                        
                        try:
                            text = r.recognize_google(audio)
                            print(f"[AUDIO] [OK] Recognized (Google): {text}")
                        except speech_recognition.UnknownValueError:
                            print("[AUDIO] Could not understand audio")
                            text = ""
                        except speech_recognition.RequestError as e:
                            print(f"[AUDIO] API Error: {e}")
                            text = ""
                    except speech_recognition.WaitTimeoutError:
                        print("[AUDIO] No speech detected (timeout)")
                        text = ""
                    except Exception as e:
                        print(f"[AUDIO] Microphone error: {e}")
                        text = ""
                
                if text:
                    # Save audio log to database
                    audio_timestamp = time.time()
                    try:
                        aegis_db.add_audio_log(text, audio_timestamp, current_incident_id)
                        print(f"[AUDIO] [OK] Audio log saved to database")
                    except Exception as e:
                        print(f"[AUDIO] [ERR] Error saving audio log: {e}")
                    
                    # Send the subtitle to the frontend
                    try:
                        with app.app_context():
                            socketio.emit('subtitle', {
                                'type': 'SPEECH',
                                'text': text,
                                'is_blackbox': False
                            }, namespace='/', skip_sid=None)
                            print(f"[AUDIO] [OK] Subtitle emitted: {text}")
                    except Exception as e:
                        print(f"[AUDIO] [ERR] Error emitting subtitle: {e}")
                
            except Exception as e:
                print(f"[AUDIO] [ERR] Error: {e}")
        else:
            time.sleep(0.5)  # Sleep when not logging

def camera_thread():
    """
    Continuously capture frames and run detections.
    This runs in a separate thread to avoid blocking.
    """
    global prev_gray, current_frame, processed_frame, frame_lock, detection_data_cache
    global liveness_reference_frame, liveness_reference_time, liveness_startup_time
    global liveness_is_frozen, liveness_status_text, clahe
    
    print("[CAMERA] Camera thread starting...")
    
    if cap is None:
        print("[CAMERA] WARNING: Camera not initialized. Running in fallback mode.")
        while True:
            with frame_lock:
                current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.5)
    
    ret, first_frame = cap.read()
    if not ret:
        print("[CAMERA] Error: Could not read first frame.")
        return
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    liveness_reference_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    liveness_reference_time = time.time()
    liveness_startup_time = time.time()  # Grace period tracker
    frame_count = 0
    
    print("[CAMERA] Camera thread initialized successfully. Starting main loop...")
    print(f"[CAMERA] Frame size: {first_frame.shape}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA] Error: Could not read frame.")
            break
        
        frame_count += 1
        if frame_count == 1:
            print("[CAMERA] [OK] First frame captured! Stream is live.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()
        
        # --- RUN DETECTIONS ---
        is_blurred, blur_variance = tamper_detector.check_blur(gray, threshold=BLUR_THRESHOLD)
        is_shaken, shake_magnitude = tamper_detector.check_shake(gray, prev_gray, threshold=SHAKE_THRESHOLD)
        is_repositioned, shift_magnitude, shift_x, shift_y = tamper_detector.detect_camera_reposition(
            gray, prev_gray, threshold_shift=REPOSITION_THRESHOLD
        )

        # Fallback for fast movements: optical flow fails when displacement > ~1/8 frame width.
        # Raw pixel difference catches the scene-change regardless of movement speed.
        # Guard: skip if mean brightness < 20 (blackout event, not a reposition).
        if not is_repositioned:
            frame_mad = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            mean_brightness_now = float(np.mean(gray))
            if frame_mad > 45.0 and mean_brightness_now > 20.0:
                is_repositioned = True
                shift_magnitude = frame_mad * 0.4
                shift_x, shift_y = 0.0, 0.0

        # <--- FIX 3: Replace old glare check with the robust one ---
        # is_glare, glare_percentage, glare_histogram = tamper_detector.check_glare(frame, threshold_pct=10.0) # <--- DELETED
        
        # Read sensor configuration (thread-safe)
        with sensor_config_lock:
            sensor_enabled = sensor_config.copy()

        if GLARE_RESCUE_AVAILABLE and sensor_enabled['glare']:
            is_glare, dark_pct, mid_pct, bright_pct, hist_for_plot, _ = get_image_viability_stats(
                frame, dark_thresh=40, bright_thresh=200
            )
        else:
            is_glare, dark_pct, mid_pct, bright_pct, hist_for_plot = False, 0.0, 100.0, 0.0, None
        
        # --- LIVENESS DETECTION ---
        diff_frame = cv2.absdiff(gray, liveness_reference_frame)
        mean_diff = np.mean(diff_frame)
        mean_brightness = np.mean(gray)
        
        # Check if grace period (10s startup) has passed
        is_liveness_active = (current_time - liveness_startup_time) > LIVENESS_ACTIVATION_TIME
        
        # Determine liveness state - frozen feed and blackout detection
        is_blackout = mean_brightness < BLACKOUT_BRIGHTNESS_THRESHOLD
        is_frozen = False
        
        if is_liveness_active and mean_diff < LIVENESS_THRESHOLD:
            is_frozen = True
        
        # Update liveness status text
        if is_blackout:
            liveness_status_text = "BLACKOUT DETECTED"
        elif is_frozen:
            liveness_status_text = "FROZEN FEED ALERT"
        elif not is_liveness_active:
            time_left = LIVENESS_ACTIVATION_TIME - (current_time - liveness_startup_time)
            liveness_status_text = f"INITIALIZING... ({time_left:.1f}s)"
        else:
            liveness_status_text = "LIVE"
        
        # Update reference frame if check interval has passed
        if current_time - liveness_reference_time >= LIVENESS_CHECK_INTERVAL:
            liveness_reference_frame = gray.copy()
            liveness_reference_time = current_time
        
        # Store frozen state globally
        liveness_is_frozen = is_frozen
        
        # Respect sensor configuration - disable detections if sensor is disabled
        if not sensor_enabled['blur']:
            is_blurred = False
        if not sensor_enabled['shake']:
            is_shaken = False
        if not sensor_enabled['glare']:
            is_glare = False
        if not sensor_enabled['liveness']:
            is_frozen = is_blackout = False
            liveness_status_text = "INITIALIZING"
        if not sensor_enabled['reposition']:
            is_repositioned = False
        
        # Determine if ANY tamper is detected (for audio logging trigger)
        any_tamper_detected = is_blurred or is_shaken or is_glare or is_frozen or is_blackout
        
        # --- RECORD DETECTIONS TO DATABASE ---
        if is_blurred:
            record_detection('blur', current_time)
        if is_shaken:
            record_detection('shake', current_time)
        if is_glare:
            record_detection('glare', current_time)
        if is_repositioned:
            record_detection('reposition', current_time)
        if is_frozen:
            record_detection('freeze', current_time)
        if is_blackout:
            record_detection('blackout', current_time)
        
        # Manage repositioning alert state
        global reposition_alert_active, reposition_alert_shown, reposition_alert_frames
        if is_repositioned:
            reposition_alert_frames = 0
            if not reposition_alert_shown:
                reposition_alert_shown = True
                reposition_alert_active = True
                print(f"🚨 REPOSITION DETECTED - Magnitude: {shift_magnitude:.2f}px, Shift: ({shift_x:.2f}, {shift_y:.2f})")
                # Emit immediately — bypass the 3-frame throttle so the frontend never misses it
                try:
                    with app.app_context():
                        socketio.emit('reposition_alert', {
                            'magnitude': float(shift_magnitude),
                            'shift_x': float(shift_x),
                            'shift_y': float(shift_y),
                        }, namespace='/', skip_sid=None)
                except Exception as e:
                    print(f"[REPOSITION] Immediate emit error: {e}")
            else:
                reposition_alert_active = False
        else:
            reposition_alert_frames += 1
            if reposition_alert_frames > 30:  # Clear alert after 30 frames without detection
                reposition_alert_shown = False  # Reset flag when motion fully stops
            reposition_alert_active = False
        
        # <--- FIX 4: Populate detection_data with the NEW glare stats ---
        detection_data = {
            'blur': {
                'detected': bool(is_blurred),
                'variance': float(blur_variance)
            },
            'shake': {
                'detected': bool(is_shaken),
                'magnitude': float(shake_magnitude)
            },
            'reposition': {
                'detected': bool(is_repositioned),
                'magnitude': float(shift_magnitude),
                'shift_x': float(shift_x),
                'shift_y': float(shift_y),
                'alert_active': bool(reposition_alert_active)
            },
            'glare': {
                'detected': bool(is_glare),
                'dark_pct': float(dark_pct),
                'mid_pct': float(mid_pct),
                'bright_pct': float(bright_pct),
                'histogram': hist_for_plot.tolist() if hist_for_plot is not None else []
            },
            'liveness': {
                'frozen': bool(is_frozen),
                'blackout': bool(is_blackout),
                'status': liveness_status_text,
                'mean_diff': float(mean_diff),
                'mean_brightness': float(mean_brightness),
                'is_active': bool(is_liveness_active)
            }
        }
        
        # --- AUDIO LOGGING TRIGGER ---
        # Audio logging is triggered when ANY tamper (blur, shake, glare) is detected
        global LOG_AUDIO_SUBTITLES
        should_enable_audio = any_tamper_detected and sensor_enabled['audio_alerts']
        
        with audio_logging_lock:
            current_audio_state = LOG_AUDIO_SUBTITLES
        
        if should_enable_audio:
            if not current_audio_state:
                with audio_logging_lock:
                    LOG_AUDIO_SUBTITLES = True
                print(f"[TRIGGER] [OK] Audio logging ENABLED - Detections: Blur={is_blurred}, Shake={is_shaken}, Glare={is_glare}, Frozen={is_frozen}")
                try:
                    with app.app_context():
                        socketio.emit('alert', {
                            'type': 'AUDIO_LOGGING',
                            'message': 'ALERT: TAMPER DETECTED! Audio logging engaged.'
                        }, namespace='/', skip_sid=None)
                except Exception as e:
                    print(f"[TRIGGER] [ERR] Error emitting alert: {e}")
        else:
            if current_audio_state:
                with audio_logging_lock:
                    LOG_AUDIO_SUBTITLES = False
                print(f"[TRIGGER] [OK] Audio logging DISABLED - No tampering detected")
                try:
                    with app.app_context():
                        socketio.emit('alert_clear', {}, namespace='/', skip_sid=None)
                except Exception as e:
                    print(f"[TRIGGER] [ERR] Error emitting alert_clear: {e}")
        
        # Cache detection data
        detection_data_cache = detection_data
        
        # Emit detection update to all connected clients (every 3 frames to reduce jank)
        if frame_count % 3 == 0:
            try:
                with app.app_context():
                    socketio.emit('detection_update', detection_data, namespace='/', skip_sid=None)
                if frame_count % 30 == 0:
                    print(f"Emitted detection data: Blur={blur_variance:.2f}, Shake={shake_magnitude:.2f}, Dark%={dark_pct:.1f}")
            except Exception as e:
                print(f"Error emitting detection data: {e}")
        
        # --- GLARE RESCUE (Applied FIRST, before blur fixing) ---
        frame_for_processing = frame.copy()  # Start with original frame
        
        # <--- FIX 5: Replaced rescue block with your tuned CLAHE pipeline ---
        if is_glare and sensor_enabled['glare_rescue'] and clahe is not None:
            try:
                with sensor_config_lock:
                    current_mode = sensor_config.get('glare_rescue_mode', 'CLAHE')
                print(f"[GLARE] Applying glare rescue (mode: {current_mode})...")

                if current_mode == 'CLAHE':
                    # CLAHE on L channel in LAB space
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l_clahe = clahe.apply(l)
                    # Gamma boost to lift dark areas (gamma < 1 = brighter)
                    l_gamma = np.power(l_clahe / 255.0, 0.65) * 255.0
                    l_final = np.clip(l_gamma, 0, 255).astype(np.uint8)
                    merged = cv2.merge((l_final, a, b))
                    clahe_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                    # Sharpening pass
                    clahe_frame = apply_unsharp_mask(clahe_frame, amount=1.2)
                    # Tame highlights aggressively — anything > 220 is blown out
                    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_raw, 220, 255, cv2.THRESH_BINARY)
                    clahe_frame[mask > 0] = (120, 120, 120)
                    frame_for_processing = clahe_frame
                    print(f"[GLARE] CLAHE + gamma + tame applied.")

                elif current_mode == 'MSR':
                    # Multi-Scale Retinex — dramatic illumination normalization
                    msr_frame = apply_msr(frame, sigmas=(15, 80, 250))
                    # Light sharpening pass to restore edge crispness
                    msr_frame = apply_unsharp_mask(msr_frame, amount=0.6)
                    frame_for_processing = msr_frame
                    print(f"[GLARE] MSR applied.")
                
                # Save glare image to storage and database
                try:
                    file_path = save_glare_image(frame_for_processing, dark_pct, current_time) # Use dark_pct
                    aegis_db.add_glare_image(file_path, dark_pct, current_time, current_incident_id)
                    print(f"[GLARE] [OK] Rescued image saved: {file_path}")
                except Exception as e:
                    print(f"[GLARE] [ERR] Error saving glare image: {e}")
                        
            except Exception as e:
                print(f"[GLARE] Rescue error: {e}")
                frame_for_processing = frame.copy()
        
        # Create processed frame with blur fixing (applied AFTER glare rescue)
        if sensor_enabled['blur_fix']:
            # Always apply unsharp masking, but dynamically adjust strength
            if blur_variance < 50:
                dynamic_strength = 8.5  # Very blurry - maximum sharpening
            elif blur_variance < 100:
                dynamic_strength = 3 + (100 - blur_variance) / 8  # Scale between 5.0-8.5
            else:
                dynamic_strength = max(3, 8.5 - (blur_variance - 100) / 40)  # Scale down
            
            # Apply unsharp masking with dynamic strength
            processed_frame_final = fix_blur_unsharp_mask(frame_for_processing, kernel_size=5, sigma=1.0, strength=dynamic_strength)
        else:
            # Blur fix disabled, use glare-rescued frame as-is
            processed_frame_final = frame_for_processing
        
        # --- EMBED WATERMARK ON PROCESSED FRAME ---
        try:
            watermark_embedder = get_watermark_embedder()
            processed_frame_final = watermark_embedder.embed(processed_frame_final)
        except Exception as e:
            print(f"[WATERMARK] Error embedding watermark: {e}")
        
        # Update previous frame
        prev_gray = gray
        
        # Store frames for streaming
        with frame_lock:
            current_frame = frame.copy()  # Raw unmodified frame
            processed_frame = processed_frame_final.copy()  # Frame with glare rescue + blur fix + watermark
        
        # Small delay to prevent CPU overload
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: Blur={blur_variance:.2f}, Shake={shake_magnitude:.2f}")
        
        # Sleep briefly to avoid 100% CPU usage (~30 FPS)
        time.sleep(0.01)

def gen_frames():
    """
    Generator function that yields frames as MJPEG encoded frames.
    """
    global current_frame, frame_lock

    while True:
        if current_frame is None:
            time.sleep(0.033)
            continue

        with frame_lock:
            if current_frame is None:
                time.sleep(0.033)
                continue
            frame = current_frame.copy()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            time.sleep(0.033)
            continue

        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
               + frame_bytes + b'\r\n')

        time.sleep(0.033)  # cap at ~30 fps, stops busy-loop hammering frame_lock

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main dashboard HTML."""
    return send_from_directory('./Frontend', 'index.html')

@app.route('/video_frame')
def video_frame():
    """Serve a single JPEG frame from the raw feed (without detection text)."""
    global current_frame, frame_lock
    
    if current_frame is None or frame_lock is None:
        return "No frame available", 503
    
    try:
        with frame_lock:
            if current_frame is None:
                return "No frame available", 503
            frame = current_frame.copy()
    except:
        return "Error accessing frame", 500
    
    # Encode frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        return "Could not encode frame", 500
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/processed_frame')
def get_processed_frame():
    """Serve a single JPEG frame from the processed feed (with detection text)."""
    global processed_frame, frame_lock
    
    if processed_frame is None or frame_lock is None:
        return "No frame available", 503
    
    try:
        with frame_lock:
            if processed_frame is None:
                return "No frame available", 503
            frame = processed_frame.copy()
    except:
        return "Error accessing frame", 500
    
    # Encode frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        return "Could not encode frame", 500
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/detection')
def get_detection():
    """Get the latest detection data as JSON."""
    global detection_data_cache
    
    if detection_data_cache is None:
        return {"error": "No detection data available"}, 503
    
    return detection_data_cache

@app.route('/video_feed')
def video_feed():
    """Stream the raw video feed with detections."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_feed')
def processed_feed():
    """Stream processed video feed (same as raw for now)."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/style.css')
def serve_css():
    """Serve the CSS file."""
    return send_from_directory('./Frontend', 'style.css')

@app.route('/script.js')
def serve_js():
    """Serve the JavaScript file."""
    return send_from_directory('./Frontend', 'script.js')

# ============================================================================
# INCIDENT & HISTORY API ENDPOINTS
# ============================================================================

@app.route('/api/incidents', methods=['GET'])
def get_incidents():
    """Get recent incidents (last 5)."""
    try:
        incidents = aegis_db.get_recent_incidents(limit=5)
        return jsonify({
            'success': True,
            'incidents': incidents,
            'count': len(incidents)
        })
    except Exception as e:
        print(f"Error retrieving incidents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/incidents/<int:incident_id>', methods=['GET'])
def get_incident(incident_id):
    """Get a specific incident with all related data."""
    try:
        incident = aegis_db.get_incident_by_id(incident_id)
        if not incident:
            return jsonify({'success': False, 'error': 'Incident not found'}), 404
        
        return jsonify({
            'success': True,
            'incident': incident
        })
    except Exception as e:
        print(f"Error retrieving incident: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/incidents/<int:incident_id>/audio', methods=['GET'])
def get_incident_audio(incident_id):
    """Get audio logs for a specific incident."""
    try:
        audio_logs = aegis_db.get_audio_logs_for_incident(incident_id)
        return jsonify({
            'success': True,
            'audio_logs': audio_logs,
            'count': len(audio_logs)
        })
    except Exception as e:
        print(f"Error retrieving audio logs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/incidents/<int:image_id>/glare-image', methods=['GET'])
def get_glare_image(image_id):
    """Serve a glare image file."""
    try:
        file_path = aegis_db.get_glare_image_path(image_id)
        if not file_path or not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error retrieving glare image: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate-liveness-video', methods=['POST'])
def validate_liveness_video():
    """
    Upload and validate a liveness video for watermark integrity.
    Checks HMAC tokens at 1-second intervals.
    """
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(f"liveness_{int(time.time())}.mp4")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"[LIVENESS] Video uploaded: {file_path}")
        
        # Get video start timestamp if provided
        video_start_timestamp = request.form.get('video_start_timestamp')
        if video_start_timestamp:
            try:
                video_start_timestamp = int(float(video_start_timestamp))
            except:
                video_start_timestamp = int(time.time())  # Default to current time
        else:
            video_start_timestamp = int(time.time())  # Default to current time
        
        # Validate watermarks
        validation_result = validate_video(file_path, video_start_timestamp)
        
        if validation_result.get('overall_status') == 'ERROR':
            print(f"[LIVENESS] Validation failed: {validation_result.get('error')}")
        else:
            print(f"[LIVENESS] Validation complete - Status: {validation_result.get('overall_status')}")
        
        # Save validation to database
        try:
            incident_id = request.form.get('incident_id')
            incident_id = int(incident_id) if incident_id else None
            
            # Convert results to frame_results dict for database
            frame_results_dict = {}
            for result in validation_result.get('results', []):
                frame_key = f'second_{result["second"]}'
                frame_results_dict[frame_key] = result
            
            aegis_db.add_liveness_validation(
                file_path,
                validation_result.get('overall_status', 'UNKNOWN'),
                frame_results_dict,
                time.time(),
                incident_id
            )
            print("[LIVENESS] [OK] Validation result saved to database")
        except Exception as e:
            print(f"[LIVENESS] [ERR] Error saving validation to database: {e}")
        
        return jsonify({
            'success': True,
            'validation': validation_result,
            'file_path': file_path
        })
        
    except Exception as e:
        print(f"[LIVENESS] Error processing video: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# SOCKET.IO EVENT HANDLERS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    emit('status_update', {'status': 'healthy', 'message': 'Connected to Aegis server'})
    # Send current state on connect
    with sensor_config_lock:
        emit('sensor_states', sensor_config)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")

@socketio.on('test_alert')
def handle_test_alert(data):
    """Emit a test alert to all clients."""
    emit('alert', {
        'type': 'TEST',
        'message': data.get('message', 'Test alert from server')
    }, broadcast=True)

@socketio.on('dismiss_reposition_alert')
def handle_dismiss_reposition_alert():
    """Handle reposition alert dismissal from frontend."""
    global reposition_alert_shown
    reposition_alert_shown = False
    print("Reposition alert dismissed by user")

@socketio.on('get_sensor_states')
def handle_get_sensor_states():
    """Send current sensor configuration to client."""
    with sensor_config_lock:
        emit('sensor_states', sensor_config)

@socketio.on('set_sensor_enabled')
def handle_set_sensor_enabled(data):
    """Handle sensor enable/disable request from frontend."""
    sensor = data.get('sensor')
    enabled = data.get('enabled', True)
    
    if sensor in sensor_config:
        with sensor_config_lock:
            sensor_config[sensor] = enabled
        print(f"Sensor '{sensor}' set to {enabled}")
        # Broadcast sensor state update to all connected clients
        emit('status_update', {'status': 'healthy', 'message': f'{sensor} toggled to {enabled}'}, broadcast=True)
        emit('sensor_states', sensor_config, broadcast=True) # Send update to all clients
    else:
        print(f"Warning: Unknown sensor '{sensor}'")

# <--- NEW: Add handler for glare_rescue_mode ---
@socketio.on('set_glare_mode')
def handle_set_glare_mode(data):
    """Handle glare rescue mode toggle request from frontend."""
    mode = data.get('mode')
    
    if mode in ['CLAHE', 'MSR']:
        with sensor_config_lock:
            sensor_config['glare_rescue_mode'] = mode
        print(f"Glare Rescue Mode set to {mode}")
        # Broadcast to all connected clients
        emit('status_update', {'status': 'healthy', 'message': f'Glare Mode set to {mode}'}, broadcast=True)
        emit('sensor_states', sensor_config, broadcast=True) # Send updated config back
    else:
        print(f"Warning: Unknown glare mode '{mode}'")

# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

def startup():
    """Initialize the application."""
    print("=" * 60)
    print("AEGIS: ACTIVE DEFENSE SYSTEM")
    print("=" * 60)
    print(f"Blur Threshold: {BLUR_THRESHOLD}")
    print(f"Shake Threshold: {SHAKE_THRESHOLD}")
    print("[OK] Dynamic Watermarking: ENABLED")
    print("[OK] PocketSphinx Speech Recognition: READY")
    print("-" * 60)
    
    if not initialize_camera():
        print("WARNING: Could not initialize camera. Continuing in fallback mode.")
    
    return True

if __name__ == '__main__':
    if startup():
        # Start audio logging thread
        audio_thread = threading.Thread(target=audio_logging_thread, daemon=True)
        audio_thread.start()
        print("Audio logging thread started.")
        
        # Start camera thread
        camera_thread_obj = threading.Thread(target=camera_thread, daemon=True)
        camera_thread_obj.start()
        print("Camera thread started.")
        
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server.")
        print("-" * 60)
        
        try:
            socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            if cap:
                cap.release()
            print("Goodbye!")
