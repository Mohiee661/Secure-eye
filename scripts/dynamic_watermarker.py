import cv2
import numpy as np
import time
import hmac
import hashlib
from datetime import datetime
import csv
import os

# --- 1. CONFIGURATION ---
INPUT_VIDEO_FILE = 0 # Set to 0 for the default webcam. Try 1, 2, etc., if 0 fails.
OUTPUT_VIDEO_FILE = 'watermarked_output.mp4'
WATERMARK_LOG_FILE = 'watermark_log.csv' # Log file for validator
LOG_DURATION_SECONDS = 30 # Limit log duration to 30 seconds for testing purposes

# **CRITICAL SECRET:** This key must be identical in the validator (test_validator.py).
SERVER_SECRET_KEY = b"YourUnbreakableWatermarkSecretKey12345" 

# Opacity/Visibility settings
VISIBILITY_FLIP_INTERVAL_SECONDS = 10 

# --- 2. CRYPTOGRAPHIC TOKEN LOGIC (HMAC-SHA256) ---

def generate_hmac_token():
    """
    Generates a unique 4-digit token based on HMAC-SHA256, ensuring non-sequential jumps.
    Returns the token string and the timestamp used to generate it.
    """
    current_time_seconds = int(time.time())
    
    # The message is the time, converted to bytes
    message = str(current_time_seconds).encode('utf-8')
    
    # 1. Compute the HMAC hash
    hmac_digest = hmac.new(
        key=SERVER_SECRET_KEY, 
        msg=message, 
        digestmod=hashlib.sha256
    ).hexdigest()
    
    # 2. TRUNCATE and Convert to a small integer (0000-9999)
    last_4_hex = hmac_digest[-4:]
    token_int = int(last_4_hex, 16)
    final_token = token_int % 10000
    
    # Format as a 4-digit string with leading zeros
    return f"{final_token:04d}", current_time_seconds

def get_current_opacity():
    """
    Determines the current watermark opacity for the adaptive visibility effect.
    """
    current_time_seconds = int(time.time())
    
    # Opacity is High (1.0) for 10s, Low (0.2) for the next 10s.
    if current_time_seconds % (VISIBILITY_FLIP_INTERVAL_SECONDS * 2) < VISIBILITY_FLIP_INTERVAL_SECONDS:
        return 1.0 # High Opacity (Fully visible text)
    else:
        return 0.2 # Low Opacity (Faint text)


def generate_watermark_text(token):
    """
    Creates the complete dynamic watermark signature.
    """
    dt_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    signature = f"TST-H:{token} | T:{dt_now}"
    return signature

# --- 3. WATERMARK LOGGING AND EMBEDDING ---

def initialize_log_file():
    """Initializes the CSV log file with headers."""
    # Delete file if it exists to ensure a clean start
    if os.path.exists(WATERMARK_LOG_FILE):
        os.remove(WATERMARK_LOG_FILE)
    
    with open(WATERMARK_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp_seconds', 'hmac_token'])

def log_watermark(timestamp, token):
    """Appends the current token and timestamp to the CSV log file."""
    with open(WATERMARK_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, token])

def embed_watermark(frame, watermark_text, rect_alpha_multiplier):
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 
    font_thickness = 2
    
    text_size = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size[0], text_size[1]
    
    padding = 10
    text_x = W - text_width - padding
    text_y = H - padding 
    
    # 1. Draw a semi-transparent black rectangle for contrast
    overlay = frame.copy()
    rect_color = (0, 0, 0)
    
    rect_alpha = 0.4 * rect_alpha_multiplier
    
    rect_top_left_x = text_x - padding
    rect_top_left_y = text_y - text_height - padding
    
    rect_bottom_right_x = W 
    rect_bottom_right_y = H 

    cv2.rectangle(overlay, 
                  (rect_top_left_x, rect_top_left_y), 
                  (rect_bottom_right_x, rect_bottom_right_y), 
                  rect_color, 
                  -1)
    
    cv2.addWeighted(overlay, rect_alpha, frame, 1 - rect_alpha, 0, frame)

    # 2. Add the text watermark
    text_opacity = rect_alpha_multiplier
    if text_opacity < 1.0:
        text_color = (150, 150, 150)
    else:
        text_color = (255, 255, 255)


    cv2.putText(frame, watermark_text, 
                (text_x, text_y), 
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return frame

# --- 4. MAIN PROCESSING LOOP ---
def process_dynamic_watermarking(input_source):
    
    cap = cv2.VideoCapture(input_source, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        print("Check if camera is available or try changing INPUT_VIDEO_FILE index (0, 1, etc.).")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0.0:
        fps = 30.0
    
    # --- Codec and VideoWriter Initialization ---
    CODECS = ['XVID', 'MJPG', 'mp4v', 'DIVX']
    out = None
    for codec_name in CODECS:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                print(f"Successfully initialized video writer with codec: **{codec_name}**")
                break 
        except Exception:
            continue
    
    if not out or not out.isOpened():
        print("Fatal Error: Could not find a working codec to write the video file.")
        cap.release()
        return
    # --- End Codec Initialization ---

    initialize_log_file() # Initialize the log file
    start_time = time.time()
    last_logged_time = 0
    
    print(f"Starting HMAC-TST watermarking. Estimated FPS: {fps:.2f}. ")
    print(f"Logging tokens for the first {LOG_DURATION_SECONDS} seconds to {WATERMARK_LOG_FILE}.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        current_time_s = int(current_time)
        
        # --- LOGIC EXECUTION ---
        opacity_multiplier = get_current_opacity()
        token, token_timestamp = generate_hmac_token() # Get the token and its source timestamp
        watermark_text = generate_watermark_text(token)
        
        # --- LOGGING (Once per second) ---
        if current_time_s > last_logged_time and current_time < start_time + LOG_DURATION_SECONDS:
            log_watermark(token_timestamp, token)
            last_logged_time = current_time_s

        # Stop processing after logging duration is met
        if current_time > start_time + LOG_DURATION_SECONDS + 5: # Give a few seconds buffer
             print("\nLOGGING COMPLETE. Ending stream now.")
             break

        # --- EMBEDDING ---
        frame = embed_watermark(frame, watermark_text, opacity_multiplier)

        out.write(frame)
        cv2.imshow('HMAC-TST Watermarked Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing. Output saved to {OUTPUT_VIDEO_FILE}")

# --- EXECUTE ---
process_dynamic_watermarking(INPUT_VIDEO_FILE)