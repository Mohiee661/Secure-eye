import cv2
import time
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
CAMERA_INDEX = 0
LIVENESS_THRESHOLD = 3.0        # LOW threshold: detects freezing/static feed (Only active after grace period)
MAJOR_TAMPER_THRESHOLD = 60.0   # HIGH threshold: detects sudden, massive scene change (Always active)
BLACKOUT_BRIGHTNESS_THRESHOLD = 25.0 # INCREASED: Mean pixel intensity threshold for blackout (0-255 range). 
                                     # Set higher to account for noise/light leaks when covered.
LIVENESS_CHECK_INTERVAL = 3.0   # Time (in seconds) between capturing a new reference frame
LIVENESS_ACTIVATION_TIME = 10.0 # Time (s) after startup before "FROZEN FEED ALERT" becomes active
CAMERA_WARMUP_FRAMES = 30       # Number of frames to read and discard at startup

# Global variable to track the definitive start time for the 10s grace period
startup_time = 0.0

# ============================================================================
# INITIALIZATION & WARMUP
# ============================================================================

# Initialize the camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera. Please check CAMERA_INDEX.")
    exit()

# Set properties for stability
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Camera Warm-up / Black Frame Fix ---
# Read and discard the first few frames to allow the camera to stabilize exposure.
print(f"Warming up camera for {CAMERA_WARMUP_FRAMES} frames...")
for _ in range(CAMERA_WARMUP_FRAMES):
    cap.read()
    time.sleep(0.033) # Delay to simulate typical frame rate

# Capture the first reliable reference frame
ret, reference_frame_bgr = cap.read()
if not ret:
    print("Error: Could not read a stable initial frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

reference_frame_gray = cv2.cvtColor(reference_frame_bgr, cv2.COLOR_BGR2GRAY)
reference_time = time.time()
startup_time = time.time() # FIX: Set the definitive start time for the grace period
print("Camera ready. Starting interval-based liveness monitoring.")

# ============================================================================
# MAIN LOOP
# ============================================================================

while True:
    ret, current_frame_bgr = cap.read()
    if not ret:
        print("Error reading current frame.")
        break

    current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
    current_time = time.time()
    
    # --- LIVENESS CHECKS ---
    
    # 1. Calculate the absolute difference between the current frame and the reference frame (for motion magnitude)
    diff_frame = cv2.absdiff(current_frame_gray, reference_frame_gray)

    # 2. Calculate the mean difference across all pixels
    mean_diff = np.mean(diff_frame)
    
    # 3. Calculate the mean brightness of the current frame (for blackout detection)
    mean_brightness = np.mean(current_frame_gray)
    
    # Check if the liveness detection grace period has passed using the non-resetting startup_time
    is_liveness_active = (current_time - startup_time) > LIVENESS_ACTIVATION_TIME
    
    # --- Determine Tamper State ---
    
    is_blackout = mean_brightness < BLACKOUT_BRIGHTNESS_THRESHOLD # High priority check
    is_major_tamper = mean_diff > MAJOR_TAMPER_THRESHOLD           # High change (Hijack)
    is_frozen = False
    
    if is_liveness_active and mean_diff < LIVENESS_THRESHOLD:
        # Only flag as frozen if the grace period is over AND the change is very low
        is_frozen = True
    
    # --- Update Liveness Status for Display (Prioritize Blackout) ---
    
    status_text = ""
    alert_color = (0, 255, 0) # Green (LIVE default)

    if is_blackout:
        # Highest priority alert: Blackout detected.
        status_text = f"!!! BLACKOUT DETECTED !!! (Brightness: {mean_brightness:.1f})"
        alert_color = (0, 0, 255) # Red
    elif is_major_tamper:
        status_text = f"!!! MAJOR TAMPER / HIJACK !!! ({mean_diff:.1f})"
        alert_color = (0, 0, 255) # Red
    elif is_frozen:
        status_text = f"!! FROZEN FEED ALERT !! ({mean_diff:.1f} < {LIVENESS_THRESHOLD:.1f})"
        alert_color = (0, 165, 255) # Orange
    elif not is_liveness_active:
        # During the grace period, calculate remaining time using startup_time
        time_left = LIVENESS_ACTIVATION_TIME - (current_time - startup_time)
        status_text = f"STATUS: INITIALIZING LIVENESS... ({time_left:.1f}s left)"
        alert_color = (255, 255, 0) # Yellow
    else:
        status_text = f"STATUS: LIVE. Mean Diff: {mean_diff:.2f}"
        
    # Display status on the frame
    display_frame = current_frame_bgr.copy()
    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
    
    # Display Metrics
    cv2.putText(display_frame, f"Activation Time Left: {LIVENESS_ACTIVATION_TIME - (current_time - startup_time):.1f}s", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Ref Age: {current_time - reference_time:.1f}s", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Change Mag: {mean_diff:.2f}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Brightness: {mean_brightness:.2f}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. Check if it's time to capture a new reference frame (resets reference_time only)
    if current_time - reference_time >= LIVENESS_CHECK_INTERVAL:
        reference_frame_gray = current_frame_gray.copy()
        reference_time = current_time
        # print(f"Reference frame updated. Next check interval starts now.")

    # Display the final frame
    cv2.imshow("AEGIS Liveness Monitor", display_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()