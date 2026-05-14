import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Reusable Functions for app.py ---

# We combine glare check and rescue, as they are related
def check_glare_and_rescue(frame, clahe, glare_threshold_pct=25.0):
    """
    Analyzes a frame for glare. If glare is detected, it applies CLAHE
    to rescue the frame.
    
    Returns:
        - processed_frame: The (potentially) enhanced frame.
        - is_glare: Boolean flag.
        - hist: The histogram data (for plotting).
    """
    # 1. Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Calculate the histogram
    # 256 bins, for pixel values 0-255
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # 3. Glare Detection Logic
    total_pixels = gray.shape[0] * gray.shape[1]
    
    # Check "pure white" pixels (value 255)
    white_pixels = hist[255][0] + hist[254][0]  + hist[253][0] + hist[252][0] # Consider top 3 values as glare
    
    percentage = (white_pixels / total_pixels) * 100
    
    is_glare = percentage > glare_threshold_pct
    
    processed_frame = frame # By default, return the original color frame
    
    #4. Active Defense (CLAHE)
    if is_glare:
        # If glare is detected, apply CLAHE to the *grayscale* frame
        # and return that as the processed frame for the demo.
        rescued_frame = clahe.apply(gray)
        
        # Convert grayscale back to BGR so it can be shown in the
        # same color window as the original.
        processed_frame = cv2.cvtColor(rescued_frame, cv2.COLOR_GRAY2BGR)
        
    return processed_frame, is_glare, hist


def check_liveness(gray_frame, prev_gray_frame, liveness_threshold=1000):
    """
    Checks for a frozen or looped feed by comparing frame differences.
    
    Returns:
        - is_frozen: Boolean flag.
    """
    if prev_gray_frame is None:
        return False # Can't compare first frame

    # 1. Calculate the absolute difference
    frame_diff = cv2.absdiff(prev_gray_frame, gray_frame)
    
    # 2. Sum the differences
    total_diff = np.sum(frame_diff)
    
    # 3. Check against threshold
    # A real video feed *always* has noise.
    # A frozen feed will have a sum of 0 (or very close).
    is_frozen = total_diff < liveness_threshold
    
    return is_frozen


# --- Standalone Test Harness ---

# This block only runs when you execute `python sensor.py` directly.
# Your app.py will NOT run this code.
if __name__ == "__main__":
    
    print("Starting sensor test harness...")
    print("Point a flashlight at the camera to test glare.")
    print("Show a static photo to the camera to test liveness.")
    print("Press 'q' to quit.")

    # --- Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        exit()

    # Initialize CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Initialize previous frame for liveness check
    ret, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Matplotlib Setup for Live Graph ---
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_title("Live Grayscale Histogram")
    ax.set_xlabel("Pixel Value (0-255)")
    ax.set_ylabel("Frequency")

    x_axis = np.arange(256)
    
    line, = ax.plot([], []) # Create an empty line object to update
    ax.set_xlim(0, 255) # Fix x-axis
    
    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't receive frame.")
            break
            
        # 1. Run Glare Check & Rescue
        processed_frame, is_glare, hist = check_glare_and_rescue(
            frame, clahe, glare_threshold_pct=10.0
        )
        
        
        # 2. Run Liveness Check
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_frozen = check_liveness(current_gray, prev_gray)
        
        # 3. Update Histogram Plot
        #line.set_ydata(hist)
        line.set_data(x_axis, hist.ravel())
        ax.set_ylim(0, np.max(hist[1:]) + 100) # Auto-adjust y-axis (ignore black spike)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 4. Put Status Text on Video Frame
        if is_glare:
            cv2.putText(processed_frame, "GLARE DETECTED (RESCUE ACTIVE)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(processed_frame, "Status: Normal", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if is_frozen:
            cv2.putText(processed_frame, "LIVENESS FAILED (FROZEN FEED)", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 5. Display the Frame
        cv2.imshow("Aegis Sensor Test - Processed Feed", processed_frame)
        cv2.imshow("Raw Feed", frame) # Show raw feed for comparison
        
        # 6. Update prev_gray for next loop
        prev_gray = current_gray

        if cv2.waitKey(1) == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()