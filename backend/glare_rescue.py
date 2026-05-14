import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---

def apply_unsharp_mask(frame, amount=1.0, kernel_size=(5, 5), sigma=1.0):
    """Applies a simple unsharp mask to sharpen the image."""
    blurred = cv2.GaussianBlur(frame, kernel_size, sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def get_image_viability_stats(frame, dark_thresh=40, bright_thresh=200):
    """Analyzes a frame using the "Loss of Detail" metric."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = gray.shape[0] * gray.shape[1]

    dark_pixels = np.sum(hist[0:dark_thresh])
    bright_pixels = np.sum(hist[bright_thresh:256])
    mid_tone_pixels = np.sum(hist[dark_thresh:bright_thresh])

    dark_pct = (dark_pixels / total_pixels) * 100
    bright_pct = (bright_pixels / total_pixels) * 100
    mid_pct = (mid_tone_pixels / total_pixels) * 100

    # More sensitive thresholds — triggers on moderate glare, not just blown-out highlights
    threshold_dark_pct = 15.0
    threshold_bright_pct = 0.3
    threshold_mid_pct = 75.0

    is_glare = (
        dark_pct > threshold_dark_pct and
        bright_pct > threshold_bright_pct and
        mid_pct < threshold_mid_pct
    )

    return is_glare, dark_pct, mid_pct, bright_pct, hist, gray


def apply_msr(frame, sigmas=(15, 80, 250)):
    """Multi-Scale Retinex — normalizes illumination across scales for a stark, evenly-lit result."""
    frame_f = frame.astype(np.float64) + 1.0
    log_f = np.log(frame_f)
    retinex = np.zeros_like(log_f)
    w = 1.0 / len(sigmas)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(frame_f, (0, 0), sigma)
        retinex += w * (log_f - np.log(np.maximum(blur, 1.0)))
    # Stretch each channel to [0, 255]
    out = np.zeros_like(retinex)
    for i in range(3):
        ch = retinex[:, :, i]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, i] = (ch - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

# --- Standalone Test Harness ---
if __name__ == "__main__":
    
    print("Starting glare rescue test (CLAHE-Only Mode)...")
    print("...Using your tuned parameters (clipLimit=16.0, thresholds 50/252).")
    print("Press 'q' in the OpenCV window to quit.")

    # --- Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        exit()
        
    # --- Using your tuned CLAHE parameters ---
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
    
    # --- Matplotlib Setup ---
    plt.ion() 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('Live Histogram Analysis', fontsize=16)

    ax1.set_title("Continuous Histogram (0-255)")
    x_continuous = np.arange(256)
    line_continuous, = ax1.plot(x_continuous, np.zeros(256), color='b')
    ax1.set_xlim(0, 255)
    # Using your tuned thresholds for the plot lines
    ax1.axvline(x=50, color='r', linestyle='--') 
    ax1.axvline(x=252, color='r', linestyle='--')

    ax2.set_title("Bucket-wise Histogram (~10 Intensity Buckets)")
    nbins = 26
    x_bucketed = np.arange(nbins)
    rects_bucketed = ax2.bar(x_bucketed, np.zeros(nbins), width=1.0, color='g')
    ax2.set_xlim(0, nbins - 1)
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Using your tuned thresholds from the file
        is_glare, dark_pct, mid_pct, bright_pct, hist, gray = get_image_viability_stats(
            frame, dark_thresh=50, bright_thresh=252
        )
        
        # 2. --- ACTIVE DEFENSE LOGIC (CLAHE-Only) ---
        if is_glare:
            # --- 1. CLAHE Rescue ---
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_frame)
            l_clahe = clahe.apply(l)
            enhanced_lab_frame = cv2.merge((l_clahe, a, b))
            clahe_rescued_frame = cv2.cvtColor(enhanced_lab_frame, cv2.COLOR_LAB2BGR)
            
            # --- 2. Sharpening ---
            processed_frame = apply_unsharp_mask(clahe_rescued_frame, amount=1.0)
            
            # --- 3. TAME HIGHLIGHTS ---
            # Find the original blown-out highlights
            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_raw, 252, 255, cv2.THRESH_BINARY)
            # Set those pixels to a neutral gray
            processed_frame[mask > 0] = (150, 150, 150) 
            # --- END OF HACK ---

            status_text = "GLARE (RESCUE: CLAHE + TAME)"
            text_color = (0, 0, 255) # Red
        
        else:
            # No glare
            processed_frame = frame
            status_text = "Status: Normal"
            text_color = (0, 255, 0) # Green
        
        # 3. Add status text
        cv2.putText(processed_frame, status_text, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, text_color, 2)
        
        # 4. Add the percentage stats
        h, w, _ = frame.shape
        stats_text_1 = f"Dark %: {dark_pct:.1f}"
        stats_text_2 = f"Mid %:  {mid_pct:.1f}"
        stats_text_3 = f"Bright %: {bright_pct:.1f}"
        
        cv2.putText(processed_frame, stats_text_1, (10, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(processed_frame, stats_text_2, (10, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(processed_frame, stats_text_3, (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 5. Calculate Bucketed Histogram
        hist_bucketed = cv2.calcHist([gray], [0], None, [nbins], [0, 256])

        # 6. Update Histogram Plots
        if np.max(hist[10:-10]) > 0: # Avoid crash on empty hist
            ax1.set_ylim(0, np.max(hist[10:-10]) + 100) 
        line_continuous.set_ydata(hist.ravel())

        if np.max(hist_bucketed[1:-1]) > 0:
            ax2.set_ylim(0, np.max(hist_bucketed[1:-1]) + 100) 
        for rect, h in zip(rects_bucketed, hist_bucketed.ravel()):
            rect.set_height(h)

        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # 7. Display BOTH Frames
        cv2.imshow("Aegis Rescued Feed (Press 'q' to quit)", processed_frame)
        cv2.imshow("Raw Feed", frame)

        # --- Keypress Handler ---
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)
    print("Test finished.")