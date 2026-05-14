import cv2
import numpy as np

# -------------------------------
# Utility Functions
# -------------------------------

def gamma_correction(image, gamma=1.8):
    """Corrects gamma for low-light enhancement."""
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(image, table)

def enhance_low_light(frame):
    """Main low-light enhancement pipeline."""
    
    # Convert to LAB for CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for brightness improvement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    enhanced_lab = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Gamma correction
    enhanced = gamma_correction(enhanced, gamma=1.6)

    # Slight sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Denoise (only if dark)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    return enhanced


def is_low_light(frame, threshold=70):
    """Detects low-light by mean intensity."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold, brightness


# -------------------------------
# MAIN - Live Low-Light Enhancer
# -------------------------------

def start_low_light_enhancer():
    print("Starting Low-Light Feed Enhancer...")
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        low_light, brightness = is_low_light(frame)

        if low_light:
            enhanced = enhance_low_light(frame)
            cv2.putText(enhanced,
                        f"LOW LIGHT MODE (Brightness={brightness:.1f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            output = enhanced
        else:
            cv2.putText(frame,
                        f"Normal Light (Brightness={brightness:.1f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            output = frame

        cv2.imshow("Low-Light Enhanced Feed", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Low-light enhancer stopped.")
    

# -------------------------------
# Run directly
# -------------------------------
if __name__ == "__main__":
    start_low_light_enhancer()
