import cv2
import numpy as np

def check_blur(gray_frame, threshold=50.0):
    """
    Checks if a grayscale frame is blurry using the Laplacian variance method.
    
    Args:
        gray_frame (numpy.ndarray): The input grayscale image.
        threshold (float): Variance threshold. Below this is blurry.

    Returns:
        bool: True if the image is blurry, False otherwise.
    """
    variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return variance < threshold, variance

def check_shake(gray_frame, prev_gray_frame, threshold=5.0):
    """
    Checks for camera shake using Dense Optical Flow.
    
    Args:
        gray_frame (numpy.ndarray): The current grayscale frame.
        prev_gray_frame (numpy.ndarray): The previous grayscale frame.
        threshold (float): Average motion magnitude. Above this is shake.

    Returns:
        bool: True if shake is detected, False otherwise.
        float: The average motion magnitude.
    """
    # Pre-allocate flow array
    flow = np.zeros((gray_frame.shape[0], gray_frame.shape[1], 2), dtype=np.float32)
    
    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_frame, 
        gray_frame, 
        flow, 
        0.5,  # pyr_scale
        3,    # levels
        15,   # winsize
        3,    # iterations
        5,    # poly_n
        1.2,  # poly_sigma
        0     # flags
    )
    
    # Calculate the magnitude (speed) of motion vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Convert to NumPy array and get the average magnitude across the entire frame
    magnitude_np = cv2.cvtColor(magnitude.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(magnitude.shape) == 2 else magnitude
    avg_magnitude = float(magnitude.mean())
    
    # If avg magnitude is high, it means the whole camera is moving (shake)
    return avg_magnitude > threshold, avg_magnitude

def check_glare(frame, threshold_pct=10.0):
    """
    Analyzes a frame to detect glare.
    
    Glare is defined as a high percentage of pixels being "pure white".
    
    Args:
        frame (numpy.ndarray): The BGR video frame from OpenCV.
        threshold_pct (float): The percentage of white pixels that triggers glare detection.

    Returns:
        tuple: (is_glare, percentage, histogram)
            - is_glare (bool): True if glare is detected.
            - percentage (float): The actual percentage of white pixels.
            - histogram (numpy.ndarray): Histogram data for visualization.
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate total number of pixels
    total_pixels = gray.shape[0] * gray.shape[1]
    
    # Find "pure white" pixels (values 250-255)
    white_pixels = np.sum((gray >= 250))
    
    # Calculate the percentage
    percentage = (white_pixels / total_pixels) * 100
    
    # Calculate histogram for visualization (26 buckets of ~10 intensity each)
    histogram = cv2.calcHist([gray], [0], None, [26], [0, 256])
    histogram = histogram.flatten().tolist()  # Convert to list for JSON serialization
    
    # Return stats
    is_glare = percentage > threshold_pct
    return is_glare, percentage, histogram

def fix_blur_unsharp_mask(frame, kernel_size=5, sigma=1.0, strength=1.5):
    """
    Fix blur using Unsharp Masking technique.
    Enhances edges and makes blurry images appear sharper in real-time.
    
    Args:
        frame (numpy.ndarray): Input BGR frame.
        kernel_size (int): Gaussian blur kernel size (must be odd, default 5).
        sigma (float): Standard deviation for Gaussian blur (default 1.0).
        strength (float): How much sharpening to apply. 
                         1.0 = no effect, >1.0 = sharpen (default 1.5).
    
    Returns:
        numpy.ndarray: Sharpened BGR frame with same dimensions as input.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create blurred version (low-pass filter)
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    # Unsharp mask formula: output = original + (original - blurred) * strength
    # This amplifies the high-frequency details (edges)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    
    # Clip values to valid range [0, 255] and convert to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

# Track shift magnitude history for repositioning detection
_shift_history = []
_direction_history = []  # Track direction vectors for consistency
_MAX_HISTORY = 15  # Track last 15 frames for better temporal analysis

def _exponential_moving_average(value, history, alpha=0.3):
    """Apply exponential moving average to smooth noisy measurements."""
    if not history:
        return value
    return alpha * value + (1 - alpha) * history[-1]

def detect_camera_reposition(gray_frame, prev_gray_frame, threshold_shift=10.0):
    """
    Detects camera repositioning by analyzing sustained directional motion.
    Improved accuracy with better direction consistency and noise filtering.
    
    Repositioning is detected when:
    1. For fast movement: High shift magnitude (>2x threshold) sustained for 2+ frames
    2. For slow movement: Consistent directional movement over 5+ frames
    
    Args:
        gray_frame (numpy.ndarray): Current grayscale frame.
        prev_gray_frame (numpy.ndarray): Previous grayscale frame.
        threshold_shift (float): Threshold for shift magnitude detection. Default 10.0.
    
    Returns:
        tuple: (is_repositioned, shift_magnitude, shift_x, shift_y)
            - is_repositioned (bool): True if sustained repositioning detected.
            - shift_magnitude (float): Directional shift magnitude.
            - shift_x (float): Average horizontal motion.
            - shift_y (float): Average vertical motion.
    """
    global _shift_history, _direction_history
    
    h, w = gray_frame.shape
    
    # Calculate dense optical flow
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_frame, gray_frame, flow,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Calculate magnitude of motion vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Use full frame but filter outliers for more accurate detection
    valid_magnitude = magnitude[magnitude > 0.1]  # Ignore noise (very small movements)
    
    if len(valid_magnitude) == 0:
        shift_x, shift_y, shift_magnitude = 0.0, 0.0, 0.0
    else:
        # Calculate shift using median of top 25% of flows (more robust than mean)
        valid_flow_x = flow[..., 0][magnitude > 0.1]
        valid_flow_y = flow[..., 1][magnitude > 0.1]
        
        # Use median for robustness against outliers
        shift_x = float(np.median(valid_flow_x)) if len(valid_flow_x) > 0 else 0.0
        shift_y = float(np.median(valid_flow_y)) if len(valid_flow_y) > 0 else 0.0
        shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
    
    # Apply noise filtering: ignore very small movements
    if shift_magnitude < 0.2:
        shift_magnitude = 0.0
        shift_x = 0.0
        shift_y = 0.0
    
    # Track shift history with smoothing
    smoothed_shift = _exponential_moving_average(shift_magnitude, _shift_history, alpha=0.25)
    _shift_history.append(smoothed_shift)
    if len(_shift_history) > _MAX_HISTORY:
        _shift_history.pop(0)
    
    # Track direction vectors (normalized)
    if shift_magnitude > 0.5:
        direction = (shift_x / shift_magnitude, shift_y / shift_magnitude)
    else:
        direction = (0.0, 0.0)
    _direction_history.append(direction)
    if len(_direction_history) > _MAX_HISTORY:
        _direction_history.pop(0)
    
    # Repositioning detection with improved dual criteria:
    is_repositioned = False
    
    if len(_shift_history) >= 2:
        # CRITERION 1: FAST REPOSITIONING (Sudden large movement)
        # Requires sustained high shift over 2+ frames (not just one spike)
        fast_reposition_threshold = threshold_shift * 1.8  # Slightly lower for sensitivity
        fast_reposition_count = sum(1 for s in _shift_history[-3:] if s > fast_reposition_threshold)
        
        if fast_reposition_count >= 2:
            is_repositioned = True
        
        # CRITERION 2: SLOW/SUSTAINED REPOSITIONING (Gradual movement)
        if len(_shift_history) >= 6:
            # Detect sustained movement over 6 frames with direction consistency
            recent_shifts = _shift_history[-6:]
            recent_directions = _direction_history[-6:]
            
            # Count frames with significant shift (>50% of threshold)
            shift_count = sum(1 for s in recent_shifts if s > threshold_shift * 0.6)
            
            # Calculate direction consistency using dot product
            if shift_count >= 4:
                # Only consider directions where shift was significant
                significant_directions = [
                    recent_directions[i] for i in range(len(recent_directions))
                    if recent_shifts[i] > threshold_shift * 0.6 and recent_directions[i] != (0.0, 0.0)
                ]
                
                if len(significant_directions) >= 3:
                    # Calculate average direction
                    avg_dir_x = np.mean([d[0] for d in significant_directions])
                    avg_dir_y = np.mean([d[1] for d in significant_directions])
                    avg_direction_mag = np.sqrt(avg_dir_x**2 + avg_dir_y**2)
                    
                    # Direction consistency: how aligned are the individual directions?
                    # Calculate average dot product with the mean direction
                    if avg_direction_mag > 0:
                        dot_products = [
                            (d[0] * avg_dir_x + d[1] * avg_dir_y) / (avg_direction_mag + 1e-6)
                            for d in significant_directions
                        ]
                        direction_consistency = float(np.mean(dot_products))
                        
                        # Slow repositioning: sustained consistent movement
                        # Direction consistency > 0.5 means angles are within ~60 degrees
                        if direction_consistency > 0.5:
                            is_repositioned = True
    
    return is_repositioned, shift_magnitude, shift_x, shift_y