"""
RGB Watermark Extractor - Extracts and compares watermark colors from frames
"""
import cv2
import numpy as np

# Configuration (must match embedder)
WATERMARK_SIZE = 40
PADDING = 10

def extract_watermark_color(frame):
    """
    Extract RGB watermark color from bottom-right corner of frame.
    
    Args:
        frame: OpenCV frame (BGR format)
        
    Returns:
        Tuple (R, G, B) extracted from watermark, or None if extraction fails
    """
    if frame is None:
        return None
    
    try:
        height, width = frame.shape[:2]
        
        # Define watermark ROI (bottom-right corner)
        x_end = width - PADDING
        x_start = x_end - WATERMARK_SIZE
        y_end = height - PADDING
        y_start = y_end - WATERMARK_SIZE
        
        # Extract ROI
        roi = frame[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            return None
        
        # Calculate mean color from ROI
        # Reshape to 2D for calculation
        roi_2d = roi.reshape((-1, 3))
        mean_rgb = np.mean(roi_2d, axis=0)
        
        # Return as RGB tuple
        return tuple(int(x) for x in mean_rgb)
        
    except Exception as e:
        print(f"[WATERMARK] Error extracting color: {e}")
        return None


def color_distance(color1, color2):
    """
    Calculate Euclidean distance between two RGB colors.
    
    Args:
        color1: Tuple (R, G, B)
        color2: Tuple (R, G, B)
        
    Returns:
        Float distance value
    """
    if color1 is None or color2 is None:
        return float('inf')
    
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    distance = np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
    return distance
