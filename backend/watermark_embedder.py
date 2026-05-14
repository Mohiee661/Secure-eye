"""
RGB Watermark Embedder - Encodes HMAC tokens as RGB colors and embeds into frames
"""
import hmac
import hashlib
import cv2
import time
import threading

# Configuration
SECRET_KEY = b"AegisSecureWatermarkKey2025"
WATERMARK_SIZE = 40  # 40x40 pixel square
PADDING = 10  # 10px from edges

def get_hmac_color(timestamp):
    """
    Generate RGB color from HMAC token based on timestamp.
    
    Args:
        timestamp: Unix timestamp (float or int)
        
    Returns:
        Tuple (R, G, B) with values 0-255
    """
    # Convert timestamp to bytes
    timestamp_bytes = str(int(timestamp)).encode('utf-8')
    
    # Generate HMAC-SHA256
    hmac_obj = hmac.new(SECRET_KEY, timestamp_bytes, hashlib.sha256)
    hmac_digest = hmac_obj.digest()
    
    # Use first 3 bytes as RGB values
    r = hmac_digest[0]
    g = hmac_digest[1]
    b = hmac_digest[2]
    
    return (int(r), int(g), int(b))


def embed_watermark(frame, timestamp):
    """
    Embed RGB watermark into frame at bottom-right corner.
    
    Args:
        frame: OpenCV frame (BGR format)
        timestamp: Unix timestamp
        
    Returns:
        Frame with watermark embedded
    """
    if frame is None:
        return frame
    
    height, width = frame.shape[:2]
    
    # Calculate watermark position (bottom-right corner with padding)
    x_end = width - PADDING
    x_start = x_end - WATERMARK_SIZE
    y_end = height - PADDING
    y_start = y_end - WATERMARK_SIZE
    
    # Get RGB color from timestamp
    rgb_color = get_hmac_color(timestamp)
    
    # OpenCV uses BGR, so convert RGB to BGR
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    
    # Draw filled rectangle
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), bgr_color, -1)
    
    return frame


class WatermarkEmbedder:
    """Simple watermark embedder that uses current timestamp."""
    
    def embed(self, frame):
        """Embed watermark with current timestamp."""
        return embed_watermark(frame, time.time())


# Global singleton
_embedder = None


def get_watermark_embedder():
    """Get or create the global watermark embedder."""
    global _embedder
    if _embedder is None:
        _embedder = WatermarkEmbedder()
    return _embedder
_embedder_lock = threading.Lock()


def get_watermark_embedder():
    """
    Get or create global watermark embedder instance (singleton).
    Thread-safe initialization.
    
    Returns:
        WatermarkEmbedder: Global embedder instance
    """
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = WatermarkEmbedder()
    return _embedder
