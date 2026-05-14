"""
RGB Watermark Validator - Validates video watermarks and compares expected vs extracted colors
"""
import cv2
import hmac
import hashlib
from backend.watermark_extractor import extract_watermark_color, color_distance

# Configuration (must match embedder)
SECRET_KEY = b"AegisSecureWatermarkKey2025"
LIVE_THRESHOLD = 0.70  # 70% of frames must match to be LIVE

def get_expected_hmac_token(timestamp):
    """
    Calculate expected HMAC token for a given timestamp.
    Uses the same algorithm as dynamic_watermarker.generate_hmac_token()
    
    Args:
        timestamp: Unix timestamp (float or int)
        
    Returns:
        HMAC token as 4-digit string (0000-9999)
    """
    # Convert timestamp to bytes
    timestamp_bytes = str(int(timestamp)).encode('utf-8')
    
    # Generate HMAC-SHA256
    hmac_obj = hmac.new(SECRET_KEY, timestamp_bytes, hashlib.sha256)
    hmac_digest = hmac_obj.hexdigest()
    
    # Get last 4 hex characters and convert to 4-digit token (0000-9999)
    last_4_hex = hmac_digest[-4:]
    token_int = int(last_4_hex, 16)
    final_token = token_int % 10000
    
    # Return as 4-digit string with leading zeros
    return f"{final_token:04d}"


def validate_video(video_path, start_timestamp):
    """
    Validate video watermarks by checking frames at 1-second intervals.
    
    Args:
        video_path: Path to video file
        start_timestamp: Unix timestamp when recording started
        
    Returns:
        Dict with validation results:
        {
            'overall_status': 'LIVE' or 'NOT_LIVE',
            'results': [
                {
                    'second': 0,
                    'timestamp': 1234567890,
                    'expected_hmac': '1a2b3c4d5e6f7890',
                    'extracted_hmac': '1a2b3c4d5e6f7890',
                    'match': 1
                },
                ...
            ],
            'matched': 5,
            'total': 6,
            'percentage': 83.3
        }
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            'overall_status': 'ERROR',
            'error': f'Could not open video: {video_path}',
            'results': []
        }
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    frame_interval = int(fps)  # Check every 1 second
    results = []
    frame_count = 0
    matched = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check frame at 1-second intervals
            if frame_count % frame_interval == 0:
                second_index = frame_count // frame_interval
                expected_timestamp = start_timestamp + second_index
                
                # Get expected HMAC token
                expected_hmac = get_expected_hmac_token(expected_timestamp)
                extracted_color = extract_watermark_color(frame)
                
                # Convert extracted RGB to HMAC token representation
                # RGB comes from first 3 bytes of HMAC, so reconstruct the token
                if extracted_color is not None:
                    # Build hex string from RGB bytes: RRGGBB + 00
                    hex_from_rgb = f"{extracted_color[0]:02x}{extracted_color[1]:02x}{extracted_color[2]:02x}00"
                    # Take last 4 hex chars and convert to 4-digit token
                    last_4_hex = hex_from_rgb[-4:]
                    token_int = int(last_4_hex, 16)
                    extracted_hmac = f"{token_int % 10000:04d}"
                    # Exact match only - no tolerance
                    match = extracted_hmac == expected_hmac
                    if match:
                        matched += 1
                else:
                    extracted_hmac = "None"
                    match = False
                
                results.append({
                    'second': second_index,
                    'timestamp': int(expected_timestamp),
                    'expected_hmac': expected_hmac,
                    'extracted_hmac': extracted_hmac,
                    'match': int(match)  # Convert bool to int for JSON (0 or 1)
                })
                
                # Print for debugging
                match_icon = '[OK]' if match else '[ERR]'
                print(f"[LIVENESS] Second {second_index}: Expected {expected_hmac} | Extracted {extracted_hmac} | {match_icon}")
            
            frame_count += 1
    
    finally:
        cap.release()
    
    # Calculate overall status
    total = len(results)
    if total == 0:
        return {
            'overall_status': 'ERROR',
            'error': 'No frames to validate',
            'results': []
        }
    
    percentage = (matched / total) * 100
    overall_status = 'LIVE' if (matched / total) >= LIVE_THRESHOLD else 'NOT_LIVE'
    
    print(f"[LIVENESS] Summary: {matched}/{total} frames matched ({percentage:.1f}%) - Status: {overall_status}")
    
    return {
        'overall_status': overall_status,
        'results': results,
        'matched': matched,
        'total': total,
        'percentage': percentage
    }
