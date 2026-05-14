"""
Fast offline speech recognition using PocketSphinx
Uncensored, lightweight, instant results
No internet required, runs locally on CPU
"""

import pyaudio
import time
import threading

try:
    from pocketsphinx import Decoder, get_model_path
    POCKETSPHINX_AVAILABLE = True
except ImportError:
    POCKETSPHINX_AVAILABLE = False
    print("Warning: PocketSphinx not installed. Install with: pip install pocketsphinx")


class PocketSphinxRecognizer:
    """Fast, offline speech recognizer using PocketSphinx."""
    
    def __init__(self):
        """Initialize PocketSphinx decoder."""
        self.is_ready = False
        self.decoder = None
        
        if not POCKETSPHINX_AVAILABLE:
            print("[POCKETSPHINX] [ERR] PocketSphinx not installed")
            print("[POCKETSPHINX] Install with: pip install pocketsphinx")
            return
        
        try:
            model_path = get_model_path()
            config = Decoder.default_config()
            config.set_string('-hmm', f'{model_path}/en-us')
            config.set_string('-dict', f'{model_path}/cmudict-en-us.dict')
            
            self.decoder = Decoder(config)
            print("[POCKETSPHINX] [OK] Initialized successfully")
            self.is_ready = True
        except Exception as e:
            print(f"[POCKETSPHINX] [ERR] Initialization error: {e}")
            print("[POCKETSPHINX] Make sure PocketSphinx models are installed")
            self.is_ready = False
    
    def listen_and_recognize(self, timeout_seconds=1.5):
        """
        Listen and recognize speech in REAL-TIME.
        Returns results within 1-2 seconds (much faster than Google).
        Completely uncensored.
        
        Args:
            timeout_seconds: How long to listen (1.5s is optimal)
        
        Returns:
            Recognized text or empty string
        """
        if not self.is_ready or self.decoder is None:
            print("[POCKETSPHINX] [ERR] Recognizer not available")
            return ""
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=2048
            )
            
            print(f"[POCKETSPHINX] Listening... ({timeout_seconds}s timeout)")
            
            start_time = time.time()
            self.decoder.start_utt()
            
            # Collect audio data
            while time.time() - start_time < timeout_seconds:
                try:
                    data = stream.read(2048, exception_on_overflow=False)
                    self.decoder.process_raw(data, False, False)
                except Exception as e:
                    print(f"[POCKETSPHINX] Audio read error: {e}")
                    continue
            
            self.decoder.end_utt()
            result = self.decoder.hyp()
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if result and result.hypstr:
                recognized_text = result.hypstr.strip()
                if recognized_text:
                    print(f"[POCKETSPHINX] [OK] Recognized: {recognized_text}")
                    return recognized_text
            
            print("[POCKETSPHINX] No speech detected")
            return ""
            
        except Exception as e:
            print(f"[POCKETSPHINX] [ERR] Error during recognition: {e}")
            return ""


# Singleton instance
_ps_recognizer = None
_recognizer_lock = threading.Lock()


def get_pocketsphinx_recognizer():
    """
    Get or create PocketSphinx recognizer instance (singleton).
    Thread-safe initialization.
    """
    global _ps_recognizer
    if _ps_recognizer is None:
        with _recognizer_lock:
            if _ps_recognizer is None:
                _ps_recognizer = PocketSphinxRecognizer()
    return _ps_recognizer


def is_pocketsphinx_available():
    """Check if PocketSphinx is available and ready."""
    recognizer = get_pocketsphinx_recognizer()
    return recognizer.is_ready
