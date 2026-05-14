"""
AegisAI Database Module
Handles SQLite persistence for detection incidents, audio logs, glare images, and liveness validations.
"""

import sqlite3
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import os

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'aegis.db')
STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'storage')
GLARE_IMAGES_DIR = os.path.join(STORAGE_DIR, 'glare_images')
LIVENESS_VIDEOS_DIR = os.path.join(STORAGE_DIR, 'liveness_videos')

# Incident grouping configuration
INCIDENT_GROUP_TIMEOUT = 5.0  # seconds - group same type detections within this window
MAX_INCIDENTS_RETAINED = 5

# Incident type grouping
PHYSICAL_TAMPER_TYPES = {'blur', 'shake', 'glare', 'reposition'}
LIVENESS_THREAT_TYPES = {'freeze', 'blackout', 'major_tamper'}

# Thread safety
db_lock = threading.Lock()


class AegisDatabase:
    """SQLite database manager for AEGIS system."""
    
    def __init__(self):
        """Initialize database connection and create tables if needed."""
        self.connection = None
        self.initialize_db()
    
    def initialize_db(self):
        """Create database tables if they don't exist."""
        with db_lock:
            Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
            Path(GLARE_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
            Path(LIVENESS_VIDEOS_DIR).mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Incidents table - stores grouped detection incidents
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_type TEXT NOT NULL,
                    primary_detection TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    count INTEGER DEFAULT 1,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Audio logs table - stores audio transcripts chronologically
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id INTEGER,
                    text TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (incident_id) REFERENCES incidents(id)
                )
            ''')
            
            # Glare images table - stores glare rescue image metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS glare_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    glare_percentage REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (incident_id) REFERENCES incidents(id)
                )
            ''')
            
            # Liveness validations table - stores video watermark validation results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS liveness_validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id INTEGER,
                    file_path TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    frame_results TEXT,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (incident_id) REFERENCES incidents(id)
                )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents(incident_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_timestamp ON audio_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_glare_incident ON glare_images(incident_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_liveness_incident ON liveness_validations(incident_id)')
            
            conn.commit()
            conn.close()
    
    def get_connection(self):
        """Get thread-local database connection."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_incident_group_type(self, detection_type):
        """
        Determine the incident group type based on detection type.
        Physical tamper types grouped together, liveness threats grouped together.
        """
        if detection_type in PHYSICAL_TAMPER_TYPES:
            return 'PHYSICAL_TAMPER'
        elif detection_type in LIVENESS_THREAT_TYPES:
            return 'LIVENESS_THREAT'
        else:
            return 'UNKNOWN'
    
    def record_detection(self, detection_type, timestamp, description=None):
        """
        Record a detection, grouping with recent incidents of the same type.
        Returns the incident_id (newly created or existing).
        """
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            group_type = self.get_incident_group_type(detection_type)
            cutoff_time = timestamp - INCIDENT_GROUP_TIMEOUT
            
            # Check if there's a recent incident of the same group type
            cursor.execute('''
                SELECT id, timestamp, count FROM incidents
                WHERE incident_type = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (group_type, cutoff_time))
            
            existing_incident = cursor.fetchone()
            
            if existing_incident:
                # Update existing incident
                incident_id = existing_incident['id']
                new_count = existing_incident['count'] + 1
                
                cursor.execute('''
                    UPDATE incidents
                    SET count = ?, timestamp = ?, description = ?
                    WHERE id = ?
                ''', (new_count, timestamp, description or existing_incident['description'], incident_id))
                
                conn.commit()
            else:
                # Create new incident
                cursor.execute('''
                    INSERT INTO incidents (incident_type, primary_detection, timestamp, count, description)
                    VALUES (?, ?, ?, 1, ?)
                ''', (group_type, detection_type, timestamp, description))
                
                conn.commit()
                incident_id = cursor.lastrowid
            
            # Cleanup old incidents (keep only last MAX_INCIDENTS_RETAINED)
            self._cleanup_old_incidents(conn, cursor)
            
            conn.close()
            return incident_id
    
    def _cleanup_old_incidents(self, conn, cursor):
        """Remove incidents older than the last MAX_INCIDENTS_RETAINED."""
        cursor.execute('''
            SELECT id FROM incidents
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (1, MAX_INCIDENTS_RETAINED - 1))
        
        result = cursor.fetchone()
        if result:
            cutoff_id = result['id']
            
            # Delete old incidents and cascade delete related data
            cursor.execute('DELETE FROM audio_logs WHERE incident_id NOT IN (SELECT id FROM incidents)')
            cursor.execute('DELETE FROM glare_images WHERE incident_id NOT IN (SELECT id FROM incidents)')
            cursor.execute('DELETE FROM liveness_validations WHERE incident_id NOT IN (SELECT id FROM incidents)')
            
            conn.commit()
    
    def add_audio_log(self, text, timestamp, incident_id=None):
        """
        Add an audio transcript to the audio log.
        Auto-finds closest incident if incident_id not provided.
        """
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # If no incident_id, find the closest incident by timestamp
            if incident_id is None:
                cursor.execute('''
                    SELECT id FROM incidents
                    ORDER BY ABS(timestamp - ?) ASC
                    LIMIT 1
                ''', (timestamp,))
                
                closest_incident = cursor.fetchone()
                incident_id = closest_incident['id'] if closest_incident else None
            
            cursor.execute('''
                INSERT INTO audio_logs (incident_id, text, timestamp)
                VALUES (?, ?, ?)
            ''', (incident_id, text, timestamp))
            
            conn.commit()
            conn.close()
    
    def add_glare_image(self, file_path, glare_percentage, timestamp, incident_id=None):
        """
        Add a glare image record to the database.
        Auto-finds incident if incident_id not provided.
        """
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # If no incident_id, find the physical tamper incident
            if incident_id is None:
                cursor.execute('''
                    SELECT id FROM incidents
                    WHERE incident_type = 'PHYSICAL_TAMPER'
                    ORDER BY ABS(timestamp - ?) ASC
                    LIMIT 1
                ''', (timestamp,))
                
                closest_incident = cursor.fetchone()
                incident_id = closest_incident['id'] if closest_incident else None
            
            cursor.execute('''
                INSERT INTO glare_images (incident_id, file_path, glare_percentage, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (incident_id, file_path, glare_percentage, timestamp))
            
            conn.commit()
            conn.close()
    
    def add_liveness_validation(self, file_path, validation_status, frame_results, timestamp, incident_id=None):
        """
        Add a liveness video validation result to the database.
        frame_results should be a dict or JSON string of per-frame validation results.
        """
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # If no incident_id, find the liveness threat incident
            if incident_id is None:
                cursor.execute('''
                    SELECT id FROM incidents
                    WHERE incident_type = 'LIVENESS_THREAT'
                    ORDER BY ABS(timestamp - ?) ASC
                    LIMIT 1
                ''', (timestamp,))
                
                closest_incident = cursor.fetchone()
                incident_id = closest_incident['id'] if closest_incident else None
            
            # Convert frame_results to JSON if it's a dict
            if isinstance(frame_results, dict):
                frame_results_json = json.dumps(frame_results)
            else:
                frame_results_json = frame_results
            
            cursor.execute('''
                INSERT INTO liveness_validations (incident_id, file_path, validation_status, frame_results, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (incident_id, file_path, validation_status, frame_results_json, timestamp))
            
            conn.commit()
            conn.close()
    
    def get_recent_incidents(self, limit=5):
        """Retrieve the last N incidents with all related data."""
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM incidents
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            incidents = cursor.fetchall()
            result = []
            
            for incident in incidents:
                # Get audio logs for this incident
                cursor.execute('''
                    SELECT text, timestamp FROM audio_logs
                    WHERE incident_id = ?
                    ORDER BY timestamp ASC
                ''', (incident['id'],))
                
                audio_logs = cursor.fetchall()
                
                # Get glare images for this incident
                cursor.execute('''
                    SELECT * FROM glare_images
                    WHERE incident_id = ?
                    ORDER BY timestamp ASC
                ''', (incident['id'],))
                
                glare_images = cursor.fetchall()
                
                # Get liveness validations for this incident
                cursor.execute('''
                    SELECT * FROM liveness_validations
                    WHERE incident_id = ?
                    ORDER BY timestamp ASC
                ''', (incident['id'],))
                
                liveness_validations = cursor.fetchall()
                
                result.append({
                    'incident': dict(incident),
                    'audio_logs': [dict(log) for log in audio_logs],
                    'glare_images': [dict(img) for img in glare_images],
                    'liveness_validations': [dict(val) for val in liveness_validations]
                })
            
            conn.close()
            return result
    
    def get_incident_by_id(self, incident_id):
        """Retrieve a specific incident with all related data."""
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM incidents WHERE id = ?', (incident_id,))
            incident = cursor.fetchone()
            
            if not incident:
                conn.close()
                return None
            
            # Get audio logs
            cursor.execute('''
                SELECT text, timestamp FROM audio_logs
                WHERE incident_id = ?
                ORDER BY timestamp ASC
            ''', (incident_id,))
            
            audio_logs = cursor.fetchall()
            
            # Get glare images
            cursor.execute('''
                SELECT * FROM glare_images
                WHERE incident_id = ?
                ORDER BY timestamp ASC
            ''', (incident_id,))
            
            glare_images = cursor.fetchall()
            
            # Get liveness validations
            cursor.execute('''
                SELECT * FROM liveness_validations
                WHERE incident_id = ?
                ORDER BY timestamp ASC
            ''', (incident_id,))
            
            liveness_validations = cursor.fetchall()
            
            conn.close()
            
            return {
                'incident': dict(incident),
                'audio_logs': [dict(log) for log in audio_logs],
                'glare_images': [dict(img) for img in glare_images],
                'liveness_validations': [dict(val) for val in liveness_validations]
            }
    
    def get_audio_logs_for_incident(self, incident_id):
        """Retrieve audio logs for a specific incident."""
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, text, timestamp FROM audio_logs
                WHERE incident_id = ?
                ORDER BY timestamp ASC
            ''', (incident_id,))
            
            logs = cursor.fetchall()
            conn.close()
            
            return [dict(log) for log in logs]
    
    def get_glare_image_path(self, image_id):
        """Retrieve the file path for a glare image."""
        with db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT file_path FROM glare_images WHERE id = ?', (image_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result['file_path'] if result else None


# Global database instance
aegis_db = AegisDatabase()


def save_glare_image(frame, glare_percentage, timestamp):
    """
    Save a glare-rescued frame as JPEG to storage and return the file path.
    """
    import cv2
    
    # Create filename with timestamp
    filename = f"glare_{int(timestamp * 1000)}.jpg"
    file_path = os.path.join(GLARE_IMAGES_DIR, filename)
    
    # Save as JPEG
    cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return file_path


def get_incident_description(detection_type):
    """Generate a description for a detection based on type."""
    descriptions = {
        'blur': 'Camera obscured or blurry feed detected',
        'shake': 'Camera vibration or shake detected',
        'glare': 'Glare or bright light detected',
        'reposition': 'Camera repositioning detected',
        'freeze': 'Frozen or static feed detected',
        'blackout': 'Camera blackout or cover detected',
        'major_tamper': 'Major scene change detected'
    }
    return descriptions.get(detection_type, f'{detection_type} detected')
