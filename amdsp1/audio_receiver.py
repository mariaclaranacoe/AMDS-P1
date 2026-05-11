#!/usr/bin/env python3
"""
Audio File Receiver for ESP32 Recordings
Saves WAV files to /home/teasis/mosquito_listener/data/mosquito_recordings/
"""

from flask import Flask, request, jsonify
import os
import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import signal
import sys

# Configuration - UPDATED WITH YOUR NEW PATH
RECORDINGS_PATH = "/home/teasis/mosquito_listener/data/mosquito_recordings/"
HOST = '0.0.0.0'
PORT = 5000
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

# Create Flask app
app = Flask(__name__)

# Setup logging
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = '/home/teasis/mosquito_listener/receiver.log'

    # Rotate log files (keep 5 backups, 1MB each)
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

    return app.logger

logger = setup_logging()

# Ensure recordings directory exists
def ensure_directory():
    """Create the recordings directory if it doesn't exist"""
    if not os.path.exists(RECORDINGS_PATH):
        os.makedirs(RECORDINGS_PATH, mode=0o755, exist_ok=True)
        logger.info(f"Created directory: {RECORDINGS_PATH}")
    else:
        logger.debug(f"Directory already exists: {RECORDINGS_PATH}")

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """
    Endpoint to receive audio files from ESP32
    Expects raw WAV data in POST body
    """
    start_time = time.time()

    try:
        # Check if file was sent
        if not request.data:
            logger.warning("No data received")
            return "No data received", 400

        # Check file size
        content_length = request.content_length
        if content_length and content_length > MAX_FILE_SIZE:
            logger.warning(f"File too large: {content_length} bytes")
            return "File too large", 400

        # Ensure directory exists before saving
        ensure_directory()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mosquito_{timestamp}.wav"
        filepath = os.path.join(RECORDINGS_PATH, filename)

        # Save the audio data
        with open(filepath, 'wb') as f:
            f.write(request.data)

        # Get file size for logging
        file_size = os.path.getsize(filepath)

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # in milliseconds

        logger.info(f"✓ Saved: {filename} ({file_size} bytes) in {process_time:.1f}ms")
        logger.info(f"  Full path: {filepath}")

        return jsonify({
            "status": "success",
            "filename": filename,
            "filepath": filepath,
            "size": file_size,
            "message": f"File saved as {filename}"
        }), 200

    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    try:
        # Count recordings
        recording_count = 0
        if os.path.exists(RECORDINGS_PATH):
            recording_count = len([f for f in os.listdir(RECORDINGS_PATH) if f.endswith('.wav')])

        return jsonify({
            "status": "healthy",
            "recordings_path": RECORDINGS_PATH,
            "recordings_count": recording_count,
            "disk_usage": get_disk_usage()
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/recordings', methods=['GET'])
def list_recordings():
    """List all recordings with details"""
    try:
        if not os.path.exists(RECORDINGS_PATH):
            return jsonify({"recordings": []}), 200

        files = []
        for f in os.listdir(RECORDINGS_PATH):
            if f.endswith('.wav'):
                filepath = os.path.join(RECORDINGS_PATH, f)
                stat = os.stat(filepath)
                files.append({
                    "name": f,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })

        # Sort by modified time, newest first
        files.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            "count": len(files),
            "path": RECORDINGS_PATH,
            "recordings": files
        }), 200

    except Exception as e:
        logger.error(f"Error listing recordings: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_disk_usage():
    """Get disk usage information for the recordings directory"""
    try:
        if os.path.exists(RECORDINGS_PATH):
            stat = os.statvfs(RECORDINGS_PATH)
            total = stat.f_frsize * stat.f_blocks
            free = stat.f_frsize * stat.f_bfree
            used = total - free
            return {
                "total_bytes": total,
                "free_bytes": free,
                "used_bytes": used,
                "usage_percent": (used / total) * 100 if total > 0 else 0
            }
    except:
        pass
    return {}

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Shutting down...")
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Ensure directory exists
    ensure_directory()

    # Print startup banner
    logger.info("="*60)
    logger.info("Audio Receiver Started")
    logger.info(f"Host: {HOST}:{PORT}")
    logger.info(f"Saving to: {RECORDINGS_PATH}")
    logger.info("="*60)

    # Run Flask app
    app.run(host=HOST, port=PORT, debug=False, threaded=True)