#!/usr/bin/env python3 (FINAL)
"""
Mosquito gender classifier - WITH INFLUXDB SUPPORT
Uses MosquitoSong+ style preprocessing:
- mono audio
- resample to 8 kHz
- 300 ms windows
- 150 ms overlap
- average probabilities across windows

Sends results to InfluxDB for Grafana visualization
Resets count every day at 7:07 AM for 24-hour running total
Saves ONLY ONE final 24-hour summary file at reset time
"""

import os
import glob
import shutil
import json
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# IMPORT CHECKS
# ============================================

try:
    from influxdb import InfluxDBClient
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False
    print("InfluxDB library not available. Install with: pip install influxdb")

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available")
    exit(1)

# ============================================
# CONFIGURATION
# ============================================

INFLUX_CONFIG = {
    'host': '192.168.0.24',
    'port': 8086,
    'database': 'mosquito_db',
    'enabled': True,
    'username': None,
    'password': None
}

BASE_DIR = "/home/teasis/mosquito_listener"
MODEL_DIR = os.path.join(BASE_DIR, "models/")
AUDIO_DIR = os.path.join(BASE_DIR, "data/mosquito_recordings/")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed_recordings/")
RESULTS_DIR = os.path.join(BASE_DIR, "classification_results/")

COUNT_FILE = os.path.join(BASE_DIR, "mosquito_24h_count.json")

RESET_HOUR = 7
RESET_MINUTE = 7

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# MODEL + PREPROCESSING SETTINGS
# ============================================

MODEL_NAME = "cnn_model_min_val_round10.h5"

TARGET_SR = 8000
WINDOW_MS = 300
HOP_MS = 150

WINDOW_SAMPLES = int(TARGET_SR * WINDOW_MS / 1000)   # 2400
HOP_SAMPLES = int(TARGET_SR * HOP_MS / 1000)         # 1200

FEMALE_CLASSES = [0, 2, 4, 6]
MALE_CLASSES   = [1, 3, 5, 7]

# ============================================
# FUNCTION: send_to_influxdb()
# ============================================
def send_to_influxdb(stats):
    if not INFLUX_AVAILABLE or not INFLUX_CONFIG['enabled']:
        return False

    try:
        client = InfluxDBClient(
            host=INFLUX_CONFIG['host'],
            port=INFLUX_CONFIG['port'],
            database=INFLUX_CONFIG['database']
        )

        json_body = [{
            "measurement": "mosquito_classification",
            "time": datetime.utcnow().isoformat(),
            "fields": {
                "total_files": stats['total'],
                "female_count": stats['female'],
                "male_count": stats['male'],
                "unknown_count": stats.get('unknown', 0),
                "error_count": stats.get('error', 0),
                "female_percentage": round(stats['female_pct'], 2),
                "male_percentage": round(stats['male_pct'], 2)
            },
            "tags": {
                "model": stats['model'],
                "device": "raspberry_pi_4",
                "location": "mosquito_listener"
            }
        }]

        client.write_points(json_body)
        return True

    except Exception as e:
        print(f"✗ Failed to send to InfluxDB: {e}")
        return False

# ============================================
# FUNCTION: save_final_summary_to_file()
# PURPOSE: Save ONLY the full 24-hour final summary at reset time
# ============================================
def save_final_summary_to_file(stats):
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(RESULTS_DIR, f"summary_{current_date}_final.txt")

        next_reset = datetime.now() + timedelta(days=1)
        next_reset_str = next_reset.strftime('%Y-%m-%d') + f" {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM"

        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MOSQUITO CLASSIFICATION FINAL 24-HOUR SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Summary type: FINAL 24-HOUR SUMMARY\n")
            f.write(f"This summary covers: Previous 24 hours (until {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM)\n")
            f.write(f"Next reset will be: {next_reset_str}\n")
            f.write("-" * 60 + "\n\n")

            f.write("24-HOUR TOTALS:\n")
            f.write(f"   Total mosquitoes: {stats['total']}\n")
            f.write(f"   Females: {stats['female']} ({stats['female_pct']:.1f}%)\n")
            f.write(f"   Males: {stats['male']} ({stats['male_pct']:.1f}%)\n")

            if stats.get('unknown', 0) > 0:
                unknown_pct = (stats['unknown'] / stats['total']) * 100 if stats['total'] > 0 else 0
                f.write(f"   Unknown: {stats['unknown']} ({unknown_pct:.1f}%)\n")

            if stats.get('error', 0) > 0:
                error_pct = (stats['error'] / stats['total']) * 100 if stats['total'] > 0 else 0
                f.write(f"   Errors: {stats['error']} ({error_pct:.1f}%)\n")

            f.write("\n" + "-" * 60 + "\n")
            f.write(f"Model used: {stats['model']}\n")
            f.write("=" * 60 + "\n")

        print(f"Final 24-hour summary saved to: {filename}")
        return True

    except Exception as e:
        print(f"Warning: Could not save final summary file: {e}")
        return False

# ============================================
# FUNCTION: preprocess_audio_windows()
# ============================================
def preprocess_audio_windows(audio_path):
    try:
        import librosa

        y, sr = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True,
            res_type='kaiser_fast'
        )

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)), mode='constant')

        windows = []
        for start in range(0, len(y) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
            segment = y[start:start + WINDOW_SAMPLES]
            segment = segment.reshape(-1, 1).astype(np.float32)
            windows.append(segment)

        if not windows:
            return None

        return np.array(windows, dtype=np.float32)

    except Exception as e:
        print(f"Error preprocessing {audio_path}: {e}")
        return None

# ============================================
# FUNCTION: classify_with_model()
# No double prediction
# ============================================
def classify_with_model(audio_path, model):
    try:
        windows = preprocess_audio_windows(audio_path)
        if windows is None:
            return "ERROR", None, 0, None

        predictions = model.predict(windows, verbose=0)
        avg_pred = np.mean(predictions, axis=0)

        predicted_class = int(np.argmax(avg_pred))
        confidence = float(np.max(avg_pred))
        window_count = len(windows)

        if predicted_class in FEMALE_CLASSES:
            gender = "FEMALE"
        elif predicted_class in MALE_CLASSES:
            gender = "MALE"
        else:
            gender = "UNKNOWN"

        return gender, confidence, window_count, predicted_class

    except Exception as e:
        print(f"Classification error for {audio_path}: {e}")
        return "ERROR", None, 0, None

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    print("\n" + "=" * 50)
    print("MOSQUITO GENDER CLASSIFIER")
    print("WITH INFLUXDB SUPPORT - 24H RUNNING TOTAL")
    print(f"MODEL: {MODEL_NAME}")
    print(f"PREPROCESSING: {TARGET_SR} Hz, {WINDOW_MS} ms windows, {HOP_MS} ms hop")
    print(f"RESETS DAILY AT {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM")
    print(f"FINAL SUMMARY DIRECTORY: {RESULTS_DIR}")
    print("=" * 50)

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now()
    print(f"Run started at: {run_time}")
    print("-" * 50)

    running_female = 0
    running_male = 0
    running_unknown = 0
    running_error = 0
    current_date = str(datetime.now().date())

    should_reset = False
    old_totals = None

    try:
        with open(COUNT_FILE, 'r') as f:
            saved_data = json.load(f)

            last_update_str = saved_data.get('last_update', '2000-01-01T00:00:00')
            last_update = datetime.fromisoformat(last_update_str)

            today_reset = datetime.now().replace(
                hour=RESET_HOUR,
                minute=RESET_MINUTE,
                second=0,
                microsecond=0
            )

            if last_update < today_reset and current_time >= today_reset:
                should_reset = True
                old_totals = {
                    'total': saved_data.get('female_total', 0) + saved_data.get('male_total', 0) +
                             saved_data.get('unknown_total', 0) + saved_data.get('error_total', 0),
                    'female': saved_data.get('female_total', 0),
                    'male': saved_data.get('male_total', 0),
                    'unknown': saved_data.get('unknown_total', 0),
                    'error': saved_data.get('error_total', 0)
                }
                print(f"Reset time reached ({RESET_HOUR:02d}:{RESET_MINUTE:02d} AM) - saving final 24-hour summary and starting new count")
            else:
                running_female = saved_data.get('female_total', 0)
                running_male = saved_data.get('male_total', 0)
                running_unknown = saved_data.get('unknown_total', 0)
                running_error = saved_data.get('error_total', 0)
                print(f"Loaded running totals: {running_female + running_male + running_unknown + running_error} total so far")
                print(f"   Last update: {last_update_str}")
                print(f"   Next reset: {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM")

    except FileNotFoundError:
        print("No previous totals found - starting fresh")
    except Exception as e:
        print(f"Warning: Error loading totals: {e}")

    if should_reset and old_totals:
        old_totals['model'] = MODEL_NAME
        old_totals['female_pct'] = (old_totals['female'] / old_totals['total'] * 100) if old_totals['total'] > 0 else 0
        old_totals['male_pct'] = (old_totals['male'] / old_totals['total'] * 100) if old_totals['total'] > 0 else 0
        save_final_summary_to_file(old_totals)

        running_female = 0
        running_male = 0
        running_unknown = 0
        running_error = 0
        print("Count has been reset to 0 for new 24-hour period")

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model_name = MODEL_NAME

    print(f"Using model: {model_name}")

    try:
        model = load_model(model_path, compile=False)
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, "*.flac")))

    if not audio_files:
        print("No audio files found!")
        return

    total_files = len(audio_files)
    print(f"\nProcessing {total_files} new files...")

    female_count = 0
    male_count = 0
    unknown_count = 0
    error_count = 0

    class_counts = {i: 0 for i in range(8)}
    confidence_values = []

    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total_files}")

        gender, confidence, window_count, predicted_class = classify_with_model(audio_file, model)

        if confidence is not None:
            confidence_values.append(confidence)

        if predicted_class is not None and predicted_class in class_counts:
            class_counts[predicted_class] += 1

        if gender == "FEMALE":
            female_count += 1
        elif gender == "MALE":
            male_count += 1
        elif gender == "UNKNOWN":
            unknown_count += 1
        else:
            error_count += 1

        try:
            shutil.move(audio_file, os.path.join(PROCESSED_DIR, os.path.basename(audio_file)))
        except Exception as e:
            print(f"Could not move file {audio_file}: {e}")

    total_female_24h = running_female + female_count
    total_male_24h = running_male + male_count
    total_unknown_24h = running_unknown + unknown_count
    total_error_24h = running_error + error_count
    total_all_24h = total_female_24h + total_male_24h + total_unknown_24h + total_error_24h

    female_pct_run = (female_count / total_files) * 100 if total_files > 0 else 0
    male_pct_run = (male_count / total_files) * 100 if total_files > 0 else 0

    female_pct_24h = (total_female_24h / total_all_24h) * 100 if total_all_24h > 0 else 0
    male_pct_24h = (total_male_24h / total_all_24h) * 100 if total_all_24h > 0 else 0

    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"New recordings this run: {total_files}")
    print(f"Female (this run): {female_count} ({female_pct_run:.1f}%)")
    print(f"Male (this run):   {male_count} ({male_pct_run:.1f}%)")

    print("\n--- 24-HOUR RUNNING TOTALS ---")
    print(f"Total last 24h: {total_all_24h}")
    print(f"Females last 24h: {total_female_24h} ({female_pct_24h:.1f}%)")
    print(f"Males last 24h:   {total_male_24h} ({male_pct_24h:.1f}%)")

    if unknown_count > 0:
        unknown_pct_run = (unknown_count / total_files) * 100 if total_files > 0 else 0
        unknown_pct_24h = (total_unknown_24h / total_all_24h) * 100 if total_all_24h > 0 else 0
        print(f"\nUnknown this run: {unknown_count} ({unknown_pct_run:.1f}%)")
        print(f"Unknown last 24h: {total_unknown_24h} ({unknown_pct_24h:.1f}%)")

    if error_count > 0:
        error_pct_run = (error_count / total_files) * 100 if total_files > 0 else 0
        error_pct_24h = (total_error_24h / total_all_24h) * 100 if total_all_24h > 0 else 0
        print(f"Errors this run: {error_count} ({error_pct_run:.1f}%)")
        print(f"Errors last 24h: {total_error_24h} ({error_pct_24h:.1f}%)")

    if confidence_values:
        print(f"\nAverage confidence this run: {np.mean(confidence_values):.3f}")

    print("\nClass distribution this run:")
    for cls, count in class_counts.items():
        if count > 0:
            print(f"  Class {cls}: {count}")

    print("=" * 50)

    try:
        with open(COUNT_FILE, 'w') as f:
            json.dump({
                'date': current_date,
                'female_total': total_female_24h,
                'male_total': total_male_24h,
                'unknown_total': total_unknown_24h,
                'error_total': total_error_24h,
                'last_update': datetime.now().isoformat(),
                'reset_time': f"{RESET_HOUR:02d}:{RESET_MINUTE:02d}",
                'reset_note': f"Count resets daily at {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM"
            }, f)
        print(f"Saved 24h running totals to {COUNT_FILE}")
    except Exception as e:
        print(f"Warning: Could not save running totals: {e}")

    if INFLUX_CONFIG['enabled']:
        stats = {
            'total': total_all_24h,
            'female': total_female_24h,
            'male': total_male_24h,
            'unknown': total_unknown_24h,
            'error': total_error_24h,
            'female_pct': female_pct_24h,
            'male_pct': male_pct_24h,
            'model': model_name
        }

        print("\nSending 24-hour running totals to InfluxDB...")
        if send_to_influxdb(stats):
            print("Data sent to InfluxDB successfully!")
            print(f"  Database: {INFLUX_CONFIG['database']}")
            print(f"  Total mosquitoes last 24h: {total_all_24h}")
            print(f"  (Count resets at {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM)")
        else:
            print("Failed to send to InfluxDB")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3 (DRAFT)
"""
Mosquito gender classifier - WITH INFLUXDB SUPPORT
Sends results to InfluxDB for Grafana visualization
Resets count every day at 7:07 AM for 24-hour running total
Saves daily summary to classification_results folder


import os
import glob
import shutil
import json
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# IMPORT CHECKS - These try/except blocks check if required libraries are installed
# ============================================

# InfluxDB imports - Checks if InfluxDB library is available for database connection
try:
    from influxdb import InfluxDBClient
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False
    print("InfluxDB library not available. Install with: pip install influxdb")

# TensorFlow imports - Checks if TensorFlow/AI library is available
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow warning messages
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available")
    exit(1)

# ============================================
# CONFIGURATION SECTION - All user-adjustable settings
# ============================================

INFLUX_CONFIG = {
    'host': '192.168.0.24',      # IP address of your InfluxDB server
    'port': 8086,                 # Default port for InfluxDB
    'database': 'mosquito_db',    # Database name for storing results
    'enabled': True,              # Toggle InfluxDB on/off
    'username': None,             # Login if needed
    'password': None
}

BASE_DIR = "/home/teasis/mosquito_listener"           # Main project folder
MODEL_DIR = os.path.join(BASE_DIR, "models/")         # Where AI models are stored
AUDIO_DIR = os.path.join(BASE_DIR, "data/mosquito_recordings/")     # New recordings go here
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed_recordings/") # Files after processing
RESULTS_DIR = os.path.join(BASE_DIR, "classification_results/")     # Where summaries are saved

# File to store 24-hour running totals (so counts persist between script runs)
COUNT_FILE = os.path.join(BASE_DIR, "mosquito_24h_count.json")

# Reset time - count resets every day at 7:07 AM
RESET_HOUR = 7
RESET_MINUTE = 7

# Create folders if not exists - prevents errors if folders missing
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class mapping - Which AI output numbers mean female vs male
# The AI model outputs numbers 0-7, these lists tell us which are which
FEMALE_CLASSES = [0, 2, 3, 6]    # These class numbers indicate FEMALE
MALE_CLASSES = [1, 4, 5, 7]       # These class numbers indicate MALE

# ============================================
# FUNCTION: send_to_influxdb()
# PURPOSE:  Sends classification data to InfluxDB for Grafana dashboards
# ============================================
def send_to_influxdb(stats):
    if not INFLUX_AVAILABLE or not INFLUX_CONFIG['enabled']:
        return False

    try:
        # Connect to InfluxDB server
        client = InfluxDBClient(
            host=INFLUX_CONFIG['host'],
            port=INFLUX_CONFIG['port'],
            database=INFLUX_CONFIG['database']
        )

        # Format data for InfluxDB (they use a specific JSON format)
        json_body = [{
            "measurement": "mosquito_classification",  # Table name in InfluxDB
            "time": datetime.utcnow().isoformat(),      # Timestamp
            "fields": {                                  # Actual numeric data
                "total_files": stats['total'],
                "female_count": stats['female'],
                "male_count": stats['male'],
                "unknown_count": stats.get('unknown', 0),
                "error_count": stats.get('error', 0),
                "female_percentage": round(stats['female_pct'], 2),
                "male_percentage": round(stats['male_pct'], 2)
            },
            "tags": {                                    # Labels for filtering in Grafana
                "model": stats['model'],
                "device": "raspberry_pi_4",
                "location": "mosquito_listener"
            }
        }]

        client.write_points(json_body)  # Send to database
        return True

    except Exception as e:
        print(f"✗ Failed to send to InfluxDB: {e}")
        return False


# ============================================
# FUNCTION: save_summary_to_file()
# PURPOSE:  Creates human-readable .txt summaries in classification_results folder
# ============================================
def save_summary_to_file(stats, is_reset=False):
    """Save the 24-hour summary to a text file in classification_results folder"""
    try:
        # Create filename with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if is_reset:
            # This is the reset summary (end of 24-hour period) - FINAL report
            filename = os.path.join(RESULTS_DIR, f"summary_{current_date}_final.txt")
            summary_type = "FINAL 24-HOUR SUMMARY"
        else:
            # This is an intermediate summary (during the day) - has timestamp
            current_time = datetime.now().strftime("%H-%M")
            filename = os.path.join(RESULTS_DIR, f"summary_{current_date}_{current_time}.txt")
            summary_type = "INTERMEDIATE SUMMARY"
        
        # Write formatted summary to file
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"MOSQUITO CLASSIFICATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Summary type: {summary_type}\n")
            
            if is_reset:
                next_reset = datetime.now() + timedelta(days=1)
                next_reset_str = next_reset.strftime('%Y-%m-%d') + f" {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM"
                f.write(f"This summary covers: Previous 24 hours (until reset time)\n")
                f.write(f"Next reset will be: {next_reset_str}\n")
            else:
                f.write(f"Counts will reset at: {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM daily\n")
            
            f.write("-" * 60 + "\n\n")
            
            # Write the actual numbers
            f.write("📊 24-HOUR RUNNING TOTALS:\n")
            f.write(f"   Total mosquitoes: {stats['total']}\n")
            f.write(f"   Females: {stats['female']} ({stats['female_pct']:.1f}%)\n")
            f.write(f"   Males: {stats['male']} ({stats['male_pct']:.1f}%)\n")
            
            # Only show unknown/errors if they exist (cleaner output)
            if stats.get('unknown', 0) > 0:
                unknown_pct = (stats['unknown'] / stats['total']) * 100 if stats['total'] > 0 else 0
                f.write(f"   Unknown: {stats['unknown']} ({unknown_pct:.1f}%)\n")
            
            if stats.get('error', 0) > 0:
                error_pct = (stats['error'] / stats['total']) * 100 if stats['total'] > 0 else 0
                f.write(f"   Errors: {stats['error']} ({error_pct:.1f}%)\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write(f"Model used: {stats['model']}\n")
            f.write("=" * 60 + "\n")
        
        print(f"📄 Summary saved to: {filename}")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not save summary file: {e}")
        return False


# ============================================
# FUNCTION: load_audio_for_model()
# PURPOSE:  Prepares audio files for the AI model (resizes, normalizes, formats)
# ============================================
def load_audio_for_model(audio_path, target_sr=22050, duration=2.0, expected_length=2400):
    try:
        import librosa
        # Load audio file at correct sample rate and duration
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration, res_type='kaiser_fast')

        # Make sure all audio files are the same length (model expects fixed size)
        if len(y) < expected_length:
            # If too short, pad with zeros
            y = np.pad(y, (0, expected_length - len(y)), mode='constant')
        else:
            # If too long, truncate
            y = y[:expected_length]

        # Normalize volume (scale to between -1 and 1)
        if np.abs(y).max() > 0:
            y = y / np.abs(y).max()

        # Reshape for the model (adds dimension for channels)
        y = y.reshape(-1, 1)
        return y

    except Exception:
        return None  # Return None if file can't be processed


# ============================================
# FUNCTION: classify_with_model()
# PURPOSE:  Runs one audio file through AI model and returns gender
# ============================================
def classify_with_model(audio_path, model):
    try:
        # Load and prepare audio
        audio_data = load_audio_for_model(audio_path)
        if audio_data is None:
            return "ERROR"

        # Add batch dimension (model expects multiple files at once)
        audio_data = np.expand_dims(audio_data, axis=0)
        
        # Run prediction (faster than predict() because it doesn't do full evaluation)
        predictions = model.predict_on_batch(audio_data)
        
        # Get the highest probability class
        predicted_class = np.argmax(predictions[0])

        # Convert class number to gender
        if predicted_class in FEMALE_CLASSES:
            return "FEMALE"
        elif predicted_class in MALE_CLASSES:
            return "MALE"
        else:
            return "UNKNOWN"

    except Exception:
        return "ERROR"


# ============================================
# MAIN FUNCTION - This is where the program starts
# ============================================
def main():
    print("\n" + "=" * 50)
    print("MOSQUITO GENDER CLASSIFIER")
    print("WITH INFLUXDB SUPPORT - 24H RUNNING TOTAL")
    print(f"RESETS DAILY AT {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM")
    print(f"SUMMARIES SAVED TO: {RESULTS_DIR}")
    print("=" * 50)

    # Record when this run started
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now()
    print(f"Run started at: {run_time}")
    print("-" * 50)

    # ============================================
    # LOAD PREVIOUS TOTALS & CHECK FOR RESET
    # This section reads the JSON file and decides if we need to reset counters
    # ============================================
    running_female = 0
    running_male = 0
    running_unknown = 0
    running_error = 0
    current_date = str(datetime.now().date())

    # Check if we should reset the count (at 7:07 AM)
    should_reset = False
    old_totals = None  # Store old totals for summary before reset

    try:
        with open(COUNT_FILE, 'r') as f:
            saved_data = json.load(f)

            # Get the last time we saved data
            last_update_str = saved_data.get('last_update', '2000-01-01')
            last_update = datetime.fromisoformat(last_update_str)

            # Create today's reset time (today at 7:07 AM)
            today_reset = datetime.now().replace(hour=RESET_HOUR, minute=RESET_MINUTE, second=0, microsecond=0)

            # RESET LOGIC: If last update was BEFORE today's reset AND current time is AFTER today's reset
            if last_update < today_reset and current_time >= today_reset:
                should_reset = True
                # Store old totals for summary before resetting
                old_totals = {
                    'total': saved_data.get('female_total', 0) + saved_data.get('male_total', 0) + 
                             saved_data.get('unknown_total', 0) + saved_data.get('error_total', 0),
                    'female': saved_data.get('female_total', 0),
                    'male': saved_data.get('male_total', 0),
                    'unknown': saved_data.get('unknown_total', 0),
                    'error': saved_data.get('error_total', 0)
                }
                print(f"⏰ Reset time reached ({RESET_HOUR:02d}:{RESET_MINUTE:02d} AM) - Starting new 24-hour count")
            else:
                # Load the running totals from JSON
                running_female = saved_data.get('female_total', 0)
                running_male = saved_data.get('male_total', 0)
                running_unknown = saved_data.get('unknown_total', 0)
                running_error = saved_data.get('error_total', 0)
                print(f"📊 Loaded running totals: {running_female + running_male + running_unknown + running_error} total so far")
                print(f"   Last update: {last_update_str}")
                print(f"   Next reset: {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM")

    except FileNotFoundError:
        print("📁 No previous totals found - starting fresh")
    except Exception as e:
        print(f"⚠️ Error loading totals: {e}")

    # If reset needed, save final summary and zero out counters
    if should_reset and old_totals:
        print("📊 Generating final summary for the completed 24-hour period...")
        old_totals['model'] = "cnn_model_min_val_round9.h5"
        old_totals['female_pct'] = (old_totals['female'] / old_totals['total'] * 100) if old_totals['total'] > 0 else 0
        old_totals['male_pct'] = (old_totals['male'] / old_totals['total'] * 100) if old_totals['total'] > 0 else 0
        save_summary_to_file(old_totals, is_reset=True)
        
        # Reset counts to 0
        running_female = 0
        running_male = 0
        running_unknown = 0
        running_error = 0
        print("🔄 Count has been reset to 0 for new 24-hour period")
    # =================================================

    # Load the AI model
    updated_model = "cnn_model_min_val_round9.h5"
    model_path = os.path.join(MODEL_DIR, updated_model)
    model_name = updated_model

    print(f"Using model: {model_name}")

    try:
        model = load_model(model_path, compile=False)  # compile=False because we only need predictions
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # ============================================
    # FIND ALL AUDIO FILES to process
    # ============================================
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, "*.flac")))

    if not audio_files:
        print("No audio files found!")
        return

    total_files = len(audio_files)
    print(f"\nProcessing {total_files} new files...")

    # Initialize counters for this batch
    female_count = 0
    male_count = 0
    unknown_count = 0
    error_count = 0

    # ============================================
    # PROCESS EACH FILE - Main loop
    # ============================================
    for i, audio_file in enumerate(audio_files):

        # Progress indicator every 100 files
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total_files}")

        # Classify the audio file
        gender = classify_with_model(audio_file, model)

        # Update counters based on result
        if gender == "FEMALE":
            female_count += 1
        elif gender == "MALE":
            male_count += 1
        elif gender == "UNKNOWN":
            unknown_count += 1
        else:
            error_count += 1

        # Move processed file to processed folder (so it won't be processed again)
        try:
            shutil.move(audio_file, os.path.join(PROCESSED_DIR, os.path.basename(audio_file)))
        except Exception as e:
            print(f"Could not move file {audio_file}: {e}")

    # ============================================
    # CALCULATE RUNNING TOTALS (add new counts to historical)
    # ============================================
    total_female_24h = running_female + female_count
    total_male_24h = running_male + male_count
    total_unknown_24h = running_unknown + unknown_count
    total_error_24h = running_error + error_count
    total_all_24h = total_female_24h + total_male_24h + total_unknown_24h + total_error_24h
    # ============================================

    # Calculate percentages
    female_pct_hour = (female_count / total_files) * 100 if total_files > 0 else 0
    male_pct_hour = (male_count / total_files) * 100 if total_files > 0 else 0

    female_pct_24h = (total_female_24h / total_all_24h) * 100 if total_all_24h > 0 else 0
    male_pct_24h = (total_male_24h / total_all_24h) * 100 if total_all_24h > 0 else 0

    # ============================================
    # PRINT RESULTS TO CONSOLE
    # ============================================
    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"New recordings this hour: {total_files}")
    print(f"Female (this hour): {female_count} ({female_pct_hour:.1f}%)")
    print(f"Male (this hour):   {male_count} ({male_pct_hour:.1f}%)")

    print("\n--- 24-HOUR RUNNING TOTALS ---")
    print(f"Total last 24h: {total_all_24h}")
    print(f"Females last 24h: {total_female_24h} ({female_pct_24h:.1f}%)")
    print(f"Males last 24h:   {total_male_24h} ({male_pct_24h:.1f}%)")

    if unknown_count > 0:
        unknown_pct_hour = (unknown_count / total_files) * 100
        unknown_pct_24h = (total_unknown_24h / total_all_24h) * 100 if total_all_24h > 0 else 0
        print(f"\nUnknown this hour: {unknown_count} ({unknown_pct_hour:.1f}%)")
        print(f"Unknown last 24h: {total_unknown_24h} ({unknown_pct_24h:.1f}%)")

    if error_count > 0:
        error_pct_hour = (error_count / total_files) * 100
        error_pct_24h = (total_error_24h / total_all_24h) * 100 if total_all_24h > 0 else 0
        print(f"Errors this hour: {error_count} ({error_pct_hour:.1f}%)")
        print(f"Errors last 24h: {total_error_24h} ({error_pct_24h:.1f}%)")

    print("=" * 50)

    # ============================================
    # SAVE UPDATED TOTALS TO JSON FILE (for next run)
    # ============================================
    try:
        with open(COUNT_FILE, 'w') as f:
            json.dump({
                'date': current_date,
                'female_total': total_female_24h,
                'male_total': total_male_24h,
                'unknown_total': total_unknown_24h,
                'error_total': total_error_24h,
                'last_update': datetime.now().isoformat(),
                'reset_time': f"{RESET_HOUR:02d}:{RESET_MINUTE:02d}",
                'reset_note': f"Count resets daily at {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM"
            }, f)
        print(f"💾 Saved 24h running totals to {COUNT_FILE}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save running totals: {e}")
    # ============================================

    # ============================================
    # SAVE SUMMARY TO TEXT FILE
    # ============================================
    current_stats = {
        'total': total_all_24h,
        'female': total_female_24h,
        'male': total_male_24h,
        'unknown': total_unknown_24h,
        'error': total_error_24h,
        'female_pct': female_pct_24h,
        'male_pct': male_pct_24h,
        'model': model_name
    }
    save_summary_to_file(current_stats, is_reset=False)
    # ===================================================

    # ============================================
    # SEND TO INFLUXDB
    # ============================================
    if INFLUX_CONFIG['enabled']:
        # Send 24-hour totals to InfluxDB
        stats = {
            'total': total_all_24h,
            'female': total_female_24h,
            'male': total_male_24h,
            'unknown': total_unknown_24h,
            'error': total_error_24h,
            'female_pct': female_pct_24h,
            'male_pct': male_pct_24h,
            'model': model_name
        }

        print("\nSending 24-hour running totals to InfluxDB...")
        if send_to_influxdb(stats):
            print("✓ Data sent to InfluxDB successfully!")
            print(f"  Database: {INFLUX_CONFIG['database']}")
            print(f"  Total mosquitoes last 24h: {total_all_24h}")
            print(f"  (Count resets at {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM)")
        else:
            print("✗ Failed to send to InfluxDB")


if __name__ == "__main__":
    main()
"""
