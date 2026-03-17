#!/usr/bin/env python3
"""
Mosquito gender classifier
Updated architecture:
- Official total mosquito count comes from ESP32 break-beam sensors via MQTT
- This script only classifies uploaded audio files
- Saves per-file classification results locally
- Maintains 24-hour gender statistics locally using TRUE 7:00 AM -> 7:00 AM periods
- Uses each file's recording timestamp (from filename if possible, otherwise file modified time)
- Sends data to InfluxDB for Grafana visualization
"""

import os
import re
import glob
import shutil
import json
import csv
import numpy as np
from datetime import datetime, timedelta
import warnings
import socket
warnings.filterwarnings('ignore')

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model
    import tensorflow as tf
except ImportError:
    print("TensorFlow not available")
    raise SystemExit(1)

# ============================================
# INFLUXDB IMPORTS
# ============================================
try:
    from influxdb import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    print("InfluxDB client not available. Install with: pip install influxdb")
    INFLUXDB_AVAILABLE = False

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = "/home/teasis/mosquito_listener"
MODEL_DIR = os.path.join(BASE_DIR, "models")
AUDIO_DIR = os.path.join(BASE_DIR, "data/mosquito_recordings")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed_recordings")
RESULTS_DIR = os.path.join(BASE_DIR, "classification_results")
STATS_DIR = os.path.join(RESULTS_DIR, "daily_gender_stats")

MODEL_NAME = "cnn_model_min_val_round10.h5"

# Backward-compatible "current stats" file
GENDER_COUNT_FILE = os.path.join(BASE_DIR, "gender_24h_stats.json")
RESULTS_CSV = os.path.join(RESULTS_DIR, "classification_log.csv")

RESET_HOUR = 7
RESET_MINUTE = 0

# ============================================
# INFLUXDB CONFIGURATION
# ============================================
INFLUXDB_HOST = "192.168.0.24"
INFLUXDB_PORT = 8086
INFLUXDB_DATABASE = "mosquito_db"
INFLUXDB_USER = "teasis"
INFLUXDB_PASSWORD = "teasis"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

# ============================================
# MODEL + PREPROCESSING SETTINGS
# ============================================

TARGET_SR = 8000
WINDOW_MS = 300
HOP_MS = 150

WINDOW_SAMPLES = int(TARGET_SR * WINDOW_MS / 1000)   # 2400
HOP_SAMPLES = int(TARGET_SR * HOP_MS / 1000)         # 1200

FEMALE_CLASSES = [0, 2, 4, 6]
MALE_CLASSES   = [1, 3, 5, 7]

# ============================================
# INFLUXDB INITIALIZATION
# ============================================

def init_influxdb():
    """Initialize InfluxDB client"""
    if not INFLUXDB_AVAILABLE:
        print("⚠️ InfluxDB client not available - install with: pip install influxdb")
        return None

    try:
        client = InfluxDBClient(
            host=INFLUXDB_HOST,
            port=INFLUXDB_PORT,
            username=INFLUXDB_USER,
            password=INFLUXDB_PASSWORD,
            database=INFLUXDB_DATABASE,
            timeout=5
        )

        client.ping()
        print(f"✅ Connected to InfluxDB at {INFLUXDB_HOST}:{INFLUXDB_PORT}")

        databases = [db['name'] for db in client.get_list_database()]
        if INFLUXDB_DATABASE not in databases:
            client.create_database(INFLUXDB_DATABASE)
            print(f"✅ Created database: {INFLUXDB_DATABASE}")

        return client
    except Exception as e:
        print(f"⚠️ InfluxDB connection failed: {e}")
        print("⚠️ Data will be saved locally only")
        return None

# ============================================
# PERIOD / TIME HELPERS
# ============================================

def get_period_start(dt):
    """
    Return the start of the 24-hour period that dt belongs to.
    Period boundary is 07:00 every day.
    """
    boundary_today = dt.replace(
        hour=RESET_HOUR,
        minute=RESET_MINUTE,
        second=0,
        microsecond=0
    )

    if dt >= boundary_today:
        return boundary_today
    return boundary_today - timedelta(days=1)

def get_period_end(period_start):
    return period_start + timedelta(days=1)

def get_period_label(period_start):
    period_end = get_period_end(period_start)
    return f"{period_start.strftime('%Y-%m-%d %H:%M')} -> {period_end.strftime('%Y-%m-%d %H:%M')}"

def get_stats_file_for_period(period_start):
    fname = f"gender_stats_{period_start.strftime('%Y%m%d_%H%M')}.json"
    return os.path.join(STATS_DIR, fname)

def make_default_stats(period_start):
    period_end = get_period_end(period_start)
    return {
        "female_total": 0,
        "male_total": 0,
        "unknown_total": 0,
        "error_total": 0,
        "last_update": datetime.now().isoformat(),
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "period_label": get_period_label(period_start),
        "reset_time": f"{RESET_HOUR:02d}:{RESET_MINUTE:02d}",
        "summary_sent": False
    }

def load_period_stats(period_start):
    stats_file = get_stats_file_for_period(period_start)
    default_stats = make_default_stats(period_start)

    if not os.path.exists(stats_file):
        return default_stats

    try:
        with open(stats_file, "r") as f:
            saved = json.load(f)

        # Make sure required keys exist
        for key, value in default_stats.items():
            if key not in saved:
                saved[key] = value

        return saved
    except Exception as e:
        print(f"Warning: could not load stats file {stats_file}: {e}")
        return default_stats

def save_period_stats(stats):
    stats["last_update"] = datetime.now().isoformat()
    period_start = datetime.fromisoformat(stats["period_start"])
    stats_file = get_stats_file_for_period(period_start)

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

def update_current_stats_alias(current_stats):
    """
    Keep a copy of the CURRENT period stats in the old path
    so other parts of your system still have a familiar file.
    """
    try:
        with open(GENDER_COUNT_FILE, "w") as f:
            json.dump(current_stats, f, indent=2)
    except Exception as e:
        print(f"Warning: could not update current stats alias: {e}")

def extract_recorded_datetime(audio_path):
    """
    Try to get recording timestamp from filename first.
    Expected pattern examples:
      mosquito_20260317_064512.wav
      anything_20260317_064512_anything.wav

    If not found, fall back to file modified time.
    """
    base_name = os.path.basename(audio_path)

    match = re.search(r'(\d{8})_(\d{6})', base_name)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        try:
            return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # Fallback: file modified time
    try:
        return datetime.fromtimestamp(os.path.getmtime(audio_path))
    except Exception:
        return datetime.now()

# ============================================
# INFLUXDB FUNCTIONS
# ============================================

def send_to_influxdb(client, measurement, fields, tags=None, timestamp=None):
    """Send data to InfluxDB"""
    if client is None:
        return False

    try:
        point_time = timestamp if timestamp else (datetime.utcnow().isoformat() + "Z")

        json_body = [
            {
                "measurement": measurement,
                "tags": tags or {},
                "fields": fields,
                "time": point_time
            }
        ]
        client.write_points(json_body)
        return True
    except Exception as e:
        print(f"⚠️ InfluxDB write failed: {e}")
        return False

def send_classification_result(client, filename, gender, confidence, predicted_class,
                               window_count, recorded_dt, period_start):
    """Send individual classification result to InfluxDB"""
    if client is None:
        return

    gender_value = 1.0 if gender == "FEMALE" else (0.0 if gender == "MALE" else 0.5)
    period_end = get_period_end(period_start)

    fields = {
        "gender_value": float(gender_value),
        "confidence": float(confidence) if confidence is not None else 0.0,
        "window_count": int(window_count),
        "predicted_class": int(predicted_class) if predicted_class is not None else -1
    }

    tags = {
        "gender": str(gender),
        "filename": str(filename),
        "model": str(MODEL_NAME),
        "host": str(socket.gethostname()),
        "period_start": period_start.strftime("%Y-%m-%d %H:%M:%S"),
        "period_end": period_end.strftime("%Y-%m-%d %H:%M:%S")
    }

    success = send_to_influxdb(
        client,
        "mosquito_classification",
        fields,
        tags,
        timestamp=recorded_dt.isoformat() + "Z"
    )
    if success:
        print("  📤 Sent classification to InfluxDB")

def send_gender_stats(client, stats, stat_type="cumulative"):
    """Send period statistics to InfluxDB"""
    if client is None:
        return

    fields = {
        "female_total": int(stats.get("female_total", 0)),
        "male_total": int(stats.get("male_total", 0)),
        "unknown_total": int(stats.get("unknown_total", 0)),
        "error_total": int(stats.get("error_total", 0)),
        "total_classified": int(
            stats.get("female_total", 0) +
            stats.get("male_total", 0) +
            stats.get("unknown_total", 0) +
            stats.get("error_total", 0)
        )
    }

    tags = {
        "period_start": str(stats.get("period_start", "")),
        "period_end": str(stats.get("period_end", "")),
        "type": stat_type
    }

    # For daily summary, use period_end as stable timestamp so resends overwrite the same period point
    if stat_type == "daily_summary":
        ts = datetime.fromisoformat(stats["period_end"]).isoformat() + "Z"
    else:
        ts = datetime.utcnow().isoformat() + "Z"

    success = send_to_influxdb(client, "mosquito_gender_stats", fields, tags, timestamp=ts)
    if success:
        if stat_type == "daily_summary":
            print("  📤 Sent daily summary to InfluxDB")
        else:
            print("  📤 Sent cumulative stats to InfluxDB")

# ============================================
# LOCAL FILE HELPERS
# ============================================

def ensure_csv_exists():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_processed",
                "timestamp_recorded",
                "period_start",
                "period_end",
                "filename",
                "gender",
                "confidence",
                "predicted_class",
                "window_count",
                "model"
            ])

def save_final_gender_summary(stats):
    try:
        period_start = datetime.fromisoformat(stats["period_start"])
        period_end = datetime.fromisoformat(stats["period_end"])

        filename = os.path.join(
            RESULTS_DIR,
            f"gender_summary_{period_start.strftime('%Y%m%d_%H%M')}_to_{period_end.strftime('%Y%m%d_%H%M')}_final.txt"
        )

        total = (
            stats["female_total"] +
            stats["male_total"] +
            stats["unknown_total"] +
            stats["error_total"]
        )

        female_pct = (stats["female_total"] / total * 100) if total > 0 else 0
        male_pct = (stats["male_total"] / total * 100) if total > 0 else 0
        unknown_pct = (stats["unknown_total"] / total * 100) if total > 0 else 0
        error_pct = (stats["error_total"] / total * 100) if total > 0 else 0

        with open(filename, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MOSQUITO GENDER CLASSIFICATION FINAL 24-HOUR SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period start: {period_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period end:   {period_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model used: {MODEL_NAME}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total classified files: {total}\n")
            f.write(f"Female:  {stats['female_total']} ({female_pct:.1f}%)\n")
            f.write(f"Male:    {stats['male_total']} ({male_pct:.1f}%)\n")
            f.write(f"Unknown: {stats['unknown_total']} ({unknown_pct:.1f}%)\n")
            f.write(f"Error:   {stats['error_total']} ({error_pct:.1f}%)\n")
            f.write("=" * 70 + "\n")

        print(f"Saved final gender summary: {filename}")

    except Exception as e:
        print(f"Warning: could not save final gender summary: {e}")

def append_result_to_csv(filename, gender, confidence, predicted_class,
                         window_count, model_name, recorded_dt, period_start):
    try:
        ensure_csv_exists()
        period_end = get_period_end(period_start)

        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                recorded_dt.strftime("%Y-%m-%d %H:%M:%S"),
                period_start.strftime("%Y-%m-%d %H:%M:%S"),
                period_end.strftime("%Y-%m-%d %H:%M:%S"),
                filename,
                gender,
                "" if confidence is None else round(float(confidence), 6),
                "" if predicted_class is None else int(predicted_class),
                int(window_count),
                model_name
            ])
    except Exception as e:
        print(f"Warning: could not append to CSV: {e}")

def save_result_json(filename, gender, confidence, predicted_class,
                     window_count, model_name, recorded_dt, period_start):
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        period_end = get_period_end(period_start)

        json_name = os.path.join(RESULTS_DIR, f"{stamp}_{filename}.json")
        payload = {
            "timestamp_processed": datetime.now().isoformat(),
            "timestamp_recorded": recorded_dt.isoformat(),
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "filename": filename,
            "gender": gender,
            "confidence": None if confidence is None else float(confidence),
            "predicted_class": None if predicted_class is None else int(predicted_class),
            "window_count": int(window_count),
            "model": model_name
        }
        with open(json_name, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save result JSON for {filename}: {e}")

# ============================================
# AUDIO PREPROCESSING
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
# CLASSIFICATION
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
# PERIOD FINALIZATION
# ============================================

def finalize_previous_period_if_needed(influx_client, current_period_start):
    """
    Finalize the period immediately before the current one.
    This is done AFTER processing files so that files recorded before 7:00
    but processed at 7:00 still get counted in the previous period.
    """
    previous_period_start = current_period_start - timedelta(days=1)
    previous_stats = load_period_stats(previous_period_start)

    total_previous = (
        previous_stats.get("female_total", 0) +
        previous_stats.get("male_total", 0) +
        previous_stats.get("unknown_total", 0) +
        previous_stats.get("error_total", 0)
    )

    # Nothing to finalize
    if total_previous == 0 and previous_stats.get("summary_sent", False):
        return

    # Write/rewrite summary file
    if total_previous > 0:
        save_final_gender_summary(previous_stats)

        if influx_client:
            send_gender_stats(influx_client, previous_stats, stat_type="daily_summary")

    previous_stats["summary_sent"] = True
    save_period_stats(previous_stats)

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "=" * 60)
    print("MOSQUITO GENDER CLASSIFIER")
    print("Classification-only mode")
    print("Official total count is handled by ESP32 break-beam sensors via MQTT")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Preprocessing: {TARGET_SR} Hz, {WINDOW_MS} ms windows, {HOP_MS} ms hop")
    print(f"Period boundary: daily at {RESET_HOUR:02d}:{RESET_MINUTE:02d} AM")
    print("Files are assigned to periods using RECORDING TIME, not processing time")
    print("-" * 60)

    influx_client = init_influxdb()
    if influx_client:
        print("✅ InfluxDB enabled - data will be sent to database")
    else:
        print("⚠️ InfluxDB disabled - local files only")
    print("-" * 60)

    now = datetime.now()
    current_period_start = get_period_start(now)
    current_period_end = get_period_end(current_period_start)

    print(f"Current period: {current_period_start.strftime('%Y-%m-%d %H:%M:%S')} "
          f"-> {current_period_end.strftime('%Y-%m-%d %H:%M:%S')}")

    current_stats = load_period_stats(current_period_start)
    update_current_stats_alias(current_stats)

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    try:
        model = load_model(model_path, compile=False)
        print(f"Using model: {MODEL_NAME}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, "*.flac")))
    audio_files = sorted(audio_files)

    if not audio_files:
        print("No audio files found.")

        # Still finalize previous period if needed
        finalize_previous_period_if_needed(influx_client, current_period_start)

        # Refresh current cumulative stats
        current_stats = load_period_stats(current_period_start)
        update_current_stats_alias(current_stats)

        if influx_client:
            send_gender_stats(influx_client, current_stats, stat_type="cumulative")
            influx_client.close()
            print("InfluxDB connection closed")
        return

    print(f"\nProcessing {len(audio_files)} uploaded audio file(s)...")

    run_female = 0
    run_male = 0
    run_unknown = 0
    run_error = 0
    class_counts = {i: 0 for i in range(8)}
    confidence_values = []

    # Cache stats objects by period_start ISO string
    touched_stats = {}

    def get_cached_stats(period_start):
        key = period_start.isoformat()
        if key not in touched_stats:
            touched_stats[key] = load_period_stats(period_start)
        return touched_stats[key]

    for idx, audio_file in enumerate(audio_files, start=1):
        base_name = os.path.basename(audio_file)
        recorded_dt = extract_recorded_datetime(audio_file)
        file_period_start = get_period_start(recorded_dt)
        file_period_end = get_period_end(file_period_start)

        print(f"\n[{idx}/{len(audio_files)}] Processing: {base_name}")
        print(f"  Recorded time: {recorded_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Assigned period: {file_period_start.strftime('%Y-%m-%d %H:%M:%S')} "
              f"-> {file_period_end.strftime('%Y-%m-%d %H:%M:%S')}")

        gender, confidence, window_count, predicted_class = classify_with_model(audio_file, model)

        if confidence is not None:
            confidence_values.append(confidence)

        if predicted_class is not None and predicted_class in class_counts:
            class_counts[predicted_class] += 1

        # Per-run counters
        if gender == "FEMALE":
            run_female += 1
        elif gender == "MALE":
            run_male += 1
        elif gender == "UNKNOWN":
            run_unknown += 1
        else:
            run_error += 1

        # Update stats for the CORRECT 7AM->7AM period
        stats_for_file = get_cached_stats(file_period_start)

        if gender == "FEMALE":
            stats_for_file["female_total"] += 1
        elif gender == "MALE":
            stats_for_file["male_total"] += 1
        elif gender == "UNKNOWN":
            stats_for_file["unknown_total"] += 1
        else:
            stats_for_file["error_total"] += 1

        # A changed old period may need summary rewritten later
        stats_for_file["summary_sent"] = False

        print(f"  Gender: {gender}")
        print(f"  Confidence: {confidence if confidence is not None else 'N/A'}")
        print(f"  Predicted class: {predicted_class if predicted_class is not None else 'N/A'}")
        print(f"  Window count: {window_count}")

        append_result_to_csv(
            base_name, gender, confidence, predicted_class, window_count,
            MODEL_NAME, recorded_dt, file_period_start
        )

        save_result_json(
            base_name, gender, confidence, predicted_class, window_count,
            MODEL_NAME, recorded_dt, file_period_start
        )

        if influx_client:
            send_classification_result(
                influx_client,
                base_name,
                gender,
                confidence,
                predicted_class,
                window_count,
                recorded_dt,
                file_period_start
            )

        try:
            shutil.move(audio_file, os.path.join(PROCESSED_DIR, base_name))
        except Exception as e:
            print(f"Warning: could not move file {audio_file}: {e}")

    # Save all touched period stats
    for stats in touched_stats.values():
        save_period_stats(stats)

    # Finalize yesterday only AFTER processing files
    finalize_previous_period_if_needed(influx_client, current_period_start)

    # Reload current stats after all updates
    current_stats = load_period_stats(current_period_start)
    update_current_stats_alias(current_stats)

    # Send current cumulative stats
    if influx_client:
        send_gender_stats(influx_client, current_stats, stat_type="cumulative")

    total_current_24h = (
        current_stats["female_total"] +
        current_stats["male_total"] +
        current_stats["unknown_total"] +
        current_stats["error_total"]
    )

    female_pct = (current_stats["female_total"] / total_current_24h * 100) if total_current_24h > 0 else 0
    male_pct = (current_stats["male_total"] / total_current_24h * 100) if total_current_24h > 0 else 0

    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Processed this run: {len(audio_files)}")
    print(f"Female this run:  {run_female}")
    print(f"Male this run:    {run_male}")
    print(f"Unknown this run: {run_unknown}")
    print(f"Errors this run:  {run_error}")

    if confidence_values:
        print(f"Average confidence this run: {np.mean(confidence_values):.3f}")

    print("\nCurrent 24-hour gender stats:")
    print(f"Current period: {current_stats['period_label']}")
    print(f"Total classified: {total_current_24h}")
    print(f"Female total:  {current_stats['female_total']} ({female_pct:.1f}%)")
    print(f"Male total:    {current_stats['male_total']} ({male_pct:.1f}%)")
    print(f"Unknown total: {current_stats['unknown_total']}")
    print(f"Error total:   {current_stats['error_total']}")

    print("\nClass distribution this run:")
    for cls, count in class_counts.items():
        if count > 0:
            print(f"  Class {cls}: {count}")

    print("=" * 60)
    print(f"CSV log: {RESULTS_CSV}")
    print(f"Current stats alias file: {GENDER_COUNT_FILE}")
    print(f"Per-period stats folder: {STATS_DIR}")
    print(f"Processed files moved to: {PROCESSED_DIR}")

    if influx_client:
        influx_client.close()
        print("InfluxDB connection closed")
    print("=" * 60)

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
