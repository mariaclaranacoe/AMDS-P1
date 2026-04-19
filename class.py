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

UPDATED LOGIC:
- If confidence < 0.60, classification is treated as ERROR
- unknown_total removed
- error_total now includes low-confidence results and real processing errors
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
INFLUXDB_HOST = "192.168.5.200"
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

# Low-confidence results become ERROR
CONFIDENCE_THRESHOLD = 0.60

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
        "error_total": int(stats.get("error_total", 0)),
        "total_classified": int(
            stats.get("female_total", 0) +
            stats.get("male_total", 0) +
            stats.get("error_total", 0)
        )
    }

    tags = {
        "period_start": str(stats.get("period_start", "")),
        "period_end": str(stats.get("period_end", "")),
        "type": stat_type
    }

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
            stats["error_total"]
        )

        female_pct = (stats["female_total"] / total * 100) if total > 0 else 0
        male_pct = (stats["male_total"] / total * 100) if total > 0 else 0
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

        # Low-confidence result is treated as ERROR
        if confidence < CONFIDENCE_THRESHOLD:
            return "ERROR", confidence, window_count, predicted_class

        if predicted_class in FEMALE_CLASSES:
            gender = "FEMALE"
        elif predicted_class in MALE_CLASSES:
            gender = "MALE"
        else:
            gender = "ERROR"

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
        previous_stats.get("error_total", 0)
    )

    if total_previous == 0 and previous_stats.get("summary_sent", False):
        return

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
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f} (below this = ERROR)")
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

        finalize_previous_period_if_needed(influx_client, current_period_start)

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
    run_error = 0
    class_counts = {i: 0 for i in range(8)}
    confidence_values = []

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

        if gender == "FEMALE":
            run_female += 1
        elif gender == "MALE":
            run_male += 1
        else:
            run_error += 1

        stats_for_file = get_cached_stats(file_period_start)

        if gender == "FEMALE":
            stats_for_file["female_total"] += 1
        elif gender == "MALE":
            stats_for_file["male_total"] += 1
        else:
            stats_for_file["error_total"] += 1

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

    for stats in touched_stats.values():
        save_period_stats(stats)

    finalize_previous_period_if_needed(influx_client, current_period_start)

    current_stats = load_period_stats(current_period_start)
    update_current_stats_alias(current_stats)

    if influx_client:
        send_gender_stats(influx_client, current_stats, stat_type="cumulative")

    total_current_24h = (
        current_stats["female_total"] +
        current_stats["male_total"] +
        current_stats["error_total"]
    )

    female_pct = (current_stats["female_total"] / total_current_24h * 100) if total_current_24h > 0 else 0
    male_pct = (current_stats["male_total"] / total_current_24h * 100) if total_current_24h > 0 else 0
    error_pct = (current_stats["error_total"] / total_current_24h * 100) if total_current_24h > 0 else 0

    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Processed this run: {len(audio_files)}")
    print(f"Female this run:  {run_female}")
    print(f"Male this run:    {run_male}")
    print(f"Errors this run:  {run_error}")

    if confidence_values:
        print(f"Average confidence this run: {np.mean(confidence_values):.3f}")

    print("\nCurrent 24-hour gender stats:")
    print(f"Current period: {current_stats['period_label']}")
    print(f"Total classified: {total_current_24h}")
    print(f"Female total:  {current_stats['female_total']} ({female_pct:.1f}%)")
    print(f"Male total:    {current_stats['male_total']} ({male_pct:.1f}%)")
    print(f"Error total:   {current_stats['error_total']} ({error_pct:.1f}%)")

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

if _name_ == "_main_":
    main()
