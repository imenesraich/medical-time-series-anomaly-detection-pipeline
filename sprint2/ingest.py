# ingest.py — Member 4 | InfluxDB ingestion
# Input: windowed/X_train.npy, X_test.npy, meta_train.csv, meta_test.csv

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from datetime import datetime, timezone
import config

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CHANNELS = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# From README: t = window_index * 2.56 + sample * 0.02
# We convert to nanoseconds for InfluxDB
SAMPLE_INTERVAL_NS  = 20_000_000       # 0.02s = 20ms in nanoseconds
WINDOW_INTERVAL_NS  = 2_560_000_000    # 2.56s in nanoseconds

# Base epoch: 2024-01-01 00:00:00 UTC
BASE_TIME_NS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)

# ─── CONNECT ──────────────────────────────────────────────────────────────────
client = InfluxDBClient(
    url=config.INFLUX_URL,
    token=config.INFLUX_TOKEN,
    org=config.INFLUX_ORG
)
write_api = client.write_api(write_options=WriteOptions(
    batch_size=5000,
    flush_interval=10_000,
))

# ==============================================================================
# MEASUREMENT 1 — har_kinematic_signals
# Each sample point (128 per window × 10299 windows = ~1.32M points)
# Tags  : subject_id, activity_name, is_anomaly, split, window_id
# Fields: body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z
# ==============================================================================

def ingest_raw_signals(X, meta, split):
    print(f"\n[har_kinematic_signals] Ingesting {split} split...")
    print(f"  Windows : {X.shape[0]}, Samples/window : {X.shape[1]}, Channels : {X.shape[2]}")
    total_points = X.shape[0] * X.shape[1]
    print(f"  Total points to write : {total_points:,}")

    points = []
    written = 0

    for i, row in meta.iterrows():
        window_idx   = int(row["window_index"])
        subject_id   = str(int(row["subject_id"]))
        activity     = str(row["activity_name"])
        is_anomaly   = str(int(row["is_anomaly"]))
        window_id    = str(row["window_id"])
        window       = X[window_idx]   # shape (128, 9)

        for sample_idx in range(128):
            # Timestamp formula from README:
            # t = window_index * 2.56 + sample * 0.02  (in seconds)
            t_ns = BASE_TIME_NS + window_idx * WINDOW_INTERVAL_NS + sample_idx * SAMPLE_INTERVAL_NS

            p = (
                Point("har_kinematic_signals")
                .tag("subject_id",   subject_id)
                .tag("activity",     activity)
                .tag("is_anomaly",   is_anomaly)
                .tag("split",        split)
                .tag("window_id",    window_id)
            )

            for ch_idx, ch_name in enumerate(CHANNELS):
                p = p.field(ch_name, float(window[sample_idx, ch_idx]))

            p = p.time(t_ns, "ns")
            points.append(p)

            # Flush every 5000 points
            if len(points) >= 5000:
                write_api.write(bucket=config.INFLUX_BUCKET, record=points)
                written += len(points)
                print(f"  → Written {written:,} / {total_points:,} points...", end="\r")
                points = []

    # Flush remaining
    if points:
        write_api.write(bucket=config.INFLUX_BUCKET, record=points)
        written += len(points)

    print(f"\n  ✅ Done — {written:,} points written for [{split}]")


# ==============================================================================
# MEASUREMENT 2 — har_windowed_features
# One point per window (7352 train + 2947 test = 10299 total)
# Tags  : subject_id, activity_name, is_anomaly, split, window_id
# Fields: mean, std, min, max per channel (9 channels × 4 stats = 36 fields)
#         + activity_label
# ==============================================================================

def compute_window_stats(window):
    """
    window: (128, 9)
    Returns 36 stats: mean/std/min/max for each of the 9 channels
    """
    stats = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        signal = window[:, ch_idx]
        stats[f"{ch_name}_mean"] = float(np.mean(signal))
        stats[f"{ch_name}_std"]  = float(np.std(signal))
        stats[f"{ch_name}_min"]  = float(np.min(signal))
        stats[f"{ch_name}_max"]  = float(np.max(signal))
    return stats

def ingest_windowed_features(X, meta, split):
    print(f"\n[har_windowed_features] Ingesting {split} split...")
    print(f"  Windows to write : {len(meta)}")

    points = []
    written = 0

    for i, row in meta.iterrows():
        window_idx   = int(row["window_index"])
        subject_id   = str(int(row["subject_id"]))
        activity     = str(row["activity_name"])
        is_anomaly   = str(int(row["is_anomaly"]))
        window_id    = str(row["window_id"])
        window       = X[window_idx]   # shape (128, 9)

        # Timestamp = start of the window
        t_ns = BASE_TIME_NS + window_idx * WINDOW_INTERVAL_NS

        p = (
            Point("har_windowed_features")
            .tag("subject_id",   subject_id)
            .tag("activity",     activity)
            .tag("is_anomaly",   is_anomaly)
            .tag("split",        split)
            .tag("window_id",    window_id)
            .field("activity_label", int(row["activity_label"]))
        )

        # Add 36 statistical features
        stats = compute_window_stats(window)
        for field_name, value in stats.items():
            p = p.field(field_name, value)

        p = p.time(t_ns, "ns")
        points.append(p)

        if len(points) >= 500:
            write_api.write(bucket=config.INFLUX_BUCKET, record=points)
            written += len(points)
            print(f"  → Written {written} / {len(meta)} windows...", end="\r")
            points = []

    if points:
        write_api.write(bucket=config.INFLUX_BUCKET, record=points)
        written += len(points)

    print(f"\n  ✅ Done — {written} windows written for [{split}]")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INFLUXDB INGESTION")
    print("=" * 60)

    for split in ("train", "test"):
        print(f"\n{'─'*40}")
        print(f"Loading {split} data...")

        X    = np.load(f"windowed/X_{split}.npy")        # (N, 128, 9)
        meta = pd.read_csv(f"windowed/meta_{split}.csv") # N rows

        print(f"  X shape  : {X.shape}")
        print(f"  Meta rows: {len(meta)}")

        ingest_raw_signals(X, meta, split)
        ingest_windowed_features(X, meta, split)

    write_api.close()
    client.close()

    print("\n" + "=" * 60)
    print("✅ ALL INGESTION COMPLETE")
    print("=" * 60)
    print("\nMeasurements written:")
    print("  • har_kinematic_signals  — ~1.32M points (50Hz raw samples)")
    print("  • har_windowed_features  — 10,299 points (one per window)")