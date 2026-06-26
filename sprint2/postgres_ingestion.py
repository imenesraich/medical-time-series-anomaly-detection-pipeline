"""
=============================================================================
MEMBRE 3 - PostgreSQL Ingestion
=============================================================================

Objectif : charger les fichiers windowed dans PostgreSQL en respectant le schéma
fourni par PotgreSQL_rational_DB.sql.

Output : données insérées dans les tables suivantes :
- subjects
- windows
- signals

Ce script :
- localise automatiquement le dossier windowed
- charge les fichiers X_train.npy, X_test.npy, meta_train.csv, meta_test.csv
- insère les sujets, fenêtres et signaux en batch
- exécute quelques validations simples

=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOWED_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "windowed"))

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "har_db",
    "user": "postgres",
    "password": "postgres",
}

BATCH_SIZE = 500

# ============================================================================
# UTILITAIRES
# ============================================================================

def log(msg, level="INFO"):
    prefix = {"INFO": "ℹ", "OK": "✅", "WARN": "⚠️", "ERR": "❌"}
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {prefix.get(level, ' ')} {msg}")


def find_windowed_dir(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Windowed directory not found: {base_dir}")

    for root, dirs, files in os.walk(base_dir):
        if "X_train.npy" in files and "meta_train.csv" in files:
            return root
        # limiter la recherche à quelques niveaux
        if root.count(os.sep) - base_dir.count(os.sep) >= 3:
            dirs.clear()
    raise FileNotFoundError(
        f"Could not find X_train.npy / meta_train.csv under {base_dir}"
    )


def connect_db(config):
    return psycopg2.connect(**config)


def load_split(windowed_dir, split):
    X_path = os.path.join(windowed_dir, f"X_{split}.npy")
    meta_path = os.path.join(windowed_dir, f"meta_{split}.csv")

    log(f"Loading {split} data from {windowed_dir}...")
    X = np.load(X_path)
    meta = pd.read_csv(meta_path)
    log(f"Loaded {split}: X shape={X.shape}, meta shape={meta.shape}", "OK")
    return X, meta


def insert_subjects(conn, meta):
    log("Inserting subjects...", "INFO")
    subjects = meta[["subject_id", "split"]].drop_duplicates().values.tolist()

    query = (
        "INSERT INTO subjects (subject_id, split) VALUES %s "
        "ON CONFLICT (subject_id) DO UPDATE SET split = EXCLUDED.split"
    )

    with conn.cursor() as cur:
        execute_values(cur, query, subjects)
    conn.commit()
    log(f"Inserted/updated {len(subjects)} subjects", "OK")


def insert_windows(conn, meta):
    log("Inserting windows...", "INFO")
    rows = []
    for _, row in meta.iterrows():
        rows.append((
            row["window_id"],
            int(row["window_index"]),
            row["split"],
            int(row["subject_id"]),
            int(row["activity_label"]),
            1,
        ))

    query = (
        "INSERT INTO windows (window_id, window_index, split, subject_id, activity_label, protocol_id) "
        "VALUES %s ON CONFLICT (window_id) DO NOTHING"
    )

    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            execute_values(cur, query, batch)
            conn.commit()
            log(f"  Inserted windows {i + 1}..{min(i + BATCH_SIZE, len(rows))}/{len(rows)}", "INFO")

    log(f"Inserted {len(rows)} windows", "OK")


def insert_signals(conn, X, meta):
    log("Inserting signals...", "INFO")

    channels = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    rows = []
    window_ids = meta["window_id"].tolist()

    for window_idx in range(X.shape[0]):
        window_id = window_ids[window_idx]
        for channel_idx, channel in enumerate(channels):
            sample_list = X[window_idx, :, channel_idx].astype(float).tolist()
            rows.append((window_id, channel, sample_list))

    query = (
        "INSERT INTO signals (window_id, channel, samples) VALUES %s "
        "ON CONFLICT (window_id, channel) DO NOTHING"
    )

    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            execute_values(cur, query, batch)
            conn.commit()
            log(f"  Inserted signals {i + 1}..{min(i + BATCH_SIZE, len(rows))}/{len(rows)}", "INFO")

    log(f"Inserted {len(rows)} signal rows", "OK")


def validate(conn):
    log("Running validation queries...", "INFO")
    with conn.cursor() as cur:
        cur.execute("SELECT split, COUNT(*) FROM windows GROUP BY split ORDER BY split")
        print("windows per split:")
        for row in cur.fetchall():
            print(row)

        cur.execute("SELECT split, COUNT(DISTINCT subject_id) FROM windows GROUP BY split ORDER BY split")
        print("subjects per split:")
        for row in cur.fetchall():
            print(row)

        cur.execute("SELECT channel, COUNT(*) FROM signals GROUP BY channel ORDER BY channel")
        print("signal rows per channel:")
        for row in cur.fetchall():
            print(row)


def main():
    try:
        windowed_dir = find_windowed_dir(WINDOWED_ROOT)
        log(f"Using windowed folder: {windowed_dir}")

        conn = connect_db(DB_CONFIG)
        log("Connected to PostgreSQL", "OK")

        for split in ["train", "test"]:
            X, meta = load_split(windowed_dir, split)
            insert_subjects(conn, meta)
            insert_windows(conn, meta)
            insert_signals(conn, X, meta)

        validate(conn)
        conn.close()
        log("PostgreSQL ingestion completed", "OK")

    except Exception as exc:
        log(f"Error: {exc}", "ERR")
        sys.exit(1)


if __name__ == "__main__":
    main()
