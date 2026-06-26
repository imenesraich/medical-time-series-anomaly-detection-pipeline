# =============================================================================
# windowing.py — Membre 2 | Sprint 2 | Topic M5: Human Activity Anomaly Detection
# =============================================================================
# ANOMALY DETECTION RULES:
# This version labels anomalies using signal behavior instead of activity names.
#
# A window is anomalous if:
#   1. Extremely low movement variance
#   2. Abnormal acceleration magnitude
#   3. Unusual gyroscope stability
# =============================================================================

import os
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEANED_DIR = "./cleaned"
OUTPUT_DIR = "./windowed"

CHANNELS = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# -----------------------------------------------------------------------------
# ANOMALY DETECTION THRESHOLDS & INDEXES
# -----------------------------------------------------------------------------
BODY_ACC_IDX  = [0, 1, 2]
BODY_GYRO_IDX = [3, 4, 5]
TOTAL_ACC_IDX = [6, 7, 8]

# These values are empirical starting points.
LOW_MOVEMENT_STD_THRESHOLD = 0.015
HIGH_ACCEL_MAG_THRESHOLD = 2.5
LOW_ACCEL_MAG_THRESHOLD  = 0.05
GYRO_STABILITY_THRESHOLD = 0.008

# =============================================================================
# ÉTAPE 1 — LABELLING DES ANOMALIES (BEHAVIORAL)
# =============================================================================

def compute_window_features(window):
    """
    window shape: (128, 9)
    returns: dict of behavioral statistics
    """
    # BODY ACCELERATION
    body_acc = window[:, BODY_ACC_IDX]
    movement_std = np.std(body_acc)
    acc_magnitude = np.sqrt(np.sum(body_acc**2, axis=1))
    mean_acc_mag = np.mean(acc_magnitude)
    max_acc_mag  = np.max(acc_magnitude)

    # GYROSCOPE
    gyro = window[:, BODY_GYRO_IDX]
    gyro_std = np.std(gyro)

    return {
        "movement_std": movement_std,
        "mean_acc_mag": mean_acc_mag,
        "max_acc_mag":  max_acc_mag,
        "gyro_std":     gyro_std,
    }

def is_behavioral_anomaly(features):
    """
    Returns: 1 -> anomaly, 0 -> normal
    """
    # RULE 1 — Extremely low movement variance
    low_variance = (features["movement_std"] < LOW_MOVEMENT_STD_THRESHOLD)

    # RULE 2 — Abnormal acceleration magnitude
    abnormal_acceleration = (
        features["mean_acc_mag"] > HIGH_ACCEL_MAG_THRESHOLD or
        features["mean_acc_mag"] < LOW_ACCEL_MAG_THRESHOLD
    )

    # RULE 3 — Unusual gyroscope stability
    overly_stable_gyro = (features["gyro_std"] < GYRO_STABILITY_THRESHOLD)

    # FINAL DECISION
    if low_variance or abnormal_acceleration or overly_stable_gyro:
        return 1

    return 0

def label_behavioral_anomalies(X):
    """
    X shape: (N, 128, 9)
    returns: anomaly_labels shape (N,)
    """
    anomaly_labels = np.zeros(len(X), dtype=int)

    for i in range(len(X)):
        window = X[i]
        features = compute_window_features(window)
        anomaly_labels[i] = is_behavioral_anomaly(features)

    return anomaly_labels

# =============================================================================
# ÉTAPE 2 — CHARGEMENT
# =============================================================================

def load_split(split):
    """
    Charge les 9 canaux et assemble : (N, 128, 9)
    """
    split_dir = os.path.join(CLEANED_DIR, split)
    print(f"\nChargement du split [{split}]...")
    
    arrays = []
    
    for ch in CHANNELS:
        path = os.path.join(split_dir, f"{ch}.npy")
        arr = np.load(path)
        print(f"   ✔ {ch}.npy — shape: {arr.shape}")
        arrays.append(arr)

    # (N, 128, 9)
    X = np.stack(arrays, axis=-1).astype(np.float32)

    labels = np.load(os.path.join(split_dir, "labels.npy"))
    subjects = np.load(os.path.join(split_dir, "subjects.npy"))

    print(f"→ Tenseur assemblé : {X.shape}")

    return X, labels, subjects

# =============================================================================
# ÉTAPE 3 — MÉTADONNÉES
# =============================================================================

def build_metadata(X, labels, subjects, split):
    """
    Modifié pour passer le tenseur X complet à l'extracteur de features
    """
    anomaly_labels = label_behavioral_anomalies(X)
    records = []

    for i in range(len(labels)):
        activity = int(labels[i])

        records.append({
            "window_id": str(uuid.uuid4()),
            "window_index": i,
            "split": split,
            "subject_id": int(subjects[i]),
            "activity_label": activity,
            "activity_name": ACTIVITY_MAP[activity],
            "is_anomaly": int(anomaly_labels[i]),
            "n_samples": 128,
            "sampling_rate_hz": 50,
            "duration_sec": 2.56,
        })

    return pd.DataFrame(records)

# =============================================================================
# ÉTAPE 4 — VALIDATION
# =============================================================================

def validate(X, meta, split):

    issues = []

    if np.any(np.isnan(X)):
        issues.append("NaN détectés dans le tenseur")

    if np.any(np.isinf(X)):
        issues.append("Inf détectés dans le tenseur")

    class_dist = meta["activity_name"].value_counts().to_dict()
    anomaly_dist = meta["is_anomaly"].value_counts().to_dict()

    report = {
        "split": split,
        "n_windows": int(len(X)),
        "tensor_shape": list(X.shape),
        "n_subjects": int(meta["subject_id"].nunique()),
        "class_distribution": class_dist,

        "anomaly_counts": {
            "normal (0)": int(anomaly_dist.get(0, 0)),
            "anomalie (1)": int(anomaly_dist.get(1, 0)),
        },

        "anomaly_ratio": round(float(meta["is_anomaly"].mean()), 4),

        "value_min": round(float(X.min()), 6),
        "value_max": round(float(X.max()), 6),
        "value_mean": round(float(X.mean()), 6),
        "value_std": round(float(X.std()), 6),

        "issues": issues if issues else ["Aucun problème détecté ✅"],
    }

    return report

# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 60)
    print("WINDOWING PIPELINE — Membre 2 | Sprint 2")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    windowing_log = {
        "run_timestamp": datetime.now().isoformat(),

        "anomaly_definition": {
            "type": "Signal Behavior",
            "thresholds": {
                "LOW_MOVEMENT_STD_THRESHOLD": LOW_MOVEMENT_STD_THRESHOLD,
                "HIGH_ACCEL_MAG_THRESHOLD": HIGH_ACCEL_MAG_THRESHOLD,
                "LOW_ACCEL_MAG_THRESHOLD": LOW_ACCEL_MAG_THRESHOLD,
                "GYRO_STABILITY_THRESHOLD": GYRO_STABILITY_THRESHOLD
            },
            "justification": (
                "Anomalies are detected purely on physical signal thresholds "
                "(movement variance, acceleration magnitude, and gyroscope stability) "
                "rather than categorical activity labels."
            ),
        },

        "splits": {},
    }

    for split in ("train", "test"):

        # Chargement
        X, labels, subjects = load_split(split)

        # Métadonnées (passage de X pour extraire les anomalies comportementales)
        meta = build_metadata(X, labels, subjects, split)

        # Validation
        report = validate(X, meta, split)
        windowing_log["splits"][split] = report

        # Sauvegarde
        np.save(os.path.join(OUTPUT_DIR, f"X_{split}.npy"), X)
        meta.to_csv(os.path.join(OUTPUT_DIR, f"meta_{split}.csv"), index=False)

        # Résumé
        print(f"\n✅ [{split}]")
        print(f"   Shape        : {X.shape}")
        print(f"   Sujets       : {report['n_subjects']}")
        print(
            f"   Anomaly ratio: {report['anomaly_ratio']} "
            f"({report['anomaly_counts']['anomalie (1)']} anomalies / "
            f"{report['anomaly_counts']['normal (0)']} normaux)"
        )
        print(f"   Distribution : {report['class_distribution']}")
        print(f"   Problèmes    : {report['issues']}")

    # Sauvegarde du log
    log_path = os.path.join(OUTPUT_DIR, "windowing_log.json")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            windowing_log,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ ✅")
    print("=" * 60)

    print(f"\nFichiers générés dans : {OUTPUT_DIR}/")
    print("  ├── X_train.npy")
    print("  ├── X_test.npy")
    print("  ├── meta_train.csv")
    print("  ├── meta_test.csv")
    print("  └── windowing_log.json")
    print("\n→ Transmettre ces fichiers aux Membres 3, 4 et 9")

if __name__ == "__main__":
    main()