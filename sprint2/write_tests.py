"""
MEMBRE 5 - Test de performance d'ingestion

Ce script mesure le débit d'ingestion (rows/sec) pour :
- PostgreSQL via `member3_postgres_ingestion.py`
- InfluxDB via `member4_influxdb_ingestion.py`

Il exécute les scripts existants et calcule le nombre attendu de lignes / points
à partir des données windowed.
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    import member3_postgres_ingestion as pg_ingest
    import member4_influxdb_ingestion as influx_ingest
except Exception as exc:
    print(f"Erreur d'import des modules d'ingestion : {exc}")
    print("Vérifiez que vous exécutez ce script depuis le dossier Partie_1 ou que Python trouve ces modules.")
    raise


def count_postgres_rows(windowed_dir):
    total_windows = 0
    for split in ["train", "test"]:
        _, meta = pg_ingest.load_split(windowed_dir, split)
        total_windows += len(meta)
    # en PostgreSQL, un signal row = une fenêtre * 9 canaux
    total_signal_rows = total_windows * 9
    return {
        "subjects": None,
        "windows": total_windows,
        "signals": total_signal_rows,
        "total_rows": total_windows + total_signal_rows,
    }


def count_influx_points(windowed_dir):
    total_points = 0
    for split in ["train", "test"]:
        X = __import__("numpy").load(os.path.join(windowed_dir, f"X_{split}.npy"))
        total_points += X.shape[0] * len(influx_ingest.CHANNELS) * X.shape[1]
    return total_points


def measure_postgres_ingestion():
    print("\n=== Mesure PostgreSQL ===")
    windowed_dir = pg_ingest.find_windowed_dir(pg_ingest.WINDOWED_ROOT)
    row_counts = count_postgres_rows(windowed_dir)

    start = time.perf_counter()
    pg_ingest.main()
    duration = time.perf_counter() - start

    print(f"Durée PostgreSQL : {duration:.3f} secondes")
    print(f"Lignes de windows attendues : {row_counts['windows']}")
    print(f"Lignes de signaux attendues : {row_counts['signals']}")
    print(f"Total attendu pour le calcul du débit : {row_counts['total_rows']}")
    print(f"Débit PostgreSQL : {row_counts['total_rows'] / duration:.1f} rows/sec")


def measure_influxdb_ingestion():
    print("\n=== Mesure InfluxDB ===")
    windowed_dir = influx_ingest.find_windowed_dir(influx_ingest.WINDOWED_DIR)
    total_points = count_influx_points(windowed_dir)

    start = time.perf_counter()
    influx_ingest.main()
    duration = time.perf_counter() - start

    print(f"Durée InfluxDB : {duration:.3f} secondes")
    print(f"Points InfluxDB attendus : {total_points}")
    print(f"Débit InfluxDB : {total_points / duration:.1f} rows/sec")


def main():
    measure_postgres_ingestion()
    measure_influxdb_ingestion()


if __name__ == "__main__":
    main()
