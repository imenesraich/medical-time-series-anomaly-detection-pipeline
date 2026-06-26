# -*- coding: utf-8 -*-
"""
=============================================================================
MEMBRE 6 - Stress Testing (Range Queries)
=============================================================================
Objectif : Mesurer la latence des requêtes de plage temporelle sur PostgreSQL.
Output : member6_results_postgres.csv (livrable pour Membre 10)
=============================================================================
"""
import psycopg2
import time
import statistics
import csv
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "har_db",
    "user": "postgres",
    "password": "postgres",
}

NUM_ITERATIONS = 50  # Nombre d'exécutions pour la moyenne
OUTPUT_FILE = "member6_results_postgres.csv"

# ============================================================================
# REQUÊTES DE TEST (Range Queries)
# ============================================================================
# Simulation d'une requête "Time-Range" : 
# On utilise window_index comme proxy du temps (chaque fenêtre = 2.56s)
# On récupère 20 fenêtres (env. 50 secondes de données) pour le Sujet 1.

QUERY_RANGE_HEAVY = """
    SELECT subject_id, channel, samples 
    FROM v_windows_full 
    WHERE subject_id = 1 
      AND window_index BETWEEN 100 AND 120
    ORDER BY window_index, channel;
"""

QUERY_RANGE_LIGHT = """
    SELECT COUNT(*) 
    FROM windows 
    WHERE subject_id = 1 
      AND window_index BETWEEN 100 AND 120;
"""

TESTS = [
    {"name": "Light Query (Count)", "sql": QUERY_RANGE_LIGHT},
    {"name": "Heavy Query (Retrieve Samples)", "sql": QUERY_RANGE_HEAVY},
]

# ============================================================================
# UTILITAIRES
# ============================================================================
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def run_benchmark():
    log(f"🚀 Starting Stress Test ({NUM_ITERATIONS} iterations)...")
    results = []

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        for test in TESTS:
            log(f" Running: {test['name']}")
            latencies = []

            for i in range(NUM_ITERATIONS):
                start_time = time.perf_counter()
                
                cur.execute(test['sql'])
                cur.fetchall() # Force fetch pour mesurer le temps réel
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            # Calcul des stats
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            log(f"✅ {test['name']} | Avg: {avg_latency:.2f}ms | Min: {min_latency:.2f}ms | Max: {max_latency:.2f}ms")

            results.append({
                "query_name": test['name'],
                "avg_latency_ms": round(avg_latency, 4),
                "min_latency_ms": round(min_latency, 4),
                "max_latency_ms": round(max_latency, 4),
                "iterations": NUM_ITERATIONS
            })

        cur.close()
        conn.close()

        # Sauvegarde CSV
        with open(OUTPUT_FILE, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["query_name", "avg_latency_ms", "min_latency_ms", "max_latency_ms", "iterations"])
            writer.writeheader()
            writer.writerows(results)
        
        log(f"💾 Results saved to {OUTPUT_FILE}")

    except Exception as e:
        log(f"❌ Error: {e}")

if __name__ == "__main__":
    run_benchmark()