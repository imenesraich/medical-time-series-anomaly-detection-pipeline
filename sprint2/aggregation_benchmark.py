#!/usr/bin/env python3
"""
member7_aggregation_benchmark.py
================================
Membre 7 — Sprint 2 | Aggregation Performance Benchmarks
Compares PostgreSQL vs InfluxDB for time-series aggregation queries

Author: Member 7
Date: 2026-05-13
"""

import psycopg2
import json
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient, QueryApi
from datetime import datetime
from pathlib import Path
import numpy as np

# =============================================================================
# CONFIGURATION — TODO: Fill in your PostgreSQL credentials
# =============================================================================

POSTGRES_CONFIG = {
    "host": "localhost",     
    "port": 5432,
    "database": "har_db",
    "user": "postgres", 
    "password": "postgres" 
}

INFLUX_CONFIG = {
    "url": "http://localhost:8086",
    "token": "YNq10cXVSJgfD9GvPnmAgTHGwZoGHUfrc1OhF2tKEaEVqy3lZjIGKLKKD4tI62AYxi88QJVAw1M5aFHqVNQgZg==",
    "org": "ESI",
    "bucket": "har_data"
}

# Benchmark settings
N_ITERATIONS = 5  # Number of times to run each query for averaging
OUTPUT_DIR = Path("./member7_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# AGGREGATION TEST CASES
# =============================================================================

class AggregationBenchmark:
    def __init__(self):
        self.pg_conn = None
        self.influx_client = None
        self.results = []
        self.process = psutil.Process()
        
    def connect_databases(self):
        """Connect to both databases"""
        print("=" * 70)
        print("CONNECTING TO DATABASES")
        print("=" * 70)
        
        # PostgreSQL connection
        try:
            self.pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
            print("✅ PostgreSQL connected")
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            raise
        
        # InfluxDB connection
        try:
            self.influx_client = InfluxDBClient(
                url=INFLUX_CONFIG["url"],
                token=INFLUX_CONFIG["token"],
                org=INFLUX_CONFIG["org"]
            )
            print("✅ InfluxDB connected")
        except Exception as e:
            print(f"❌ InfluxDB connection failed: {e}")
            raise
            
    def close_connections(self):
        """Close database connections"""
        if self.pg_conn:
            self.pg_conn.close()
            print("🔌 PostgreSQL disconnected")
        if self.influx_client:
            self.influx_client.close()
            print("🔌 InfluxDB disconnected")
    
    def measure_memory(self):
        """Measure current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def run_query_pg(self, query, description):
        """Execute PostgreSQL query and measure performance"""
        print(f"\n📊 PostgreSQL: {description}")
        
        cursor = self.pg_conn.cursor()
        times = []
        memory_before = self.measure_memory()
        result_rows = 0
        
        for i in range(N_ITERATIONS):
            start = time.perf_counter()
            cursor.execute(query)
            _ = cursor.fetchall()
            end = time.perf_counter()
            
            elapsed = (end - start) * 1000  # Convert to ms
            times.append(elapsed)
            
            if i == 0:
                result_rows = cursor.rowcount if cursor.rowcount >= 0 else "N/A"
            
            print(f"  Iteration {i+1}/{N_ITERATIONS}: {elapsed:.2f} ms", end='\r')
        
        memory_after = self.measure_memory()
        cursor.close()
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n  ✅ Avg: {avg_time:.2f} ± {std_time:.2f} ms | Rows: {result_rows}")
        
        return {
            "database": "PostgreSQL",
            "test": description,
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "memory_mb": memory_after - memory_before,
            "result_rows": result_rows,
            "iterations": N_ITERATIONS
        }
    
    def run_query_influx(self, flux_query, description):
        """Execute InfluxDB Flux query and measure performance"""
        print(f"\n📊 InfluxDB: {description}")
        
        query_api = self.influx_client.query_api()
        times = []
        memory_before = self.measure_memory()
        result_rows = 0
        
        for i in range(N_ITERATIONS):
            start = time.perf_counter()
            result = query_api.query(query=flux_query, org=INFLUX_CONFIG["org"])
            end = time.perf_counter()
            
            elapsed = (end - start) * 1000  # Convert to ms
            times.append(elapsed)
            
            # Count result rows
            if i == 0:
                result_rows = sum(len(table.records) for table in result)
            
            print(f"  Iteration {i+1}/{N_ITERATIONS}: {elapsed:.2f} ms", end='\r')
        
        memory_after = self.measure_memory()
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n  ✅ Avg: {avg_time:.2f} ± {std_time:.2f} ms | Rows: {result_rows}")
        
        return {
            "database": "InfluxDB",
            "test": description,
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "memory_mb": memory_after - memory_before,
            "result_rows": result_rows,
            "iterations": N_ITERATIONS
        }
    
    # ==========================================================================
    # TEST CASES
    # ==========================================================================
    
    def test_agg_01_downsample_1s(self):
        """AGG-01: Downsample 50Hz → 1Hz (mean aggregation)"""
        description = "AGG-01: Downsample 50Hz→1Hz (body_acc_x mean)"
        
        # PostgreSQL: unnest array + group by 1-second bins
        pg_query = """
        SELECT 
            w.window_id,
            FLOOR(s.sample_idx / 50.0) AS second_bin,
            AVG(s.value) AS mean_acc_x
        FROM (
            SELECT 
                window_id,
                generate_series(0, 127) AS sample_idx,
                unnest(samples) AS value
            FROM signals
            WHERE channel = 'body_acc_x'
        ) s
        JOIN windows w ON w.window_id = s.window_id
        GROUP BY w.window_id, FLOOR(s.sample_idx / 50.0)
        ORDER BY w.window_id, second_bin;
        """
        
        # InfluxDB: aggregateWindow function
        influx_query = f'''
        from(bucket: "{INFLUX_CONFIG["bucket"]}")
          |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "har_kinematic_signals")
          |> filter(fn: (r) => r._field == "body_acc_x")
          |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''
        
        self.results.append(self.run_query_pg(pg_query, description))
        self.results.append(self.run_query_influx(influx_query, description))
    
    def test_agg_02_window_stats(self):
        """AGG-02: Compute mean/std per window (all 9 channels)"""
        description = "AGG-02: Window statistics (mean/std for 9 channels)"
        
        # PostgreSQL: compute stats from arrays
        pg_query = """
        SELECT 
            w.window_id,
            s.channel,
            AVG(v.value) AS mean_val,
            STDDEV(v.value) AS std_val
        FROM signals s
        JOIN windows w ON w.window_id = s.window_id,
        unnest(s.samples) AS v(value)
        GROUP BY w.window_id, s.channel
        ORDER BY w.window_id, s.channel;
        """
        
        # InfluxDB: direct read from windowed_features measurement
        influx_query = f'''
        from(bucket: "{INFLUX_CONFIG["bucket"]}")
          |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "har_windowed_features")
          |> filter(fn: (r) => r._field =~ /.*_(mean|std)/)
        '''
        
        self.results.append(self.run_query_pg(pg_query, description))
        self.results.append(self.run_query_influx(influx_query, description))
    
    def test_agg_03_group_by_activity(self):
        """AGG-03: Aggregate statistics by activity type"""
        description = "AGG-03: Group by activity (mean body_acc_x)"
        
        # PostgreSQL
        pg_query = """
        SELECT 
            a.activity_name,
            COUNT(DISTINCT w.window_id) AS n_windows,
            AVG(v.value) AS mean_acc_x,
            STDDEV(v.value) AS std_acc_x
        FROM windows w
        JOIN activities a ON a.activity_label = w.activity_label
        JOIN signals s ON s.window_id = w.window_id
        CROSS JOIN unnest(s.samples) AS v(value)
        WHERE s.channel = 'body_acc_x'
        GROUP BY a.activity_name
        ORDER BY n_windows DESC;
        """
        
        # InfluxDB
        influx_query = f'''
        from(bucket: "{INFLUX_CONFIG["bucket"]}")
          |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "har_windowed_features")
          |> filter(fn: (r) => r._field == "body_acc_x_mean")
          |> group(columns: ["activity"])
          |> mean()
        '''
        
        self.results.append(self.run_query_pg(pg_query, description))
        self.results.append(self.run_query_influx(influx_query, description))
    
    def test_agg_04_group_by_subject(self):
        """AGG-04: Aggregate by subject (30 subjects)"""
        description = "AGG-04: Group by subject (mean body_gyro_z)"
        
        # PostgreSQL
        pg_query = """
        SELECT 
            w.subject_id,
            COUNT(DISTINCT w.window_id) AS n_windows,
            AVG(v.value) AS mean_gyro_z,
            STDDEV(v.value) AS std_gyro_z
        FROM windows w
        JOIN signals s ON s.window_id = w.window_id
        CROSS JOIN unnest(s.samples) AS v(value)
        WHERE s.channel = 'body_gyro_z'
        GROUP BY w.subject_id
        ORDER BY w.subject_id;
        """
        
        # InfluxDB
        influx_query = f'''
        from(bucket: "{INFLUX_CONFIG["bucket"]}")
          |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "har_windowed_features")
          |> filter(fn: (r) => r._field == "body_gyro_z_mean")
          |> group(columns: ["subject_id"])
          |> mean()
        '''
        
        self.results.append(self.run_query_pg(pg_query, description))
        self.results.append(self.run_query_influx(influx_query, description))
    
    def test_agg_05_hourly_rollup(self):
        """AGG-05: Hourly time-based rollup (synthetic timeline)"""
        description = "AGG-05: Hourly rollup (mean all channels)"
        
        # PostgreSQL: use synthetic timestamp based on window_index
        pg_query = """
        SELECT 
            DATE_TRUNC('hour', 
                TIMESTAMP '2024-01-01 00:00:00' + 
                (w.window_index * INTERVAL '2.56 seconds')
            ) AS hour_bucket,
            COUNT(*) AS n_windows,
            AVG(v.value) AS mean_value
        FROM windows w
        JOIN signals s ON s.window_id = w.window_id
        CROSS JOIN unnest(s.samples) AS v(value)
        WHERE s.channel = 'body_acc_x'
        GROUP BY hour_bucket
        ORDER BY hour_bucket;
        """
        
        # InfluxDB: aggregateWindow with 1h
        influx_query = f'''
        from(bucket: "{INFLUX_CONFIG["bucket"]}")
          |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "har_kinematic_signals")
          |> filter(fn: (r) => r._field == "body_acc_x")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
          |> yield(name: "hourly_mean")
        '''
        
        self.results.append(self.run_query_pg(pg_query, description))
        self.results.append(self.run_query_influx(influx_query, description))
    
    def test_agg_06_anomaly_detection(self):
        """AGG-06: Anomaly aggregation (is_anomaly=1 vs 0)"""
        description = "AGG-06: Anomaly vs Normal aggregation"
        
        # PostgreSQL
        pg_query = """
        SELECT 
            CASE w.is_anomaly 
                WHEN 1 THEN 'Anomaly' 
                WHEN 0 THEN 'Normal' 
            END AS anomaly_status,
            COUNT(DISTINCT w.window_id) AS n_windows,
            AVG(v.value) AS mean_acc_y,
            STDDEV(v.value) AS std_acc_y,
            MIN(v.value) AS min_acc_y,
            MAX(v.value) AS max_acc_y
        FROM windows w
        JOIN signals s ON s.window_id = w.window_id
        CROSS JOIN unnest(s.samples) AS v(value)
        WHERE s.channel = 'body_acc_y'
        GROUP BY w.is_anomaly
        ORDER BY anomaly_status;
        """
        
        # InfluxDB
        influx_query = f'''
        from(bucket: "{INFLUX_CONFIG["bucket"]}")
          |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "har_windowed_features")
          |> filter(fn: (r) => r._field == "body_acc_y_mean")
          |> group(columns: ["is_anomaly"])
          |> mean()
        '''
        
        self.results.append(self.run_query_pg(pg_query, description))
        self.results.append(self.run_query_influx(influx_query, description))
    
    # ==========================================================================
    # RESULTS EXPORT
    # ==========================================================================
    
    def export_results(self):
        """Export results to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV export
        df = pd.DataFrame(self.results)
        csv_path = OUTPUT_DIR / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # JSON export (for Member 10)
        json_path = OUTPUT_DIR / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 JSON saved to: {json_path}")
        
        return df
    
    def generate_charts(self, df):
        """Generate comparison charts"""
        print("\n📈 Generating comparison charts...")
        
        # Filter for unique tests
        tests = df['test'].unique()
        
        for test in tests:
            test_df = df[df['test'] == test]
            
            plt.figure(figsize=(10, 6))
            plt.bar(
                test_df['database'], 
                test_df['avg_time_ms'],
                yerr=test_df['std_time_ms'],
                capsize=5,
                color=['#3366CC', '#DC3912']
            )
            plt.title(f'{test}\n(Average Execution Time)', fontsize=12, fontweight='bold')
            plt.ylabel('Time (ms)')
            plt.xlabel('Database')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            chart_path = OUTPUT_DIR / f"chart_{test.replace(' ', '_').replace(':', '')}.png"
            plt.savefig(chart_path, dpi=300)
            plt.close()
            print(f"  ✅ Chart: {chart_path.name}")
        
        # Summary comparison chart
        summary_df = df.groupby('database')['avg_time_ms'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(
            summary_df['database'],
            summary_df['avg_time_ms'],
            color=['#3366CC', '#DC3912']
        )
        plt.title('Overall Average Execution Time\n(All Aggregation Tests)', 
                 fontsize=12, fontweight='bold')
        plt.ylabel('Time (ms)')
        plt.xlabel('Database')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        summary_path = OUTPUT_DIR / "chart_overall_comparison.png"
        plt.savefig(summary_path, dpi=300)
        plt.close()
        print(f"  ✅ Summary chart: {summary_path.name}")
    
    def generate_summary_report(self, df):
        """Generate summary report for Member 10"""
        print("\n📋 Generating summary report...")
        
        report = []
        report.append("=" * 70)
        report.append("MEMBER 7 — AGGREGATION BENCHMARK SUMMARY")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")
        
        # Overall statistics
        pg_avg = df[df['database'] == 'PostgreSQL']['avg_time_ms'].mean()
        influx_avg = df[df['database'] == 'InfluxDB']['avg_time_ms'].mean()
        speedup = pg_avg / influx_avg if influx_avg > 0 else float('inf')
        
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  PostgreSQL average: {pg_avg:.2f} ms")
        report.append(f"  InfluxDB average:   {influx_avg:.2f} ms")
        report.append(f"  Speedup factor:     {speedup:.2f}x {'(InfluxDB faster)' if speedup > 1 else '(PostgreSQL faster)'}")
        report.append("")
        
        # Per-test breakdown
        report.append("PER-TEST BREAKDOWN:")
        report.append("-" * 70)
        
        for _, row in df.iterrows():
            report.append(f"\nTest: {row['test']}")
            report.append(f"  Database: {row['database']}")
            report.append(f"    Avg Time: {row['avg_time_ms']:.2f} ± {row['std_time_ms']:.2f} ms")
            report.append(f"    Min/Max:  {row['min_time_ms']:.2f} / {row['max_time_ms']:.2f} ms")
            report.append(f"    Memory:   {row['memory_mb']:.2f} MB")
            report.append(f"    Rows:     {row['result_rows']}")
        
        report.append("")
        report.append("=" * 70)
        report.append("CONCLUSION:")
        report.append("=" * 70)
        
        if speedup > 1.5:
            report.append(f"InfluxDB is {speedup:.1f}x faster on average for aggregation queries.")
            report.append("This validates the choice of TSDB for time-series workloads.")
        elif speedup < 0.67:
            report.append(f"PostgreSQL is {1/speedup:.1f}x faster on average.")
            report.append("Array-based storage with proper indexing can be competitive.")
        else:
            report.append("Both databases show comparable performance for these aggregations.")
            report.append("The choice depends on other factors (scalability, ecosystem, etc.)")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = OUTPUT_DIR / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"💾 Summary report: {report_path}")
        print("\n" + report_text)
        
        return report_text
    
    def run_all_tests(self):
        """Execute all benchmark tests"""
        print("\n" + "=" * 70)
        print("STARTING AGGREGATION BENCHMARKS")
        print(f"Iterations per test: {N_ITERATIONS}")
        print("=" * 70)
        
        self.test_agg_01_downsample_1s()
        self.test_agg_02_window_stats()
        self.test_agg_03_group_by_activity()
        self.test_agg_04_group_by_subject()
        self.test_agg_05_hourly_rollup()
        self.test_agg_06_anomaly_detection()
        
        print("\n" + "=" * 70)
        print("BENCHMARKS COMPLETE")
        print("=" * 70)
        
        # Export and visualize
        df = self.export_results()
        self.generate_charts(df)
        self.generate_summary_report(df)
        
        print("\n✅ All deliverables generated in ./member7_results/")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    benchmark = AggregationBenchmark()
    
    try:
        benchmark.connect_databases()
        benchmark.run_all_tests()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        benchmark.close_connections()