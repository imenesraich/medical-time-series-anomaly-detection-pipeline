# verify.py — FINAL clean version

from influxdb_client import InfluxDBClient
import config

client = InfluxDBClient(
    url=config.INFLUX_URL,
    token=config.INFLUX_TOKEN,
    org=config.INFLUX_ORG
)
query_api = client.query_api()
RANGE = 'range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)'

print("=" * 60)
print("VERIFICATION")
print("=" * 60)

# ── 1. Raw signal points ──────────────────────────────────────
q = f'''
from(bucket: "har_data")
  |> {RANGE}
  |> filter(fn: (r) => r._measurement == "har_kinematic_signals")
  |> filter(fn: (r) => r._field == "body_acc_x")
  |> group()
  |> count()
'''
total = sum(r.get_value() for t in query_api.query(q) for r in t.records)
status = "✅" if total == 1_318_272 else "⚠️"
print(f"{status} Raw signal points      : {total:>10,}  (expected 1,318,272)")

# ── 2. Windowed feature rows ──────────────────────────────────
q = f'''
from(bucket: "har_data")
  |> {RANGE}
  |> filter(fn: (r) => r._measurement == "har_windowed_features")
  |> filter(fn: (r) => r._field == "activity_label")
  |> group()
  |> count()
'''
total = sum(r.get_value() for t in query_api.query(q) for r in t.records)
status = "✅" if total == 10_299 else "⚠️"
print(f"{status} Windowed feature rows  : {total:>10,}  (expected    10,299)")

# ── 3. Activity breakdown ─────────────────────────────────────
print(f"\n{'─'*40}")
print("Activity breakdown (har_windowed_features):")
activities = [
    ("WALKING",            "Normal"),
    ("WALKING_UPSTAIRS",   "Normal"),
    ("WALKING_DOWNSTAIRS", "Normal"),
    ("SITTING",            "Anomaly"),
    ("STANDING",           "Anomaly"),
    ("LAYING",             "Anomaly"),
]
grand_total = 0
for act, kind in activities:
    q = f'''
from(bucket: "har_data")
  |> {RANGE}
  |> filter(fn: (r) => r._measurement == "har_windowed_features")
  |> filter(fn: (r) => r._field == "activity_label")
  |> filter(fn: (r) => r.activity == "{act}")
  |> group()
  |> count()
'''
    count = sum(r.get_value() for t in query_api.query(q) for r in t.records)
    grand_total += count
    print(f"   {act:<25} [{kind:<7}] : {count:>5} windows")

print(f"   {'─'*45}")
print(f"   {'TOTAL':<25}           : {grand_total:>5} windows  {'✅' if grand_total == 10_299 else '⚠️'}")

# ── 4. Train / Test split ─────────────────────────────────────
print(f"\n{'─'*40}")
print("Train / Test split:")
for split, expected in [("train", 7352), ("test", 2947)]:
    q = f'''
from(bucket: "har_data")
  |> {RANGE}
  |> filter(fn: (r) => r._measurement == "har_windowed_features")
  |> filter(fn: (r) => r._field == "activity_label")
  |> filter(fn: (r) => r.split == "{split}")
  |> group()
  |> count()
'''
    count = sum(r.get_value() for t in query_api.query(q) for r in t.records)
    status = "✅" if count == expected else "⚠️"
    print(f"   {status} {split:<8} : {count:>5} windows  (expected {expected})")

# ── 5. Anomaly distribution (behavioral) ─────────────────────
print(f"\n{'─'*40}")
print("Anomaly distribution (behavioral signal thresholds):")
for label, val in [("Normal  (0)", "0"), ("Anomaly (1)", "1")]:
    q = f'''
from(bucket: "har_data")
  |> {RANGE}
  |> filter(fn: (r) => r._measurement == "har_windowed_features")
  |> filter(fn: (r) => r._field == "activity_label")
  |> filter(fn: (r) => r.is_anomaly == "{val}")
  |> group()
  |> count()
'''
    count = sum(r.get_value() for t in query_api.query(q) for r in t.records)
    print(f"   {label} : {count:>5} windows")

# ── 6. Downsampling test ──────────────────────────────────────
print(f"\n{'─'*40}")
print("Downsampling test (aggregateWindow 1s, subject_id=1):")
q = f'''
from(bucket: "har_data")
  |> {RANGE}
  |> filter(fn: (r) => r._measurement == "har_kinematic_signals")
  |> filter(fn: (r) => r._field == "body_acc_x")
  |> filter(fn: (r) => r.subject_id == "1")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
'''
rows = [r for t in query_api.query(q) for r in t.records]
print(f"   ✅ {len(rows)} x 1s bins returned for subject 1")
if rows:
    print(f"   Sample values: {[round(r.get_value(),4) for r in rows[:5]]}")

client.close()

print("\n" + "=" * 60)
print("✅ VERIFICATION COMPLETE")
print("=" * 60)