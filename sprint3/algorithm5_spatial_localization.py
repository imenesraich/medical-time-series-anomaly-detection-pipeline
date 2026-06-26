"""
=============================================================================
MEMBRE 4 — Algorithm 5: Spatial Localization
Sprint 3 — Phase 2 — Deliverable: Table III (Localization Accuracy + 95% CI)
=============================================================================
INPUT FILES:
    - window_errors.csv   : window_id, mse_error, true_label, pred_label
    - meta_test.csv       : window_id, window_index, subject_id, activity_name, is_anomaly
    - eval_results.json   : threshold_tau, signal_metrics, ...
OUTPUT:
    - table3_localization.csv : Localization accuracy per granularity level
=============================================================================
"""

import pandas as pd
import numpy as np
import json
from math import sqrt

# =============================================================================
# STEP 0 — LOAD DATA
# =============================================================================
we         = pd.read_csv('window_errors.csv')
meta_test  = pd.read_csv('meta_test.csv')
eval_res   = json.load(open('eval_results.json'))
tau        = eval_res['threshold_tau']

print("=" * 60)
print("ALGORITHM 5 — Spatial Localization")
print("=" * 60)
print(f"Threshold tau        : {tau:.6f}")
print(f"Total test windows   : {len(we)}")

# =============================================================================
# STEP 1 — COMPUTE ANOMALY PREDICTION PER WINDOW
# is_anomaly_pred = 1 if mse_error > tau (from Algorithm 4)
# =============================================================================
we['is_anomaly_pred'] = (we['mse_error'] > tau).astype(int)

# Verify against eval_results
tp = ((we['true_label']==1) & (we['is_anomaly_pred']==1)).sum()
fp = ((we['true_label']==0) & (we['is_anomaly_pred']==1)).sum()
fn = ((we['true_label']==1) & (we['is_anomaly_pred']==0)).sum()
tn = ((we['true_label']==0) & (we['is_anomaly_pred']==0)).sum()
print(f"\nVerification vs eval_results.json:")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# =============================================================================
# STEP 2 — MERGE WITH META_TEST TO GET subject_id AND activity_name
# =============================================================================
merged = we.merge(
    meta_test[['window_index', 'subject_id', 'activity_name', 'is_anomaly']],
    left_on='window_id',
    right_on='window_index'
)
print(f"Merged shape         : {merged.shape}")
print(f"Subjects             : {sorted(merged['subject_id'].unique())}")

# =============================================================================
# STEP 3 — SENSOR TO REGION MAPPING (HAR adaptation of Algorithm 5)
# Original algorithm: sensor → anatomical region (e.g. Left/Right breast)
# HAR adaptation   : activity → movement region (DYNAMIC / STATIC)
#   DYNAMIC = activities with high movement (WALKING*)
#   STATIC  = activities with low movement  (SITTING, STANDING, LAYING)
# =============================================================================
sensor_to_region = {
    'WALKING':            'DYNAMIC',
    'WALKING_UPSTAIRS':   'DYNAMIC',
    'WALKING_DOWNSTAIRS': 'DYNAMIC',
    'SITTING':            'STATIC',
    'STANDING':           'STATIC',
    'LAYING':             'STATIC',
}

# =============================================================================
# STEP 4 — WILSON SCORE 95% CONFIDENCE INTERVAL
# =============================================================================
def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p     = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# =============================================================================
# STEP 5 — LOCALIZE FOR EACH SUBJECT (Algorithm 5 Steps 1 & 2)
# =============================================================================
subjects     = sorted(merged['subject_id'].unique())
correct_l1   = 0
correct_l2   = 0
total        = len(subjects)
rows         = []

for sid in subjects:
    sub = merged[merged['subject_id'] == sid]

    # --- Anomaly % per sensor (activity) ---
    sensor_pct = {}
    for act in merged['activity_name'].unique():
        act_w = sub[sub['activity_name'] == act]
        sensor_pct[act] = (
            act_w['is_anomaly_pred'].sum() / len(act_w) * 100
        ) if len(act_w) > 0 else 0.0

    # --- Sort descending, take top 5 ---
    sorted_s   = sorted(sensor_pct.items(), key=lambda x: x[1], reverse=True)
    top5       = sorted_s[:5]
    top5_names = [s[0] for s in top5]

    # --- Map to regions ---
    pred_regions = [sensor_to_region[s] for s in top5_names]
    dyn_count    = pred_regions.count('DYNAMIC')
    sta_count    = pred_regions.count('STATIC')
    pred_side    = 'STATIC' if sta_count >= dyn_count else 'DYNAMIC'

    # --- Ground truth from meta_test is_anomaly column ---
    gt_activity = sub[sub['is_anomaly'] == 1]['activity_name'].mode()
    gt_act      = gt_activity.iloc[0] if len(gt_activity) > 0 else 'UNKNOWN'
    gt_region   = sensor_to_region.get(gt_act, 'UNKNOWN')

    # --- Check correctness ---
    l1_ok = (pred_side == gt_region)
    l2_ok = (gt_act in top5_names)
    if l1_ok: correct_l1 += 1
    if l2_ok: correct_l2 += 1

    rows.append({
        'Subject':          sid,
        'GT Activity':      gt_act,
        'GT Region':        gt_region,
        'Top Sensor':       top5[0][0],
        'Top Anomaly %':    round(top5[0][1], 1),
        'Predicted Region': pred_side,
        'Level 1':          '✓' if l1_ok else '✗',
        'Level 2':          '✓' if l2_ok else '✗',
    })

# =============================================================================
# STEP 6 — COMPUTE ACCURACIES + WILSON CI (Algorithm 5 Step 3)
# =============================================================================
acc_l1 = correct_l1 / total
acc_l2 = correct_l2 / total
ci_l1  = wilson_interval(correct_l1, total)
ci_l2  = wilson_interval(correct_l2, total)

# =============================================================================
# STEP 7 — PRINT RESULTS
# =============================================================================
print("\n" + "="*70)
print("TABLE III — Spatial Localization Accuracy")
print("="*70)
print(pd.DataFrame(rows).to_string(index=False))

print("\n")
print(f"{'Granularity':<28} {'Correct':>7} {'Incorrect':>9} {'Accuracy':>9} {'95% CI':>22}")
print("-"*78)
print(f"{'Level 1 (STATIC/DYNAMIC)':<28} {correct_l1:>7} {total-correct_l1:>9} "
      f"{acc_l1*100:>8.1f}%  [{ci_l1[0]*100:.1f}%, {ci_l1[1]*100:.1f}%]")
print(f"{'Level 2 (Specific Activity)':<28} {correct_l2:>7} {total-correct_l2:>9} "
      f"{acc_l2*100:>8.1f}%  [{ci_l2[0]*100:.1f}%, {ci_l2[1]*100:.1f}%]")

# =============================================================================
# STEP 8 — SAVE TABLE III AS CSV
# =============================================================================
table3 = pd.DataFrame([
    {
        'Granularity':    'Level 1 (STATIC/DYNAMIC)',
        'Correct':        correct_l1,
        'Incorrect':      total - correct_l1,
        'Accuracy (%)':   round(acc_l1 * 100, 1),
        'CI_lower (%)':   round(ci_l1[0] * 100, 1),
        'CI_upper (%)':   round(ci_l1[1] * 100, 1),
    },
    {
        'Granularity':    'Level 2 (Specific Activity)',
        'Correct':        correct_l2,
        'Incorrect':      total - correct_l2,
        'Accuracy (%)':   round(acc_l2 * 100, 1),
        'CI_lower (%)':   round(ci_l2[0] * 100, 1),
        'CI_upper (%)':   round(ci_l2[1] * 100, 1),
    },
])
table3.to_csv('table3_localization.csv', index=False)
print("\n✅ Table III saved to table3_localization.csv")
