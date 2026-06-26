"""
=============================================================================
UCI HAR Dataset — Signal Cleaning & Normalization ETL Pipeline
Sprint 2 | Topic M5: Human Activity Anomaly Detection
=============================================================================

USAGE:
    python etl_cleaning.py

OUTPUT:
    cleaned/
    ├── train/          → cleaned .npy arrays per channel
    ├── test/           → cleaned .npy arrays per channel
    ├── norm_params.json        → z-score mean/std per channel (fit on train)
    └── cleaning_log.json       → full audit & transformation log
=============================================================================
"""

import os
import json
import time
import warnings
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# =============================================================================
#  ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
# ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
# ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
# ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
# =============================================================================
# Tweak any value here — no need to touch the code below.
# =============================================================================

CONFIG = {

    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------
    # Root folder of the extracted UCI HAR Dataset
    "dataset_root": "./UCI HAR Dataset",

    # Where to write cleaned outputs
    "output_dir": "./cleaned",

    # ------------------------------------------------------------------
    # MISSING VALUE STRATEGY
    # Strategy choices:
    #   "interpolate"  → fill gaps using linear interpolation between
    #                    neighboring valid samples (recommended)
    #   "zero"         → replace NaN / Inf with 0.0
    #   "mean"         → replace with the channel mean (computed on train)
    #   "drop"         → remove entire windows that contain any NaN/Inf
    # ------------------------------------------------------------------
    "missing_strategy": "interpolate",

    # ------------------------------------------------------------------
    # FLAT-WINDOW (SENSOR DROPOUT) DETECTION
    # A window is considered "dead" if its standard deviation across
    # all 128 samples is below this threshold.
    # Set to 0.0 to disable dropout detection.
    # ------------------------------------------------------------------
    "dropout_std_threshold": 1e-6,

    # What to do with dropout windows:
    #   "interpolate"  → replace with the mean of the two neighboring windows
    #   "drop"         → mark and remove from the dataset
    # ------------------------------------------------------------------
    "dropout_strategy": "interpolate",

    # ------------------------------------------------------------------
    # OUTLIER CLIPPING
    # Clip sample values that exceed ± N standard deviations.
    # Computed per channel across the entire training set.
    # Set to None to disable clipping.
    # ------------------------------------------------------------------
    "outlier_clip_std": 3.0,

    # ------------------------------------------------------------------
    # NOISE FILTERING  (Butterworth low-pass)
    # The UCI HAR signals were originally filtered with a median filter
    # and a 3rd-order Butterworth at 20 Hz. We apply an additional
    # light filter to smooth any residual high-frequency noise.
    #
    # "filter_enabled"  : True / False
    # "filter_cutoff_hz": cutoff frequency (Hz) — 20 Hz is standard
    # "filter_order"    : filter order (3–5 is typical)
    # "sampling_rate_hz": original sampling rate of UCI HAR = 50 Hz
    # ------------------------------------------------------------------
    "filter_enabled": True,
    "filter_cutoff_hz": 20.0,
    "filter_order": 3,
    "sampling_rate_hz": 50,

    # ------------------------------------------------------------------
    # RESAMPLING
    # Set target_hz to a value < 50 to downsample, or to 50 to keep
    # the original rate (no resampling performed).
    # ------------------------------------------------------------------
    "target_hz": 50,           # change to e.g. 25 to downsample

    # ------------------------------------------------------------------
    # NORMALIZATION
    # Strategy choices:
    #   "zscore"   → subtract mean, divide by std  (recommended)
    #   "minmax"   → scale to [0, 1]
    #   "none"     → skip normalization
    #
    # IMPORTANT: statistics are ALWAYS fit on the training set only,
    # then applied to the test set — no data leakage.
    # ------------------------------------------------------------------
    "norm_strategy": "zscore",
}

# =============================================================================
# CHANNEL DEFINITIONS
# =============================================================================

CHANNELS = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

ACTIVITY_LABELS = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}

# =============================================================================
# LOGGING HELPERS
# =============================================================================

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg, level="INFO"):
    prefix = {"INFO": "ℹ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "STEP": "🔹"}
    print(f"[{_ts()}] {prefix.get(level, ' ')} {msg}")

# =============================================================================
# STEP 1 — DATA LOADING
# =============================================================================

def load_signals(root: str, split: str) -> dict:
    """Load all 9 raw inertial signal files for a given split."""
    data = {}
    sig_dir = os.path.join(root, split, "Inertial Signals")
    for ch in CHANNELS:
        path = os.path.join(sig_dir, f"{ch}_{split}.txt")
        arr = np.loadtxt(path)          # shape: (n_windows, 128)
        data[ch] = arr
        log(f"  Loaded {ch}_{split}.txt  → shape {arr.shape}")
    return data

def load_labels(root: str, split: str) -> np.ndarray:
    path = os.path.join(root, split, f"y_{split}.txt")
    return np.loadtxt(path, dtype=int)

def load_subjects(root: str, split: str) -> np.ndarray:
    path = os.path.join(root, split, f"subject_{split}.txt")
    return np.loadtxt(path, dtype=int)

# =============================================================================
# STEP 2 — AUDIT (before any cleaning)
# =============================================================================

def audit_split(data: dict, split_name: str) -> dict:
    """Return a per-channel audit report."""
    report = {"split": split_name, "channels": {}}
    for ch, arr in data.items():
        n_win, n_samp = arr.shape
        nan_count  = int(np.sum(np.isnan(arr)))
        inf_count  = int(np.sum(np.isinf(arr)))
        flat_mask  = np.std(arr, axis=1) < CONFIG["dropout_std_threshold"]
        flat_count = int(np.sum(flat_mask))
        # Outliers beyond ±3σ (computed locally for audit only)
        mu, sigma = np.nanmean(arr), np.nanstd(arr)
        outlier_count = int(np.sum(np.abs(arr - mu) > 3 * sigma)) if sigma > 0 else 0
        report["channels"][ch] = {
            "windows": n_win, "samples_per_window": n_samp,
            "nan_count": nan_count, "inf_count": inf_count,
            "flat_windows": flat_count,
            "outlier_samples (>3σ)": outlier_count,
            "value_min": float(np.nanmin(arr)),
            "value_max": float(np.nanmax(arr)),
            "value_mean": round(float(np.nanmean(arr)), 6),
            "value_std":  round(float(np.nanstd(arr)),  6),
        }
    return report

# =============================================================================
# STEP 3 — MISSING VALUE HANDLING
# =============================================================================

def handle_missing(arr: np.ndarray, strategy: str,
                   fill_value: float = 0.0) -> tuple[np.ndarray, dict]:
    """
    Replace NaN and Inf values in arr (n_windows, 128).
    Returns cleaned array and a stats dict.
    """
    n_nan  = int(np.sum(np.isnan(arr)))
    n_inf  = int(np.sum(np.isinf(arr)))
    total  = n_nan + n_inf
    arr    = arr.copy()
    arr[np.isinf(arr)] = np.nan     # treat Inf as NaN

    dropped_windows = []

    if total == 0:
        return arr, {"strategy": strategy, "replaced": 0, "dropped_windows": []}

    if strategy == "interpolate":
        for w in range(arr.shape[0]):
            row = arr[w]
            nans = np.isnan(row)
            if nans.any():
                x     = np.arange(len(row))
                valid = ~nans
                if valid.sum() < 2:
                    arr[w] = 0.0    # fallback if almost whole window is NaN
                else:
                    f = interp1d(x[valid], row[valid],
                                 kind="linear", fill_value="extrapolate")
                    arr[w] = f(x)

    elif strategy == "zero":
        arr = np.nan_to_num(arr, nan=0.0)

    elif strategy == "mean":
        col_means = np.nanmean(arr, axis=0)     # shape (128,)
        for w in range(arr.shape[0]):
            mask = np.isnan(arr[w])
            arr[w, mask] = col_means[mask]
        arr = np.nan_to_num(arr, nan=0.0)       # safety net

    elif strategy == "drop":
        bad_rows = np.any(np.isnan(arr), axis=1)
        dropped_windows = list(np.where(bad_rows)[0])
        arr = arr[~bad_rows]

    return arr, {
        "strategy": strategy,
        "nan_found": n_nan,
        "inf_found": n_inf,
        "replaced": total,
        "dropped_windows": dropped_windows,
    }

# =============================================================================
# STEP 4 — FLAT-WINDOW (DROPOUT) DETECTION
# =============================================================================

def handle_dropouts(arr: np.ndarray, strategy: str,
                    threshold: float) -> tuple[np.ndarray, dict]:
    """Detect and fix sensor dropout windows (near-zero variance)."""
    if threshold <= 0.0:
        return arr, {"dropout_windows_found": 0, "strategy": "disabled"}

    stds = np.std(arr, axis=1)
    flat_mask = stds < threshold
    flat_idx  = list(np.where(flat_mask)[0])

    if not flat_idx:
        return arr, {"dropout_windows_found": 0, "strategy": strategy}

    arr = arr.copy()

    if strategy == "interpolate":
        for idx in flat_idx:
            prev = idx - 1 if idx > 0 else idx + 1
            nxt  = idx + 1 if idx < len(arr) - 1 else idx - 1
            arr[idx] = (arr[prev] + arr[nxt]) / 2.0

    elif strategy == "drop":
        arr = arr[~flat_mask]

    return arr, {
        "dropout_windows_found": len(flat_idx),
        "dropout_indices": flat_idx[:20],    # log first 20 only
        "strategy": strategy,
    }

# =============================================================================
# STEP 5 — NOISE FILTERING
# =============================================================================

def butterworth_lowpass(arr: np.ndarray, cutoff: float,
                        fs: float, order: int) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter row-by-row."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered = np.apply_along_axis(
        lambda row: filtfilt(b, a, row), axis=1, arr=arr
    )
    return filtered

# =============================================================================
# STEP 6 — OUTLIER CLIPPING
# =============================================================================

def clip_outliers(arr: np.ndarray, n_std: float,
                  mu: float = None, sigma: float = None
                  ) -> tuple[np.ndarray, dict]:
    """
    Clip values beyond ± n_std standard deviations.
    If mu / sigma are provided (from train), use them (for test set).
    """
    if mu is None:
        mu    = float(np.mean(arr))
        sigma = float(np.std(arr))

    lo = mu - n_std * sigma
    hi = mu + n_std * sigma
    n_clipped = int(np.sum((arr < lo) | (arr > hi)))
    arr = np.clip(arr, lo, hi)
    return arr, {
        "clip_lo": round(lo, 6), "clip_hi": round(hi, 6),
        "n_clipped": n_clipped, "channel_mean": round(mu, 6),
        "channel_std": round(sigma, 6),
    }

# =============================================================================
# STEP 7 — RESAMPLING
# =============================================================================

def resample_signals(data: dict, original_hz: int,
                     target_hz: int) -> tuple[dict, dict]:
    """Downsample windows from original_hz to target_hz if needed."""
    if target_hz == original_hz:
        log("  Resampling: skipped (target == original Hz)", "INFO")
        return data, {"resampled": False, "original_hz": original_hz,
                      "target_hz": target_hz}

    factor      = original_hz / target_hz
    orig_len    = next(iter(data.values())).shape[1]   # 128
    new_len     = int(orig_len / factor)
    x_old       = np.linspace(0, 1, orig_len)
    x_new       = np.linspace(0, 1, new_len)
    resampled   = {}

    for ch, arr in data.items():
        out = np.zeros((arr.shape[0], new_len))
        for w in range(arr.shape[0]):
            f = interp1d(x_old, arr[w], kind="linear")
            out[w] = f(x_new)
        resampled[ch] = out

    log(f"  Resampled {orig_len} → {new_len} samples/window "
        f"({original_hz} Hz → {target_hz} Hz)", "OK")
    return resampled, {
        "resampled": True,
        "original_hz": original_hz, "target_hz": target_hz,
        "original_samples_per_window": orig_len,
        "new_samples_per_window": new_len,
    }

# =============================================================================
# STEP 8 — NORMALIZATION
# =============================================================================

def compute_norm_params(data: dict, strategy: str) -> dict:
    """Compute normalization parameters from training data."""
    params = {}
    for ch, arr in data.items():
        if strategy == "zscore":
            params[ch] = {"mean": float(np.mean(arr)),
                          "std":  float(np.std(arr))}
        elif strategy == "minmax":
            params[ch] = {"min": float(np.min(arr)),
                          "max": float(np.max(arr))}
    return params

def apply_normalization(data: dict, params: dict,
                        strategy: str) -> dict:
    """Apply normalization using pre-computed parameters."""
    normalized = {}
    for ch, arr in data.items():
        if strategy == "zscore":
            mu    = params[ch]["mean"]
            sigma = params[ch]["std"]
            normalized[ch] = (arr - mu) / (sigma + 1e-8)
        elif strategy == "minmax":
            lo    = params[ch]["min"]
            hi    = params[ch]["max"]
            normalized[ch] = (arr - lo) / ((hi - lo) + 1e-8)
        elif strategy == "none":
            normalized[ch] = arr
    return normalized

# =============================================================================
# STEP 9 — FULL PIPELINE
# =============================================================================

def run_pipeline():
    start_time  = time.time()
    cfg         = CONFIG
    root        = cfg["dataset_root"]
    out_dir     = cfg["output_dir"]
    full_log    = {
        "pipeline_config": cfg,
        "run_timestamp": datetime.now().isoformat(),
        "audit_before": {},
        "audit_after":  {},
        "transformations": {},
        "normalization_params": {},
        "resampling": {},
    }

    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"),  exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    log("=" * 60, "STEP")
    log("STEP 1 — Loading raw signals", "STEP")
    log("=" * 60, "STEP")
    raw = {}
    for split in ("train", "test"):
        log(f"Loading {split} split...")
        raw[split] = {
            "signals":  load_signals(root, split),
            "labels":   load_labels(root, split),
            "subjects": load_subjects(root, split),
        }
    log("All splits loaded.", "OK")

    # ── Audit BEFORE ─────────────────────────────────────────────────
    log("=" * 60, "STEP")
    log("STEP 2 — Auditing raw data (before cleaning)", "STEP")
    log("=" * 60, "STEP")
    for split in ("train", "test"):
        audit = audit_split(raw[split]["signals"], split)
        full_log["audit_before"][split] = audit
        for ch, info in audit["channels"].items():
            log(f"  [{split}] {ch}: NaN={info['nan_count']}, "
                f"Inf={info['inf_count']}, flat_windows={info['flat_windows']}, "
                f"outliers={info['outlier_samples (>3σ)']}")

    # ── Clean each split ─────────────────────────────────────────────
    cleaned = {}
    for split in ("train", "test"):
        log("=" * 60, "STEP")
        log(f"STEP 3-7 — Cleaning [{split}]", "STEP")
        log("=" * 60, "STEP")
        signals = raw[split]["signals"].copy()
        trans_log = {}

        for ch in CHANNELS:
            arr = signals[ch]
            ch_log = {}

            # — Missing values ——————————————————————————————————————
            arr, mv_info = handle_missing(arr, cfg["missing_strategy"])
            ch_log["missing_values"] = mv_info
            if mv_info["replaced"] > 0:
                log(f"  [{split}][{ch}] Missing: {mv_info}", "WARN")

            # — Dropout windows ————————————————————————————————————
            arr, do_info = handle_dropouts(
                arr, cfg["dropout_strategy"], cfg["dropout_std_threshold"]
            )
            ch_log["dropout"] = do_info
            if do_info["dropout_windows_found"] > 0:
                log(f"  [{split}][{ch}] Dropout: {do_info['dropout_windows_found']} windows", "WARN")

            # — Noise filter ———————————————————————————————————————
            if cfg["filter_enabled"]:
                arr = butterworth_lowpass(
                    arr, cfg["filter_cutoff_hz"],
                    cfg["sampling_rate_hz"], cfg["filter_order"]
                )
                ch_log["noise_filter"] = {
                    "type": "Butterworth low-pass",
                    "cutoff_hz": cfg["filter_cutoff_hz"],
                    "order": cfg["filter_order"],
                    "applied": True,
                }

            # — Outlier clipping ———————————————————————————————————
            if cfg["outlier_clip_std"] is not None:
                # For test set, use train statistics to avoid leakage
                if split == "test" and ch in full_log.get("clip_params", {}):
                    p = full_log["clip_params"][ch]
                    arr, clip_info = clip_outliers(
                        arr, cfg["outlier_clip_std"], p["mean"], p["std"]
                    )
                else:
                    arr, clip_info = clip_outliers(arr, cfg["outlier_clip_std"])
                    if split == "train":
                        full_log.setdefault("clip_params", {})[ch] = {
                            "mean": float(np.mean(signals[ch])),
                            "std":  float(np.std(signals[ch])),
                        }
                ch_log["outlier_clipping"] = clip_info
                if clip_info["n_clipped"] > 0:
                    log(f"  [{split}][{ch}] Clipped {clip_info['n_clipped']} samples", "WARN")

            signals[ch] = arr
            trans_log[ch] = ch_log

        full_log["transformations"][split] = trans_log

        # — Resampling ————————————————————————————————————————————
        log(f"  [{split}] Applying resampling check...")
        signals, rs_info = resample_signals(
            signals, cfg["sampling_rate_hz"], cfg["target_hz"]
        )
        full_log["resampling"][split] = rs_info

        cleaned[split] = {
            "signals":  signals,
            "labels":   raw[split]["labels"],
            "subjects": raw[split]["subjects"],
        }

    # ── Normalization (fit on TRAIN, apply to both) ───────────────────
    log("=" * 60, "STEP")
    log("STEP 8 — Normalization", "STEP")
    log("=" * 60, "STEP")
    norm_params = {}

    if cfg["norm_strategy"] != "none":
        log(f"  Computing {cfg['norm_strategy']} params from TRAIN set...")
        norm_params = compute_norm_params(
            cleaned["train"]["signals"], cfg["norm_strategy"]
        )
        for split in ("train", "test"):
            cleaned[split]["signals"] = apply_normalization(
                cleaned[split]["signals"], norm_params, cfg["norm_strategy"]
            )
            log(f"  [{split}] Normalization applied.", "OK")
    else:
        log("  Normalization: skipped (strategy='none')")

    full_log["normalization_params"] = norm_params

    # ── Audit AFTER ──────────────────────────────────────────────────
    log("=" * 60, "STEP")
    log("STEP 9 — Auditing cleaned data (after cleaning)", "STEP")
    log("=" * 60, "STEP")
    for split in ("train", "test"):
        audit = audit_split(cleaned[split]["signals"], split)
        full_log["audit_after"][split] = audit
        for ch, info in audit["channels"].items():
            log(f"  [{split}] {ch}: range=[{info['value_min']:.3f}, "
                f"{info['value_max']:.3f}]  mean={info['value_mean']:.4f}  "
                f"std={info['value_std']:.4f}")

    # ── Save outputs ─────────────────────────────────────────────────
    log("=" * 60, "STEP")
    log("STEP 10 — Saving cleaned outputs", "STEP")
    log("=" * 60, "STEP")
    for split in ("train", "test"):
        split_dir = os.path.join(out_dir, split)
        for ch, arr in cleaned[split]["signals"].items():
            path = os.path.join(split_dir, f"{ch}.npy")
            np.save(path, arr)
        np.save(os.path.join(split_dir, "labels.npy"),   cleaned[split]["labels"])
        np.save(os.path.join(split_dir, "subjects.npy"), cleaned[split]["subjects"])
        log(f"  [{split}] Saved {len(CHANNELS)} signal arrays + labels + subjects → {split_dir}", "OK")

    # Normalization params
    norm_path = os.path.join(out_dir, "norm_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    log(f"  Normalization params saved → {norm_path}", "OK")

    # Full cleaning log
    elapsed = round(time.time() - start_time, 2)
    full_log["elapsed_seconds"] = elapsed
    log_path = os.path.join(out_dir, "cleaning_log.json")
    with open(log_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)
    log(f"  Cleaning log saved → {log_path}", "OK")

    # ── Summary ──────────────────────────────────────────────────────
    log("=" * 60, "STEP")
    log("PIPELINE COMPLETE", "OK")
    log("=" * 60, "STEP")
    log(f"  Total runtime     : {elapsed}s")
    log(f"  Missing strategy  : {cfg['missing_strategy']}")
    log(f"  Dropout strategy  : {cfg['dropout_strategy']}")
    log(f"  Noise filter      : {'Yes (' + str(cfg['filter_cutoff_hz']) + ' Hz)' if cfg['filter_enabled'] else 'No'}")
    log(f"  Outlier clipping  : ±{cfg['outlier_clip_std']}σ" if cfg['outlier_clip_std'] else "  Outlier clipping  : disabled")
    log(f"  Normalization     : {cfg['norm_strategy']}")
    log(f"  Resampling        : {cfg['sampling_rate_hz']} Hz → {cfg['target_hz']} Hz")
    log(f"  Output directory  : {out_dir}")

    return cleaned, full_log


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    cleaned_data, log_data = run_pipeline()