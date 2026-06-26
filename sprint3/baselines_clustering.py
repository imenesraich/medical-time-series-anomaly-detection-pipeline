# =============================================================================
# baselines_clustering.py — Membre 2 | Sprint 3 Phase 2
# =============================================================================
# Implémente deux baselines de clustering pour le Tableau I :
#   - Feature K-means   (Algorithm 7)
#   - TSKmeans DTW      (Algorithm 8)
#
# Usage :
#   python baselines_clustering.py --dataset_root ./cleaned
#
# Output :
#   - Affichage console des métriques Tableau I
#   - tableau1_clustering.csv  (à transmettre à Membre 9)
# =============================================================================

import argparse
import time
import warnings
import numpy as np
import csv
from pathlib import Path
from scipy.stats import skew, kurtosis

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    silhouette_score, calinski_harabasz_score
)

warnings.filterwarnings("ignore")

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

CHANNEL_FILES = [
    "body_acc_x.npy", "body_acc_y.npy", "body_acc_z.npy",
    "body_gyro_x.npy", "body_gyro_y.npy", "body_gyro_z.npy",
    "total_acc_x.npy", "total_acc_y.npy", "total_acc_z.npy",
]

NORMAL_LABEL = 1  # WALKING = normal dans UCI-HAR


def load_split(dataset_root: str, split: str):
    """
    Charge un split depuis les fichiers .npy nettoyés.
    Retourne X (N, 128, 9), labels (N,), subjects (N,)
    """
    root = Path(dataset_root) / split

    arrays = []
    for fname in CHANNEL_FILES:
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        arrays.append(np.load(str(path)).astype(np.float32))

    X = np.stack(arrays, axis=-1)                           # (N, 128, 9)
    labels = np.load(str(root / "labels.npy")).ravel()
    subjects = np.load(str(root / "subjects.npy")).ravel() \
               if (root / "subjects.npy").exists() else None

    return X, labels, subjects


def binarize(labels):
    """0 = normal (WALKING), 1 = anomalie (tout le reste)"""
    return np.where(labels == NORMAL_LABEL, "normal", "anomaly")


# =============================================================================
# ALGORITHM 7 — FEATURE K-MEANS
# =============================================================================

def time_to_percentile(x, p):
    """Retourne l'index temporel où le signal atteint le p-ième percentile."""
    threshold = np.percentile(x, p)
    indices = np.where(x >= threshold)[0]
    return indices[0] if len(indices) > 0 else len(x) - 1


def extract_features(X):
    """
    Extrait les features statistiques et de gradient pour chaque fenêtre.
    X : (N, L, F) → retourne features (N, n_features)
    """
    N, L, F = X.shape
    all_features = []

    for i in range(N):
        window = X[i]          # (L, F)
        f = []

        for ch in range(F):    # pour chaque canal
            sig = window[:, ch]

            # Features statistiques
            f.append(np.mean(sig))
            f.append(np.var(sig))
            f.append(np.std(sig))
            f.append(np.min(sig))
            f.append(np.max(sig))
            f.append(np.max(sig) - np.min(sig))      # range
            f.append(float(skew(sig)))
            f.append(float(kurtosis(sig)))

            # Temps pour atteindre les percentiles
            for p in [20, 40, 60, 80]:
                f.append(time_to_percentile(sig, p))

            # Features gradient
            grad = np.diff(sig)
            f.append(np.mean(grad))
            f.append(np.max(grad))
            f.append(np.std(grad))
            q75, q25 = np.percentile(grad, [75, 25])
            f.append(q75 - q25)    # IQR

        all_features.append(f)

    return np.array(all_features, dtype=np.float32)


def run_feature_kmeans(X_train, y_train_bin, X_test, y_test_bin):
    """
    Algorithm 7 — Feature K-Means.
    Entraîne sur train, évalue sur test.
    """
    print("\n" + "─" * 55)
    print("ALGORITHM 7 — Feature K-Means")
    print("─" * 55)

    # ── Step 1 : Extraction des features ─────────────────────
    print("  [1/5] Extraction des features...")
    t0 = time.time()
    feats_train = extract_features(X_train)
    feats_test  = extract_features(X_test)
    print(f"        Train : {feats_train.shape}  Test : {feats_test.shape}  ({time.time()-t0:.1f}s)")

    # ── Step 2 : Normalisation ────────────────────────────────
    print("  [2/5] Normalisation StandardScaler...")
    scaler = StandardScaler()
    feats_train_sc = scaler.fit_transform(feats_train)
    feats_test_sc  = scaler.transform(feats_test)

    # ── Step 3 : Sélection des 30 meilleures features (RF) ───
    print("  [3/5] Sélection features (Random Forest)...")
    y_train_int = (y_train_bin == "anomaly").astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(feats_train_sc, y_train_int)
    top_indices = np.argsort(rf.feature_importances_)[-30:]
    feats_train_sel = feats_train_sc[:, top_indices]
    feats_test_sel  = feats_test_sc[:, top_indices]
    print(f"        Top 30 features sélectionnées sur {feats_train_sc.shape[1]}")

    # ── Step 4 : K-Means clustering ───────────────────────────
    print("  [4/5] KMeans (k=2)...")
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(feats_train_sel)
    labels_test = kmeans.predict(feats_test_sel)

    # ── Step 5 : Alignement clusters → ground truth ───────────
    print("  [5/5] Alignement et évaluation...")
    y_test_int  = (y_test_bin == "anomaly").astype(int)
    labels_train_pred = kmeans.predict(feats_train_sel)

    c0_normal = np.sum((labels_train_pred == 0) & (y_train_int == 0))
    c1_normal = np.sum((labels_train_pred == 1) & (y_train_int == 0))
    normal_cluster = 0 if c0_normal > c1_normal else 1

    y_pred = np.where(labels_test == normal_cluster, "normal", "anomaly")

    # ── Métriques ────────────────────────────────────────────
    acc  = accuracy_score(y_test_bin, y_pred)
    prec = precision_score(y_test_bin, y_pred, pos_label="anomaly", zero_division=0)
    rec  = recall_score(y_test_bin, y_pred, pos_label="anomaly", zero_division=0)
    f1   = f1_score(y_test_bin, y_pred, pos_label="anomaly", zero_division=0)
    sil  = silhouette_score(feats_test_sel, labels_test)
    ch   = calinski_harabasz_score(feats_test_sel, labels_test)

    # Individual Accuracy (par sujet) — approximation via majority vote
    indiv_acc = _compute_individual_acc(y_test_bin, y_pred)

    metrics = {
        "method":       "Feature K-means",
        "f1":           round(f1 * 100, 2),
        "precision":    round(prec * 100, 2),
        "recall":       round(rec * 100, 2),
        "accuracy":     round(acc * 100, 2),
        "indiv_acc":    round(indiv_acc * 100, 2),
        "silhouette":   round(sil, 4),
        "calinski":     round(ch, 2),
    }

    _print_metrics(metrics)
    return metrics


# =============================================================================
# ALGORITHM 8 — TSKmeans DTW
# =============================================================================

def run_tskmeans_dtw(X_train, y_train_bin, X_test, y_test_bin):
    """
    Algorithm 8 — Time-Series K-Means with DTW.
    """
    print("\n" + "─" * 55)
    print("ALGORITHM 8 — TSKmeans DTW")
    print("─" * 55)

    try:
        from tslearn.clustering import TimeSeriesKMeans
    except ImportError:
        print("  ❌ tslearn non installé. Lancer : pip install tslearn")
        return None

    # TSKmeans travaille sur un seul canal (on prend body_acc_x = index 0)
    # Shape attendu par tslearn : (N, L, 1)
    X_train_ts = X_train[:, :, 0:1].astype(np.float64)
    X_test_ts  = X_test[:, :, 0:1].astype(np.float64)

    # Sous-échantillonnage si trop lent (TSKmeans DTW est O(N²))
    MAX_SAMPLES = 1000
    if len(X_train_ts) > MAX_SAMPLES:
        print(f"  ⚠ Train trop grand ({len(X_train_ts)}) → sous-échantillonnage à {MAX_SAMPLES}")
        idx = np.random.RandomState(42).choice(len(X_train_ts), MAX_SAMPLES, replace=False)
        X_train_ts_sub = X_train_ts[idx]
        y_train_sub    = y_train_bin[idx]
    else:
        X_train_ts_sub = X_train_ts
        y_train_sub    = y_train_bin

    print(f"  Fitting TSKmeans DTW sur {len(X_train_ts_sub)} séries...")
    t0 = time.time()

    model = TimeSeriesKMeans(
        n_clusters=2,
        metric="dtw",
        max_iter=10,
        random_state=42,
        n_init=2,
        n_jobs=-1 if hasattr(TimeSeriesKMeans, 'n_jobs') else 1,
    )
    model.fit(X_train_ts_sub)
    print(f"  Fit terminé en {time.time()-t0:.1f}s")

    # Prédiction sur le test
    labels_test = model.predict(X_test_ts)

    # Alignement clusters → ground truth (sur le train sub)
    labels_train_pred = model.predict(X_train_ts_sub)
    y_train_int = (y_train_sub == "anomaly").astype(int)
    c0_normal = np.sum((labels_train_pred == 0) & (y_train_int == 0))
    c1_normal = np.sum((labels_train_pred == 1) & (y_train_int == 0))
    normal_cluster = 0 if c0_normal > c1_normal else 1

    y_pred = np.where(labels_test == normal_cluster, "normal", "anomaly")

    # Métriques
    y_test_int = (y_test_bin == "anomaly").astype(int)
    acc  = accuracy_score(y_test_bin, y_pred)
    prec = precision_score(y_test_bin, y_pred, pos_label="anomaly", zero_division=0)
    rec  = recall_score(y_test_bin, y_pred, pos_label="anomaly", zero_division=0)
    f1   = f1_score(y_test_bin, y_pred, pos_label="anomaly", zero_division=0)
    indiv_acc = _compute_individual_acc(y_test_bin, y_pred)

    metrics = {
        "method":    "TSKmeans (DTW)",
        "f1":        round(f1 * 100, 2),
        "precision": round(prec * 100, 2),
        "recall":    round(rec * 100, 2),
        "accuracy":  round(acc * 100, 2),
        "indiv_acc": round(indiv_acc * 100, 2),
        "silhouette": "N/A",
        "calinski":   "N/A",
    }

    _print_metrics(metrics)
    return metrics


# =============================================================================
# UTILS
# =============================================================================

def _compute_individual_acc(y_true, y_pred):
    """
    Individual accuracy : pour chaque 'sujet' simulé (groupes de 100 fenêtres),
    majority vote → accuracy.
    """
    N = len(y_true)
    GROUP = 100
    correct = 0
    total = 0
    for start in range(0, N, GROUP):
        chunk_true = y_true[start:start+GROUP]
        chunk_pred = y_pred[start:start+GROUP]
        majority_pred = "anomaly" if np.sum(chunk_pred == "anomaly") > len(chunk_pred)/2 else "normal"
        majority_true = "anomaly" if np.sum(chunk_true == "anomaly") > len(chunk_true)/2 else "normal"
        correct += int(majority_pred == majority_true)
        total += 1
    return correct / total if total > 0 else 0.0


def _print_metrics(m):
    print(f"\n  ┌─ Résultats {m['method']} ───────────────────────────")
    print(f"  │  F1-Score          : {m['f1']} %")
    print(f"  │  Précision         : {m['precision']} %")
    print(f"  │  Rappel            : {m['recall']} %")
    print(f"  │  Accuracy          : {m['accuracy']} %")
    print(f"  │  Individual Acc.   : {m['indiv_acc']} %")
    if m['silhouette'] != "N/A":
        print(f"  │  Silhouette        : {m['silhouette']}")
        print(f"  │  Calinski-Harabasz : {m['calinski']}")
    print(f"  └─────────────────────────────────────────────────")


def save_csv(results, output_path="tableau1_clustering.csv"):
    """Sauvegarde les résultats dans un CSV pour Membre 9."""
    fieldnames = ["method", "f1", "precision", "recall", "accuracy", "indiv_acc"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r is not None:
                writer.writerow({k: r[k] for k in fieldnames})
    print(f"\n  ✔ CSV sauvegardé → {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baselines Clustering — Feature K-means + TSKmeans DTW"
    )
    parser.add_argument(
        "--dataset_root", type=str, default="./cleaned",
        help="Chemin vers le dossier cleaned/ contenant train/ et test/"
    )
    parser.add_argument(
        "--output", type=str, default="tableau1_clustering.csv",
        help="Fichier CSV de sortie pour Membre 9"
    )
    parser.add_argument(
        "--skip_dtw", action="store_true",
        help="Ignorer TSKmeans DTW (plus lent, nécessite tslearn)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("BASELINES CLUSTERING — Membre 2 | Sprint 3 Phase 2")
    print("=" * 55)

    # ── Chargement ─────────────────────────────────────────────
    print(f"\nChargement des données depuis : {args.dataset_root}")
    X_train, labels_train, _ = load_split(args.dataset_root, "train")
    X_test,  labels_test,  _ = load_split(args.dataset_root, "test")

    y_train_bin = binarize(labels_train)
    y_test_bin  = binarize(labels_test)

    print(f"  Train : {X_train.shape}  "
          f"(normal={np.sum(y_train_bin=='normal')}, "
          f"anomalie={np.sum(y_train_bin=='anomaly')})")
    print(f"  Test  : {X_test.shape}   "
          f"(normal={np.sum(y_test_bin=='normal')}, "
          f"anomalie={np.sum(y_test_bin=='anomaly')})")

    results = []

    # ── Algorithm 7 : Feature K-Means ──────────────────────────
    m7 = run_feature_kmeans(X_train, y_train_bin, X_test, y_test_bin)
    results.append(m7)

    # ── Algorithm 8 : TSKmeans DTW ─────────────────────────────
    if not args.skip_dtw:
        m8 = run_tskmeans_dtw(X_train, y_train_bin, X_test, y_test_bin)
        results.append(m8)
    else:
        print("\n  ⚠ TSKmeans DTW ignoré (--skip_dtw)")

    # ── Tableau I — Résumé ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("TABLEAU I — Résumé (à transmettre à Membre 9)")
    print("=" * 55)
    print(f"  {'Méthode':<22} {'F1':>8} {'Précision':>10} {'Rappel':>8} {'Indiv.Acc':>10}")
    print(f"  {'─'*60}")
    for r in results:
        if r:
            print(f"  {r['method']:<22} {r['f1']:>7}% {r['precision']:>9}% "
                  f"{r['recall']:>7}% {r['indiv_acc']:>9}%")
    print(f"  {'─'*60}")
    print("  Supervised BiLSTM     → résultats Membre 3")
    print("  MedAttnAID (Ours)     → résultats Membre 1")

    # ── Sauvegarde CSV ──────────────────────────────────────────
    save_csv(results, args.output)

    print("\n✅ Baselines clustering terminées.")
    print(f"   → Transmettre {args.output} à Membre 9")


if __name__ == "__main__":
    main()