# =============================================================================
# baseline_bilstm.py — Membre 3 | Sprint 3 Phase 2
# =============================================================================
# Implémente la baseline Supervised BiLSTM pour le Tableau I
#
# Usage :
#   python baseline_bilstm.py --dataset_root ./cleaned
#
# Output :
#   - Affichage console des métriques Tableau I
#   - tableau1_bilstm.csv  (à transmettre à Membre 9)
# =============================================================================

import argparse
import time
import warnings
import numpy as np
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
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
    root = Path(dataset_root) / split
    arrays = []
    for fname in CHANNEL_FILES:
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        arrays.append(np.load(str(path)).astype(np.float32))
    X = np.stack(arrays, axis=-1)                    # (N, 128, 9)
    labels   = np.load(str(root / "labels.npy")).ravel()
    subjects = np.load(str(root / "subjects.npy")).ravel() \
               if (root / "subjects.npy").exists() else None
    return X, labels, subjects


def binarize(labels):
    """0 = normal (WALKING), 1 = anomalie"""
    return np.where(labels == NORMAL_LABEL, 0, 1).astype(np.int64)


# =============================================================================
# MODEL — Supervised BiLSTM (Algorithm 6)
# =============================================================================

class BaselineBiLSTM(nn.Module):
    def __init__(self, input_features=9, hidden=64, dropout=0.3, num_classes=2):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_features, hidden, batch_first=True,
                                bidirectional=True)
        self.drop1   = nn.Dropout(dropout)
        self.bilstm2 = nn.LSTM(hidden * 2, hidden, batch_first=True,
                                bidirectional=True)
        self.drop2      = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:   nn.init.xavier_uniform_(param)
            elif "weight_hh" in name: nn.init.orthogonal_(param)
            elif "bias" in name:      nn.init.zeros_(param)

    def forward(self, x):
        out, _ = self.bilstm1(x)       # (B, T, hidden*2)
        out    = self.drop1(out)
        out, _ = self.bilstm2(out)     # (B, T, hidden*2)
        out    = self.drop2(out)
        out    = out[:, -1, :]         # last timestep
        return self.classifier(out)    # (B, 2)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def run_bilstm(X_train, y_train, X_test, y_test, device):
    print("\n" + "─" * 55)
    print("ALGORITHM 6 — Supervised BiLSTM")
    print("─" * 55)

    # ── Tensors & DataLoaders ─────────────────────────────────
    X_tr = torch.tensor(X_train)
    y_tr = torch.tensor(y_train)
    X_te = torch.tensor(X_test)
    y_te = torch.tensor(y_test)

    # validation split (50% of train as per algorithm6)
    n_val   = len(X_tr) // 2
    X_val, y_val = X_tr[:n_val], y_tr[:n_val]
    X_tr,  y_tr  = X_tr[n_val:], y_tr[n_val:]

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=256)
    test_loader  = DataLoader(TensorDataset(X_te, y_te),  batch_size=256)

    # ── Model, optimizer, loss ────────────────────────────────
    model     = BaselineBiLSTM(input_features=9, hidden=64, dropout=0.3).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux : {total_params:,}")
    print(f"  Device : {device}")

    # ── Early stopping ────────────────────────────────────────
    best_val_loss  = float('inf')
    patience       = 25
    patience_count = 0
    best_weights   = None

    print(f"\n  [Training] max_epochs=1000, patience={patience}, batch=128")
    t0 = time.time()

    for epoch in range(1, 1001):
        # — Train —
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # — Validation —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 50 == 0:
            print(f"    Epoch {epoch:4d} | val_loss={val_loss:.4f} | "
                  f"patience={patience_count}/{patience}")

        if patience_count >= patience:
            print(f"  Early stopping à l'époque {epoch}")
            break

    model.load_state_dict(best_weights)
    print(f"  Entraînement terminé en {time.time()-t0:.1f}s")

    # ── Evaluation ────────────────────────────────────────────
    print("\n  [Evaluation] sur le test set...")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    indiv_acc = _compute_individual_acc(y_true, y_pred)

    metrics = {
        "method":    "Supervised BiLSTM",
        "f1":        round(f1   * 100, 2),
        "precision": round(prec * 100, 2),
        "recall":    round(rec  * 100, 2),
        "accuracy":  round(acc  * 100, 2),
        "indiv_acc": round(indiv_acc * 100, 2),
    }

    _print_metrics(metrics)
    return metrics


# =============================================================================
# UTILS
# =============================================================================

def _compute_individual_acc(y_true, y_pred, group=100):
    correct, total = 0, 0
    for start in range(0, len(y_true), group):
        ct = y_true[start:start+group]
        cp = y_pred[start:start+group]
        maj_pred = 1 if np.sum(cp == 1) > len(cp) / 2 else 0
        maj_true = 1 if np.sum(ct == 1) > len(ct) / 2 else 0
        correct += int(maj_pred == maj_true)
        total   += 1
    return correct / total if total > 0 else 0.0


def _print_metrics(m):
    print(f"\n  ┌─ Résultats {m['method']} ───────────────────────────")
    print(f"  │  F1-Score          : {m['f1']} %")
    print(f"  │  Précision         : {m['precision']} %")
    print(f"  │  Rappel            : {m['recall']} %")
    print(f"  │  Accuracy          : {m['accuracy']} %")
    print(f"  │  Individual Acc.   : {m['indiv_acc']} %")
    print(f"  └─────────────────────────────────────────────────")


def save_csv(results, output_path="tableau1_bilstm.csv"):
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
        description="Baseline Supervised BiLSTM — Membre 3 | Sprint 3 Phase 2"
    )
    parser.add_argument("--dataset_root", type=str, default="./cleaned")
    parser.add_argument("--output", type=str, default="tableau1_bilstm.csv")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 55)
    print("BASELINE SUPERVISED BiLSTM — Membre 3 | Sprint 3 Phase 2")
    print("=" * 55)

    print(f"\nChargement des données depuis : {args.dataset_root}")
    X_train, labels_train, _ = load_split(args.dataset_root, "train")
    X_test,  labels_test,  _ = load_split(args.dataset_root, "test")

    y_train = binarize(labels_train)
    y_test  = binarize(labels_test)

    print(f"  Train : {X_train.shape}  "
          f"(normal={np.sum(y_train==0)}, anomalie={np.sum(y_train==1)})")
    print(f"  Test  : {X_test.shape}   "
          f"(normal={np.sum(y_test==0)}, anomalie={np.sum(y_test==1)})")

    metrics = run_bilstm(X_train, y_train, X_test, y_test, device)

    print("\n" + "=" * 55)
    print("TABLEAU I — Résumé (à transmettre à Membre 9)")
    print("=" * 55)
    print(f"  {'Méthode':<22} {'F1':>8} {'Précision':>10} {'Rappel':>8} {'Indiv.Acc':>10}")
    print(f"  {'─'*60}")
    print(f"  {metrics['method']:<22} {metrics['f1']:>7}% {metrics['precision']:>9}% "
          f"{metrics['recall']:>7}% {metrics['indiv_acc']:>9}%")
    print(f"  {'─'*60}")
    print("  Feature K-means       → résultats Membre 2")
    print("  TSKmeans (DTW)        → résultats Membre 2")
    print("  MedAttnAID (Ours)     → résultats Membre 1")

    save_csv([metrics], args.output)

    print("\n✅ Baseline BiLSTM terminée.")
    print(f"   → Transmettre {args.output} à Membre 4,5 et 6")


if __name__ == "__main__":
    main()