# InfluxDB Ingestion — Membre 4 | Sprint 2
**Topic M5 : Human Activity Anomaly Detection**
**Date de génération :** 2026-05-12

---

## Vue d'ensemble

Ce dossier contient les scripts d'ingestion des données HAR (Human Activity Recognition)
dans InfluxDB. Les données proviennent de **Membre 1** (nettoyage) et **Membre 2** (windowing).

```
member4_influxdb/
├── config.py           → Paramètres de connexion InfluxDB
├── ingest.py           → Script d'ingestion (raw signals + windowed features)
├── verify.py           → Script de vérification
└── README_influxdb.md  → Ce fichier
```

---

## Connexion à InfluxDB

| Paramètre    | Valeur                  |
|---|---|
| URL          | `http://localhost:8086` |
| Organisation | `ESI`                   |
| Bucket       | `har_data`              |
| Token        | *(voir config.py)*      |

> **Note :** InfluxDB tourne via Docker sur la machine de Membre 4.
> Assurez-vous d'être sur le même réseau ou demandez l'IP locale.

### Connexion Python rapide

```python
from influxdb_client import InfluxDBClient

client = InfluxDBClient(
    url="http://localhost:8086",
    token="VOTRE_TOKEN_ICI",
    org="ESI"
)
query_api = client.query_api()
```

---

## Measurements chargées

### 1. `har_kinematic_signals`
Signaux bruts à 50 Hz — **1,318,272 points** au total.

| Propriété     | Valeur                        |
|---|---|
| Résolution    | 20 ms (50 Hz)                 |
| Points total  | 1,318,272                     |
| Train         | 941,056 points (7352 fenêtres × 128 samples) |
| Test          | 377,216 points (2947 fenêtres × 128 samples) |

**Tags :**
| Tag          | Description                        | Exemple         |
|---|---|---|
| `subject_id` | Identifiant sujet (1–30)           | `"1"`           |
| `activity`   | Nom de l'activité                  | `"WALKING"`     |
| `is_anomaly` | 0 = Normal, 1 = Anomalie           | `"0"`           |
| `split`      | Jeu de données                     | `"train"`       |
| `window_id`  | UUID unique de la fenêtre          | `"a3f2bc..."`   |

**Fields (9 canaux) :**
| Field          | Description                  |
|---|---|
| `body_acc_x`   | Accélération corporelle X    |
| `body_acc_y`   | Accélération corporelle Y    |
| `body_acc_z`   | Accélération corporelle Z    |
| `body_gyro_x`  | Gyroscope corporel X         |
| `body_gyro_y`  | Gyroscope corporel Y         |
| `body_gyro_z`  | Gyroscope corporel Z         |
| `total_acc_x`  | Accélération totale X        |
| `total_acc_y`  | Accélération totale Y        |
| `total_acc_z`  | Accélération totale Z        |

---

### 2. `har_windowed_features`
Statistiques agrégées par fenêtre — **10,299 points** au total.

| Propriété    | Valeur                              |
|---|---|
| Résolution   | 2.56 s / fenêtre                    |
| Points total | 10,299                              |
| Train        | 7,352 fenêtres                      |
| Test         | 2,947 fenêtres                      |

**Tags :** identiques à `har_kinematic_signals` (subject_id, activity, is_anomaly, split, window_id)

**Fields (37 au total) :**
- `activity_label` — label numérique de l'activité (1–6)
- Pour chacun des 9 canaux : `{canal}_mean`, `{canal}_std`, `{canal}_min`, `{canal}_max`
- Exemple : `body_acc_x_mean`, `body_acc_x_std`, `body_acc_x_min`, `body_acc_x_max`

---

## Distribution des données

### Par activité

| Activité             | Type    | Fenêtres |
|---|---|---|
| WALKING              | Normal  | 1,722    |
| WALKING_UPSTAIRS     | Normal  | 1,544    |
| WALKING_DOWNSTAIRS   | Normal  | 1,406    |
| SITTING              | Anomalie| 1,777    |
| STANDING             | Anomalie| 1,906    |
| LAYING               | Anomalie| 1,944    |
| **TOTAL**            |         | **10,299** |

### Par split

| Split | Fenêtres | Sujets |
|---|---|---|
| Train | 7,352    | 21     |
| Test  | 2,947    | 9      |

### Anomalies (détection comportementale)

| Label      | Fenêtres |
|---|---|
| Normal (0) | 8,866    |
| Anomalie (1)| 1,433   |

> **Important :** Les anomalies sont détectées par **seuils de signal** (variance,
> magnitude d'accélération, stabilité gyroscope) — pas par label d'activité.
> C'est la stratégie de Membre 2. SITTING/STANDING/LAYING ne sont donc pas
> systématiquement des anomalies si leurs signaux restent dans les seuils normaux.

---

## Logique des timestamps

```
t_ns = BASE_TIME + window_index × 2,560,000,000 ns + sample_index × 20,000,000 ns
```

- `BASE_TIME` = 2024-01-01 00:00:00 UTC en nanosecondes
- Chaque fenêtre démarre à `window_index × 2.56s`
- Chaque sample est espacé de `0.02s` (50 Hz)

---

## Exemples de requêtes Flux

### Récupérer les signaux d'un sujet

```flux
from(bucket: "har_data")
  |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "har_kinematic_signals")
  |> filter(fn: (r) => r._field == "body_acc_x")
  |> filter(fn: (r) => r.subject_id == "1")
```

### Downsampling à 1 seconde

```flux
from(bucket: "har_data")
  |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "har_kinematic_signals")
  |> filter(fn: (r) => r._field == "body_acc_x")
  |> filter(fn: (r) => r.subject_id == "1")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
```

### Récupérer uniquement les anomalies

```flux
from(bucket: "har_data")
  |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "har_windowed_features")
  |> filter(fn: (r) => r._field == "body_acc_x_mean")
  |> filter(fn: (r) => r.is_anomaly == "1")
```

### Récupérer une activité spécifique

```flux
from(bucket: "har_data")
  |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "har_windowed_features")
  |> filter(fn: (r) => r._field == "activity_label")
  |> filter(fn: (r) => r.activity == "WALKING")
  |> filter(fn: (r) => r.split == "train")
```

---

## Instructions par membre

### → Membres 5, 6, 7 (Stress tests)
- Connectez-vous à `http://localhost:8086` avec le token dans `config.py`
- Bucket cible : `har_data`
- Pour les **write tests** (Membre 5) : utilisez `ingest.py` comme référence de débit
- Pour les **range query tests** (Membre 6) : utilisez `har_kinematic_signals` avec des filtres `range()`
- Pour les **aggregation tests** (Membre 7) : utilisez `aggregateWindow()` sur `body_acc_x`

### → Membre 8 (Storage analysis)
- Accédez à l'UI InfluxDB : `http://localhost:8086`
- Allez dans **Settings → Usage** pour voir la taille du bucket
- Ou via CLI Docker :
```bash
docker exec influxdb du -sh /var/lib/influxdb2/engine/data/
```

### → Membre 9 (Documentation)
- Measurement 1 : `har_kinematic_signals` — 9 fields, tags: subject_id/activity/is_anomaly/split/window_id
- Measurement 2 : `har_windowed_features` — 37 fields, mêmes tags
- Stratégie d'anomalie : **comportementale** (seuils de signal, pas labels d'activité)
- Timestamp base : 2024-01-01T00:00:00Z, résolution nanoseconde

### → Membre 10 (Rapport final)
- Voir le screenshot `verify_output.png` pour les résultats de vérification
- Points ingérés : **1,318,272** raw + **10,299** windowed = **1,328,571 total**
- Downsampling vérifié : 1,222 bins à 1s pour sujet 1

---

## Validation — Résumé

| Vérification                  | Résultat                    |
|---|---|
| Raw signal points             | ✅ 1,318,272                |
| Windowed feature rows         | ✅ 10,299                   |
| Train split                   | ✅ 7,352 fenêtres           |
| Test split                    | ✅ 2,947 fenêtres           |
| Total activités               | ✅ 10,299 (6 activités)     |
| Downsampling 1s (sujet 1)     | ✅ 1,222 bins               |
| NaN / Inf détectés            | ✅ Aucun                    |

---

*Produit par : Membre 4 — Sprint 2 — Topic M5*
*Pipeline : ingest.py*
*Input : windowed/ (Membre 2) → Output : InfluxDB bucket har_data*
