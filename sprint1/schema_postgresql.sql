-- =============================================================================
-- 1. SCHÉMA RELATIONNEL
-- =============================================================================

-- ---------------------------------------------------------------
-- Table : subjects
-- Un enregistrement par participant (30 sujets UCI HAR).
-- Chaque sujet appartient strictement à un seul split
-- (21 sujets train, 9 sujets test — source : meta_train/test.csv).
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subjects (
    subject_id  SMALLINT    PRIMARY KEY,
    split       VARCHAR(5)  NOT NULL CHECK (split IN ('train', 'test'))
);


-- ---------------------------------------------------------------
-- Table : activities
-- Référentiel des 6 activités UCI HAR.
-- is_anomaly représente la tendance par défaut de la catégorie.
-- ATTENTION (v3) : la valeur réelle d'anomalie par fenêtre est dans
-- windows.is_anomaly (calculée sur seuils physiques par Membre 2).
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS activities (
    activity_label  SMALLINT    PRIMARY KEY,
    activity_name   VARCHAR(30) NOT NULL UNIQUE,
    is_anomaly      SMALLINT    NOT NULL CHECK (is_anomaly IN (0, 1))
    -- Référence catégorielle (pas la vérité terrain par fenêtre)
);


-- ---------------------------------------------------------------
-- Table : capture_protocol
-- Paramètres constants du protocole d'acquisition UCI HAR.
-- Extraits de windows (3NF) : ces valeurs ne dépendent pas d'une
-- fenêtre individuelle mais du protocole de capture.
-- Extensible si de nouveaux capteurs/fréquences sont intégrés.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capture_protocol (
    protocol_id         SMALLINT        PRIMARY KEY,
    n_samples           SMALLINT        NOT NULL,   -- 128 samples/fenêtre
    sampling_rate_hz    SMALLINT        NOT NULL,   -- 50 Hz
    duration_sec        NUMERIC(5,2)    NOT NULL,   -- 2.56 s = 128 / 50
    description         TEXT
);


-- ---------------------------------------------------------------
-- Table : windows
-- Une ligne = une fenêtre de 2.56 secondes.
-- Clé primaire UUID fournie par meta_train.csv / meta_test.csv.
--
-- CHANGEMENT v3 :
--   - is_anomaly RÉTABLI dans windows.
--     Raison : la nouvelle stratégie Membre 2 calcule is_anomaly sur
--     des seuils physiques de signal — une même activité (ex: STANDING)
--     peut être 0 ou 1 selon la fenêtre. La valeur n'est plus dérivable
--     depuis activities.is_anomaly → elle doit être stockée ici.
--
-- Colonnes toujours absentes (déjà normalisées en v2) :
--   - activity_name  → JOIN activities
--   - n_samples / sampling_rate_hz / duration_sec → FK capture_protocol
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS windows (
    window_id       UUID        PRIMARY KEY,
    window_index    INT         NOT NULL,
    split           VARCHAR(5)  NOT NULL CHECK (split IN ('train', 'test')),
    subject_id      SMALLINT    NOT NULL REFERENCES subjects(subject_id),
    activity_label  SMALLINT    NOT NULL REFERENCES activities(activity_label),
    is_anomaly      SMALLINT    NOT NULL CHECK (is_anomaly IN (0, 1)),
    protocol_id     SMALLINT    NOT NULL REFERENCES capture_protocol(protocol_id),
    ingested_at     TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_windows_split      ON windows(split);
CREATE INDEX IF NOT EXISTS idx_windows_subject    ON windows(subject_id);
CREATE INDEX IF NOT EXISTS idx_windows_activity   ON windows(activity_label);
CREATE INDEX IF NOT EXISTS idx_windows_is_anomaly ON windows(is_anomaly);


-- ---------------------------------------------------------------
-- Table : signals
-- Une ligne = un canal complet (128 samples) pour une fenêtre donnée.
-- PK composite (window_id, channel) — pas de BIGSERIAL inutile.
--
-- Choix REAL[] vs. modèle scalaire :
--   Modèle scalaire (v1) : 10 299 fenêtres × 9 canaux × 128 samples
--                        = ~11,9 M lignes
--   Modèle tableau  (v2) : 10 299 × 9 = ~92 700 lignes  (×128 moins)
--
-- Les 128 samples sont ordonnés par index temporel (0 → 127).
-- La contrainte CHECK garantit l'intégrité du tableau.
-- Les requêtes d'agrégation utilisent unnest(samples).
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    window_id   UUID        NOT NULL REFERENCES windows(window_id),
    channel     VARCHAR(15) NOT NULL CHECK (channel IN (
                    'body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x','body_gyro_y','body_gyro_z',
                    'total_acc_x','total_acc_y','total_acc_z'
                )),
    samples     REAL[]      NOT NULL CHECK (cardinality(samples) = 128),
    PRIMARY KEY (window_id, channel)
);

CREATE INDEX IF NOT EXISTS idx_signals_channel ON signals(channel);


-- =============================================================================
-- 2. DONNÉES DE RÉFÉRENCE (INSERT statiques — Membre 3)
-- =============================================================================

-- Protocole UCI HAR (unique dans ce dataset)
INSERT INTO capture_protocol (protocol_id, n_samples, sampling_rate_hz, duration_sec, description)
VALUES (1, 128, 50, 2.56, 'UCI HAR — fenêtre 2.56 s @ 50 Hz, overlap 50 %')
ON CONFLICT (protocol_id) DO NOTHING;

-- Référentiel des activités (stratégie anomalie définie par Membre 2)
INSERT INTO activities (activity_label, activity_name, is_anomaly) VALUES
    (1, 'WALKING',            0),
    (2, 'WALKING_UPSTAIRS',   0),
    (3, 'WALKING_DOWNSTAIRS', 0),
    (4, 'SITTING',            1),
    (5, 'STANDING',           1),
    (6, 'LAYING',             1)
ON CONFLICT (activity_label) DO NOTHING;


-- =============================================================================
-- 3. REQUÊTES DE VALIDATION POST-INGESTION
-- Ces requêtes servent uniquement à contrôler que le chargement
-- du Membre 3 est correct. Elles ne constituent pas des tests de
-- performance (Membres 5-7) ni d'analyse de stockage (Membre 8).
-- =============================================================================

-- 3.1 Contrôle du volume : nombre de fenêtres par split
SELECT split, COUNT(*) AS n_windows
FROM windows
GROUP BY split
ORDER BY split;
-- Attendu : train = 7352, test = 2947

-- 3.2 Contrôle de la couverture sujets
SELECT split, COUNT(DISTINCT subject_id) AS n_subjects
FROM windows
GROUP BY split
ORDER BY split;
-- Attendu : train = 21 sujets, test = 9 sujets

-- 3.3 Contrôle de la distribution des activités (train)
SELECT a.activity_name, COUNT(*) AS n_windows,
       SUM(w.is_anomaly) AS n_anomalies
FROM windows w
JOIN activities a ON a.activity_label = w.activity_label
WHERE w.split = 'train'
GROUP BY a.activity_name
ORDER BY n_windows DESC;
-- Attendu : LAYING 1407, STANDING 1374, SITTING 1286,
--           WALKING 1226, WALKING_UPSTAIRS 1073, WALKING_DOWNSTAIRS 986
-- Note : une même activité peut avoir des anomalies ET des normaux (v3)

-- 3.4 Contrôle du ratio anomalie / normal par split
-- CHANGEMENT v3 : utilise windows.is_anomaly (plus activities.is_anomaly)
SELECT
    split,
    SUM(is_anomaly)                        AS n_anomalies,
    SUM(1 - is_anomaly)                    AS n_normal,
    ROUND(AVG(is_anomaly::NUMERIC), 4)     AS anomaly_ratio
FROM windows
GROUP BY split;
-- Attendu : train ≈ 0.1575 (1158 anomalies), test ≈ 0.0933 (275 anomalies)

-- 3.5 Contrôle de la table signals : 9 canaux × n_windows lignes attendues
SELECT channel, COUNT(*) AS n_rows
FROM signals
GROUP BY channel
ORDER BY channel;
-- Chaque canal doit avoir 7352 lignes (train) + 2947 lignes (test) = 10 299

-- 3.6 Contrôle de l'intégrité : fenêtres sans les 9 canaux (anomalie d'ingestion)
SELECT w.window_id, COUNT(s.channel) AS n_channels
FROM windows w
LEFT JOIN signals s ON s.window_id = w.window_id
GROUP BY w.window_id
HAVING COUNT(s.channel) <> 9;
-- Attendu : 0 lignes (toutes les fenêtres ont exactement 9 canaux)

-- 3.7 Contrôle des valeurs : bornes globales après normalisation z-score
SELECT
    channel,
    ROUND(MIN(v)::NUMERIC, 4)    AS val_min,
    ROUND(MAX(v)::NUMERIC, 4)    AS val_max,
    ROUND(AVG(v)::NUMERIC, 6)    AS val_mean,
    ROUND(STDDEV(v)::NUMERIC, 6) AS val_std
FROM signals, unnest(samples) AS v
GROUP BY channel
ORDER BY channel;
-- Attendu : valeurs bornées entre ±3.35 (clipping Membre 1),
--           mean ≈ 0, std ≈ 1 (normalisation z-score)

-- 3.8 Contrôle de cohérence UUID : chaque window_id est unique
SELECT COUNT(*) AS total, COUNT(DISTINCT window_id) AS distinct_ids
FROM windows;
-- Attendu : total = distinct_ids = 10 299


-- =============================================================================
-- 4. VUE UTILITAIRE — Reconstruction tenseur
-- Fournie aux autres membres pour accéder aux données sans connaître
-- le schéma interne. Expose window_id, métadonnées et tableau de samples
-- par canal — équivalent SQL du tenseur numpy (N, 128, 9).
-- =============================================================================

CREATE OR REPLACE VIEW v_windows_full AS
SELECT
    w.window_id,
    w.window_index,
    w.split,
    w.subject_id,
    w.activity_label,
    a.activity_name,
    w.is_anomaly,        -- valeur réelle par fenêtre (seuils physiques v3)
    s.channel,
    s.samples
FROM windows w
JOIN activities a ON a.activity_label = w.activity_label
JOIN signals    s ON s.window_id      = w.window_id;

-- Exemple d'utilisation par les membres aval :
-- SELECT channel, samples
-- FROM v_windows_full
-- WHERE window_id = '<uuid>'
-- ORDER BY channel;