# Human Activity Recognition Anomaly Detection Project

This repository contains a multi-sprint project for human activity recognition (HAR) and anomaly detection using smartphone sensor data. The project combines data preprocessing, time-series windowing, database ingestion, and deep learning for reconstruction-based anomaly detection.

## Project goal

The goal is to process the UCI HAR Dataset, transform raw sensor signals into windowed temporal features, store them in a database, and train a temporal autoencoder to detect abnormal behavior patterns.

## Project structure

- sprint1/: dataset preparation and SQL schema
  - UCI HAR Dataset/: original HAR dataset files
  - schema_postgresql.sql: PostgreSQL schema definition
  - metadata.json: dataset metadata

- sprint2/: cleaning, windowing, and ingestion
  - cleaning & windowing/: preprocessing and windowing scripts
  - ingest.py, postgres_ingestion.py, timeseries_ingest.py: data loading and ingestion
  - README_influxdb.md: InfluxDB usage notes

- sprint3/: modeling and evaluation
  - model.py, loss.py, train.py: autoencoder training pipeline
  - eval.py: evaluation and metrics
  - baselines/: baseline models and clustering methods
  - Figures/, Reports/, Tables/: generated results and plots

## Data source

The project uses the UCI HAR Dataset, which contains smartphone sensor recordings for six activities:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## Workflow overview

1. Sprint 1
   - Prepare the raw dataset
   - Define database schema and metadata

2. Sprint 2
   - Clean and normalize signals
   - Create sliding windows
   - Ingest data into PostgreSQL or InfluxDB

3. Sprint 3
   - Train a temporal autoencoder model
   - Evaluate reconstruction errors and anomaly scores
   - Generate reports and visualizations

## Main scripts

- sprint2/cleaning & windowing/etl_cleaning.py: data cleaning pipeline
- sprint2/cleaning & windowing/windowing.py: window generation
- sprint2/ingest.py: ingestion entry point
- sprint3/train.py: model training
- sprint3/eval.py: evaluation pipeline

## Environment requirements

A Python environment with the following dependencies is typically required:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- scikit-learn
- influxdb-client (if using InfluxDB)
- psycopg2 (if using PostgreSQL)

## Suggested usage

### 1. Prepare the dataset

Place the UCI HAR Dataset inside sprint1/UCI HAR Dataset/.

### 2. Run preprocessing and windowing

Use the scripts under sprint2/cleaning & windowing/.

### 3. Ingest the processed data

Choose the relevant ingestion script depending on the target database.

### 4. Train the model

Run the training script in sprint3/:

```bash
python sprint3/train.py
```

## Outputs

The project produces:

- cleaned and windowed sensor arrays
- database records for time-series storage
- trained model checkpoints
- evaluation metrics and figures

## Notes

This project is structured as an academic, multi-sprint workflow. Some folders contain generated artifacts, so it is helpful to keep source code, data, and results clearly separated.

