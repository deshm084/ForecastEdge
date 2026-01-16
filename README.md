# ForecastEdge: Enterprise Demand Forecasting Engine

![Status](https://img.shields.io/badge/Status-Production%%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.8%%2B-blue)
![Azure](https://img.shields.io/badge/Cloud-Azure%%20Ready-0078D4)

ForecastEdge is a scalable, end-to-end demand forecasting pipeline designed to predict daily sales across thousands of product SKUs. Unlike traditional single-model approaches, this engine utilizes a **Many Models architecture**, training unique Prophet estimators for each product category in parallel to capture distinct seasonality and trend patterns.

## Key Features

- **Parallel Training Architecture:** Uses `joblib` to parallelize model training across all available CPU cores, enabling scalability to 10,000+ SKUs.
- **Automated Anomaly Handling:** Custom holiday mask flags systematic outages (e.g., Jan 1 closures) so the model learns the real pattern, not a fake demand crash.
- **Business-Centric Evaluation:** Optimizes **WMAPE (Weighted Mean Absolute Percentage Error)** so high-volume revenue drivers matter more than low-impact items.
- **Robust Seasonality Modeling:** Ecuador holidays + multiplicative seasonality handle holiday spikes (Easter/Christmas) and organic growth.

## Technical Architecture

This repo mirrors a production-style pipeline:

1. **Ingestion:** `DataLoader` ingests raw CSVs, standardizes timestamps, and aggregates sales to Category/Family level.
2. **Processing:** Splits into Category/Family clusters for a Many Models training pattern.
3. **Training:** `ForecastEngine` spins up parallel workers. Each worker:
   - Initializes Prophet with tuned priors (`changepoint_prior_scale=0.05`)
   - Injects a "Closed Store" holiday mask for anomalies
   - Fits and forecasts 30 days out
4. **Evaluation:** Aggregates predictions to compute **Global WMAPE**

## Performance

| Metric | Score | Notes |
| :--- | :---: | :--- |
| **Global WMAPE** | **7.8911%%** | Production-grade accuracy (<10%%) |
| **Top Category** | Automotive | 7.1%% MAPE |
| **Training Time** | <30s | Parallelized on local compute (Mock Data) |

## How to Run

1) **Install Dependencies**
```bash
pip install pandas numpy prophet joblib matplotlib scikit-learn
```

2) **Run the Pipeline**
```bash
python src/train.py
```

## Azure System Design (Whiteboard Prep)

If this needed to run daily on **10TB** of data, the cloud version looks like this:

```mermaid
flowchart LR
  A[ERP / SQL (SAP, etc.)] --> B[Azure Data Factory (1 AM ingest)]
  B --> C[ADLS Gen2 (Raw Zone)]
  C --> D[Databricks / Synapse Spark (Preprocess)]
  D --> E[Azure ML Pipeline]
  E --> F[Parallel Training: AML ParallelRunStep]
  F --> G[Azure ML Model Registry (Versioned Models)]
  G --> H[AKS / Managed Online Endpoint]
  H --> I[Dashboard / API Consumers]
  D --> J[ADLS Gen2 (Curated Zone)]
```

**Flow to explain out loud:**
1. **ADLS Gen2:** Raw sales data lands in a lake (source of truth).
2. **ADF nightly ingest:** Copies incremental data from SQL/ERP into ADLS.
3. **Spark preprocessing:** Databricks/Synapse cleans + aggregates (10TB doesn't fit in pandas).
4. **Azure ML pipeline:** Many Models training via **ParallelRunStep** across nodes (cloud version of joblib).
5. **Model Registry:** Models are versioned; best model is tagged "Production".
6. **Inference:** Served on AKS / managed endpoint; dashboard calls REST API for predictions.

## Future Roadmap (Azure Migration)

- **Containerization:** Dockerize training for AKS / AML jobs
- **Experiment Tracking:** Add MLflow for hyperparameter + metric logging
- **Orchestration:** Move ingestion + preprocessing into ADF + AML pipelines
