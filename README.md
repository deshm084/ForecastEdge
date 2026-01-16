# ForecastEdge: Enterprise Demand Forecasting Engine

![Status](https://img.shields.io/badge/Status-Production_Ready-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Azure](https://img.shields.io/badge/Cloud-Azure_Ready-0078D4)

ForecastEdge is a scalable, end-to-end demand forecasting pipeline designed to predict daily sales across thousands of product SKUs. Unlike traditional single-model approaches, this engine utilizes a **"Many Models" architecture**, training unique Prophet estimators for each product category in parallel to capture distinct seasonality and trend patterns.

## 🚀 Key Features

* **Parallel Training Architecture:** Utilizes `joblib` to parallelize model training across all available CPU cores, enabling scalability to 10,000+ SKUs.
* **Automated Anomaly Detection:** Features a custom Data Quality Gate that detects and imputes systematic data outages (e.g., closed stores on Jan 1st), preventing model drift and false trend learning.
* **Business-Centric Evaluation:** Optimizes for **WMAPE (Weighted Mean Absolute Percentage Error)** rather than simple MAPE, ensuring high-volume revenue drivers are prioritized over low-impact items.
* **Robust Seasonality Handling:** Incorporates Country-Specific Holidays (Ecuador) and multiplicative seasonality to handle holiday spikes (Easter, Christmas) and organic growth.

## 🛠️ Technical Architecture

The system is built to mimic a production Azure Machine Learning pipeline:

1.  **Ingestion:** `DataLoader` class ingests raw CSVs, standardizes timestamps, and handles missing values.
2.  **Processing:** Data is split into Category/Family clusters.
3.  **Training:** The `ForecastEngine` spins up parallel workers. Each worker:
    * Initializes a Prophet model with custom priors (`changepoint_prior_scale=0.05`).
    * Injects a "Closed Store" holiday mask for anomalies.
    * Fits the model and forecasts 30 days out.
4.  **Evaluation:** Aggregates predictions to calculate Global WMAPE.

## 📊 Performance

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Global WMAPE** | **7.89%** | Production-grade accuracy (<10%) |
| **Top Category** | Automotive | 7.1% MAPE |
| **Training Time** | <30s | Parallelized on local compute (Mock Data) |

## 💻 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy prophet joblib matplotlib scikit-learn
    ```

2.  **Run the Pipeline:**
    ```bash
    python src/train.py
    ```

## ☁️ Azure System Design (Whiteboard Prep)

If this needed to run daily on **10TB** of data, the cloud version looks like this:

```mermaid
flowchart LR
  A["ERP / SQL (SAP, etc.)"] --> B["Azure Data Factory (1 AM ingest)"]
  B --> C["ADLS Gen2 (Raw Zone)"]
  C --> D["Databricks / Synapse Spark (Preprocess)"]
  D --> E["Azure ML Pipeline"]
  E --> F["Parallel Training: AML ParallelRunStep"]
  F --> G["Azure ML Model Registry (Versioned Models)"]
  G --> H["AKS / Managed Online Endpoint"]
  H --> I["Dashboard / API Consumers"]
  D --> J["ADLS Gen2 (Curated Zone)"]
