# Customer Segmentation — RFM + KMeans (Olist Brazilian E-Commerce)

## Overview

End-to-end pipeline that downloads the **Olist Brazilian E-Commerce** dataset from Kaggle, computes RFM (Recency, Frequency, Monetary) metrics from real transactional data, and segments customers using KMeans clustering with automated optimal-k selection via WCSS (Elbow Method) and Silhouette Score.

---

## Project Structure

```
customer-segmentation-rfm/
├── .env                    # Environment variables (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── config.py               # Loads .env via python-dotenv
├── main.py                 # Pipeline orchestrator
├── data/                   # Auto-populated by kagglehub on first run
│   └── .gitkeep
├── sql/
│   └── rfm_query.sql       # Reference SQL for RFM logic
├── src/
│   ├── data_loader.py      # Kaggle download + RFM builder + synthetic fallback
│   ├── rfm_processing.py   # Data cleaning and RFM scoring
│   └── ml_model.py         # Feature engineering, KMeans, plots
└── outputs/                # Generated CSV and PNG files (auto-created)
```

---

## Dataset

| Source | Kaggle |
|--------|--------|
| Dataset | [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| Key files | `olist_orders_dataset.csv`, `olist_order_payments_dataset.csv` |
| Auto-download | `kagglehub.dataset_download("olistbr/brazilian-ecommerce")` |

> Requires a valid `~/.kaggle/kaggle.json` credentials file.

---

## Setup

### 1. Configure Kaggle credentials

Download your API token from [kaggle.com → Settings → API](https://www.kaggle.com/settings) and place it at:

```
C:\Users\TuUsuario\.kaggle\kaggle.json
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

### 5. Run the pipeline

```bash
python main.py
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_loader.py` | Download Olist via `kagglehub`, cache CSVs to `data/` |
| 2 | `data_loader.py` | Merge orders + payments, build RFM table |
| 3 | `rfm_processing.py` | Drop nulls, remove invalid rows, compute RFM scores |
| 4 | `ml_model.py` | Apply `log1p` + `StandardScaler` to features |
| 5 | `ml_model.py` | Evaluate k=2..10 via WCSS and Silhouette Score |
| 6 | `ml_model.py` | Train KMeans with optimal k, generate plots |

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/rfm_segmented_customers.csv` | Customers with cluster labels and RFM scores |
| `outputs/elbow_silhouette.png` | WCSS and Silhouette plots for k selection |
| `outputs/cluster_profiles.png` | Bar charts of mean RFM values per cluster |

---

## RFM Definitions

| Metric | Definition |
|--------|-----------|
| **Recency** | Days since last delivered order (lower = better) |
| **Frequency** | Number of distinct delivered orders |
| **Monetary** | Total payment value across all orders |

---

## Git Setup & Push to GitHub

```bash
cd "C:///"
git init
git status
git add .
git commit -m "feat: initial commit customer segmentation RFM model"
git branch -M main
git remote add origin https://github.com/dan2004rojas-prog/customer-segmentation-rfm.git
git push -u origin main
```
