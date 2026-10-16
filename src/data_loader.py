import os
import pathlib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("KAGGLE_TOKEN", os.getenv("KAGGLE_TOKEN", ""))

DATA_DIR = pathlib.Path("data")
DATASET_SLUG = "olistbr/brazilian-ecommerce"


def _download_olist() -> str:
    import kagglehub
    path = kagglehub.dataset_download(DATASET_SLUG)
    return path


def _synthetic_fallback() -> dict:
    rng = np.random.default_rng(42)
    n = 5000
    customer_ids = [f"cust_{i:05d}" for i in rng.integers(0, 1500, size=n)]
    order_ids = [f"order_{i:06d}" for i in range(n)]
    dates = pd.date_range("2017-01-01", "2018-08-31", periods=n)
    payments = rng.uniform(10.0, 800.0, size=n).round(2)

    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "order_status": "delivered",
        "order_purchase_timestamp": dates,
    })
    payments_df = pd.DataFrame({
        "order_id": order_ids,
        "payment_value": payments,
    })
    return {"orders": orders, "payments": payments_df}


def load_raw_data() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    orders_path = DATA_DIR / "olist_orders_dataset.csv"
    payments_path = DATA_DIR / "olist_order_payments_dataset.csv"

    if orders_path.exists() and payments_path.exists():
        orders = pd.read_csv(orders_path, parse_dates=["order_purchase_timestamp"])
        payments = pd.read_csv(payments_path)
        return {"orders": orders, "payments": payments}

    try:
        import shutil
        raw_path = pathlib.Path(_download_olist())
        for csv_file in raw_path.glob("*.csv"):
            shutil.copy(csv_file, DATA_DIR / csv_file.name)
        orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
        payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
        return {"orders": orders, "payments": payments}
    except Exception as e:
        print(f"[WARNING] Kaggle download failed: {e}. Using synthetic fallback.")
        return _synthetic_fallback()


def build_rfm(data: dict, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    orders = data["orders"].copy()
    payments = data["payments"].copy()

    orders = orders[orders["order_status"] == "delivered"]
    orders = orders.dropna(subset=["customer_id", "order_purchase_timestamp"])

    pay_agg = payments.groupby("order_id", as_index=False)["payment_value"].sum()
    merged = orders.merge(pay_agg, on="order_id", how="inner")
    merged = merged[merged["payment_value"] > 0]

    if snapshot_date is None:
        snapshot_date = merged["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = merged.groupby("customer_id").agg(
        recency_days=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum"),
    ).reset_index()

    return rfm
