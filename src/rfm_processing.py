import pandas as pd


def clean_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(subset=["customer_id", "recency_days", "frequency", "monetary"], inplace=True)
    df = df[df["monetary"] > 0]
    df = df[df["frequency"] > 0]
    df = df[df["recency_days"] >= 0]
    df = df.drop_duplicates(subset=["customer_id"])
    df["recency_days"] = df["recency_days"].astype(int)
    df["frequency"] = df["frequency"].astype(int)
    df["monetary"] = df["monetary"].astype(float)
    return df


def score_rfm(df: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["r_score"] = pd.qcut(df["recency_days"], q=quantiles, labels=range(quantiles, 0, -1))
    df["f_score"] = pd.qcut(df["frequency"].rank(method="first"), q=quantiles, labels=range(1, quantiles + 1))
    df["m_score"] = pd.qcut(df["monetary"].rank(method="first"), q=quantiles, labels=range(1, quantiles + 1))
    df["rfm_score"] = (
        df["r_score"].astype(int) * 100
        + df["f_score"].astype(int) * 10
        + df["m_score"].astype(int)
    )
    return df
