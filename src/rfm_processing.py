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


def assign_segment_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    profile = df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()

    r_med = profile["recency_days"].median()
    f_med = profile["frequency"].median()
    m_med = profile["monetary"].median()

    def label(row):
        r, f, m = row["recency_days"], row["frequency"], row["monetary"]
        if r <= r_med and f >= f_med and m >= m_med:
            return "Champions"
        elif r <= r_med and f >= f_med and m < m_med:
            return "Loyal Customers"
        elif r <= r_med and f < f_med and m >= m_med:
            return "Potential Loyalists"
        elif r > r_med and f >= f_med and m >= m_med:
            return "At Risk"
        elif r > r_med and f >= f_med and m < m_med:
            return "Need Attention"
        elif r > r_med and f < f_med and m >= m_med:
            return "Cant Lose Them"
        else:
            return "Hibernating"

    cluster_labels = profile.apply(label, axis=1).to_dict()
    df["segment"] = df["cluster"].map(cluster_labels)
    return df
