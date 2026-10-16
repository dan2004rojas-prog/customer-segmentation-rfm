import os
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    features = df[["recency_days", "frequency", "monetary"]].copy()
    features["recency_days"] = np.log1p(features["recency_days"])
    features["frequency"] = np.log1p(features["frequency"])
    features["monetary"] = np.log1p(features["monetary"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def find_optimal_k(
    scaled_features: np.ndarray,
    k_range: Sequence[int] = range(2, 11),
    silhouette_sample: int = 5000,
) -> dict:
    wcss = []
    silhouette_scores = []

    n = scaled_features.shape[0]
    sample_size = min(silhouette_sample, n)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample = scaled_features[sample_idx]

    for k in k_range:
        print(f"      Evaluating k={k}...", flush=True)
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, max_iter=200, random_state=42)
        km.fit(scaled_features)
        wcss.append(km.inertia_)
        sample_labels = km.predict(sample)
        silhouette_scores.append(silhouette_score(sample, sample_labels))

    return {"k_range": list(k_range), "wcss": wcss, "silhouette": silhouette_scores}


def plot_elbow_silhouette(metrics: dict, output_path: str = "outputs/elbow_silhouette.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(metrics["k_range"], metrics["wcss"], marker="o", color="#4C72B0", linewidth=2)
    axes[0].set_title("Elbow Method (WCSS)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("WCSS")

    axes[1].plot(metrics["k_range"], metrics["silhouette"], marker="s", color="#DD8452", linewidth=2)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_kmeans(scaled_features: np.ndarray, n_clusters: int = 4) -> KMeans:
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    km.fit(scaled_features)
    return km


def assign_clusters(df: pd.DataFrame, km: KMeans, scaled_features: np.ndarray | None = None) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = km.labels_
    return df


def plot_cluster_profiles(df: pd.DataFrame, output_path: str = "outputs/cluster_profiles.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile = df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_cols = ["recency_days", "frequency", "monetary"]
    colors = sns.color_palette("husl", len(profile))

    for i, col in enumerate(metrics_cols):
        axes[i].bar(profile["cluster"].astype(str), profile[col], color=colors)
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("Mean Value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pca_clusters(
    df: pd.DataFrame,
    scaled_features: np.ndarray,
    output_path: str = "outputs/pca_clusters.png",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_features)

    label_col = "segment" if "segment" in df.columns else "cluster"
    segments = df[label_col].astype(str).values
    unique_segs = sorted(set(segments))
    palette = sns.color_palette("husl", len(unique_segs))
    color_map = {s: palette[i] for i, s in enumerate(unique_segs)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for seg in unique_segs:
        idx = segments == seg
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=[color_map[seg]],
            label=seg,
            alpha=0.6,
            s=18,
            edgecolors="none",
        )

    var_explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
    ax.set_title("Customer Segments — PCA 2D Projection")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
