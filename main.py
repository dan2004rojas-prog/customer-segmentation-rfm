import os
from src.data_loader import load_raw_data, build_rfm
from src.rfm_processing import clean_rfm, score_rfm
from src.ml_model import (
    prepare_features,
    find_optimal_k,
    plot_elbow_silhouette,
    train_kmeans,
    assign_clusters,
    plot_cluster_profiles,
)


def main():
    os.makedirs("outputs", exist_ok=True)

    print("[1/6] Loading Online Retail II dataset...")
    df_raw = load_raw_data()

    print("[2/6] Building RFM table from raw transactions...")
    df_rfm = build_rfm(df_raw)

    print("[3/6] Cleaning and scoring RFM data...")
    df_clean = clean_rfm(df_rfm)
    df_scored = score_rfm(df_clean)

    print("[4/6] Preparing features (log1p + StandardScaler)...")
    scaled_features, scaler = prepare_features(df_scored)

    print("[5/6] Evaluating optimal clusters (WCSS & Silhouette)...")
    metrics = find_optimal_k(scaled_features, k_range=range(2, 11))
    plot_elbow_silhouette(metrics, output_path="outputs/elbow_silhouette.png")

    optimal_k = metrics["k_range"][metrics["silhouette"].index(max(metrics["silhouette"]))]
    print(f"      -> Optimal k selected: {optimal_k}")

    print("[6/6] Training KMeans, assigning clusters, saving outputs...")
    km = train_kmeans(scaled_features, n_clusters=optimal_k)
    df_final = assign_clusters(df_scored, km, scaled_features)
    plot_cluster_profiles(df_final, output_path="outputs/cluster_profiles.png")
    df_final.to_csv("outputs/rfm_segmented_customers.csv", index=False)

    print("\n[OK] Pipeline complete. Outputs saved to /outputs/")
    print(df_final.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean().round(2))


if __name__ == "__main__":
    main()
