import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


# ============================================================================
# HYPERPARAMETERS - Adjust these as needed
# ============================================================================
# K-Means clustering parameters
# Default: three resolutions of cluster granularity
KMEANS_K_VALUES = [3,4,5,7,10,13,16,20,25,30,40,50,75,100]

# DBSCAN clustering parameters (tune as needed)
DBSCAN_EPS = 0.7
DBSCAN_MIN_SAMPLES = 100

# ============================================================================
# DEFAULT PATHS
# ============================================================================
DEFAULT_INPUT_PATH = Path("output/data/extracted_features/pca_embeddings.csv")
DEFAULT_RAW_FEATURES_PATH = Path("output/data/extracted_features/extracted_features.csv")
DEFAULT_OUTPUT_DIR = Path("output/data/clustering_results")


def load_feature_matrix(
    input_path: Path, label: str, standardize: bool = False
) -> pd.DataFrame:
    """Load feature matrix (PCA embeddings or raw features). Optionally standardize."""
    df = pd.read_csv(input_path, index_col=0)
    print(f"Loaded {label} shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")

    if standardize:
        print("Standardizing features (zero mean, unit variance)...")
        scaler = StandardScaler()
        df = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns,
        )

    return df


def cluster_kmeans(
    embeddings: pd.DataFrame, k_values: list[int]
) -> dict[int, pd.Series]:
    """
    Perform K-Means clustering for multiple k values.
    
    Returns:
        Dictionary mapping k value to cluster assignments (Series with index=id, values=cluster)
    """
    results = {}
    X = embeddings.values

    for k in k_values:
        print(f"\nRunning K-Means with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Create Series with id as index and cluster as values
        cluster_series = pd.Series(
            cluster_labels, index=embeddings.index, name=f"cluster_k{k}"
        )
        results[k] = cluster_series

        # Print cluster sizes
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        print(f"Cluster sizes for k={k}:")
        print(cluster_counts)

    return results


def cluster_dbscan(embeddings: pd.DataFrame, eps: float, min_samples: int) -> pd.Series:
    """
    Perform DBSCAN clustering.

    Returns:
        Series with index=id and values=cluster labels (-1 denotes noise).
    """
    print(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")
    X = embeddings.values

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X)

    cluster_series = pd.Series(labels, index=embeddings.index, name="cluster_dbscan")

    # Print cluster sizes (including noise as cluster -1)
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    print("DBSCAN cluster label counts (including -1=noise):")
    print(cluster_counts)

    return cluster_series


def save_cluster_assignments(
    cluster_assignments: pd.Series, output_path: Path, method_name: str
) -> None:
    """Save cluster assignments to CSV (id -> cluster mapping only)."""
    # Create DataFrame with just id and cluster
    df = pd.DataFrame(
        {
            "id": cluster_assignments.index,
            "assigned_cluster": cluster_assignments.values,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n{method_name} cluster assignments saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Number of clusters: {df['assigned_cluster'].nunique()}")


def run_all_clustering_methods(
    embeddings: pd.DataFrame,
    output_dir: Path,
    kmeans_k_values: list[int],
    include_dbscan: bool = True,
) -> None:
    """
    Run all configured clustering methods and save results.
    
    This function is easily extensible - just add new clustering method calls here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # K-Means Clustering
    # ========================================================================
    if kmeans_k_values:
        kmeans_results = cluster_kmeans(embeddings, kmeans_k_values)

        # Save results for each k value (no implicit 'default' file to avoid confusion)
        for k, cluster_assignments in kmeans_results.items():
            output_path = output_dir / f"clustered_catchments_kmeans_k{k}.csv"
            save_cluster_assignments(
                cluster_assignments, output_path, f"K-Means (k={k})"
            )

    # ========================================================================
    # DBSCAN Clustering
    # ========================================================================
    if include_dbscan:
        dbscan_assignments = cluster_dbscan(
            embeddings, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES
        )
        dbscan_output = output_dir / "clustered_catchments_dbscan.csv"
        save_cluster_assignments(dbscan_assignments, dbscan_output, "DBSCAN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster catchment embeddings using various methods."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to PCA embeddings CSV",
    )
    parser.add_argument(
        "--raw-input-path",
        type=Path,
        default=DEFAULT_RAW_FEATURES_PATH,
        help="Path to extracted feature CSV (used when --use-raw-features is set).",
    )
    parser.add_argument(
        "--use-raw-features",
        action="store_true",
        help="Cluster directly on raw tsfresh features (will be standardized internally).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save clustering results",
    )
    parser.add_argument(
        "--kmeans-k",
        type=int,
        nargs="+",
        default=KMEANS_K_VALUES,
        help=f"K values for K-Means clustering (default: {KMEANS_K_VALUES})",
    )
    parser.add_argument(
        "--no-dbscan",
        action="store_true",
        help="Skip DBSCAN clustering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load embeddings or raw features
    if args.use_raw_features:
        embeddings = load_feature_matrix(
            args.raw_input_path,
            label="raw features",
            standardize=True,
        )
    else:
        embeddings = load_feature_matrix(
            args.input_path,
            label="PCA embeddings",
            standardize=False,
        )

    # Run all clustering methods
    run_all_clustering_methods(
        embeddings=embeddings,
        output_dir=args.output_dir,
        kmeans_k_values=args.kmeans_k,
        include_dbscan=not args.no_dbscan,
    )

    print("\n" + "=" * 60)
    print("Clustering complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

