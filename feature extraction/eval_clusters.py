import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler


DEFAULT_CLUSTER_DIR = Path("output/data/clustering_results")
DEFAULT_EMBEDDINGS_PATH = Path("output/data/extracted_features/pca_embeddings.csv")
DEFAULT_RAW_FEATURES_PATH = Path("output/data/extracted_features/extracted_features.csv")
DEFAULT_EVAL_DIR = Path("output/data/eval_clusters")

N_BOOTSTRAP = 200
MAX_UNIQUE_BUCKET = 6  # report 1..6 unique clusters per org_id


def load_cluster_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or "assigned_cluster" not in df.columns:
        raise ValueError("Cluster file must contain 'id' and 'assigned_cluster' columns.")
    return df


def load_feature_matrix(
    path: Path, *, standardize: bool = False, label: str = ""
) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if label:
        print(f"Loaded {label}: shape={df.shape}")
    if standardize:
        print("Standardizing features for silhouette calculations...")
        scaler = StandardScaler()
        df = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns,
        )
    return df


def extract_org_id(id_with_year: str) -> str:
    parts = str(id_with_year).split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else str(id_with_year)


def unique_cluster_distribution(df: pd.DataFrame) -> Counter:
    df = df.copy()
    df["org_id"] = df["id"].apply(extract_org_id)
    counts = Counter()
    for _, group in df.groupby("org_id"):
        n_unique = group["assigned_cluster"].nunique()
        bucket = min(n_unique, MAX_UNIQUE_BUCKET)
        counts[bucket] += 1
    return counts


def analyze_real_clusters(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df["org_id"] = df["id"].apply(extract_org_id)
    org_groups = df.groupby("org_id")

    stats_rows = []
    most_common_clusters_per_org = []
    for org_id, group in org_groups:
        labels = group["assigned_cluster"].values
        counts = Counter(labels)
        most_cluster, most_count = counts.most_common(1)[0]
        n_years = len(labels)
        n_unique = len(counts)
        ratio = most_count / n_years
        stats_rows.append(
            {
                "org_id": org_id,
                "n_years": n_years,
                "n_unique_clusters": n_unique,
                "most_common_cluster": most_cluster,
                "most_common_count": most_count,
                "consistency_ratio": ratio,
                "is_fully_consistent": n_unique == 1,
            }
        )
        most_common_clusters_per_org.append(most_cluster)

    stats_df = pd.DataFrame(stats_rows)
    n_orgs = len(stats_df)

    fully_consistent = stats_df["is_fully_consistent"].sum()
    fully_consistent_pct = 100 * fully_consistent / n_orgs
    avg_ratio_pct = 100 * stats_df["consistency_ratio"].mean()

    most_common_overall, count_overall = Counter(most_common_clusters_per_org).most_common(1)[0]
    most_common_overall_pct = 100 * count_overall / n_orgs

    dist_unique = unique_cluster_distribution(df)
    
    # Calculate cluster sizes (number of datapoints per cluster)
    cluster_sizes = df["assigned_cluster"].value_counts().sort_index()
    cluster_sizes_pct = (cluster_sizes / len(df) * 100).round(2)

    return {
        "stats_df": stats_df,
        "n_orgs": n_orgs,
        "n_samples": len(df),
        "fully_consistent_pct": fully_consistent_pct,
        "avg_ratio_pct": avg_ratio_pct,
        "most_common_overall": most_common_overall,
        "most_common_overall_pct": most_common_overall_pct,
        "unique_cluster_dist": dist_unique,
        "cluster_sizes": cluster_sizes,
        "cluster_sizes_pct": cluster_sizes_pct,
    }


def bootstrap_random_baseline(df: pd.DataFrame, n_bootstrap: int) -> Dict[int, float]:
    df = df.copy()
    df["org_id"] = df["id"].apply(extract_org_id)
    labels = df["assigned_cluster"].values

    counts_acc = Counter()
    for _ in range(n_bootstrap):
        permuted = np.random.permutation(labels)
        df["assigned_cluster"] = permuted
        dist = unique_cluster_distribution(df)
        counts_acc.update(dist)

    return {k: counts_acc[k] / n_bootstrap for k in range(1, MAX_UNIQUE_BUCKET + 1)}


def silhouette_for_clusters(
    df_clusters: pd.DataFrame, embeddings: pd.DataFrame
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """
    Compute silhouette scores for clusters.
    Returns: (overall_score, sample_scores, cluster_labels) or (None, None, None) if invalid.
    """
    ids = df_clusters["id"].astype(str).tolist()
    ids = [i for i in ids if i in embeddings.index]
    if len(ids) == 0:
        return None, None, None

    clusters = df_clusters.set_index("id").loc[ids]["assigned_cluster"]
    X = embeddings.loc[ids].values

    if len(set(clusters)) < 2:
        return None, None, None

    overall_score = silhouette_score(X, clusters)
    sample_scores = silhouette_samples(X, clusters)
    
    return overall_score, sample_scores, clusters.values


def plot_silhouette_distribution(
    overall_score: float | None,
    sample_scores: np.ndarray | None,
    cluster_labels: np.ndarray | None,
    output_path: Path,
    title: str,
) -> None:
    """
    Create a silhouette plot showing the distribution of silhouette scores per cluster.
    This is the standard silhouette visualization from sklearn documentation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if overall_score is None or sample_scores is None or cluster_labels is None:
        # Fallback for invalid cases
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "N/A\n(only one cluster or insufficient data)", 
                ha="center", va="center", fontsize=16, 
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
        plt.xticks([])
        plt.yticks([])
        plt.title(title, fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    
    # Get unique clusters and sort them
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, n_clusters * 0.8)))
    
    # Silhouette score ranges from -1 to 1
    y_lower = 10
    
    # Use a colormap for different clusters
    cmap = plt.cm.tab10
    
    for i, cluster_id in enumerate(unique_clusters):
        # Get silhouette scores for this cluster
        cluster_silhouette_scores = sample_scores[cluster_labels == cluster_id]
        cluster_silhouette_scores.sort()
        
        size_cluster_i = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Get color for this cluster (use modulo to cycle through colors if needed)
        color = cmap(i % 10)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_scores,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id), fontsize=10, fontweight="bold")
        
        # Compute the average silhouette score for this cluster
        avg_score = cluster_silhouette_scores.mean()
        ax.axvline(x=avg_score, color=color, linestyle="--", linewidth=2, alpha=0.8)
        ax.text(avg_score + 0.01, y_lower + 0.5 * size_cluster_i, f"{avg_score:.3f}", 
               fontsize=9, fontweight="bold")
        
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    # Add overall average silhouette score line
    ax.axvline(x=overall_score, color="red", linestyle="-", linewidth=2, 
              label=f"Overall avg: {overall_score:.3f}")
    
    ax.set_xlabel("Silhouette Coefficient Values", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cluster Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    
    # Set x-axis limits
    ax.set_xlim([-0.1, 1])
    
    # Remove y-axis ticks and labels (we use cluster labels instead)
    ax.set_yticks([])
    
    # Add legend
    ax.legend(loc="upper right", fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def infer_method_and_hyperparams(cluster_path: Path) -> tuple[str, str]:
    stem = cluster_path.stem.lower()
    if "kmeans_k" in stem:
        try:
            k_str = stem.split("kmeans_k")[1]
            k_val = int(k_str)
            return "KMeans", f"k={k_val}"
        except Exception:
            return "KMeans", "k=?"
    if "dbscan" in stem:
        return "DBSCAN", "eps=0.7, min_samples=5"
    return "Unknown", stem


def write_report(
    output_dir: Path,
    input_clusters: Path,
    method_name: str,
    hyperparams: str,
    real_stats: Dict[str, Any],
    random_dist: Dict[int, float],
    silhouette: float | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_clusters.stem
    report_path = output_dir / f"{stem}_evaluation.txt"

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CLUSTER EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Input clusters: {input_clusters}")
    lines.append(f"Method: {method_name}")
    lines.append(f"Hyperparameters: {hyperparams}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("BASIC STATS")
    lines.append("-" * 80)
    lines.append(f"Total original IDs: {real_stats['n_orgs']}")
    lines.append(f"Total samples (ID-year): {real_stats['n_samples']}")
    if silhouette is not None:
        lines.append(f"Silhouette score: {silhouette:.4f}")
    else:
        lines.append("Silhouette score: N/A (only one cluster label)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("CLUSTER SIZES")
    lines.append("-" * 80)
    lines.append("Number of datapoints assigned to each cluster:")
    cluster_sizes = real_stats.get("cluster_sizes", pd.Series())
    cluster_sizes_pct = real_stats.get("cluster_sizes_pct", pd.Series())
    if len(cluster_sizes) > 0:
        for cluster_id in sorted(cluster_sizes.index):
            count = cluster_sizes[cluster_id]
            pct = cluster_sizes_pct.get(cluster_id, 0)
            lines.append(f"  Cluster {cluster_id}: {count} datapoints ({pct:.2f}%)")
    else:
        lines.append("  (No cluster size data available)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("CONSISTENCY ACROSS YEARS (REAL CLUSTERS)")
    lines.append("-" * 80)
    lines.append(
        f"Fully consistent IDs (one cluster only): "
        f"{real_stats['unique_cluster_dist'].get(1, 0)} "
        f"({100 * real_stats['unique_cluster_dist'].get(1, 0) / real_stats['n_orgs']:.2f}%)"
    )
    lines.append(
        f"Average consistency ratio (most-common cluster): {real_stats['avg_ratio_pct']:.2f}%"
    )
    lines.append(
        f"Most common cluster across IDs: {real_stats['most_common_overall']} "
        f"({real_stats['most_common_overall_pct']:.2f}% of IDs)"
    )
    lines.append("")
    lines.append("Distribution of number of unique clusters per ID (REAL):")
    lines.append(f"(Bucket {MAX_UNIQUE_BUCKET} represents {MAX_UNIQUE_BUCKET}+ unique clusters)")
    for k in range(1, MAX_UNIQUE_BUCKET + 1):
        label = f"{k}+ unique cluster(s)" if k == MAX_UNIQUE_BUCKET else f"{k} unique cluster(s)"
        real_count = real_stats["unique_cluster_dist"].get(k, 0)
        pct = 100 * real_count / real_stats["n_orgs"]
        lines.append(f"  {label}: {real_count} IDs ({pct:.2f}%)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("VS RANDOM (BOOTSTRAPPED)")
    lines.append("-" * 80)
    lines.append(
        "Random baseline created by shuffling cluster labels while preserving "
        "label frequencies and recomputing unique-cluster-per-ID statistics."
    )
    lines.append("")
    for k in range(1, MAX_UNIQUE_BUCKET + 1):
        label = f"{k}+ unique cluster(s)" if k == MAX_UNIQUE_BUCKET else f"{k} unique cluster(s)"
        real_count = real_stats["unique_cluster_dist"].get(k, 0)
        real_pct = 100 * real_count / real_stats["n_orgs"]
        rand_avg = random_dist.get(k, 0.0)
        rand_pct = 100 * rand_avg / real_stats["n_orgs"]
        lines.append(
            f"  {label}: REAL={real_count} ({real_pct:.2f}%), "
            f"RANDOM≈{rand_avg:.1f} ({rand_pct:.2f}%)"
        )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Evaluation report written to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clustering consistency across years and compare to random baseline."
    )
    parser.add_argument(
        "--clusters-dir",
        type=Path,
        default=DEFAULT_CLUSTER_DIR,
        help="Directory containing clustering result CSV files.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to PCA embeddings CSV (index=id, columns=PCs).",
    )
    parser.add_argument(
        "--raw-features-path",
        type=Path,
        default=DEFAULT_RAW_FEATURES_PATH,
        help="Path to extracted features CSV (used when --use-raw-features is set).",
    )
    parser.add_argument(
        "--use-raw-features",
        action="store_true",
        help="Evaluate clusters using raw tsfresh features instead of PCA embeddings (features will be standardized).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Directory to store evaluation outputs.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap runs for random baseline.",
    )
    return parser.parse_args()


def extract_k_value(cluster_path: Path) -> int | None:
    """Extract k value from cluster file path (e.g., 'kmeans_k20' -> 20)."""
    stem = cluster_path.stem.lower()
    if "kmeans_k" in stem:
        try:
            k_str = stem.split("kmeans_k")[1]
            # Remove any trailing non-digit characters
            k_str = ''.join(filter(str.isdigit, k_str))
            return int(k_str) if k_str else None
        except Exception:
            return None
    return None


def create_summary_csv(
    summary_rows: list[dict],
    output_path: Path,
) -> None:
    """Create a CSV summary file with all evaluation results."""
    if not summary_rows:
        print("No summary data to write.")
        return
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Sort by k value if available
    if 'k' in df_summary.columns:
        df_summary = df_summary.sort_values('k')
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_summary.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSummary CSV written to: {output_path}")
    print(f"Summary contains {len(df_summary)} evaluations")


def main() -> None:
    args = parse_args()

    if args.use_raw_features:
        feature_matrix = load_feature_matrix(
            args.raw_features_path,
            standardize=True,
            label="raw features",
        )
    else:
        feature_matrix = load_feature_matrix(
            args.embeddings_path,
            standardize=False,
            label="PCA embeddings",
        )
    cluster_dir = args.clusters_dir

    if not cluster_dir.exists():
        raise FileNotFoundError(f"Clusters directory not found: {cluster_dir}")

    cluster_files = sorted(cluster_dir.glob("*.csv"))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster CSV files found in {cluster_dir}")

    # Accumulate results for summary CSV
    summary_rows = []

    for cluster_path in cluster_files:
        print("\n" + "=" * 80)
        print(f"Evaluating clusters from: {cluster_path}")
        print("=" * 80)

        df_clusters = load_cluster_assignments(cluster_path)

        print("Analyzing real clustering statistics...")
        real_stats = analyze_real_clusters(df_clusters)

        print("Computing random baseline via bootstrapping...")
        random_dist = bootstrap_random_baseline(df_clusters, args.n_bootstrap)

        print("Computing silhouette score...")
        overall_sil, sample_sil, cluster_labels = silhouette_for_clusters(
            df_clusters, feature_matrix
        )

        method_name, hyperparams = infer_method_and_hyperparams(cluster_path)

        write_report(
            output_dir=args.eval_dir,
            input_clusters=cluster_path,
            method_name=method_name,
            hyperparams=hyperparams,
            real_stats=real_stats,
            random_dist=random_dist,
            silhouette=overall_sil,
        )

        sil_plot_path = args.eval_dir / f"{cluster_path.stem}_silhouette.png"
        plot_title = f"{method_name} ({hyperparams})"
        plot_silhouette_distribution(overall_sil, sample_sil, cluster_labels, sil_plot_path, plot_title)

        # Accumulate data for summary CSV
        k_value = extract_k_value(cluster_path)
        
        # Prepare cluster sizes as a dictionary (cluster_id -> size)
        cluster_sizes = real_stats.get("cluster_sizes", pd.Series())
        if isinstance(cluster_sizes, pd.Series):
            cluster_sizes_dict = cluster_sizes.to_dict()
        else:
            cluster_sizes_dict = {}
        
        # Count significant clusters (>10 datapoints)
        n_significant_clusters = sum(1 for size in cluster_sizes_dict.values() if size > 10)
        
        # Prepare bootstrap results as a dictionary
        bootstrap_dict = {
            f"bootstrap_n_unique_{i}": random_dist.get(i, 0.0)
            for i in range(1, MAX_UNIQUE_BUCKET + 1)
        }
        
        summary_row = {
            "k": k_value if k_value is not None else -1,
            "method": method_name,
            "hyperparams": hyperparams,
            "file_name": cluster_path.name,
            "silhouette_score": overall_sil if overall_sil is not None else None,
            "n_samples": real_stats["n_samples"],
            "n_orgs": real_stats["n_orgs"],
            "n_clusters": len(cluster_sizes_dict),
            "n_significant_clusters": n_significant_clusters,
            "cluster_sizes": str(cluster_sizes_dict),  # Store as string representation
        }
        
        # Add bootstrap results
        summary_row.update(bootstrap_dict)
        
        # Add unique cluster distribution
        unique_dist = real_stats.get("unique_cluster_dist", Counter())
        for i in range(1, MAX_UNIQUE_BUCKET + 1):
            summary_row[f"real_n_unique_{i}"] = unique_dist.get(i, 0)
        
        summary_rows.append(summary_row)

    # Write summary CSV
    if summary_rows:
        summary_path = args.eval_dir / "summary_results" / "cluster_evaluation_summary.csv"
        create_summary_csv(summary_rows, summary_path)


if __name__ == "__main__":
    main()

