import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================================
# HYPERPARAMETERS - Adjust these as needed
# ============================================================================
# Number of principal components to keep in the output dataset
N_COMPONENTS = 80

# Number of components to include in the explained variance plot
N_COMPONENTS_FOR_VARIANCE_PLOT = 80

# Number of top features to show for each principal component
TOP_FEATURES_PER_PC = 10

# Number of top principal components to analyze for feature importance
TOP_PCS_TO_ANALYZE = 4

# ============================================================================
# DEFAULT PATHS
# ============================================================================
DEFAULT_INPUT_PATH = Path("output/data/extracted_features/extracted_features.csv")
DEFAULT_OUTPUT_PATH = Path("output/data/extracted_features/pca_embeddings.csv")
DEFAULT_VARIANCE_PLOT_PATH = Path(
    "output/data/extracted_features/pca_explained_variance.png"
)
DEFAULT_FEATURE_IMPORTANCE_PLOT_PATH = Path(
    "output/data/extracted_features/pca_feature_importance.png"
)


def load_features(input_path: Path) -> pd.DataFrame:
    """Load extracted features from CSV."""
    df = pd.read_csv(input_path, index_col=0)
    print(f"Loaded features shape: {df.shape}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Number of samples: {len(df)}")
    return df


def standardize_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    features_std = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns,
    )
    print("Features standardized (zero mean, unit variance)")
    return features_std, scaler


def fit_pca(
    features_std: pd.DataFrame, n_components: int
) -> tuple[PCA, pd.DataFrame]:
    """Fit PCA and transform features."""
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(features_std)

    embeddings_df = pd.DataFrame(
        pca_embeddings,
        index=features_std.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    print(f"\nPCA fitted with {n_components} components")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(
        f"First 5 components explained variance: {pca.explained_variance_ratio_[:5]}"
    )

    return pca, embeddings_df


def plot_explained_variance(
    pca: PCA, n_components: int, output_path: Path
) -> None:
    """Plot explained variance for the first n_components."""
    explained_var = pca.explained_variance_ratio_[:n_components]
    cumulative_var = np.cumsum(explained_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Individual explained variance
    ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title(f"Explained Variance by Component (first {n_components})", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Cumulative explained variance
    ax2.plot(
        range(1, len(cumulative_var) + 1),
        cumulative_var,
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax2.axhline(y=0.95, color="r", linestyle="--", label="95% variance")
    ax2.axhline(y=0.99, color="g", linestyle="--", label="99% variance")
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance Ratio", fontsize=12)
    ax2.set_title(f"Cumulative Explained Variance (first {n_components})", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Explained variance plot saved to: {output_path}")


def plot_feature_importance(
    pca: PCA,
    feature_names: list[str],
    top_pcs: int,
    top_features: int,
    output_path: Path,
) -> None:
    """Plot the most important features for the top principal components."""
    n_features = len(feature_names)
    components = pca.components_[:top_pcs]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for pc_idx in range(top_pcs):
        ax = axes[pc_idx]
        pc_weights = components[pc_idx]

        # Get top features by absolute weight
        top_indices = np.argsort(np.abs(pc_weights))[-top_features:][::-1]
        top_weights = pc_weights[top_indices]
        top_names = [feature_names[i] for i in top_indices]

        # Create bar plot
        colors = ["red" if w < 0 else "blue" for w in top_weights]
        ax.barh(range(len(top_weights)), top_weights, color=colors, alpha=0.7, edgecolor="black")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel("Component Weight", fontsize=10)
        ax.set_title(f"PC{pc_idx+1} - Top {top_features} Features", fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply PCA to extracted features and create embeddings."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to extracted_features.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save PCA embeddings CSV",
    )
    parser.add_argument(
        "--variance-plot-path",
        type=Path,
        default=DEFAULT_VARIANCE_PLOT_PATH,
        help="Path to save explained variance plot",
    )
    parser.add_argument(
        "--feature-importance-plot-path",
        type=Path,
        default=DEFAULT_FEATURE_IMPORTANCE_PLOT_PATH,
        help="Path to save feature importance plot",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=N_COMPONENTS,
        help=f"Number of principal components to keep (default: {N_COMPONENTS})",
    )
    parser.add_argument(
        "--n-components-for-variance-plot",
        type=int,
        default=N_COMPONENTS_FOR_VARIANCE_PLOT,
        help=f"Number of components for variance plot (default: {N_COMPONENTS_FOR_VARIANCE_PLOT})",
    )
    parser.add_argument(
        "--top-features-per-pc",
        type=int,
        default=TOP_FEATURES_PER_PC,
        help=f"Number of top features per PC (default: {TOP_FEATURES_PER_PC})",
    )
    parser.add_argument(
        "--top-pcs-to-analyze",
        type=int,
        default=TOP_PCS_TO_ANALYZE,
        help=f"Number of top PCs to analyze (default: {TOP_PCS_TO_ANALYZE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load features
    features = load_features(args.input_path)

    # Standardize features
    features_std, scaler = standardize_features(features)

    # Fit PCA with enough components for variance plot
    max_components = max(args.n_components, args.n_components_for_variance_plot)
    pca, embeddings_df = fit_pca(features_std, n_components=max_components)

    # Keep only the requested number of components in output
    if args.n_components < max_components:
        embeddings_df = embeddings_df.iloc[:, : args.n_components]

    # Save PCA embeddings
    embeddings_df.to_csv(args.output_path)
    print(f"\nPCA embeddings saved to: {args.output_path}")
    print(f"Embeddings shape: {embeddings_df.shape}")

    # Plot explained variance
    plot_explained_variance(pca, args.n_components_for_variance_plot, args.variance_plot_path)

    # Plot feature importance for top PCs
    plot_feature_importance(
        pca,
        list(features.columns),
        args.top_pcs_to_analyze,
        args.top_features_per_pc,
        args.feature_importance_plot_path,
    )

    print("\nPCA processing complete!")


if __name__ == "__main__":
    main()

