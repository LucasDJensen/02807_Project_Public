import argparse
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

# ============================================================================
# HYPERPARAMETERS - Adjust these as needed
# ============================================================================
# Number of principal components to plot (default: first 10)
DEFAULT_N_FEATURES = 10

# Number of datapoints to plot (default: first 200)
DEFAULT_N_DATAPOINTS = 200

# Number of features per plot when plotting all features
FEATURES_PER_PLOT = 5

# ============================================================================
# DEFAULT PATHS
# ============================================================================
DEFAULT_INPUT_PATH = Path("output/data/extracted_features/pca_embeddings.csv")
DEFAULT_OUTPUT_DIR = Path("output/data/extracted_features/pca_plots")


def extract_catch_id(id_with_year: str) -> str:
    """Extract catchment ID without year suffix (e.g., '12000001_2012' -> '12000001')."""
    parts = str(id_with_year).split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else str(id_with_year)


def extract_year(id_with_year: str) -> str:
    """Extract year from ID (e.g., '12000001_2012' -> '2012')."""
    parts = str(id_with_year).split("_")
    return parts[-1] if len(parts) > 1 else "unknown"


def load_pca_embeddings(input_path: Path) -> pd.DataFrame:
    """Load PCA embeddings from CSV."""
    df = pd.read_csv(input_path, index_col=0)
    print(f"Loaded PCA embeddings shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features (PCs): {len(df.columns)}")
    return df


def create_color_mapping_by_catch_id(df: pd.DataFrame) -> dict[str, int]:
    """Create a mapping from catch_id to a unique color index."""
    catch_ids = df.index.map(extract_catch_id)
    unique_catch_ids = sorted(catch_ids.unique())
    return {catch_id: idx for idx, catch_id in enumerate(unique_catch_ids)}


def create_color_mapping_by_year(df: pd.DataFrame) -> dict[str, int]:
    """Create a mapping from year to a unique color index."""
    years = df.index.map(extract_year)
    unique_years = sorted(years.unique())
    return {year: idx for idx, year in enumerate(unique_years)}


def plot_feature_pairs(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_datapoints: int,
    output_path: Path,
    color_by: str = "catch_id",
    title_suffix: str = "",
) -> None:
    """
    Create all-against-all scatter plots for the given features.
    
    Args:
        df: DataFrame with PCA embeddings (index=id, columns=PC names)
        feature_cols: List of feature column names to plot
        n_datapoints: Number of datapoints to plot (first N)
        output_path: Path to save the plot
        color_by: Either "catch_id" or "year" to determine coloring scheme
        title_suffix: Optional suffix to add to plot title
    """
    # Limit to first N datapoints
    df_plot = df.head(n_datapoints).copy()
    
    # Create color mapping based on scheme
    if color_by == "catch_id":
        df_plot["color_key"] = df_plot.index.map(extract_catch_id)
        color_map = create_color_mapping_by_catch_id(df_plot)
        color_label = "catchment ID"
    elif color_by == "year":
        df_plot["color_key"] = df_plot.index.map(extract_year)
        color_map = create_color_mapping_by_year(df_plot)
        color_label = "year"
    else:
        raise ValueError(f"color_by must be 'catch_id' or 'year', got '{color_by}'")
    
    df_plot["color_idx"] = df_plot["color_key"].map(color_map)
    
    # Get all pairs of features
    feature_pairs = list(combinations(feature_cols, 2))
    n_pairs = len(feature_pairs)
    
    if n_pairs == 0:
        print(f"No pairs to plot for features: {feature_cols}")
        return
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get colormap with better contrast
    n_unique_colors = len(color_map)
    if n_unique_colors <= 12:
        # Use Set3 for excellent contrast (12 distinct colors)
        cmap = plt.cm.get_cmap("Set3")
    elif n_unique_colors <= 20:
        # Use tab20 for good contrast (20 distinct colors)
        cmap = plt.cm.get_cmap("tab20")
    else:
        # For more colors, cycle through Set3 and tab20 for better contrast
        # Create a custom colormap by combining high-contrast colormaps
        colors_set3 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_dark2 = plt.cm.Dark2(np.linspace(0, 1, 8))
        # Combine and cycle through them
        combined_colors = np.vstack([colors_set3, colors_tab20, colors_dark2])
        cmap = ListedColormap(combined_colors)
    
    # Plot each pair
    for idx, (feat_x, feat_y) in enumerate(feature_pairs):
        ax = axes[idx]
        
        # Scatter plot with colors based on color_by scheme
        scatter = ax.scatter(
            df_plot[feat_x],
            df_plot[feat_y],
            c=df_plot["color_idx"],
            cmap=cmap,
            alpha=0.6,
            s=20,
            edgecolors="black",
            linewidths=0.3,
        )
        
        ax.set_xlabel(feat_x, fontsize=10, fontweight="bold")
        ax.set_ylabel(feat_y, fontsize=10, fontweight="bold")
        ax.set_title(f"{feat_x} vs {feat_y}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    title = f"PCA Feature Pairwise Scatter Plots (first {n_datapoints} datapoints, colored by {color_label})"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    print(f"  Features plotted: {feature_cols}")
    print(f"  Number of pairs: {n_pairs}")
    print(f"  Datapoints: {n_datapoints}")
    print(f"  Unique {color_label}s: {len(color_map)}")


def plot_all_features_in_batches(
    df: pd.DataFrame,
    n_datapoints: int,
    features_per_plot: int,
    output_dir: Path,
) -> None:
    """
    Plot all features in batches of features_per_plot.
    Creates both org_id and year colored versions for each batch.
    
    Args:
        df: DataFrame with PCA embeddings
        n_datapoints: Number of datapoints to plot
        features_per_plot: Number of features per plot
        output_dir: Directory to save plots
    """
    all_features = list(df.columns)
    n_features = len(all_features)
    n_batches = int(np.ceil(n_features / features_per_plot))
    
    print(f"\nPlotting all {n_features} features in {n_batches} batches of {features_per_plot}...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * features_per_plot
        end_idx = min(start_idx + features_per_plot, n_features)
        batch_features = all_features[start_idx:end_idx]
        
        # Create title suffix
        if len(batch_features) == 1:
            title_suffix = f"{batch_features[0]}"
        else:
            title_suffix = f"{batch_features[0]} to {batch_features[-1]}"
        
        # Create base filename
        base_filename = f"pca_scatter_{batch_features[0]}_to_{batch_features[-1]}"
        
        # Plot by catch_id
        output_path_catch = output_dir / f"{base_filename}_by_catch_id.png"
        plot_feature_pairs(
            df=df,
            feature_cols=batch_features,
            n_datapoints=n_datapoints,
            output_path=output_path_catch,
            color_by="catch_id",
            title_suffix=title_suffix,
        )
        
        # Plot by year
        output_path_year = output_dir / f"{base_filename}_by_year.png"
        plot_feature_pairs(
            df=df,
            feature_cols=batch_features,
            n_datapoints=n_datapoints,
            output_path=output_path_year,
            color_by="year",
            title_suffix=title_suffix,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create all-against-all scatter plots for PCA features. Generates two versions: one colored by catchment ID and one colored by year."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to PCA embeddings CSV (index=id, columns=PCs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save scatter plots.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=DEFAULT_N_FEATURES,
        help=f"Number of features (PCs) to plot (default: {DEFAULT_N_FEATURES}).",
    )
    parser.add_argument(
        "--n-datapoints",
        type=int,
        default=DEFAULT_N_DATAPOINTS,
        help=f"Number of datapoints to plot (default: {DEFAULT_N_DATAPOINTS}).",
    )
    parser.add_argument(
        "--plot-all-features",
        action="store_true",
        help="Plot all features in batches (overrides --n-features).",
    )
    parser.add_argument(
        "--features-per-plot",
        type=int,
        default=FEATURES_PER_PLOT,
        help=f"Number of features per plot when using --plot-all-features (default: {FEATURES_PER_PLOT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load PCA embeddings
    df = load_pca_embeddings(args.input_path)
    
    if args.plot_all_features:
        # Plot all features in batches (both org_id and year versions)
        plot_all_features_in_batches(
            df=df,
            n_datapoints=args.n_datapoints,
            features_per_plot=args.features_per_plot,
            output_dir=args.output_dir,
        )
    else:
        # Plot first N features (both org_id and year versions)
        feature_cols = list(df.columns[: args.n_features])
        base_filename = f"pca_scatter_{feature_cols[0]}_to_{feature_cols[-1]}"
        
        # Plot by catch_id
        output_path_catch = args.output_dir / f"{base_filename}_by_catch_id.png"
        plot_feature_pairs(
            df=df,
            feature_cols=feature_cols,
            n_datapoints=args.n_datapoints,
            output_path=output_path_catch,
            color_by="catch_id",
        )
        
        # Plot by year
        output_path_year = args.output_dir / f"{base_filename}_by_year.png"
        plot_feature_pairs(
            df=df,
            feature_cols=feature_cols,
            n_datapoints=args.n_datapoints,
            output_path=output_path_year,
            color_by="year",
        )
    
    print("\n" + "=" * 60)
    print("PCA plotting complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

