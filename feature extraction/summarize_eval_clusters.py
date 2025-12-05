"""
Summarize and visualize cluster evaluation results across different k values.

This script reads the summary CSV file created by eval_clusters.py and creates:
1. A plot showing silhouette score distribution across different k values
2. A plot showing the number of significant clusters (>10 datapoints) for each k

Usage:
    python summarize_eval_clusters.py [--summary-csv PATH] [--output-dir PATH]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# DEFAULT PATHS
# ============================================================================
DEFAULT_SUMMARY_CSV = Path("output/data/eval_clusters/summary_results/cluster_evaluation_summary.csv")
DEFAULT_OUTPUT_DIR = Path("output/data/eval_clusters/summary_results")


def load_summary_csv(summary_path: Path) -> pd.DataFrame:
    """Load the summary CSV file with evaluation results."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    
    df = pd.read_csv(summary_path)
    print(f"Loaded summary CSV: {len(df)} evaluations")
    print(f"Columns: {list(df.columns)}")
    return df


def plot_silhouette_scores(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot silhouette scores across different k values.
    
    Args:
        df: DataFrame with evaluation summary
        output_path: Path to save the plot
    """
    # Filter rows with valid k values and silhouette scores
    df_plot = df[(df['k'] > 0) & (df['silhouette_score'].notna())].copy()
    
    if len(df_plot) == 0:
        print("Warning: No valid data for silhouette score plot.")
        return
    
    # Sort by k value
    df_plot = df_plot.sort_values('k')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    bars = ax.bar(
        df_plot['k'].astype(str),
        df_plot['silhouette_score'],
        color='steelblue',
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5,
    )
    
    # Add value labels on top of bars
    for i, (k, score) in enumerate(zip(df_plot['k'], df_plot['silhouette_score'])):
        ax.text(
            i,
            score,
            f'{score:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title(
        'Silhouette Score Distribution Across Different k Values',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(df_plot['silhouette_score']) * 1.1])
    
    # Rotate x-axis labels if many k values
    if len(df_plot) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Silhouette score plot saved to: {output_path}")


def plot_significant_clusters(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot the number of significant clusters (>10 datapoints) for each k value.
    
    Args:
        df: DataFrame with evaluation summary
        output_path: Path to save the plot
    """
    # Filter rows with valid k values
    df_plot = df[df['k'] > 0].copy()
    
    if len(df_plot) == 0:
        print("Warning: No valid data for significant clusters plot.")
        return
    
    # Sort by k value
    df_plot = df_plot.sort_values('k')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    bars = ax.bar(
        df_plot['k'].astype(str),
        df_plot['n_significant_clusters'],
        color='forestgreen',
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5,
        label='Significant clusters (>10 datapoints)',
    )
    
    # Also plot total number of clusters for comparison
    ax.plot(
        df_plot['k'].astype(str),
        df_plot['n_clusters'],
        marker='o',
        markersize=8,
        linewidth=2,
        color='orange',
        label='Total clusters',
        linestyle='--',
        alpha=0.8,
    )
    
    # Add value labels on top of bars
    for i, (k, n_sig) in enumerate(zip(df_plot['k'], df_plot['n_significant_clusters'])):
        ax.text(
            i,
            n_sig,
            f'{int(n_sig)}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax.set_title(
        'Significant Clusters (>10 datapoints) Across Different k Values',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=10)
    
    # Rotate x-axis labels if many k values
    if len(df_plot) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Significant clusters plot saved to: {output_path}")


def plot_silhouette_vs_k_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a scatter/line plot of silhouette scores vs k values.
    
    Args:
        df: DataFrame with evaluation summary
        output_path: Path to save the plot
    """
    # Filter rows with valid k values and silhouette scores
    df_plot = df[(df['k'] > 0) & (df['silhouette_score'].notna())].copy()
    
    if len(df_plot) == 0:
        print("Warning: No valid data for silhouette vs k scatter plot.")
        return
    
    # Sort by k value
    df_plot = df_plot.sort_values('k')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create line plot with markers
    ax.plot(
        df_plot['k'],
        df_plot['silhouette_score'],
        marker='o',
        markersize=10,
        linewidth=2.5,
        color='steelblue',
        markerfacecolor='lightblue',
        markeredgecolor='black',
        markeredgewidth=2,
        label='Silhouette Score',
    )
    
    # Add value labels next to points
    for k, score in zip(df_plot['k'], df_plot['silhouette_score']):
        ax.annotate(
            f'{score:.3f}',
            (k, score),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            fontweight='bold',
        )
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title(
        'Silhouette Score vs Number of Clusters (k)',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Silhouette vs k scatter plot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize cluster evaluation results across different k values."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to cluster evaluation summary CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save summary plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("="*60)
    print("Cluster Evaluation Summary Visualization")
    print("="*60 + "\n")
    
    # Load summary CSV
    print("Loading summary CSV...")
    df = load_summary_csv(args.summary_csv)
    
    # Print summary statistics
    if 'k' in df.columns and 'silhouette_score' in df.columns:
        df_valid = df[(df['k'] > 0) & (df['silhouette_score'].notna())]
        if len(df_valid) > 0:
            print(f"\nSummary Statistics:")
            print(f"  Number of k values evaluated: {len(df_valid)}")
            print(f"  k range: {df_valid['k'].min()} - {df_valid['k'].max()}")
            print(f"  Mean silhouette score: {df_valid['silhouette_score'].mean():.4f}")
            print(f"  Max silhouette score: {df_valid['silhouette_score'].max():.4f}")
            best_k = df_valid.loc[df_valid['silhouette_score'].idxmax(), 'k']
            print(f"  Best k (highest silhouette): {int(best_k)}")
    
    # Create plots
    print("\nCreating visualizations...")
    
    # Plot 1: Silhouette score distribution (bar chart)
    silhouette_bar_path = args.output_dir / "silhouette_scores_bar.png"
    plot_silhouette_scores(df, silhouette_bar_path)
    
    # Plot 2: Silhouette score vs k (line/scatter plot)
    silhouette_line_path = args.output_dir / "silhouette_scores_line.png"
    plot_silhouette_vs_k_scatter(df, silhouette_line_path)
    
    # Plot 3: Significant clusters
    significant_clusters_path = args.output_dir / "significant_clusters.png"
    plot_significant_clusters(df, significant_clusters_path)
    
    print("\n" + "="*60)
    print("Summary visualization complete!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"  - Silhouette scores (bar): {silhouette_bar_path.name}")
    print(f"  - Silhouette scores (line): {silhouette_line_path.name}")
    print(f"  - Significant clusters: {significant_clusters_path.name}")


if __name__ == "__main__":
    main()

