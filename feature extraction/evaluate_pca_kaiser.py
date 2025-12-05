"""
Evaluate the number of PCA components using the Kaiser criterion.

The Kaiser criterion states that components with eigenvalues greater than 1
should be retained. This is a common rule of thumb for PCA dimensionality reduction.

Usage:
    python evaluate_pca_kaiser.py [--input-path PATH] [--output-plot-path PATH]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# DEFAULT PATHS
# ============================================================================
DEFAULT_INPUT_PATH = Path("output/data/extracted_features/extracted_features.csv")
DEFAULT_OUTPUT_PLOT_PATH = Path("output/data/extracted_features/pca_kaiser_criterion.png")
DEFAULT_OUTPUT_TXT_PATH = Path("output/data/extracted_features/pca_kaiser_result.txt")


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


def apply_kaiser_criterion(pca: PCA) -> tuple[int, np.ndarray]:
    """
    Apply Kaiser criterion to determine number of components to keep.
    
    The Kaiser criterion: keep components with eigenvalues > 1.
    
    Args:
        pca: Fitted PCA object
        
    Returns:
        Tuple of (number of components to keep, eigenvalues)
    """
    # Get eigenvalues (explained variance)
    eigenvalues = pca.explained_variance_
    
    # Kaiser criterion: keep components with eigenvalue > 1
    components_to_keep = np.sum(eigenvalues > 1.0)
    
    return components_to_keep, eigenvalues


def plot_kaiser_criterion(
    eigenvalues: np.ndarray,
    explained_variance_ratio: np.ndarray,
    n_components_kaiser: int,
    output_path: Path,
    max_components_to_show: int = None,
) -> None:
    """
    Plot eigenvalues and Kaiser criterion threshold.
    
    Args:
        eigenvalues: Array of eigenvalues for all components
        n_components_kaiser: Number of components recommended by Kaiser criterion
        output_path: Path to save the plot
        max_components_to_show: Maximum number of components to display (None = show all)
    """
    if max_components_to_show is None:
        max_components_to_show = len(eigenvalues)
    else:
        max_components_to_show = min(max_components_to_show, len(eigenvalues))
    
    eigenvalues_to_show = eigenvalues[:max_components_to_show]
    component_numbers = np.arange(1, len(eigenvalues_to_show) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot eigenvalues
    ax.plot(
        component_numbers,
        eigenvalues_to_show,
        marker='o',
        markersize=6,
        linewidth=2,
        label='Eigenvalue',
        color='blue',
    )
    
    # Draw Kaiser threshold line (eigenvalue = 1)
    ax.axhline(
        y=1.0,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Kaiser Criterion (eigenvalue = 1)',
        alpha=0.7,
    )
    
    # Calculate how many components above threshold are shown in this plot
    n_components_above_threshold_visible = np.sum(eigenvalues_to_show > 1.0)
    
    # Highlight components above threshold
    mask_above = eigenvalues_to_show > 1.0
    if np.any(mask_above):
        ax.scatter(
            component_numbers[mask_above],
            eigenvalues_to_show[mask_above],
            color='green',
            s=100,
            zorder=5,
            label=f'Components to keep (shown: {n_components_above_threshold_visible}, total: {n_components_kaiser})',
            edgecolors='black',
            linewidth=1.5,
        )
    
    # Highlight components below threshold
    mask_below = eigenvalues_to_show <= 1.0
    if np.any(mask_below):
        ax.scatter(
            component_numbers[mask_below],
            eigenvalues_to_show[mask_below],
            color='orange',
            s=100,
            zorder=5,
            label='Components to discard',
            edgecolors='black',
            linewidth=1.5,
        )
    
    ax.set_xlabel('Principal Component Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eigenvalue', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Kaiser Criterion for PCA Component Selection\n'
        f'Recommended Components: {n_components_kaiser} (eigenvalue > 1)',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add text box with summary
    if n_components_kaiser > 0:
        explained_var = np.sum(explained_variance_ratio[:n_components_kaiser])
    else:
        explained_var = 0.0
    
    # Check if plot is truncated
    plot_truncated = max_components_to_show < len(eigenvalues)
    
    textstr = (
        f'Total components: {len(eigenvalues)}\n'
        f'Components to keep: {n_components_kaiser}\n'
        f'Components to discard: {len(eigenvalues) - n_components_kaiser}\n'
        f'Explained variance: {explained_var:.2%}'
    )
    if plot_truncated:
        textstr += f'\n(Plot shows first {max_components_to_show} components)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
    )
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Kaiser criterion plot saved to: {output_path}")


def save_kaiser_results(
    n_components: int,
    eigenvalues: np.ndarray,
    explained_variance_ratio: np.ndarray,
    output_path: Path,
) -> None:
    """Save Kaiser criterion results to a text file."""
    cumulative_var = np.cumsum(explained_variance_ratio)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PCA Kaiser Criterion Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total number of components: {len(eigenvalues)}\n")
        f.write(f"Recommended number of components (Kaiser criterion): {n_components}\n")
        f.write(f"  (Components with eigenvalue > 1.0)\n\n")
        
        f.write(f"Explained variance by selected components:\n")
        f.write(f"  Cumulative explained variance: {cumulative_var[n_components-1]:.4f} ({cumulative_var[n_components-1]:.2%})\n")
        f.write(f"  Individual variance: {np.sum(explained_variance_ratio[:n_components]):.4f} ({np.sum(explained_variance_ratio[:n_components]):.2%})\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Component Analysis (first 100 components):\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Component':<12} {'Eigenvalue':<15} {'Eigenvalue>1':<15} {'Explained Var':<15} {'Cumulative':<15}\n")
        f.write("-"*60 + "\n")
        
        for i in range(min(100, len(eigenvalues))):
            component_name = f"PC{i+1}"
            eigenvalue = eigenvalues[i]
            above_threshold = "YES" if eigenvalue > 1.0 else "NO"
            explained = explained_variance_ratio[i]
            cumulative = cumulative_var[i]
            
            f.write(
                f"{component_name:<12} "
                f"{eigenvalue:<15.4f} "
                f"{above_threshold:<15} "
                f"{explained:<15.4f} "
                f"{cumulative:<15.4f}\n"
            )
        
        if len(eigenvalues) > 100:
            f.write(f"\n... (showing first 100 of {len(eigenvalues)} components)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("RECOMMENDATION:\n")
        f.write("="*60 + "\n")
        f.write(f"Use {n_components} principal components based on Kaiser criterion.\n")
        f.write(f"This will retain {cumulative_var[n_components-1]:.2%} of the total variance.\n")
    
    print(f"Kaiser criterion results saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate number of PCA components using Kaiser criterion (eigenvalue > 1)."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to extracted_features.csv",
    )
    parser.add_argument(
        "--output-plot-path",
        type=Path,
        default=DEFAULT_OUTPUT_PLOT_PATH,
        help="Path to save Kaiser criterion plot",
    )
    parser.add_argument(
        "--output-txt-path",
        type=Path,
        default=DEFAULT_OUTPUT_TXT_PATH,
        help="Path to save detailed results text file",
    )
    parser.add_argument(
        "--max-components-to-show",
        type=int,
        default=None,
        help="Maximum number of components to show in plot (None = show all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("="*60)
    print("PCA Kaiser Criterion Evaluation")
    print("="*60 + "\n")
    
    # Load features
    print("Loading features...")
    features = load_features(args.input_path)
    
    # Standardize features
    print("\nStandardizing features...")
    features_std, scaler = standardize_features(features)
    
    # Fit PCA with all components to get all eigenvalues
    print("\nFitting PCA with all components...")
    n_features = len(features.columns)
    pca = PCA(n_components=n_features)
    pca.fit(features_std)
    
    # Apply Kaiser criterion
    print("\nApplying Kaiser criterion...")
    n_components_kaiser, eigenvalues = apply_kaiser_criterion(pca)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total number of components: {len(eigenvalues)}")
    print(f"Recommended number of components (Kaiser criterion): {n_components_kaiser}")
    print(f"  (Components with eigenvalue > 1.0)")
    
    if n_components_kaiser > 0:
        explained_var = np.sum(pca.explained_variance_ratio_[:n_components_kaiser])
        print(f"\nExplained variance by selected components: {explained_var:.4f} ({explained_var:.2%})")
        
        # Show eigenvalues for first few components
        print(f"\nFirst 10 eigenvalues:")
        for i in range(min(10, len(eigenvalues))):
            above_threshold = "[KEEP]" if eigenvalues[i] > 1.0 else "[DISCARD]"
            print(f"  PC{i+1}: {eigenvalues[i]:.4f} {above_threshold}")
    else:
        print("\nWARNING: Kaiser criterion suggests keeping 0 components!")
        print("This may indicate issues with the data or that PCA is not suitable.")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_kaiser_criterion(
        eigenvalues,
        pca.explained_variance_ratio_,
        n_components_kaiser,
        args.output_plot_path,
        args.max_components_to_show,
    )
    
    # Save detailed results
    print("\nSaving detailed results...")
    save_kaiser_results(
        n_components_kaiser,
        eigenvalues,
        pca.explained_variance_ratio_,
        args.output_txt_path,
    )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    print(f"Recommended number of components: {n_components_kaiser}")
    print(f"Plot saved to: {args.output_plot_path}")
    print(f"Results saved to: {args.output_txt_path}")


if __name__ == "__main__":
    main()

