import argparse
import os
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

from tsfresh import extract_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute


# Feature extraction presets:
#   minimal        -> ~10 summary statistics per series (fast, default)
#   efficient      -> ~200–300 features per series (balanced coverage)
#   comprehensive  -> 500+ features per series (slow & memory heavy)
# Switch via --feature-mode or by editing DEFAULT_FEATURE_MODE.
DEFAULT_FEATURE_MODE = "efficient"
DEFAULT_DATA_PATH = Path("output/data/discharge_tables/discharge_tables_by_year.csv")
DEFAULT_OUTPUT_ROOT = Path("output/data/extracted_features")
DEFAULT_FEATURE_OUTPUT = DEFAULT_OUTPUT_ROOT / "extracted_features.csv"
DEFAULT_HIST_OUTPUT = DEFAULT_OUTPUT_ROOT / "feature_histograms.png"
DEFAULT_QUANTILE_OUTPUT = DEFAULT_OUTPUT_ROOT / "feature_quantile_plots.png"
DEFAULT_COVARIANCE_OUTPUT = DEFAULT_OUTPUT_ROOT / "feature_covariance_matrix.png"


def load_discharge_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_numeric(df.index, errors="coerce")
    df.index.name = "time"
    print(f"Data shape: {df.shape}")
    print(f"Index range: {df.index.min()} to {df.index.max()}")
    print(f"Number of time series (columns): {len(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    return df


def prepare_extraction_frame(df: pd.DataFrame) -> pd.DataFrame:
    extraction_data: list[dict[str, object]] = []

    for col_id in df.columns:
        series = df[col_id].dropna()
        if series.empty:
            continue

        extraction_data.extend(
            {
                "id": col_id,
                "time": int(time_val),
                "value": value,
            }
            for time_val, value in series.items()
        )

    extraction_df = pd.DataFrame(extraction_data)
    print(f"Total number of id_year combinations: {extraction_df['id'].nunique()}")
    print(f"Total number of data points: {len(extraction_df)}")
    print("\nSample data:")
    print(extraction_df.head(10))
    return extraction_df


def load_custom_feature_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Custom feature config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("Custom feature config must be a JSON object mapping calculator names to parameters.")

    return config


def get_feature_parameters(mode: str, custom_config: dict[str, object] | None = None):
    match mode.lower():
        case "minimal":
            return MinimalFCParameters()
        case "efficient":
            return EfficientFCParameters()
        case "comprehensive":
            return ComprehensiveFCParameters()
        case "custom":
            if not custom_config:
                raise ValueError("Custom feature mode requires a feature configuration provided via --feature-config.")
            return custom_config
        case _:
            warnings.warn(
                f"Unknown mode '{mode}', falling back to comprehensive parameters."
            )
            return ComprehensiveFCParameters()


def extract_tsfresh_features(
    extraction_df: pd.DataFrame, mode: str, custom_config: dict[str, object] | None = None
) -> pd.DataFrame:
    feature_params = get_feature_parameters(mode, custom_config)
    print(f"Extracting features using '{mode}' mode. This may take a while...")

    features = extract_features(
        extraction_df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=feature_params,
    )
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of features: {len(features.columns)}")

    features = impute(features)
    print("\nFeatures after imputation:")
    print(features.head())
    return features


def save_features(features: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)
    print(
        f"Features saved to: {output_path} "
        f"(rows={len(features)}, columns={len(features.columns)})"
    )


def plot_histograms(features: pd.DataFrame, output_path: Path) -> None:
    n_features = len(features.columns)
    if n_features == 0:
        print("No features to plot histograms.")
        return

    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, col in enumerate(features.columns):
        ax = axes[i]
        series = features[col].dropna()
        if series.empty or series.nunique(dropna=True) <= 1:
            ax.set_visible(False)
            continue

        values = series.to_numpy()
        lower, upper = np.nanpercentile(values, [1, 99])
        clipped = np.clip(values, lower, upper)

        unique_vals = np.unique(clipped)
        unique_count = len(unique_vals)
        if unique_count <= 1:
            ax.set_visible(False)
            continue

        bins = min(50, max(5, int(np.sqrt(len(clipped)))))
        bins = max(2, min(bins, unique_count - 1))

        try:
            counts, bins_edges, _ = ax.hist(
                clipped, bins=bins, edgecolor="black", alpha=0.7
            )
        except ValueError:
            try:
                counts, bins_edges, _ = ax.hist(
                    clipped, bins="auto", edgecolor="black", alpha=0.7
                )
            except ValueError:
                ax.set_visible(False)
                continue

        ax.set_yscale("log", nonpositive="clip")
        positive_counts = counts[counts > 0]
        if positive_counts.size:
            ax.set_ylim(bottom=max(1e-2, positive_counts.min() * 0.8))

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value (1st–99th pct clipped)")
        ax.set_ylabel("Frequency (log scale)")
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histograms saved to: {output_path}")


def plot_quantile_functions(features: pd.DataFrame, output_path: Path) -> None:
    n_features = len(features.columns)
    if n_features == 0:
        print("No features to plot quantile functions.")
        return

    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, col in enumerate(features.columns):
        ax = axes[i]
        values = features[col].dropna().sort_values()
        if values.empty or values.nunique() <= 1:
            ax.set_visible(False)
            continue

        positive_values = values[values > 0]
        if positive_values.empty:
            ax.set_visible(False)
            continue

        quantiles = np.linspace(0, 1, len(positive_values))
        ax.plot(positive_values, quantiles, linewidth=2)
        ax.set_xscale("log")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value (log scale)")
        ax.set_ylabel("Quantile")
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Quantile plots saved to: {output_path}")


def plot_correlation_matrix(features: pd.DataFrame, output_path: Path) -> None:
    if features.empty:
        print("No features available for correlation plot.")
        return

    corr_matrix = features.corr(numeric_only=True)
    if corr_matrix.empty:
        print("Correlation matrix is empty; skipping plot.")
        return

    fig_size = min(24, max(8, 0.25 * len(corr_matrix)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title("Feature Correlation Matrix")
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr_matrix.index, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Correlation matrix plot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tsfresh features from discharge data."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the discharge CSV file.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["minimal", "efficient", "comprehensive", "custom"],
        default=DEFAULT_FEATURE_MODE,
        help="Feature extraction preset to use.",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        help="Path to JSON file describing custom tsfresh feature calculators (required when --feature-mode custom).",
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=DEFAULT_FEATURE_OUTPUT,
        help="Where to store the extracted features CSV.",
    )
    parser.add_argument(
        "--hist-output",
        type=Path,
        default=DEFAULT_HIST_OUTPUT,
        help="Where to store the feature histogram figure.",
    )
    parser.add_argument(
        "--quantile-output",
        type=Path,
        default=DEFAULT_QUANTILE_OUTPUT,
        help="Where to store the quantile plot figure.",
    )
    parser.add_argument(
        "--covariance-output",
        type=Path,
        default=DEFAULT_COVARIANCE_OUTPUT,
        help="Where to store the feature covariance heatmap.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()

    df = load_discharge_data(args.data_path)
    extraction_df = prepare_extraction_frame(df)

    if extraction_df.empty:
        raise ValueError(
            "No time series contained any data after preprocessing. "
            "Inspect the segmented dataset for missing values."
        )

    custom_config = (
        load_custom_feature_config(args.feature_config)
        if args.feature_mode == "custom"
        else None
    )

    features = extract_tsfresh_features(extraction_df, args.feature_mode, custom_config)
    save_features(features, args.feature_output)

    # Plotting only works with < 20 features
    #plot_histograms(features, args.hist_output)
    #plot_quantile_functions(features, args.quantile_output)
    #plot_correlation_matrix(features, args.covariance_output)


if __name__ == "__main__":
    main()

