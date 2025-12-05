import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_PATH = Path("output/data/discharge_tables/discharge_table_2001_2022.csv")
DEFAULT_OUTPUT_PATH = Path("output/data/discharge_tables/discharge_tables_by_year.csv")
DEFAULT_PLOT_PATH = Path("output/data/plots_dataprocessing/yearly_datapoint_distribution.png")
DEFAULT_STATS_PATH = Path("output/data/plots_dataprocessing/yearly_stats.txt")
MIN_POINTS = 365


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def segment_by_year(df: pd.DataFrame, filter_min_points: bool = True) -> pd.DataFrame:
    """Segment time series by year. If filter_min_points is False, includes all segments."""
    segmented_columns: dict[str, pd.Series] = {}
    for column in df.columns:
        ts = df[column].dropna()
        if ts.empty:
            continue

        ts_df = ts.to_frame(name="value")
        ts_df["year"] = ts_df.index.year
        grouped = ts_df.groupby("year")

        for year, group in grouped:
            non_null_count = group["value"].count()
            if filter_min_points and non_null_count <= MIN_POINTS - 1:
                continue

            series_name = f"{column}_{year}"
            series = pd.Series(
                group["value"].values,
                index=group.index.dayofyear,
                name=series_name,
            )
            segmented_columns[series_name] = series.reindex(range(1, 367))

    if not segmented_columns:
        return pd.DataFrame()

    segmented_df = pd.DataFrame(segmented_columns)
    segmented_df.index.name = "day_of_year"
    return segmented_df


def analyze_yearly_datapoints(df: pd.DataFrame) -> dict:
    """Analyze datapoint counts per year and categorize them."""
    stats = {
        "yearly_counts": {},  # year -> list of datapoint counts
        "yearly_full_years": {},  # year -> count of full years (>=365)
        "categories": {"<365": 0, "365": 0, "366": 0}
    }
    
    for column in df.columns:
        ts = df[column].dropna()
        if ts.empty:
            continue
        
        ts_df = ts.to_frame(name="value")
        ts_df["year"] = ts_df.index.year
        grouped = ts_df.groupby("year")
        
        for year, group in grouped:
            count = group["value"].count()
            
            # Track yearly counts
            if year not in stats["yearly_counts"]:
                stats["yearly_counts"][year] = []
            stats["yearly_counts"][year].append(count)
            
            # Count full years (>=365)
            if count >= 365:
                if year not in stats["yearly_full_years"]:
                    stats["yearly_full_years"][year] = 0
                stats["yearly_full_years"][year] += 1
            
            # Categorize datapoints
            if count < 365:
                stats["categories"]["<365"] += 1
            elif count == 365:
                stats["categories"]["365"] += 1
            elif count == 366:
                stats["categories"]["366"] += 1
    
    return stats


def plot_full_years_per_year(stats: dict, output_path: Path) -> None:
    """Plot the number of full years (>=365 datapoints) for each year."""
    yearly_full_years = stats["yearly_full_years"]
    
    if not yearly_full_years:
        print("No full years data available for plotting.")
        return
    
    years = sorted(yearly_full_years.keys())
    counts = [yearly_full_years[year] for year in years]
    
    plt.figure(figsize=(14, 8))
    plt.bar(years, counts, edgecolor="black", alpha=0.7)
    plt.title("Number of Full Years (>=365 datapoints) per Year", fontsize=14, fontweight="bold")
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Full Years", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Full years plot saved to: {output_path}")


def write_statistics(stats: dict, output_path: Path) -> None:
    """Write statistics to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("Yearly Datapoint Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # Write category counts
        f.write("Total number of datapoints by category:\n")
        f.write("-" * 50 + "\n")
        for category, count in stats["categories"].items():
            f.write(f"{category}: {count}\n")
        f.write("\n")
        
        # Write full years per year
        f.write("Number of full years (>=365 datapoints) per year:\n")
        f.write("-" * 50 + "\n")
        yearly_full_years = stats["yearly_full_years"]
        if yearly_full_years:
            for year in sorted(yearly_full_years.keys()):
                f.write(f"{year}: {yearly_full_years[year]}\n")
        else:
            f.write("No full years found.\n")
        f.write("\n")
        
        # Write summary statistics
        f.write("Summary:\n")
        f.write("-" * 50 + "\n")
        total_datapoints = sum(stats["categories"].values())
        f.write(f"Total number of year-segments: {total_datapoints}\n")
        
        total_full_years = sum(stats["yearly_full_years"].values())
        f.write(f"Total number of full years (>=365): {total_full_years}\n")
        
        if total_datapoints > 0:
            percentage_full = (total_full_years / total_datapoints) * 100
            f.write(f"Percentage of full years: {percentage_full:.2f}%\n")
    
    print(f"Statistics saved to: {output_path}")


def plot_datapoint_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data available for plotting.")
        return

    counts = df.count()
    plt.figure(figsize=(16, 9))
    plt.hist(counts, bins=50, edgecolor="black", alpha=0.7)
    plt.yscale("log")
    plt.title("Distribution of datapoint counts per segmented time series")
    plt.xlabel("Number of datapoints")
    plt.ylabel("Frequency (log scale)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment discharge table into yearly time series per ID."
    )
    parser.add_argument(
        "--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input CSV path."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV for segmented data.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Output path for datapoint distribution plot.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help="Output path for statistics text file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input_path)
    
    # Analyze yearly datapoints
    print("Analyzing yearly datapoints...")
    stats = analyze_yearly_datapoints(df)
    
    # Create plot for full years per year
    full_years_plot_path = args.plot_path.parent / "full_years_per_year.png"
    plot_full_years_per_year(stats, full_years_plot_path)
    
    # Write statistics to text file
    write_statistics(stats, args.stats_path)
    
    # Segment without filtering to plot distribution of all data
    segmented_unfiltered = segment_by_year(df, filter_min_points=False)
    plot_datapoint_distribution(segmented_unfiltered, args.plot_path)
    
    # Now filter and save the filtered dataset
    segmented_filtered = segment_by_year(df, filter_min_points=True)
    segmented_filtered.to_csv(args.output_path)
    print(f"Filtered dataset saved: {len(segmented_filtered.columns)} columns (removed {len(segmented_unfiltered.columns) - len(segmented_filtered.columns)} with <= {MIN_POINTS - 1} datapoints)")


if __name__ == "__main__":
    main()

