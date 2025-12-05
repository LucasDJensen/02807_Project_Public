"""
Reformat the clusters file:
Input, a path to a cluster file .csv:
    columns = ['id', 'assigned_cluster']

Output:
    a folder in the same dir as the file:
        The folder should contain csv files with the dataset split by year:
        ID should then ONLY contain the ID part, not the year part. 
        fx if there are 4 different years, there should be 4 files. 
        The file should be named: <year>.csv 
        Additionally, a majority_vote.csv file is created with the most common
        cluster assignment for each catchment across all years.
"""
import pandas as pd
import os
from pathlib import Path


def reformat_cluster_file(cluster_file_path: str):
    """
    Reformat a cluster file by splitting it into separate files per year.
    
    Args:
        cluster_file_path: Path to the input CSV file with columns ['id', 'assigned_cluster']
    """
    # Read the cluster file
    df = pd.read_csv(cluster_file_path)
    
    # Verify required columns exist
    if 'id' not in df.columns or 'assigned_cluster' not in df.columns:
        raise ValueError("CSV file must contain 'id' and 'assigned_cluster' columns")
    
    # Split the id column into catchment_id and year
    # Format: {catchment_id}_{year}
    df[['catchment_id', 'year']] = df['id'].str.rsplit('_', n=1, expand=True)
    df['year'] = df['year'].astype(int)
    
    # Create output folder in the same directory as the input file
    input_path = Path(cluster_file_path)
    output_dir = input_path.parent / f"{input_path.stem}_by_year"
    output_dir.mkdir(exist_ok=True)
    
    # Group by year and save separate files
    for year, group in df.groupby('year'):
        # Create output dataframe with only catchment_id and assigned_cluster
        output_df = group[['catchment_id', 'assigned_cluster']].copy()
        output_df.columns = ['id', 'assigned_cluster']  # Rename to match expected format
        
        # Save to CSV file named <year>.csv
        output_file = output_dir / f"{year}.csv"
        output_df.to_csv(output_file, index=False)
        print(f"Saved {len(output_df)} records for year {year} to {output_file}")
    
    print(f"\nReformatting complete! Output directory: {output_dir}")
    
    # Create majority vote file
    create_majority_vote_file(output_dir)
    
    return output_dir


def create_majority_vote_file(output_dir: Path):
    """
    Create a majority vote CSV file by combining all year files and voting on cluster assignments.
    
    Args:
        output_dir: Path to the directory containing the year CSV files
    """
    print("\n" + "="*50)
    print("Creating majority vote file...")
    print("="*50 + "\n")
    
    # Get all year CSV files (exclude majority_vote.csv if it exists)
    year_files = sorted([f for f in output_dir.glob('*.csv') if f.name != 'majority_vote.csv'])
    
    if not year_files:
        print("No CSV files found for majority voting.")
        return
    
    print(f"Found {len(year_files)} year files to process for majority vote...")
    
    # Collect all cluster assignments for each catchment across all years
    all_assignments = {}
    
    for year_file in year_files:
        print(f"Processing {year_file.name}...")
        df = pd.read_csv(year_file, dtype=str)
        
        # Verify required columns exist
        if 'id' not in df.columns:
            print(f"  Warning: 'id' column not found in {year_file.name}, skipping...")
            continue
        
        # Determine cluster column name (could be 'assigned_cluster' or 'cluster')
        cluster_col = None
        if 'assigned_cluster' in df.columns:
            cluster_col = 'assigned_cluster'
        elif 'cluster' in df.columns:
            cluster_col = 'cluster'
        else:
            print(f"  Warning: No cluster column found in {year_file.name}, skipping...")
            continue
        
        # Collect assignments
        for _, row in df.iterrows():
            catchment_id = str(row['id'])
            cluster_val = str(row[cluster_col])
            
            if catchment_id not in all_assignments:
                all_assignments[catchment_id] = []
            all_assignments[catchment_id].append(cluster_val)
    
    if not all_assignments:
        print("No valid assignments found for majority voting.")
        return
    
    # Perform majority vote for each catchment
    print("\nPerforming majority vote...")
    majority_vote_results = []
    
    for catchment_id, cluster_assignments in all_assignments.items():
        # Count occurrences of each cluster
        cluster_counts = pd.Series(cluster_assignments).value_counts()
        # Get the cluster with the highest count (majority vote)
        # If there's a tie, value_counts() returns them in order of frequency, so index[0] is fine
        majority_cluster = cluster_counts.index[0]
        majority_vote_results.append({
            'id': catchment_id,
            'cluster': majority_cluster
        })
    
    # Create DataFrame and save to CSV
    majority_vote_df = pd.DataFrame(majority_vote_results)
    majority_vote_df = majority_vote_df.sort_values(by='id')
    
    # Save to the same folder as the year files
    majority_vote_path = output_dir / 'majority_vote.csv'
    majority_vote_df.to_csv(majority_vote_path, index=False)
    
    print(f"\n✓ Majority vote file created: {majority_vote_path}")
    print(f"  Total unique catchments: {len(majority_vote_df)}")
    print(f"  Cluster distribution:")
    print(majority_vote_df['cluster'].value_counts().to_string())
    print()


if __name__ == "__main__":
    import sys
    # python clustering_by_year/reformat_cluster.py output/data/clustering_results/clustered_catchments_kmeans_k20.csv
    if len(sys.argv) < 2:
        print("Usage: python reformat_cluster.py <path_to_cluster_file.csv>")
        sys.exit(1)
    
    cluster_file_path = sys.argv[1]
    reformat_cluster_file(cluster_file_path)
