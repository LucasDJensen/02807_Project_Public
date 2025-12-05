
# Order to run scripts in!
1 segment_by_year
2 extract_features:
  2.1 "clustering_by_year\evaluate_pca_kaiser.py"  evaluate number of PCs
3 pca_extracted_features
3.1 (optional) plot_pca.py 
  python clustering_by_year/plot_pca.py
4 clustering_catchments
  4.1 "reformat_clusters" if using Visualize_clustering.ipynb! 
5 eval_clusters:
  5.1 summarize_eval_clusters.py

## How to run the clustering pipeline

- **Input assumptions**  
  - `segment_by_year` produces `output/data/discharge_tables/discharge_tables_by_year.csv`.  
  - `extract_features` consumes that file and writes feature tables to `output/data/extracted_features/extracted_features.csv`.  
  - `pca_extracted_features` reads the feature CSV and writes PCA embeddings to `output/data/extracted_features/pca_embeddings.csv`.

- **4. clustering_catchments**  
  - Script: `clustering_by_year/clustering_catchments.py`  
  - Default input: `output/data/extracted_features/pca_embeddings.csv`  
  - Output directory (created if needed): `output/data/clustering_results/`  
  - Runs:
    - K‑Means with `k = 3, 8, 20` → CSVs `clustered_catchments_kmeans_k3.csv`, `..._k8.csv`, `..._k20.csv`  
    - The first K‑Means result (k=3 by default) is also saved as `clustered_catchments.csv`.  
    - DBSCAN with `(eps=0.7, min_samples=5)` → `clustered_catchments_dbscan.csv`.  
  - Example:
    - `python clustering_by_year/clustering_catchments.py`

- **5. eval_clusters**  
  - Script: `clustering_by_year/eval_clusters.py`  
  - Default clustering input: `output/data/clustering_results/clustered_catchments.csv`  
  - Default embeddings input: `output/data/extracted_features/pca_embeddings.csv`  
  - Output directory (created if needed): `output/data/eval_clusters/`  

## Script reference & CLI cheat sheet

### 1. `segment_by_year.py`
- **Goal**: Split `discharge_table_2001_2022.csv` into `id_year` series, plot datapoint distribution (log y), and drop segments with ≤364 valid samples.
- **Key arguments**:
  - `--input-path` (`DEFAULT_INPUT_PATH`): raw discharge table.
  - `--output-path` (`DEFAULT_OUTPUT_PATH`): segmented CSV written to `output/data/discharge_tables/discharge_tables_by_year.csv`.
  - `--plot-path` (`DEFAULT_PLOT_PATH`): distribution plot under `output/data/plots_dataprocessing/`.
- **Example**:  
  `python clustering_by_year/segment_by_year.py --input-path output/data/discharge_tables/discharge_table_2001_2022.csv`

### 2. `extract_features.py`
- **Goal**: Run tsfresh on the segmented table and save features/plots.
- **Main hyperparameters**:
  - `--feature-mode {minimal|efficient|comprehensive|custom}` (default `efficient`; see in-code comments for feature counts).
  - `--feature-config path/to/features.json` (required when `--feature-mode custom`).
  - Output paths: `--feature-output`, `--hist-output`, `--quantile-output`, `--covariance-output`.
- **Example**:  
  `python clustering_by_year/extract_features.py --feature-mode efficient`
- **Feature Categories**: See `TSFRESH_FEATURE_CATEGORIES.md` for an overview of the types of features tsfresh extracts (statistical, temporal, frequency domain, entropy, etc.).

### 3. `pca_extracted_features.py`
- **Goal**: Optional dimensionality reduction + diagnostics.
- **Key hyperparameters**:
  - `--n-components` (default **10**) → controls embedding dimensionality.
  - `--n-components-for-variance-plot` (default 100) → explains more variance than saved PCs if needed.
  - `--top-features-per-pc` / `--top-pcs-to-analyze` → control feature-importance plots.
- **Example (standard run)**:  
  `python clustering_by_year/pca_extracted_features.py --n-components 10`

### 4. `clustering_catchments.py`
- **Goal**: Cluster either PCA embeddings *or* the raw tsfresh feature table.
- **Inputs**:
  - PCA path (`--input-path`, default `output/data/extracted_features/pca_embeddings.csv`).
  - Raw feature path (`--raw-input-path`, default `output/data/extracted_features/extracted_features.csv`).
- **Switching between PCA and raw**:
  - **With PCA (default):** `python clustering_by_year/clustering_catchments.py`
  - **Without PCA:** `python clustering_by_year/clustering_catchments.py --use-raw-features`
    - Raw features are standardized internally before clustering.
- **Other knobs**:
  - `--kmeans-k 3 8 20` (list of cluster counts).
  - `--no-dbscan` to skip DBSCAN; adjust `DBSCAN_EPS` / `DBSCAN_MIN_SAMPLES` inside the script.

### 5. `eval_clusters.py`
- **Goal**: Score cluster assignments, compare to random baseline, and render silhouette distributions.
- **Inputs**:
  - Cluster CSV directory (`--clusters-dir`, default `output/data/clustering_results/`).
  - Feature matrix for silhouette calculations:
    - **PCA (default):** `--embeddings-path output/data/extracted_features/pca_embeddings.csv`
    - **Raw features:** add `--use-raw-features` (optionally override `--raw-features-path`). Raw features are standardized automatically.
- **Example commands**:
  - With PCA: `python clustering_by_year/eval_clusters.py`
  - Without PCA: `python clustering_by_year/eval_clusters.py --use-raw-features`
- **Other parameters**:
  - `--n-bootstrap` controls the "vs random" baseline iterations.
  - Outputs land in `output/data/eval_clusters/` (text reports + silhouette plots).

## PCA Visualization: `plot_pca.py`

### Overview
The `plot_pca.py` script creates all-against-all pairwise scatter plots of PCA features. **For each set of features, it generates two plots**: one colored by catchment ID (without the year suffix) and one colored by year. This allows you to visually inspect how different catchments cluster in the PCA space and how temporal patterns emerge across years. The script uses high-contrast color schemes for better visibility.

### How it works
1. **Loads PCA embeddings** from the input CSV file (default: `output/data/extracted_features/pca_embeddings.csv`).
2. **Extracts identifiers**:
   - Catchment IDs by removing the year suffix (e.g., `12000001_2012` → `12000001`)
   - Years by extracting the year suffix (e.g., `12000001_2012` → `2012`)
3. **Generates two color schemes with high contrast**:
   - **By catchment ID**: All years of the same catchment share the same color
   - **By year**: All samples from the same year share the same color
   - Uses `Set3` colormap (up to 12 colors), `tab20` (up to 20 colors), or a custom high-contrast combination for more colors
4. **Creates scatter plots** for all pairwise combinations of the selected principal components.
5. **Limits datapoints** to the first N samples (default: 200) for readability.
6. **Saves both versions** with appropriate filenames (`_by_catch_id.png` and `_by_year.png`).

### Key Hyperparameters
- `--n-features` (default: **10**): Number of principal components to plot (PC1, PC2, ..., PCN).
- `--n-datapoints` (default: **200**): Number of datapoints to include in each plot (uses first N rows).
- `--features-per-plot` (default: **5**): When using `--plot-all-features`, controls how many features are included per plot.

### Usage Examples

**Basic usage (first 10 PCs, first 200 datapoints):**
```bash
python clustering_by_year/plot_pca.py
```

**Custom number of features and datapoints:**
```bash
python clustering_by_year/plot_pca.py --n-features 10 --n-datapoints 500
```

**Plot all features in batches:**
```bash
python clustering_by_year/plot_pca.py --plot-all-features
```
This creates multiple plots, each containing 5 consecutive features (PC1-PC5, PC6-PC10, etc.) until all features are plotted.

**Custom batch size for all-features mode:**
```bash
python clustering_by_year/plot_pca.py --plot-all-features --features-per-plot 3
```

**Custom input/output paths:**
```bash
python clustering_by_year/plot_pca.py \
  --input-path output/data/extracted_features/pca_embeddings.csv \
  --output-dir output/data/extracted_features/pca_plots
```

### Output
- **Default mode**: Two plot files for each feature set:
  - `pca_scatter_PC1_to_PC10_by_catch_id.png` - colored by catchment ID
  - `pca_scatter_PC1_to_PC10_by_year.png` - colored by year
- **All-features mode**: Multiple plot pairs, each covering consecutive feature ranges:
  - `pca_scatter_PC1_to_PC5_by_catch_id.png` and `pca_scatter_PC1_to_PC5_by_year.png`
  - `pca_scatter_PC6_to_PC10_by_catch_id.png` and `pca_scatter_PC6_to_PC10_by_year.png`
  - And so on...

All plots are saved to `output/data/extracted_features/pca_plots/` (or custom `--output-dir`).

### Color Coding
- **By catchment ID** (`_by_catch_id.png`):
  - Each unique catchment ID (e.g., `12000001`) gets a unique color with high contrast.
  - All years of the same catchment (e.g., `12000001_2012`, `12000001_2013`) share the same color.
  - Useful for identifying catchment-specific patterns and temporal consistency.
  - Uses high-contrast colormaps (`Set3`, `tab20`, or custom combination) for better visibility.
- **By year** (`_by_year.png`):
  - Each unique year (e.g., `2012`, `2013`) gets a unique color with high contrast.
  - All samples from the same year share the same color, regardless of catchment.
  - Useful for identifying temporal trends and year-specific patterns in the PCA space.
  - Uses high-contrast colormaps for better visibility.






# Explaining process of clustering

How to create the section. 
3 Include relevant figures. Only do 4 figures in total!

Write this all to "report.md" a new file!

Write the equivalent of around 1 A4 page, excluding the figures. 

Follow the following structure relatively tightly, but add more explanation, especially as it relates to the figures. Every figure should be understandable given the code is included in the scripts. 

Be relatively concise, and dont try to add words via fluff. The purpose is to describe how the pipeline works, justify our decisions, explain figures etc. 

"pre-processing":
    - Features such as count_above_mean, count_below_mean and many others are affected by number of datapoints in the time series. 
    Hence we decided that choosing a harsh cutoff (only including timeseries with 365 or 366 elements) would make for a clearer picture!
    - The proportion of the dataset for which this is the case is 92.46% The number of datapoints per year ranges from 245 to 383. 
    - PCA: From the 777 features produces by tsfresh, we used 80 PCs as estimated using the kaiser criterion. This explains a total of 88% of total variance (ie had eigenvalue > 1).
    - Write a small overview about the nature of these 777 feartures, see tshresh for reference!
   - Include a table of how many where removed, and the number of datapoints and the number of datapoints pertaining to each year!
   - show the plot output\data\extracted_features\pca_explained_variance.png

"method": 
    - In the preliminary stages we tried to use dbscan, and heirrarchical clustering techniques such as agglomorative herirarchical clustering with average ward linkage. However these often only produced a single cluster so we decided to use k-means instead.
    - We switched to the k-means method and tried KMEANS_K_VALUES = [3,4,5,7,10,13,16,20,25,30,40,50,75,100]
    - Note, we cluster ALL data together, meaning all datapoint belonging to every year are mixed into the combiend dataset.
  
"evaluation of cluster quality":
    - We evaluated the k-means clusters in 2 different ways
    1 Silhouette score:
      We computed the silhoutte score and plotted the silhoutte plot for every k-means result
      This is our primary method of choosing k = 20, as the method, ie. lowst sillhoutte score!
    2 Catchment cluster consistency:
      Given we had an implicitly heirarchical dataset, given multiple datapoints pertains to the same catchment, we could use the fact to evaluate the quality of the clsutering method.
      Specifically, we assume that the same catchment should be significantly more likely to be clustered with itself in a different year, then a random other catchment.
      We therefore compared the number of unique clusters assigned to each group of datapoints pertaining to the same catchment id with a random boostrapped dataset.
      This random boostrapped dataset had the same size / distribution of clusters, but each datapoint is randomly assigned.
  - Include the silhoutte scores plot!  output\data\eval_clusters\summary_results\silhouette_scores_line.png
  - Also show the number of significantly sized clusters! 
  output\data\eval_clusters\summary_results\significant_clusters.png

#4 After producing the plots we can visualize the results by overlaying the cluster assignments ontop of a map of denmark
  - describe how there is generel geographic consistency indicating succesfull clustering.
  - Coastal regions are disproportionaltely clustered seperately from more inlands regions
  - Show the majority vote plot output\data\clustering_results\clustered_catchments_kmeans_k20_by_year\plot\majority_vote.png
    Datapoints with >10 are grouped into the same cluster!
    Cluster sizes are highly bimodal, in that they are either very significant in size or not at all, like 1-5 datapoints. See here: output\data\eval_clusters\clustered_catchments_kmeans_k20_evaluation.txt
    Describe this.
  - Also describe the results and write into a table the VS RANDOM (BOOTSTRAPPED) section for our chosen k = 20!


