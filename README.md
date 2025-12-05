
# Hydrological Time Series Clustering for Danish Catchments

This repository contains a full workflow for **clustering hydrological discharge time series** from Danish catchments.  

The project explores **three complementary approaches**:

1. **Yearly feature–based clustering**  
   - Extracts time series features with TSFresh  
   - Reduces dimensionality with PCA  
   - Applies clustering (primarily K-means; also DBSCAN)  
2. **Seasonal clustering analysis**  
   - Focuses on intra-annual (seasonal) patterns  
   - Clusters catchments based on seasonal behaviour  
3. **DTW-based clustering**  
   - Uses Dynamic Time Warping (DTW) distances between resampled discharge time series  
   - Applies K-Medoids and DBSCAN on the DTW distance matrix  

The goal is to identify hydrologically similar catchments and link these clusters to **geographic and physical characteristics**.

---

## Project Structure
```
text
02807_Project_Public/
├── .cadence/                      # (Optional) Tooling/CI metadata
├── .venv/                         # Local virtual environment (not tracked in VCS)
├── dtw/
│   ├── dtw.ipynb                  # DTW pipeline: distance matrix + clustering
│   └── dtw_groups.ipynb           # Grouping/visual analysis based on DTW clusters
├── feature extraction/
│   ├── clustering_catchments.py   # Main orchestration: feature-based clustering workflow
│   ├── eval_clusters.py           # Evaluate cluster quality (metrics, summaries)
│   ├── evaluate_pca_kaiser.py     # PCA + Kaiser criterion (eigenvalues > 1)
│   ├── extract_features.py        # TSFresh-based feature extraction from time series
│   ├── pca_extracted_features.py  # PCA on TSFresh features and export of scores
│   ├── plot_pca.py                # Visualization of PCA results (scatterplots, loadings)
│   ├── readme.md                  # (Local) documentation of feature-extraction submodule
│   ├── reformat_cluster.py        # Reformat cluster outputs (e.g., for mapping/analysis)
│   ├── segment_by_year.py         # Split time series into annual segments
│   ├── summarize_eval_clusters.py # Summaries over clustering evaluations
│   └── TSFRESH_FEATURE_CATEGORIES.md # Description of TSFresh feature groups
├── seasonal/
│   ├── season_clustering/         # (Folder for code/results related to seasonal clustering)
│   ├── SeasonalClustering.ipynb   # Main seasonal clustering notebook (by seasons)
│   └── SeasonalVisualClustering.ipynb # Visualization of seasonal clusters
├── yearly/
│   ├── yearly_clustering_dbscan.py     # Yearly-based clustering using DBSCAN
│   ├── yearly_clustering_kmeans.py     # Yearly-based clustering using K-means
│   └── yearly_clustering_plotmerger.py # Merge/compose multiple cluster plots
├── .env_template                  # Template for environment variables (paths, credentials)
├── _config.py                     # Central configuration (paths, constants, parameters)
├── _tools.py                      # Shared helper utilities used across scripts
├── per_catchment_exclude_all.csv      # Catchments to fully exclude from analysis
├── per_catchment_exclude_periods.csv  # Catchment-specific periods to exclude
├── produce_discharge_table.py          # Build long-term discharge table from raw data
├── produce_discharge_table_seasonal.py # Build seasonal discharge data tables
├── pyproject.toml                 # Project dependencies and build configuration
├── README.md                      # Main project documentation (this file)
├── simple_load_and_plot_nc_file.ipynb # Example: load & visualize raw NetCDF
├── uv.lock                        # Dependency lock file (for `uv` / modern pip tools)
└── Visualize_clustering.ipynb     # Map-based/geospatial cluster visualization
```
High-level grouping of approaches:

- **Yearly feature-based workflows**
  - `produce_discharge_table.py`
  - `feature extraction/segment_by_year.py`
  - `feature extraction/extract_features.py`
  - `feature extraction/pca_extracted_features.py`
  - `feature extraction/evaluate_pca_kaiser.py`
  - `feature extraction/clustering_catchments.py`
  - `yearly/yearly_clustering_kmeans.py`
  - `yearly/yearly_clustering_dbscan.py`
  - `yearly/yearly_clustering_plotmerger.py`
  - `feature extraction/eval_clusters.py`, `summarize_eval_clusters.py`, `reformat_cluster.py`
  - `Visualize_clustering.ipynb`

- **Seasonal clustering workflows**
  - `produce_discharge_table_seasonal.py`
  - `seasonal/season_clustering/` (scripts, if any)
  - `seasonal/SeasonalClustering.ipynb`
  - `seasonal/SeasonalVisualClustering.ipynb`

- **DTW-based workflows**
  - `dtw/dtw.ipynb`
  - `dtw/dtw_groups.ipynb`

---

## Installation

### 1. Python version

The project targets **Python 3.12.7**.  
Using a virtual environment is strongly recommended.
```
bash
# Example using venv
python3.12 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```
### 2. Install dependencies

All dependencies are defined in `pyproject.toml`.  
You can install them with your preferred tool, e.g.:
```
bash
# Using pip
pip install .

# or using uv (if available)
uv sync
```
If you prefer to install in editable (development) mode:
```
bash
pip install -e .
```
### 3. Environment configuration (`.env`)

Use the provided `.env_template` as a starting point:
```
bash
cp .env_template .env
```
Edit `.env` to set:

- **Data paths**
  - `RAW_NETCDF_DIR` – directory containing raw hydrological NetCDF files
  - `SHAPEFILE_PATH` – path to catchment shapefile(s) for mapping
  - `STATION_METADATA_CSV` – mapping between station IDs and catchments/coordinates
- **Output paths**
  - `OUTPUT_ROOT` – root folder where computed tables, features, and plots are written
  - Optional subfolders for:  
    - `DISCHARGE_TABLE_DIR`  
    - `FEATURES_DIR`  
    - `PCA_DIR`  
    - `CLUSTERS_DIR`  
    - `FIGURES_DIR`
- Any other paths or secrets required by `_config.py` and `_tools.py`

The scripts read configuration typically via `_config.py`, which in turn can read `.env` variables.

---

## Data Requirements

To run the workflows, the following data are expected:

1. **Discharge time series (NetCDF)**  
   - One or multiple NetCDF files with:
     - Time dimension (e.g. daily discharge values)
     - Station/catchment dimension (e.g. `station_id`, `catchment_id`)
     - Discharge variable (e.g. `q`, `discharge`, `flow`)
   - All workflows assume consistent units (e.g. m³/s) and a regular time grid (or at least resample-able).

2. **Shapefiles for catchments or stations**  
   - Polygon or point shapefile containing:
     - Catchment or station identifiers matching those used in the time series
     - Coordinate reference system appropriate for Denmark
   - Used by visualization notebooks (e.g. `Visualize_clustering.ipynb`, seasonal visualization).

3. **Station / catchment mapping** (CSV or similar)  
   - A table with at least:
     - `station_id` / `catchment_id`
     - Latitude, longitude (if not in shapefile)
     - Optional attributes: basin area, elevation, land use, etc.
   - Used to:
     - Filter and join metadata with clusters
     - Support geographic visualizations and interpretations

4. **Exclusion lists**  
   - `per_catchment_exclude_all.csv`  
     - Complete catchments to exclude (e.g. insufficient data quality)
   - `per_catchment_exclude_periods.csv`  
     - Specific time spans to exclude per catchment (e.g. gaps, anomalies)

---

## Usage Guides: Main Workflows

This section emphasises the **structure and flow** of each approach.

### 1. Yearly Clustering Pipeline (Feature Extraction → PCA → K-means)

**Goal:** Cluster catchments based on **yearly statistical behaviour**, using TSFresh + PCA + clustering.

#### 1.1 Build discharge tables
```
bash
python produce_discharge_table.py
```
- Reads raw NetCDF data (`RAW_NETCDF_DIR`)  
- Produces long-form discharge tables (e.g. `discharge_table_YYYY_YYYY.csv`) in the configured output directory  
- Applies exclusion rules from:
  - `per_catchment_exclude_all.csv`
  - `per_catchment_exclude_periods.csv`

#### 1.2 Segment by year
```
bash
python "feature extraction/segment_by_year.py"
```
- Takes long discharge tables
- Splits each catchment time series into **yearly segments**
- Outputs a structure suitable for TSFresh (e.g. one row per catchment–year)

#### 1.3 Feature extraction (TSFresh)
```
bash
python "feature extraction/extract_features.py"
```
- Uses TSFresh to compute **time-series features** per yearly segment:
  - Basic statistics, quantiles
  - Frequency-domain features
  - Autocorrelation properties
  - And many more (see `TSFRESH_FEATURE_CATEGORIES.md`)
- Outputs feature matrices (e.g. `features_yearly.csv`) in the feature directory

#### 1.4 PCA on features
```
bash
python "feature extraction/pca_extracted_features.py"
python "feature extraction/evaluate_pca_kaiser.py"
```
- Standardizes features and runs PCA
- `evaluate_pca_kaiser.py` evaluates **number of components** to retain (e.g. eigenvalues > 1, explained variance)
- Outputs:
  - PCA scores per catchment
  - Diagnostics (explained variance, scree plots, eigenvalue tables)

#### 1.5 Clustering (K-means / DBSCAN) and orchestration

There are two levels of structure:

**(a) Feature-extraction module orchestration**
```
bash
python "feature extraction/clustering_catchments.py"
```
- Wraps together:
  - Reading PCA features
  - Running one or more clustering algorithms
  - Saving cluster labels (e.g. catchment → cluster ID)
- Uses helper scripts:
  - `eval_clusters.py` – compute metrics (e.g. silhouette, Davies–Bouldin)
  - `summarize_eval_clusters.py` – summarise results across multiple configurations
  - `reformat_cluster.py` – reformat labels to join with shapefiles/metadata

**(b) Yearly-specific cluster scripts**
```
bash
python "yearly/yearly_clustering_kmeans.py"
python "yearly/yearly_clustering_dbscan.py"
```
- Implement specific configurations for:
  - **K-means** clustering on PCA scores
  - **DBSCAN** clustering on distances or feature spaces
- Save:
  - Cluster labels (e.g. CSV)
  - Possibly intermediate metrics for evaluation

The structure allows you to:
- Experiment with different `k` for K-means
- Compare DBSCAN parameter combinations (`eps`, `min_samples`)

#### 1.6 Plotting & visualization
```
bash
python "yearly/yearly_clustering_plotmerger.py"
```
- Collects plots from various clustering runs and merges them into combined figures (e.g. multi-page PDFs or side-by-side PNGs)
```
bash
jupyter notebook Visualize_clustering.ipynb
```
- Maps cluster labels onto geographic locations
- Produces:
  - Cluster maps over Denmark
  - Possibly cluster “centroids” or typical hydrographs

---

### 2. Seasonal Clustering Analysis

**Goal:** Understand **intra-annual/seasonal patterns** of discharge and cluster catchments accordingly.

#### 2.1 Build seasonal discharge tables
```
bash
python produce_discharge_table_seasonal.py
```
- Aggregates or segments discharge into **seasonal units** (e.g. DJF, MAM, JJA, SON, or custom seasonal definitions)
- Produces seasonal discharge tables in the configured output directory

#### 2.2 Seasonal clustering

Open:

- `seasonal/SeasonalClustering.ipynb`

Typical steps within the notebook:

1. Load seasonal discharge tables  
2. Optionally standardize within seasons  
3. Compute derived seasonal features (e.g. mean seasonal discharge, low-flow indices)  
4. Apply clustering (often similar to yearly: K-means, DBSCAN, etc.)  
5. Evaluate clusters using silhouette and other indices  

Any additional scripts under `seasonal/season_clustering/` support:

- Scripted versions of seasonal clustering
- Batch execution over multiple seasonal definitions

#### 2.3 Seasonal visualization

Open:

- `seasonal/SeasonalVisualClustering.ipynb`

This notebook:

- Loads seasonal cluster labels
- Joins with shapefile and/or metadata
- Produces:
  - Seasonal cluster maps
  - Seasonal hydrograph plots grouped by cluster

---

### 3. DTW (Dynamic Time Warping) Analysis

**Goal:** Cluster catchments based on **shape similarity** of discharge time series, independent of simple time shifts or scaling.

Work primarily happens in:

- `dtw/dtw.ipynb`
- `dtw/dtw_groups.ipynb`

#### 3.1 Preprocessing and resampling

Inside `dtw.ipynb`:

1. Load a discharge table (e.g. multi-year daily series for each catchment)
2. For each catchment:
   - Drop missing values
   - Resample each time series to a **fixed length** (e.g. 500 time steps) using interpolation
3. Normalize series (typically by mean and standard deviation) so that clustering focuses on **shape** rather than absolute magnitude

#### 3.2 DTW distance matrix

- Compute a **pairwise DTW distance matrix** between all catchment series:
  - Optionally using fast implementations and parallelization
- Convert the compact DTW result into a square distance matrix
- Visualize a subset (e.g. 100 series) via a heatmap to inspect patterns

#### 3.3 Clustering on DTW distances

Two main clustering structures:

1. **K-Medoids (K-Medoids on DTW distances)**
   - Explore a range of `k` (e.g. 2–10)
   - For each `k`:
     - Fit K-Medoids using the **precomputed DTW distance matrix**
     - Compute:
       - Silhouette score (higher is better)
       - Davies–Bouldin Index (lower is better)
       - A distortion / elbow-like measure based on sum of distances to medoids
     - Store and analyse metrics to select suitable `k`
   - Save cluster assignments (e.g. one CSV per chosen `k`) for further visualization:
     - `catchment_id`, `cluster`

2. **DBSCAN on DTW distances**
   - Sweep across `eps` values (and optionally `min_samples`)
   - For each combination:
     - Fit DBSCAN with `metric='precomputed'`
     - Compute:
       - Number of clusters (excluding noise)
       - Number of noise points
       - Silhouette score and Davies–Bouldin Index (where applicable)
   - Choose optimal `eps`:
     - Maximize silhouette score
     - Minimize Davies–Bouldin Index
   - Save DBSCAN labels, e.g.:
     - `catchment_id`, `cluster` (with `-1` denoting noise)

#### 3.4 Visual analysis (2D embeddings & time series)

Within `dtw.ipynb` and `dtw_groups.ipynb`:

- Apply **t-SNE** to the DTW distance matrix to obtain a 2D embedding
- Colour points by DTW-based clusters (K-Medoids or DBSCAN)
- For each cluster:
  - Plot overlaid resampled discharge series to inspect typical shapes
- Save cluster labels for integration with spatial visualization notebooks

---

## Pipeline Details: Script Purposes

Below is a concise reference for major scripts and how they fit into the three approaches.

### Root scripts

- **`produce_discharge_table.py`**  
  - Build long-format discharge tables from NetCDF  
  - Handles filtering and exclusion lists  
  - Used by **yearly** and **feature-based** workflows

- **`produce_discharge_table_seasonal.py`**  
  - Same logic but for **seasonal aggregations**  
  - First step in the **seasonal clustering** workflow

- **`_config.py`**  
  - Central location of:
    - Paths to data and outputs
    - Global constants (e.g. date ranges, variable names)
  - Used in most Python scripts

- **`_tools.py`**  
  - Reusable utilities (I/O, plotting, helper math functions, etc.)  
  - Reduces duplication across scripts and notebooks

### Feature-extraction module (`feature extraction/`)

- **`segment_by_year.py`**  
  - Slices discharge series into **yearly chunks**.

- **`extract_features.py`**  
  - Uses TSFresh to compute large sets of time-series features per catchment–year.

- **`pca_extracted_features.py`**  
  - Standardizes feature matrices and runs PCA; outputs loadings and scores.

- **`evaluate_pca_kaiser.py`**  
  - Evaluates PCA output using the **Kaiser criterion** (and other heuristics).

- **`clustering_catchments.py`**  
  - Orchestrates clustering on PCA scores (e.g. multiple k, algorithms).

- **`eval_clusters.py`**  
  - Computes cluster validity indices (silhouette, Davies–Bouldin, etc.).

- **`summarize_eval_clusters.py`**  
  - Aggregates results from many clustering runs for easy comparison.

- **`reformat_cluster.py`**  
  - Cleans/reformats cluster label files for joining with shapefiles or metadata.

- **`plot_pca.py`**  
  - Produces PCA diagnostic and visual plots (scores, loadings, scree, etc.).

- **`TSFRESH_FEATURE_CATEGORIES.md`**  
  - Human-readable documentation of TSFresh feature groups used in the analysis.

### Yearly clustering scripts (`yearly/`)

- **`yearly_clustering_kmeans.py`**  
  - Implements K-means-based clustering on yearly PCA scores or features.  
  - Likely includes loops over potential ks for cluster selection.

- **`yearly_clustering_dbscan.py`**  
  - Implements DBSCAN on yearly features or distance matrices.

- **`yearly_clustering_plotmerger.py`**  
  - Combines cluster plots from multiple runs into consolidated figures.

### Seasonal notebooks and code (`seasonal/`)

- **`seasonal/SeasonalClustering.ipynb`**  
  - End-to-end seasonal clustering: load seasonal data → feature computation → clustering.

- **`seasonal/SeasonalVisualClustering.ipynb`**  
  - Visualizes seasonal clusters in geographic space.

- **`seasonal/season_clustering/`**  
  - Container for modular seasonal clustering scripts and outputs.

### DTW notebooks (`dtw/`)

- **`dtw/dtw.ipynb`**  
  - Complete DTW pipeline:
    - Resample or normalize time series
    - Compute DTW distance matrix
    - K-Medoids and DBSCAN on DTW distances
    - Evaluation metrics, t-SNE visualization
    - Export of cluster labels as CSV

- **`dtw/dtw_groups.ipynb`**  
  - Additional analyses and plotting of DTW-based clusters:
    - Group statistics, hydrograph plots
    - Potential linking to geography/metadata

### Visualization notebooks

- **`simple_load_and_plot_nc_file.ipynb`**  
  - Minimal example of loading NetCDF data and plotting raw discharge.

- **`Visualize_clustering.ipynb`**  
  - Spatial visualization of cluster results (maps, legends, overlays).

---

## Output Structure

The exact folder names depend on `_config.py` and `.env`, but a typical layout is:
```
text
output/
├── data/
│   ├── discharge_tables/
│   │   ├── discharge_table_YYYY_YYYY.csv
│   │   └── discharge_table_seasonal_*.csv
│   ├── features/
│   │   ├── tsfresh_features_yearly.csv
│   │   └── tsfresh_features_seasonal.csv
│   └── pca/
│       ├── pca_scores_yearly.csv
│       ├── pca_loadings_yearly.csv
│       └── pca_eigenvalues_yearly.csv
├── clusters/
│   ├── yearly_kmeans/
│   │   ├── labels_k=<k>.csv
│   │   └── metrics_k=<k>.csv
│   ├── yearly_dbscan/
│   │   └── labels_dbscan_*.csv
│   ├── seasonal/
│   │   └── labels_seasonal_*.csv
│   └── dtw/
│       ├── kmedoids_labels_k=<k>.csv
│       └── dbscan_labels_*.csv
└── figures/
    ├── pca/
    ├── yearly_clustering/
    ├── seasonal_clustering/
    └── dtw/
```
Check `_config.py` and `.env` to confirm and/or adapt the actual output locations.

---

## Key Features

1. **Multiple, structured approaches to time-series clustering**
   - **Feature-based yearly clustering** using TSFresh + PCA + K-means/DBSCAN
   - **Seasonal clustering** capturing intra-annual behaviour
   - **DTW-based shape clustering** robust to time shifts and non-linear alignment

2. **Rich feature extraction (TSFresh)**  
   - Hundreds of standard and advanced time-series features  
   - Feature groups documented in `TSFRESH_FEATURE_CATEGORIES.md`  
   - Easily extendable or filterable feature sets

3. **Dimensionality reduction with PCA**
   - PCA used to combat feature redundancy and noise
   - Kaiser criterion to select an interpretable number of components
   - Visualization of PCs for interpretability (e.g. which features drive clusters)

4. **Flexible clustering methods**
   - **K-means**:
     - Simple, well-known baseline on PCA scores
   - **DBSCAN**:
     - Density-based, identifies noise and outliers
   - **K-Medoids on DTW distances**:
     - Prototype-based clustering using actual series as “medoids”
   - Support for exploring parameter grids and evaluation metrics

5. **Comprehensive evaluation and visualization**
   - Silhouette scores, Davies–Bouldin Index, elbow metrics
   - PCA plots, t-SNE embeddings, cluster hydrograph plots
   - Geographic visualizations linking hydrological clusters to spatial patterns

6. **Configurable and modular design**
   - Centralised configuration in `_config.py` and `.env`
   - Separate modules for:
     - Data preparation
     - Feature extraction
     - Dimensionality reduction
     - Clustering and evaluation
     - Visualization
   - Easy to plug in new algorithms or additional data sources

---

## Getting Started

1. **Set up environment**  
   - Create and activate a Python 3.12.7 environment  
   - Install dependencies via `pip install .` or `uv sync`  
   - Create `.env` from `.env_template` and configure all paths

2. **Prepare data**  
   - Place NetCDF files, shapefiles, and station metadata as configured in `.env`  
   - Ensure exclusion CSVs reflect your quality-control decisions

3. **Run a pipeline**  
   - For yearly feature-based clustering:
     - `produce_discharge_table.py`  
     - `feature extraction/segment_by_year.py`  
     - `feature extraction/extract_features.py`  
     - `feature extraction/pca_extracted_features.py`  
     - `feature extraction/evaluate_pca_kaiser.py`  
     - `yearly/yearly_clustering_kmeans.py` (and/or `yearly_clustering_dbscan.py`)  
     - Visualize via `Visualize_clustering.ipynb`
   - For seasonal analysis:
     - `produce_discharge_table_seasonal.py`  
     - `seasonal/SeasonalClustering.ipynb`  
     - `seasonal/SeasonalVisualClustering.ipynb`
   - For DTW analysis:
     - `dtw/dtw.ipynb`  
     - `dtw/dtw_groups.ipynb`  

4. **Customize and extend**  
   - Adjust feature sets, PCA thresholds, clustering parameters  
   - Add new indices or extra visualization notebooks as needed

